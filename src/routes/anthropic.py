"""Anthropic Messages API routes."""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..adapters.anthropic_adapter import AnthropicAdapter
from ..middleware.auth import AuthResult, verify_api_key
from ..models.anthropic import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    InputJsonDelta,
    Message,
    MessageDelta,
    MessageStart,
    MessageStop,
    MessagesRequest,
    MessagesResponse,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
)
from ..services.claude import ClaudeService
from ..services.session import SessionManager
from ..services.tool_proxy import (
    ToolProxy,
    create_tool_proxy,
    get_proxy,
    register_proxy,
    remove_proxy,
)
from ..utils.debug_logger import (
    log_incoming_request,
    log_outgoing_response,
    log_stream_chunk,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Anthropic"])

# Global service instances (will be set by main.py)
claude_service: Optional[ClaudeService] = None
session_manager: Optional[SessionManager] = None


def init_services(claude: ClaudeService, sessions: SessionManager) -> None:
    """Initialize service instances."""
    global claude_service, session_manager
    claude_service = claude
    session_manager = sessions


@dataclass
class MessagesContext:
    """Shared context for messages processing."""

    request: MessagesRequest
    request_id: str
    prompt: str  # Text prompt (images converted to [Image: path] references)
    system_prompt: Any  # str, SystemPromptPreset dict, or None
    claude_session_id: Optional[str]
    mcp_servers_dict: Optional[Dict[str, Any]]
    session_id: Optional[str] = None  # Determined session ID (via hash or explicit)
    messages_for_hash: Optional[List[Dict[str, Any]]] = None  # Original messages for hash storage
    external_tools: Optional[List[Dict[str, Any]]] = None  # Client-provided tool definitions


async def _build_messages_context(
    request: MessagesRequest,
    auth: AuthResult,
) -> MessagesContext:
    """Build shared context for messages processing."""
    request_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Convert to Claude format (files saved to cache in cwd)
    cwd = claude_service.cwd if claude_service else None

    # Convert Anthropic messages to internal format for prompt
    internal_messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            internal_messages.append({"role": msg.role, "content": msg.content})
        else:
            # Convert content blocks to dict format
            content_blocks = []
            for block in msg.content:
                if hasattr(block, "model_dump"):
                    content_blocks.append(block.model_dump())
                elif isinstance(block, dict):
                    content_blocks.append(block)
            internal_messages.append({"role": msg.role, "content": content_blocks})

    # Session identification via hash-based lookup only
    # External session_id (e.g., from Anthropic API) cannot be used for resumption
    session_id, messages_to_send = session_manager.find_session_and_extract_new(internal_messages)
    # - If hash matched: messages_to_send = [last_user_message], SDK resumes context
    # - If hash failed: messages_to_send = full history, new conversation

    # Get Claude session ID for SDK resume (only if hash matched)
    claude_session_id = None
    if session_id:
        claude_session_id = session_manager.get_claude_session(session_id)
        logger.debug(f"Session {session_id}: claude_session={claude_session_id}, sending {len(messages_to_send)} msgs")

    # Generate new session_id if needed (for new conversations)
    if not session_id:
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        logger.debug(f"Created new session: {session_id}")

    # Extract system prompt (handle both string and array formats)
    system_text = None
    if request.system:
        if isinstance(request.system, str):
            system_text = request.system
        elif isinstance(request.system, list):
            # Array of SystemBlock objects
            texts = []
            for block in request.system:
                if hasattr(block, "text"):
                    texts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            system_text = "\n".join(texts)

    prompt, system_prompt = await AnthropicAdapter.to_claude_prompt(
        [Message(**m) for m in messages_to_send],
        cwd,
        system_text
    )

    # Convert MCP servers to dict format
    mcp_servers_dict = None
    if request.mcp_servers:
        mcp_servers_dict = {
            name: config.model_dump(exclude_none=True)
            for name, config in request.mcp_servers.items()
        }

    # Extract external tool definitions for proxy support
    external_tools = None
    if request.tools:
        external_tools = [t.model_dump() for t in request.tools]
        logger.info(f"External tools: {[t['name'] for t in external_tools]}")

    logger.debug(f"Request {request_id}: stream={request.stream}, model={request.model}")

    return MessagesContext(
        request=request,
        request_id=request_id,
        prompt=prompt,
        system_prompt=system_prompt,
        claude_session_id=claude_session_id,
        mcp_servers_dict=mcp_servers_dict,
        session_id=session_id,
        messages_for_hash=internal_messages,  # Store for hash mapping after response
        external_tools=external_tools,
    )


def _extract_tool_results(
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Extract tool_result blocks from the last user message.

    Returns dict mapping tool_use_id → {"content": ..., "is_error": ...}
    or None if no tool results found.
    """
    if not messages:
        return None

    # Look at the last user message
    last_msg = messages[-1] if messages else None
    if not last_msg or last_msg.get("role") != "user":
        return None

    content = last_msg.get("content")
    if not isinstance(content, list):
        return None

    results = {}
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if tool_use_id:
                # Extract content - can be string, list, or None
                block_content = block.get("content", "")
                is_error = block.get("is_error", False)
                results[tool_use_id] = {"content": block_content, "is_error": is_error}

    return results if results else None


def _build_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """Build system prompt instructions for external tools."""
    lines = [
        "\n\n# External Tools",
        "You have access to the following external tools via MCP. Use them to help answer the user's query.",
        "",
    ]
    for tool in tools:
        desc = tool.get("description", "No description")
        lines.append(f"- {tool['name']}: {desc}")

    return "\n".join(lines)


def _apply_tool_prompt(ctx: MessagesContext) -> None:
    """Modify ctx.system_prompt to include external tool instructions.

    If no custom system prompt exists, uses SystemPromptPreset with append
    to preserve Claude Code's default system prompt.
    """
    if not ctx.external_tools:
        return

    tool_prompt = _build_tool_prompt(ctx.external_tools)

    if ctx.system_prompt:
        # Append to existing custom system prompt
        ctx.system_prompt = ctx.system_prompt + tool_prompt
    else:
        # Preserve Claude Code's default prompt, append tool instructions
        ctx.system_prompt = {
            "type": "preset",
            "preset": "claude_code",
            "append": tool_prompt,
        }


def _sse_message_start(request_id: str, model: str) -> str:
    """Format message_start SSE event."""
    msg = MessageStart(
        message={
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    )
    return f"event: message_start\ndata: {msg.model_dump_json()}\n\n"


def _sse_text_block_start(index: int) -> str:
    """Format content_block_start SSE event for a text block."""
    block_start = ContentBlockStart(index=index, content_block=TextBlock(text=""))
    return f"event: content_block_start\ndata: {block_start.model_dump_json()}\n\n"


def _sse_text_delta(index: int, text: str) -> str:
    """Format content_block_delta SSE event for text."""
    delta = ContentBlockDelta(index=index, delta=TextDelta(text=text))
    return f"event: content_block_delta\ndata: {delta.model_dump_json()}\n\n"


def _sse_tool_use_start(index: int, tool_use_id: str, name: str) -> str:
    """Format content_block_start SSE event for a tool_use block."""
    block_start = ContentBlockStart(
        index=index,
        content_block=ToolUseBlock(id=tool_use_id, name=name, input={}),
    )
    return f"event: content_block_start\ndata: {block_start.model_dump_json()}\n\n"


def _sse_input_json_delta(index: int, input_dict: dict) -> str:
    """Format content_block_delta SSE event for tool input JSON."""
    partial_json = json.dumps(input_dict, ensure_ascii=False)
    delta = ContentBlockDelta(
        index=index,
        delta=InputJsonDelta(partial_json=partial_json),
    )
    return f"event: content_block_delta\ndata: {delta.model_dump_json()}\n\n"


def _sse_block_stop(index: int) -> str:
    """Format content_block_stop SSE event."""
    stop = ContentBlockStop(index=index)
    return f"event: content_block_stop\ndata: {stop.model_dump_json()}\n\n"


def _sse_message_delta(stop_reason: str, usage_data: Dict[str, int]) -> str:
    """Format message_delta SSE event."""
    delta = MessageDelta(
        delta={"stop_reason": stop_reason},
        usage=Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
        ),
    )
    return f"event: message_delta\ndata: {delta.model_dump_json()}\n\n"


def _sse_message_stop() -> str:
    """Format message_stop SSE event."""
    stop = MessageStop()
    return f"event: message_stop\ndata: {stop.model_dump_json()}\n\n"


@router.post("/messages", response_model=MessagesResponse)
async def anthropic_messages(
    request: MessagesRequest,
    req: Request,
    auth: AuthResult = Depends(verify_api_key),
):
    """
    Anthropic Messages API compatible endpoint.

    Tool availability is configured via TOOLS env var (see config.py).
    """
    if not claude_service or not session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    ctx = await _build_messages_context(request, auth)

    # Log incoming request
    log_incoming_request(
        request_id=ctx.request_id,
        method="POST",
        path="/v1/messages",
        headers=dict(req.headers) if req.headers else None,
        body=request.model_dump(exclude_none=True),
    )

    # 1. Check for tool_result resume flow
    if ctx.session_id:
        proxy = get_proxy(ctx.session_id)
        if proxy and proxy.has_pending_calls:
            tool_results = _extract_tool_results(
                [m.model_dump() if hasattr(m, "model_dump") else m
                 for m in request.messages]
            )
            if tool_results:
                logger.info(
                    f"Resuming tool proxy for session {ctx.session_id} "
                    f"with {len(tool_results)} results"
                )
                if request.stream:
                    return StreamingResponse(
                        generate_tool_resume_stream(ctx, proxy, tool_results),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )
                return await generate_tool_proxy_response(
                    ctx, proxy=proxy, tool_results=tool_results
                )

    # 2. Check for new external tools
    if ctx.external_tools:
        if request.stream:
            return StreamingResponse(
                generate_tool_proxy_stream(ctx),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return await generate_tool_proxy_response(ctx)

    # 3. Existing flow (no external tools)
    if request.stream:
        return StreamingResponse(
            generate_anthropic_stream(ctx),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return await generate_anthropic_response(ctx)


def _get_delta_content(filtered: str, accumulated: str) -> str:
    """Calculate new content delta to avoid duplicates."""
    if filtered.startswith(accumulated):
        return filtered[len(accumulated) :]
    if accumulated and (filtered == accumulated or accumulated in filtered):
        return ""
    return filtered


def _extract_text(content: Any) -> Optional[str]:
    """Extract text from content (string or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return AnthropicAdapter.extract_text_from_content(content)
    return None


def _extract_content(chunk: dict) -> Optional[str]:
    """Extract text content from a message chunk."""
    # StreamEvent with content_block_delta/start
    event = chunk.get("event")
    if isinstance(event, dict):
        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                return delta.get("text")
        elif event_type == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "text":
                return block.get("text")

    # Direct content field
    if "content" in chunk:
        return _extract_text(chunk["content"])

    # Nested message.content
    message = chunk.get("message")
    if isinstance(message, str):
        return message
    if isinstance(message, dict) and "content" in message:
        return _extract_text(message["content"])

    # Direct text field
    return chunk.get("text")


async def generate_anthropic_stream(ctx: MessagesContext):
    """Generate Anthropic SSE streaming response."""
    messages_buffer = []
    accumulated_content = ""
    content_started = False
    request = ctx.request
    chunk_index = 0

    # Log streaming response start
    log_outgoing_response(
        request_id=ctx.request_id,
        status_code=200,
        is_stream=True,
    )

    try:
        # Send message_start event
        yield _sse_message_start(ctx.request_id, request.model)

        async for chunk in claude_service.run_completion(
            prompt=ctx.prompt,
            system_prompt=ctx.system_prompt,
            model=request.model,

            session_id=ctx.claude_session_id,
            mcp_servers=ctx.mcp_servers_dict,
            stream=True,

        ):
            messages_buffer.append(chunk)

            # Extract and store Claude session ID for resumption
            if chunk.get("subtype") == "init" and ctx.session_id:
                new_session_id = claude_service.get_session_id([chunk])
                if new_session_id:
                    session_manager.set_claude_session(ctx.session_id, new_session_id)

            # Extract content
            content = _extract_content(chunk)
            if not content:
                continue

            # Filter response (don't strip for streaming to preserve spaces)
            filtered = AnthropicAdapter.filter_response(content, strip=False)
            if not filtered:
                continue

            # Send content_block_start on first content
            if not content_started:
                yield _sse_text_block_start(0)
                content_started = True

            # Calculate delta to avoid duplicates
            new_content = _get_delta_content(filtered, accumulated_content)
            if new_content:
                accumulated_content = filtered
                chunk_index += 1
                log_stream_chunk(
                    request_id=ctx.request_id,
                    chunk_index=chunk_index,
                    event_type="content_block_delta",
                    data={"text": new_content},
                )
                yield _sse_text_delta(0, new_content)

        # Store response in session and hash mapping for future lookups
        if ctx.session_id and accumulated_content:
            session_manager.add_response(
                ctx.session_id,
                Message(role="assistant", content=accumulated_content),
            )
            # Store hash mapping of original request messages
            if ctx.messages_for_hash:
                session_manager.store_hash_mapping(ctx.messages_for_hash, ctx.session_id)

        # Send content_block_stop if we started content
        if content_started:
            yield _sse_block_stop(0)

        # Get usage (real or estimated)
        usage_data = claude_service.extract_usage(messages_buffer)
        if not usage_data:
            usage_data = claude_service.estimate_usage(
                ctx.prompt, accumulated_content, request.model
            )

        yield _sse_message_delta("end_turn", usage_data)
        yield _sse_message_stop()

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        error_data = json.dumps({"type": "error", "error": {"type": "api_error", "message": "Internal server error"}})
        yield f"event: error\ndata: {error_data}\n\n"
        return


async def generate_anthropic_response(ctx: MessagesContext) -> MessagesResponse:
    """Generate non-streaming Anthropic response."""
    messages_buffer = []
    request = ctx.request

    try:
        async for chunk in claude_service.run_completion(
            prompt=ctx.prompt,
            system_prompt=ctx.system_prompt,
            model=request.model,

            session_id=ctx.claude_session_id,
            mcp_servers=ctx.mcp_servers_dict,

        ):
            messages_buffer.append(chunk)

            # Extract and store Claude session ID
            if chunk.get("subtype") == "init" and ctx.session_id:
                new_session_id = claude_service.get_session_id([chunk])
                if new_session_id:
                    session_manager.set_claude_session(ctx.session_id, new_session_id)

        # Extract and filter response
        raw_content = claude_service.extract_response(messages_buffer)
        content = AnthropicAdapter.filter_response(raw_content or "")

        # Store in session and hash mapping for future lookups
        if ctx.session_id and content:
            session_manager.add_response(
                ctx.session_id,
                Message(role="assistant", content=content),
            )
            # Store hash mapping of original request messages
            if ctx.messages_for_hash:
                session_manager.store_hash_mapping(ctx.messages_for_hash, ctx.session_id)

        # Get usage (real or estimated)
        usage_data = claude_service.extract_usage(messages_buffer)
        if not usage_data:
            usage_data = claude_service.estimate_usage(ctx.prompt, content, request.model)

        response = MessagesResponse(
            id=ctx.request_id,
            model=request.model,
            content=[TextBlock(text=content)],
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            ),
        )

        # Log outgoing response
        log_outgoing_response(
            request_id=ctx.request_id,
            status_code=200,
            body=response.model_dump(),
        )

        return response

    except Exception as e:
        logger.error(f"Messages error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# Tool Proxy Streaming / Non-Streaming
# =============================================================================


async def _run_sdk_with_proxy(ctx: MessagesContext, proxy: ToolProxy) -> None:
    """Run SDK completion in background, pushing events to proxy.event_queue."""
    try:
        async for chunk in claude_service.run_completion(
            prompt=ctx.prompt,
            system_prompt=ctx.system_prompt,
            model=ctx.request.model,

            session_id=ctx.claude_session_id,
            mcp_servers=ctx.mcp_servers_dict,
            stream=True,

        ):
            # Extract and store Claude session ID for resumption
            if chunk.get("subtype") == "init" and ctx.session_id:
                new_session_id = claude_service.get_session_id([chunk])
                if new_session_id:
                    session_manager.set_claude_session(ctx.session_id, new_session_id)

            await proxy.event_queue.put({"type": "sdk_chunk", "chunk": chunk})

        await proxy.event_queue.put({"type": "sdk_complete"})
    except asyncio.CancelledError:
        await proxy.event_queue.put({"type": "sdk_complete"})
    except Exception as e:
        logger.error(f"SDK task error: {e}", exc_info=True)
        await proxy.event_queue.put({"type": "sdk_error", "error": str(e)})


async def _consume_proxy_events(
    ctx: MessagesContext,
    proxy: ToolProxy,
    *,
    emit_message_start: bool = True,
):
    """Consume events from proxy queue, yielding SSE strings.

    Handles both text chunks and tool_use_pending events.
    Returns when SDK completes or a tool_use is pending (client must provide result).
    """
    request = ctx.request
    accumulated_content = ""
    content_started = False
    block_index = 0
    chunk_index = 0

    if emit_message_start:
        yield _sse_message_start(ctx.request_id, request.model)

    while True:
        try:
            event = await asyncio.wait_for(proxy.event_queue.get(), timeout=310)
        except asyncio.TimeoutError:
            logger.error("Proxy event queue timed out")
            error_data = json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": "Proxy timeout"},
            })
            yield f"event: error\ndata: {error_data}\n\n"
            return

        etype = event.get("type")

        if etype == "sdk_chunk":
            chunk = event["chunk"]
            text = _extract_content(chunk)
            if not text:
                continue
            filtered = AnthropicAdapter.filter_response(text, strip=False)
            if not filtered:
                continue

            if not content_started:
                yield _sse_text_block_start(block_index)
                content_started = True

            new_content = _get_delta_content(filtered, accumulated_content)
            if new_content:
                accumulated_content = filtered
                chunk_index += 1
                log_stream_chunk(
                    request_id=ctx.request_id,
                    chunk_index=chunk_index,
                    event_type="content_block_delta",
                    data={"text": new_content},
                )
                yield _sse_text_delta(block_index, new_content)

        elif etype == "tool_use_pending":
            # Close text block if open
            if content_started:
                yield _sse_block_stop(block_index)
                block_index += 1

            # Emit tool_use block
            tool_use_id = event["tool_use_id"]
            tool_name = event["name"]
            tool_input = event["input"]

            yield _sse_tool_use_start(block_index, tool_use_id, tool_name)
            yield _sse_input_json_delta(block_index, tool_input)
            yield _sse_block_stop(block_index)
            block_index += 1

            # Store session/hash mappings before returning to client
            if ctx.session_id and ctx.messages_for_hash:
                session_manager.store_hash_mapping(ctx.messages_for_hash, ctx.session_id)

            # Usage: estimate since we don't have final usage yet
            usage_data = claude_service.estimate_usage(
                ctx.prompt, accumulated_content, request.model
            )
            yield _sse_message_delta("tool_use", usage_data)
            yield _sse_message_stop()
            return

        elif etype == "sdk_complete":
            # Close text block if open
            if content_started:
                yield _sse_block_stop(block_index)

            # Store session response
            if ctx.session_id and accumulated_content:
                session_manager.add_response(
                    ctx.session_id,
                    Message(role="assistant", content=accumulated_content),
                )
                if ctx.messages_for_hash:
                    session_manager.store_hash_mapping(ctx.messages_for_hash, ctx.session_id)

            # Clean up proxy
            remove_proxy(ctx.session_id)

            usage_data = claude_service.estimate_usage(
                ctx.prompt, accumulated_content, request.model
            )
            yield _sse_message_delta("end_turn", usage_data)
            yield _sse_message_stop()
            return

        elif etype == "sdk_error":
            if content_started:
                yield _sse_block_stop(block_index)
            remove_proxy(ctx.session_id)
            error_data = json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": event.get("error", "SDK error")},
            })
            yield f"event: error\ndata: {error_data}\n\n"
            return


async def generate_tool_proxy_stream(ctx: MessagesContext):
    """SSE generator for initial request with external tools.

    Creates an in-process MCP proxy, starts the SDK in a background task,
    and streams events until a tool_use is pending or the SDK completes.
    """
    log_outgoing_response(
        request_id=ctx.request_id,
        status_code=200,
        is_stream=True,
    )

    try:
        proxy = await create_tool_proxy(ctx.session_id, ctx.external_tools)

        # Merge proxy MCP config with any existing MCP servers
        merged_mcp = {**(ctx.mcp_servers_dict or {}), **(proxy.mcp_server_config or {})}
        ctx.mcp_servers_dict = merged_mcp

        # Append tool usage instructions to system prompt
        _apply_tool_prompt(ctx)

        # Register proxy for resume flow
        register_proxy(ctx.session_id, proxy)

        # Start SDK in background
        proxy._sdk_task = asyncio.create_task(_run_sdk_with_proxy(ctx, proxy))

        async for sse_event in _consume_proxy_events(ctx, proxy):
            yield sse_event

    except Exception as e:
        logger.error(f"Tool proxy stream error: {e}", exc_info=True)
        error_data = json.dumps({
            "type": "error",
            "error": {"type": "api_error", "message": "Internal server error"},
        })
        yield f"event: error\ndata: {error_data}\n\n"


async def generate_tool_resume_stream(
    ctx: MessagesContext,
    proxy: ToolProxy,
    tool_results: Dict[str, Dict[str, Any]],
):
    """SSE generator for resuming after tool_result submission.

    Resolves pending Futures so the MCP handlers unblock,
    then continues consuming events from the proxy queue.
    """
    log_outgoing_response(
        request_id=ctx.request_id,
        status_code=200,
        is_stream=True,
    )

    try:
        # Resolve all pending tool calls
        for tool_use_id, result in tool_results.items():
            proxy.submit_tool_result(
                tool_use_id,
                content=result["content"],
                is_error=result.get("is_error", False),
            )

        # Continue consuming events (SDK resumes after handler unblocks)
        async for sse_event in _consume_proxy_events(
            ctx, proxy, emit_message_start=True
        ):
            yield sse_event

    except Exception as e:
        logger.error(f"Tool resume stream error: {e}", exc_info=True)
        error_data = json.dumps({
            "type": "error",
            "error": {"type": "api_error", "message": "Internal server error"},
        })
        yield f"event: error\ndata: {error_data}\n\n"


async def generate_tool_proxy_response(
    ctx: MessagesContext,
    proxy: Optional[ToolProxy] = None,
    tool_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> MessagesResponse:
    """Non-streaming response for tool proxy flow.

    Collects all SSE events into a buffer and returns a MessagesResponse.
    """
    try:
        content_blocks = []
        stop_reason = "end_turn"

        if proxy and tool_results:
            # Resume flow
            for tool_use_id, result in tool_results.items():
                proxy.submit_tool_result(
                    tool_use_id,
                    content=result["content"],
                    is_error=result.get("is_error", False),
                )
        else:
            # New flow: create proxy
            proxy = await create_tool_proxy(ctx.session_id, ctx.external_tools)
            merged_mcp = {**(ctx.mcp_servers_dict or {}), **(proxy.mcp_server_config or {})}
            ctx.mcp_servers_dict = merged_mcp
            _apply_tool_prompt(ctx)
            register_proxy(ctx.session_id, proxy)
            proxy._sdk_task = asyncio.create_task(_run_sdk_with_proxy(ctx, proxy))

        # Collect events
        accumulated_text = ""

        while True:
            try:
                event = await asyncio.wait_for(proxy.event_queue.get(), timeout=310)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Proxy timeout")

            etype = event.get("type")

            if etype == "sdk_chunk":
                chunk = event["chunk"]
                text = _extract_content(chunk)
                if text:
                    filtered = AnthropicAdapter.filter_response(text, strip=False)
                    if filtered:
                        new_content = _get_delta_content(filtered, accumulated_text)
                        if new_content:
                            accumulated_text = filtered

            elif etype == "tool_use_pending":
                # Add accumulated text as a block
                if accumulated_text:
                    content_blocks.append(TextBlock(text=accumulated_text))
                    accumulated_text = ""

                content_blocks.append(ToolUseBlock(
                    id=event["tool_use_id"],
                    name=event["name"],
                    input=event["input"],
                ))
                stop_reason = "tool_use"

                # Store hash mapping
                if ctx.session_id and ctx.messages_for_hash:
                    session_manager.store_hash_mapping(ctx.messages_for_hash, ctx.session_id)
                break

            elif etype == "sdk_complete":
                if accumulated_text:
                    content_blocks.append(TextBlock(text=accumulated_text))
                stop_reason = "end_turn"
                remove_proxy(ctx.session_id)

                if ctx.session_id and accumulated_text:
                    session_manager.add_response(
                        ctx.session_id,
                        Message(role="assistant", content=accumulated_text),
                    )
                    if ctx.messages_for_hash:
                        session_manager.store_hash_mapping(ctx.messages_for_hash, ctx.session_id)
                break

            elif etype == "sdk_error":
                remove_proxy(ctx.session_id)
                raise HTTPException(status_code=500, detail=event.get("error", "SDK error"))

        if not content_blocks:
            content_blocks = [TextBlock(text="")]

        usage_data = claude_service.estimate_usage(
            ctx.prompt, accumulated_text, ctx.request.model
        )

        response = MessagesResponse(
            id=ctx.request_id,
            model=ctx.request.model,
            content=content_blocks,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            ),
        )

        log_outgoing_response(
            request_id=ctx.request_id,
            status_code=200,
            body=response.model_dump(),
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tool proxy response error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
