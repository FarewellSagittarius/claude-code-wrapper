"""Anthropic Messages API routes."""

import json
import logging
import time
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
    Message,
    MessageDelta,
    MessageStart,
    MessageStop,
    MessagesRequest,
    MessagesResponse,
    TextBlock,
    TextDelta,
    Usage,
)
from ..services.claude import ClaudeService
from ..services.session import SessionManager

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
    prompt: Union[str, List[Dict[str, Any]]]  # Support multimodal
    system_prompt: Optional[str]
    claude_session_id: Optional[str]
    enable_internal_tools: bool
    enable_internal_mcp: bool
    mcp_servers_dict: Optional[Dict[str, Any]]
    max_thinking_tokens: Optional[int] = None


async def _build_messages_context(
    request: MessagesRequest,
    auth: AuthResult,
) -> MessagesContext:
    """Build shared context for messages processing."""
    request_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Get Claude session ID if resuming
    claude_session_id = None
    if request.session_id:
        claude_session_id = session_manager.get_claude_session(request.session_id)

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

    prompt, system_prompt = await AnthropicAdapter.to_claude_prompt(
        [Message(**m) for m in internal_messages],
        cwd,
        request.system
    )

    # Determine tool mode based on API key
    # Anthropic style: tools enabled by default for basic/heavy
    if auth.tool_mode == "heavy":
        enable_internal_tools = True
        enable_internal_mcp = True
    elif auth.tool_mode == "basic":
        enable_internal_tools = True
        enable_internal_mcp = False
    elif auth.tool_mode == "custom":
        # For custom mode, enable tools by default (Anthropic style)
        enable_internal_tools = True
        enable_internal_mcp = False
    else:  # light or default
        enable_internal_tools = False
        enable_internal_mcp = False

    # Convert MCP servers to dict format
    mcp_servers_dict = None
    if request.mcp_servers:
        mcp_servers_dict = {
            name: config.model_dump(exclude_none=True)
            for name, config in request.mcp_servers.items()
        }

    # Extract max_thinking_tokens from native Anthropic thinking config
    max_thinking_tokens = None
    if request.thinking is not None and request.thinking.type == "enabled":
        max_thinking_tokens = request.thinking.budget_tokens
        logger.debug(f"thinking.budget_tokens={max_thinking_tokens}")

    logger.debug(
        f"Request {request_id}: stream={request.stream}, model={request.model}, "
        f"tool_mode={auth.tool_mode}, internal_tools={enable_internal_tools}"
    )

    return MessagesContext(
        request=request,
        request_id=request_id,
        prompt=prompt,
        system_prompt=system_prompt,
        claude_session_id=claude_session_id,
        enable_internal_tools=enable_internal_tools,
        enable_internal_mcp=enable_internal_mcp,
        mcp_servers_dict=mcp_servers_dict,
        max_thinking_tokens=max_thinking_tokens,
    )


@router.post("/messages", response_model=MessagesResponse)
async def anthropic_messages(
    request: MessagesRequest,
    req: Request,
    auth: AuthResult = Depends(verify_api_key),
):
    """
    Anthropic Messages API compatible endpoint.

    Tool availability depends on the API key used:
    - sk-light-*: No tools (chat only)
    - sk-basic-*: All built-in tools (enabled by default)
    - sk-heavy-*: All tools + plugins + MCP servers
    - sk-custom-*: Tools enabled by default (Anthropic style)
    """
    if not claude_service or not session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    ctx = await _build_messages_context(request, auth)

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

    try:
        # Send message_start event
        message_start = MessageStart(
            message={
                "id": ctx.request_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        )
        yield f"event: message_start\ndata: {message_start.model_dump_json()}\n\n"

        async for chunk in claude_service.run_completion(
            prompt=ctx.prompt,
            system_prompt=ctx.system_prompt,
            model=request.model,
            enable_internal_tools=ctx.enable_internal_tools,
            enable_internal_mcp=ctx.enable_internal_mcp,
            max_turns=10,
            session_id=ctx.claude_session_id,
            allowed_tools=request.allowed_tools,
            disallowed_tools=request.disallowed_tools,
            mcp_servers=ctx.mcp_servers_dict,
            stream=True,
            max_thinking_tokens=ctx.max_thinking_tokens,
        ):
            messages_buffer.append(chunk)

            # Extract and store Claude session ID
            if chunk.get("subtype") == "init" and request.session_id:
                new_session_id = claude_service.get_session_id([chunk])
                if new_session_id:
                    session_manager.set_claude_session(request.session_id, new_session_id)

            # Extract content
            content = _extract_content(chunk)
            if not content:
                continue

            # Filter response
            filtered = AnthropicAdapter.filter_response(content)
            if not filtered:
                continue

            # Send content_block_start on first content
            if not content_started:
                block_start = ContentBlockStart(
                    index=0,
                    content_block=TextBlock(text=""),
                )
                yield f"event: content_block_start\ndata: {block_start.model_dump_json()}\n\n"
                content_started = True

            # Calculate delta to avoid duplicates
            new_content = _get_delta_content(filtered, accumulated_content)
            if new_content:
                accumulated_content = filtered
                block_delta = ContentBlockDelta(
                    index=0,
                    delta=TextDelta(text=new_content),
                )
                yield f"event: content_block_delta\ndata: {block_delta.model_dump_json()}\n\n"

        # Store response in session
        if request.session_id and accumulated_content:
            session_manager.add_response(
                request.session_id,
                Message(role="assistant", content=accumulated_content),
            )

        # Send content_block_stop if we started content
        if content_started:
            block_stop = ContentBlockStop(index=0)
            yield f"event: content_block_stop\ndata: {block_stop.model_dump_json()}\n\n"

        # Get usage (real or estimated)
        usage_data = claude_service.extract_usage(messages_buffer)
        if not usage_data:
            usage_data = claude_service.estimate_usage(
                ctx.prompt, accumulated_content, request.model
            )

        # Send message_delta with final usage
        message_delta = MessageDelta(
            delta={"stop_reason": "end_turn"},
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
            ),
        )
        yield f"event: message_delta\ndata: {message_delta.model_dump_json()}\n\n"

        # Send message_stop
        message_stop = MessageStop()
        yield f"event: message_stop\ndata: {message_stop.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_data = json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e)}})
        yield f"event: error\ndata: {error_data}\n\n"


async def generate_anthropic_response(ctx: MessagesContext) -> MessagesResponse:
    """Generate non-streaming Anthropic response."""
    messages_buffer = []
    request = ctx.request

    try:
        async for chunk in claude_service.run_completion(
            prompt=ctx.prompt,
            system_prompt=ctx.system_prompt,
            model=request.model,
            enable_internal_tools=ctx.enable_internal_tools,
            enable_internal_mcp=ctx.enable_internal_mcp,
            max_turns=10,
            session_id=ctx.claude_session_id,
            allowed_tools=request.allowed_tools,
            disallowed_tools=request.disallowed_tools,
            mcp_servers=ctx.mcp_servers_dict,
            max_thinking_tokens=ctx.max_thinking_tokens,
        ):
            messages_buffer.append(chunk)

            # Extract and store Claude session ID
            if chunk.get("subtype") == "init" and request.session_id:
                new_session_id = claude_service.get_session_id([chunk])
                if new_session_id:
                    session_manager.set_claude_session(request.session_id, new_session_id)

        # Extract and filter response
        raw_content = claude_service.extract_response(messages_buffer)
        content = AnthropicAdapter.filter_response(raw_content or "")

        # Store in session
        if request.session_id and content:
            session_manager.add_response(
                request.session_id,
                Message(role="assistant", content=content),
            )

        # Get usage (real or estimated)
        usage_data = claude_service.extract_usage(messages_buffer)
        if not usage_data:
            usage_data = claude_service.estimate_usage(ctx.prompt, content, request.model)

        return MessagesResponse(
            id=ctx.request_id,
            model=request.model,
            content=[TextBlock(text=content)],
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
            ),
        )

    except Exception as e:
        logger.error(f"Messages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
