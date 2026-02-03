"""Chat completion routes."""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..adapters.openai_adapter import OpenAIAdapter
from ..config import settings
from ..middleware.auth import AuthResult, verify_api_key
from ..models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    Message,
    ResponseMessage,
    StreamChoice,
    Usage,
)
from ..services.claude import ClaudeService
from ..services.session import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

# Global service instances (will be set by main.py)
claude_service: Optional[ClaudeService] = None
session_manager: Optional[SessionManager] = None


def init_services(claude: ClaudeService, sessions: SessionManager) -> None:
    """Initialize service instances."""
    global claude_service, session_manager
    claude_service = claude
    session_manager = sessions


@dataclass
class CompletionContext:
    """Shared context for completion processing."""
    request: ChatCompletionRequest
    request_id: str
    prompt: Union[str, List[Dict[str, Any]]]  # Support multimodal
    system_prompt: Optional[str]
    claude_session_id: Optional[str]
    enable_internal_tools: bool
    enable_internal_mcp: bool
    mcp_servers_dict: Optional[Dict[str, Any]]
    max_thinking_tokens: Optional[int] = None  # From reasoning_effort mapping


async def _build_completion_context(
    request: ChatCompletionRequest,
    auth: AuthResult,
) -> CompletionContext:
    """Build shared context for completion processing."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Get Claude session ID if resuming
    claude_session_id = None
    if request.session_id:
        claude_session_id = session_manager.get_claude_session(request.session_id)

    # Convert messages to Claude format (files saved to cache in cwd)
    cwd = claude_service.cwd if claude_service else None
    prompt, system_prompt = await OpenAIAdapter.to_claude_prompt(request.messages, cwd)

    # Determine tool mode based on API key
    if auth.tool_mode == "heavy":
        enable_internal_tools = True
        enable_internal_mcp = True
    elif auth.tool_mode == "basic":
        enable_internal_tools = True
        enable_internal_mcp = False
    elif auth.tool_mode == "custom":
        enable_internal_tools = request.enable_tools or False
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

    # Map reasoning_effort to max_thinking_tokens
    max_thinking_tokens = None
    if request.reasoning_effort is not None:
        max_thinking_tokens = settings.REASONING_EFFORT_MAP.get(request.reasoning_effort)
        logger.debug(f"reasoning_effort={request.reasoning_effort} -> max_thinking_tokens={max_thinking_tokens}")

    logger.debug(
        f"Request {request_id}: stream={request.stream}, model={request.model}, "
        f"tool_mode={auth.tool_mode}, internal_tools={enable_internal_tools}"
    )

    return CompletionContext(
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


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request,
    auth: AuthResult = Depends(verify_api_key),
):
    """
    OpenAI-compatible chat completions endpoint.

    Tool availability depends on the API key used:
    - sk-light-*: No tools (chat only)
    - sk-basic-*: All built-in tools
    - sk-heavy-*: All tools + plugins + MCP servers
    - sk-custom-*: Use request settings
    """
    if not claude_service or not session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    ctx = await _build_completion_context(request, auth)

    if request.stream:
        return StreamingResponse(
            generate_stream(ctx),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return await generate_response(ctx)


def _get_delta_content(filtered: str, accumulated: str) -> str:
    """Calculate new content delta to avoid duplicates."""
    if filtered.startswith(accumulated):
        return filtered[len(accumulated):]
    if accumulated and (filtered == accumulated or accumulated in filtered):
        return ""
    return filtered


async def generate_stream(ctx: CompletionContext):
    """Generate SSE streaming response."""
    messages_buffer = []
    role_sent = False
    accumulated_content = ""
    created = int(time.time())
    request = ctx.request

    try:
        async for chunk in claude_service.run_completion(
            prompt=ctx.prompt,
            system_prompt=ctx.system_prompt,
            model=request.model,
            enable_internal_tools=ctx.enable_internal_tools,
            enable_internal_mcp=ctx.enable_internal_mcp,
            max_turns=request.max_turns or 10,
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

            # Extract and stream content
            content = _extract_content(chunk)
            if not content:
                continue

            # Send role on first content
            if not role_sent:
                initial = ChatCompletionStreamResponse(
                    id=ctx.request_id,
                    created=created,
                    model=request.model,
                    choices=[StreamChoice(delta={"role": "assistant", "content": ""}, finish_reason=None)],
                )
                yield f"data: {initial.model_dump_json()}\n\n"
                role_sent = True

            # Filter and send content delta
            filtered = OpenAIAdapter.filter_response(content)
            if not filtered:
                continue

            new_content = _get_delta_content(filtered, accumulated_content)
            if new_content:
                accumulated_content = filtered
                stream_chunk = ChatCompletionStreamResponse(
                    id=ctx.request_id,
                    created=created,
                    model=request.model,
                    choices=[StreamChoice(delta={"content": new_content}, finish_reason=None)],
                )
                yield f"data: {stream_chunk.model_dump_json()}\n\n"

        # Store response in session
        if request.session_id and accumulated_content:
            session_manager.add_response(
                request.session_id,
                Message(role="assistant", content=accumulated_content),
            )

        # Send final chunk with usage
        usage_data = claude_service.extract_usage(messages_buffer)
        if not usage_data:
            usage_data = claude_service.estimate_usage(ctx.prompt, accumulated_content, request.model)

        final = ChatCompletionStreamResponse(
            id=ctx.request_id,
            created=created,
            model=request.model,
            choices=[StreamChoice(delta={}, finish_reason="stop")],
            usage=Usage(**usage_data),
        )
        yield f"data: {final.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {{'error': {{'message': '{str(e)}', 'type': 'server_error'}}}}\n\n"
        yield "data: [DONE]\n\n"


async def generate_response(ctx: CompletionContext) -> ChatCompletionResponse:
    """Generate non-streaming response."""
    messages_buffer = []
    request = ctx.request

    try:
        async for chunk in claude_service.run_completion(
            prompt=ctx.prompt,
            system_prompt=ctx.system_prompt,
            model=request.model,
            enable_internal_tools=ctx.enable_internal_tools,
            enable_internal_mcp=ctx.enable_internal_mcp,
            max_turns=request.max_turns or 10,
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
        content = OpenAIAdapter.filter_response(raw_content or "")

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

        return ChatCompletionResponse(
            id=ctx.request_id,
            model=request.model,
            choices=[Choice(message=ResponseMessage(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(**usage_data),
        )

    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_text(content: Any) -> Optional[str]:
    """Extract text from content (string or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return OpenAIAdapter.extract_text_from_content(content)
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
