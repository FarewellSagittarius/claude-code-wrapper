"""Debug logging utility for request/response payload inspection."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ..config import settings

logger = logging.getLogger("debug.payloads")


def _truncate(text: str, max_length: int = 0) -> str:
    """Truncate text if max_length is set."""
    if max_length <= 0 or len(text) <= max_length:
        return text
    return text[:max_length] + f"... [truncated, total {len(text)} chars]"


def _safe_json(obj: Any, indent: int = 2) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
    except (TypeError, ValueError) as e:
        return f"<serialization error: {e}>"


def log_incoming_request(
    request_id: str,
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
) -> None:
    """Log incoming external request."""
    if not settings.DEBUG_LOG_PAYLOADS:
        return

    timestamp = datetime.now().isoformat()
    max_len = settings.DEBUG_LOG_MAX_LENGTH

    log_parts = [
        f"\n{'='*60}",
        f"[{timestamp}] INCOMING REQUEST: {request_id}",
        f"{'='*60}",
        f"Method: {method}",
        f"Path: {path}",
    ]

    if headers:
        # Mask sensitive headers
        safe_headers = {
            k: ("***" if k.lower() in ("authorization", "x-api-key") else v)
            for k, v in headers.items()
        }
        log_parts.append(f"Headers: {_safe_json(safe_headers)}")

    if body is not None:
        body_str = _safe_json(body)
        log_parts.append(f"Body:\n{_truncate(body_str, max_len)}")

    log_parts.append("=" * 60)
    logger.info("\n".join(log_parts))


def log_outgoing_response(
    request_id: str,
    status_code: int,
    body: Optional[Any] = None,
    is_stream: bool = False,
) -> None:
    """Log outgoing response to external client."""
    if not settings.DEBUG_LOG_PAYLOADS:
        return

    timestamp = datetime.now().isoformat()
    max_len = settings.DEBUG_LOG_MAX_LENGTH

    log_parts = [
        f"\n{'-'*60}",
        f"[{timestamp}] OUTGOING RESPONSE: {request_id}",
        f"{'-'*60}",
        f"Status: {status_code}",
        f"Stream: {is_stream}",
    ]

    if body is not None and not is_stream:
        body_str = _safe_json(body)
        log_parts.append(f"Body:\n{_truncate(body_str, max_len)}")
    elif is_stream:
        log_parts.append("Body: <streaming response>")

    log_parts.append("-" * 60)
    logger.info("\n".join(log_parts))


def log_stream_chunk(
    request_id: str,
    chunk_index: int,
    event_type: str,
    data: Optional[Any] = None,
) -> None:
    """Log individual stream chunk."""
    if not settings.DEBUG_LOG_PAYLOADS:
        return

    max_len = settings.DEBUG_LOG_MAX_LENGTH
    data_str = _truncate(_safe_json(data), max_len) if data else ""
    logger.debug(f"[{request_id}] Stream #{chunk_index} ({event_type}): {data_str}")


def log_internal_request(
    request_id: str,
    prompt: Any,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> None:
    """Log internal request to Claude SDK."""
    if not settings.DEBUG_LOG_PAYLOADS:
        return

    timestamp = datetime.now().isoformat()
    max_len = settings.DEBUG_LOG_MAX_LENGTH

    log_parts = [
        f"\n{'>'*60}",
        f"[{timestamp}] INTERNAL REQUEST (to SDK): {request_id}",
        f"{'>'*60}",
        f"Model: {model}",
    ]

    if system_prompt:
        log_parts.append(f"System Prompt:\n{_truncate(system_prompt, max_len)}")

    prompt_str = _safe_json(prompt) if not isinstance(prompt, str) else prompt
    log_parts.append(f"Prompt:\n{_truncate(prompt_str, max_len)}")

    if options:
        # Filter sensitive options
        safe_options = {k: v for k, v in options.items() if k not in ("api_key",)}
        log_parts.append(f"Options: {_safe_json(safe_options)}")

    log_parts.append(">" * 60)
    logger.info("\n".join(log_parts))


def log_internal_response(
    request_id: str,
    messages: Any,
    final_content: Optional[str] = None,
) -> None:
    """Log internal response from Claude SDK."""
    if not settings.DEBUG_LOG_PAYLOADS:
        return

    timestamp = datetime.now().isoformat()
    max_len = settings.DEBUG_LOG_MAX_LENGTH

    log_parts = [
        f"\n{'<'*60}",
        f"[{timestamp}] INTERNAL RESPONSE (from SDK): {request_id}",
        f"{'<'*60}",
    ]

    if final_content:
        log_parts.append(f"Final Content:\n{_truncate(final_content, max_len)}")

    if messages:
        msg_count = len(messages) if isinstance(messages, list) else 1
        log_parts.append(f"Message Count: {msg_count}")

        # Log last few messages for context
        if isinstance(messages, list) and messages:
            last_msgs = messages[-3:] if len(messages) > 3 else messages
            log_parts.append(f"Last Messages:\n{_truncate(_safe_json(last_msgs), max_len)}")

    log_parts.append("<" * 60)
    logger.info("\n".join(log_parts))


def log_sdk_chunk(
    request_id: str,
    chunk: Dict[str, Any],
) -> None:
    """Log individual SDK message chunk."""
    if not settings.DEBUG_LOG_PAYLOADS:
        return

    max_len = settings.DEBUG_LOG_MAX_LENGTH
    chunk_type = chunk.get("type", chunk.get("subtype", "unknown"))
    chunk_str = _truncate(_safe_json(chunk), max_len)
    logger.debug(f"[{request_id}] SDK chunk ({chunk_type}): {chunk_str}")
