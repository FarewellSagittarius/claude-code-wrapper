"""Anthropic messages response models."""

import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .content import TextBlock


class Usage(BaseModel):
    """Anthropic token usage."""

    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[TextBlock]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence"]] = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Usage


# =============================================================================
# Streaming Events
# =============================================================================


class MessageStart(BaseModel):
    """message_start event data."""

    type: Literal["message_start"] = "message_start"
    message: Dict[str, Any]


class ContentBlockStart(BaseModel):
    """content_block_start event data."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: TextBlock


class TextDelta(BaseModel):
    """Text delta for content_block_delta."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ContentBlockDelta(BaseModel):
    """content_block_delta event data."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: TextDelta


class ContentBlockStop(BaseModel):
    """content_block_stop event data."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDelta(BaseModel):
    """message_delta event data."""

    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]
    usage: Usage


class MessageStop(BaseModel):
    """message_stop event data."""

    type: Literal["message_stop"] = "message_stop"
