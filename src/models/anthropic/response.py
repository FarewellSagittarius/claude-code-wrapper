"""Anthropic messages response models."""

import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .content import TextBlock, ToolUseBlock


class Usage(BaseModel):
    """Anthropic token usage."""

    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Union[TextBlock, ToolUseBlock]]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = "end_turn"
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
    content_block: Union[TextBlock, ToolUseBlock]


class TextDelta(BaseModel):
    """Text delta for content_block_delta."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(BaseModel):
    """Input JSON delta for streaming tool_use input."""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ContentBlockDelta(BaseModel):
    """content_block_delta event data."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Union[TextDelta, InputJsonDelta]


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
