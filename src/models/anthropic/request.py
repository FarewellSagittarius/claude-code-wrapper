"""Anthropic messages request models."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..common import McpServerConfig
from .content import ContentBlock


class Message(BaseModel):
    """Anthropic message format."""

    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


class ThinkingConfig(BaseModel):
    """Anthropic extended thinking configuration.

    DISABLED: Extended thinking is fully disabled. This config is accepted
    for API compatibility but not processed. SDK does not support adaptive
    thinking, and budget_tokens is deprecated on Opus 4.6.
    """

    type: Literal["enabled", "disabled", "adaptive"] = "enabled"
    budget_tokens: Optional[int] = Field(default=None, ge=1024, description="Deprecated on Opus 4.6")


class ToolDefinition(BaseModel):
    """Anthropic tool definition format."""

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class SystemBlock(BaseModel):
    """Anthropic system content block."""

    type: Literal["text"] = "text"
    text: str


class MessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None  # Deprecated: accepted but ignored, SDK manages token limits
    system: Optional[Union[str, List[SystemBlock]]] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None

    # Extended thinking — DISABLED (accepted but ignored)
    thinking: Optional[ThinkingConfig] = None

    # Tool definitions (Anthropic native format)
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Dict[str, Any]] = None  # auto, any, tool, none

    # Wrapper-specific extensions
    mcp_servers: Optional[Dict[str, McpServerConfig]] = None
