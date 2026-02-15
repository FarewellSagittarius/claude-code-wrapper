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

    Accepted for API compatibility but ignored. The wrapper internally uses
    adaptive thinking with effort=high. Client-provided thinking config
    (type, budget_tokens) is dropped.
    """

    type: Literal["enabled", "disabled", "adaptive"] = "enabled"
    budget_tokens: Optional[int] = Field(default=None, ge=1024)


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

    # Extended thinking — accepted but ignored (wrapper uses adaptive + effort=high internally)
    thinking: Optional[ThinkingConfig] = None

    # Tool definitions (Anthropic native format)
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Dict[str, Any]] = None  # auto, any, tool, none

    # Wrapper-specific extensions
    mcp_servers: Optional[Dict[str, McpServerConfig]] = None
