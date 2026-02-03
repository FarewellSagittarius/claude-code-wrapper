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
    """Anthropic extended thinking configuration."""

    type: Literal["enabled", "disabled"] = "enabled"
    budget_tokens: int = Field(ge=1024, description="Thinking token budget, minimum 1024")


class ToolDefinition(BaseModel):
    """Anthropic tool definition format."""

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class MessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    model: str
    messages: List[Message]
    max_tokens: int = 4096
    system: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None

    # Extended thinking (native Anthropic format)
    thinking: Optional[ThinkingConfig] = None

    # Tool definitions (Anthropic native format)
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Dict[str, Any]] = None  # auto, any, tool, none

    # Wrapper-specific extensions
    session_id: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    disallowed_tools: Optional[List[str]] = None
    mcp_servers: Optional[Dict[str, McpServerConfig]] = None
