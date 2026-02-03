"""OpenAI chat completion request models."""

import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from ..common import McpServerConfig
from .content import ContentPart


class Message(BaseModel):
    """Chat message model."""

    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v: Any) -> Union[str, List[Dict[str, Any]]]:
        """Normalize content, preserving multimodal blocks when present."""
        if isinstance(v, str):
            return v

        if isinstance(v, list):
            has_multimodal = any(
                isinstance(item, dict) and item.get("type") in ("image_url", "file")
                for item in v
            )

            if has_multimodal:
                return v
            else:
                parts = []
                for item in v:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return " ".join(parts)

        return str(v)


class StreamOptions(BaseModel):
    """Stream options for chat completion."""

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "claude-sonnet-4-20250514"
    messages: List[Message]
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1, le=1)
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    stream_options: Optional[StreamOptions] = None

    # Claude-specific extensions
    session_id: Optional[str] = None
    enable_tools: Optional[bool] = True
    allowed_tools: Optional[List[str]] = None
    disallowed_tools: Optional[List[str]] = None
    max_turns: Optional[int] = None
    mcp_servers: Optional[Dict[str, McpServerConfig]] = None

    # Extended thinking control (OpenAI o1/o3 compatible)
    reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = None

    def get_effective_max_tokens(self) -> Optional[int]:
        """Get effective max tokens from either field."""
        return self.max_completion_tokens or self.max_tokens
