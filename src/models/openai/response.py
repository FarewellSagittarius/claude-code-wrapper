"""OpenAI chat completion response models."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PromptTokensDetails(BaseModel):
    """Details about prompt tokens."""

    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    """Details about completion tokens."""

    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[PromptTokensDetails] = None
    completion_tokens_details: Optional[CompletionTokensDetails] = None


class ResponseMessage(BaseModel):
    """Response message in choice."""

    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    """Non-streaming response choice."""

    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class DeltaContent(BaseModel):
    """Delta content for streaming."""

    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """Streaming response choice."""

    index: int = 0
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    owned_by: str = "anthropic"


class ModelListResponse(BaseModel):
    """Model list response."""

    object: Literal["list"] = "list"
    data: List[ModelInfo]
