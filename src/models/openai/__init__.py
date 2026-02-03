"""OpenAI-compatible data models."""

from .content import (
    ContentPart,
    FileContent,
    FileData,
    ImageUrl,
    ImageUrlContent,
    TextContent,
)
from .request import (
    ChatCompletionRequest,
    Message,
    StreamOptions,
)
from .response import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    CompletionTokensDetails,
    ModelInfo,
    ModelListResponse,
    PromptTokensDetails,
    ResponseMessage,
    StreamChoice,
    Usage,
)

__all__ = [
    # Content
    "ContentPart",
    "FileContent",
    "FileData",
    "ImageUrl",
    "ImageUrlContent",
    "TextContent",
    # Request
    "ChatCompletionRequest",
    "Message",
    "StreamOptions",
    # Response
    "ChatCompletionResponse",
    "ChatCompletionStreamResponse",
    "Choice",
    "CompletionTokensDetails",
    "ModelInfo",
    "ModelListResponse",
    "PromptTokensDetails",
    "ResponseMessage",
    "StreamChoice",
    "Usage",
]
