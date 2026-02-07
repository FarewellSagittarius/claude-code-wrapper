"""Anthropic content types for messages."""

from typing import Literal, Optional, Union

from pydantic import BaseModel


# =============================================================================
# Text Block
# =============================================================================


class TextBlock(BaseModel):
    """Anthropic text content block."""

    type: Literal["text"] = "text"
    text: str


# =============================================================================
# Image Block
# =============================================================================


class Base64ImageSource(BaseModel):
    """Anthropic base64 image source."""

    type: Literal["base64"] = "base64"
    media_type: str  # e.g., "image/png", "image/jpeg"
    data: str


class UrlImageSource(BaseModel):
    """Anthropic URL image source."""

    type: Literal["url"] = "url"
    url: str


class ImageBlock(BaseModel):
    """Anthropic image content block."""

    type: Literal["image"] = "image"
    source: Union[Base64ImageSource, UrlImageSource]


# =============================================================================
# Document Block
# =============================================================================


class Base64DocumentSource(BaseModel):
    """Anthropic base64 document source (PDF)."""

    type: Literal["base64"] = "base64"
    media_type: Literal["application/pdf"] = "application/pdf"
    data: str


class UrlDocumentSource(BaseModel):
    """Anthropic URL document source."""

    type: Literal["url"] = "url"
    url: str


class TextDocumentSource(BaseModel):
    """Anthropic plain text document source."""

    type: Literal["text"] = "text"
    media_type: Literal["text/plain"] = "text/plain"
    data: str


DocumentSource = Union[Base64DocumentSource, UrlDocumentSource, TextDocumentSource]


class DocumentBlock(BaseModel):
    """Anthropic document content block."""

    type: Literal["document"] = "document"
    source: DocumentSource
    title: Optional[str] = None
    context: Optional[str] = None


# =============================================================================
# Tool Use Block (assistant response)
# =============================================================================


class ToolUseBlock(BaseModel):
    """Anthropic tool_use content block for assistant responses."""

    type: Literal["tool_use"] = "tool_use"
    id: str  # e.g., "toolu_01abc..."
    name: str
    input: dict


# =============================================================================
# Tool Result Block (user message)
# =============================================================================


class ToolResultBlock(BaseModel):
    """Anthropic tool_result content block for user messages."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[Union[str, list]] = None
    is_error: bool = False


# =============================================================================
# Union Type
# =============================================================================


ContentBlock = Union[TextBlock, ImageBlock, DocumentBlock, ToolUseBlock, ToolResultBlock]
