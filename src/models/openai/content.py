"""OpenAI content types for chat messages."""

from typing import Literal, Optional, Union

from pydantic import BaseModel


class TextContent(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    """Image URL specification."""

    url: str  # https://... or data:image/png;base64,...
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ImageUrlContent(BaseModel):
    """Image URL content block."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


class FileData(BaseModel):
    """File data for inline file uploads."""

    filename: str
    file_data: str  # base64 encoded


class FileContent(BaseModel):
    """File content block (OpenAI standard)."""

    type: Literal["file"] = "file"
    file: FileData


ContentPart = Union[TextContent, ImageUrlContent, FileContent]
