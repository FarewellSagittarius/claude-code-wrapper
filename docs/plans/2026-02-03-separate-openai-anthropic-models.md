# OpenAI/Anthropic 模型分离 + 文件支持 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 OpenAI 和 Anthropic 的数据模型分离到独立目录，并为 OpenAI API 添加标准的 `file` 内容类型支持。

**Architecture:** 按目录分离两套 API 的模型定义，提取共享类型到 common.py，分离适配器为独立文件，各自处理对应格式的转换逻辑。

**Tech Stack:** Python 3.12, Pydantic v2, FastAPI

---

## Task 1: 创建共享类型 (models/common.py)

**Files:**
- Create: `src/models/common.py`

**Step 1: 创建 common.py 文件**

```python
"""Shared types for OpenAI and Anthropic APIs."""

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# =============================================================================
# MCP Server Configurations (shared by both APIs)
# =============================================================================


class McpStdioServerConfig(BaseModel):
    """MCP server config for stdio transport."""

    type: Optional[Literal["stdio"]] = "stdio"
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None


class McpSSEServerConfig(BaseModel):
    """MCP server config for SSE (streaming HTTP) transport."""

    type: Literal["sse"]
    url: str
    headers: Optional[Dict[str, str]] = None


class McpHttpServerConfig(BaseModel):
    """MCP server config for HTTP transport."""

    type: Literal["http"]
    url: str
    headers: Optional[Dict[str, str]] = None


McpServerConfig = Union[McpStdioServerConfig, McpSSEServerConfig, McpHttpServerConfig]


# =============================================================================
# Token Usage (internal representation)
# =============================================================================


class TokenUsage(BaseModel):
    """Internal token usage representation with conversion methods."""

    input_tokens: int
    output_tokens: int

    def to_openai(self) -> dict:
        """Convert to OpenAI usage format."""
        return {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }

    def to_anthropic(self) -> dict:
        """Convert to Anthropic usage format."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
```

**Step 2: 验证文件创建**

Run: `python -c "from src.models.common import McpServerConfig, TokenUsage; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/models/common.py
git commit -m "feat(models): add common.py with shared types"
```

---

## Task 2: 创建 OpenAI 内容类型 (models/openai/content.py)

**Files:**
- Create: `src/models/openai/__init__.py`
- Create: `src/models/openai/content.py`

**Step 1: 创建目录和 __init__.py**

```python
"""OpenAI-compatible data models."""

from .content import (
    ContentPart,
    FileContent,
    FileData,
    ImageUrl,
    ImageUrlContent,
    TextContent,
)

__all__ = [
    "ContentPart",
    "FileContent",
    "FileData",
    "ImageUrl",
    "ImageUrlContent",
    "TextContent",
]
```

**Step 2: 创建 content.py**

```python
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
```

**Step 3: 验证导入**

Run: `python -c "from src.models.openai import ContentPart, FileContent; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add src/models/openai/
git commit -m "feat(models): add OpenAI content types with file support"
```

---

## Task 3: 创建 OpenAI 请求/响应模型 (models/openai/request.py, response.py)

**Files:**
- Modify: `src/models/openai/__init__.py`
- Create: `src/models/openai/request.py`
- Create: `src/models/openai/response.py`

**Step 1: 创建 request.py**

```python
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
```

**Step 2: 创建 response.py**

```python
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
```

**Step 3: 更新 __init__.py**

```python
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
```

**Step 4: 验证导入**

Run: `python -c "from src.models.openai import ChatCompletionRequest, ChatCompletionResponse, Message, FileContent; print('OK')"`
Expected: OK

**Step 5: Commit**

```bash
git add src/models/openai/
git commit -m "feat(models): add OpenAI request/response models"
```

---

## Task 4: 创建 Anthropic 内容类型 (models/anthropic/content.py)

**Files:**
- Create: `src/models/anthropic/__init__.py`
- Create: `src/models/anthropic/content.py`

**Step 1: 创建目录和 __init__.py (初始)**

```python
"""Anthropic-compatible data models."""

from .content import (
    Base64DocumentSource,
    Base64ImageSource,
    ContentBlock,
    DocumentBlock,
    DocumentSource,
    ImageBlock,
    TextBlock,
    TextDocumentSource,
    UrlDocumentSource,
    UrlImageSource,
)

__all__ = [
    "Base64DocumentSource",
    "Base64ImageSource",
    "ContentBlock",
    "DocumentBlock",
    "DocumentSource",
    "ImageBlock",
    "TextBlock",
    "TextDocumentSource",
    "UrlDocumentSource",
    "UrlImageSource",
]
```

**Step 2: 创建 content.py**

```python
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
# Union Type
# =============================================================================


ContentBlock = Union[TextBlock, ImageBlock, DocumentBlock]
```

**Step 3: 验证导入**

Run: `python -c "from src.models.anthropic import ContentBlock, DocumentBlock; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add src/models/anthropic/
git commit -m "feat(models): add Anthropic content types"
```

---

## Task 5: 创建 Anthropic 请求/响应模型 (models/anthropic/request.py, response.py)

**Files:**
- Modify: `src/models/anthropic/__init__.py`
- Create: `src/models/anthropic/request.py`
- Create: `src/models/anthropic/response.py`

**Step 1: 创建 request.py**

```python
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

    # Wrapper-specific extensions
    session_id: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    disallowed_tools: Optional[List[str]] = None
    mcp_servers: Optional[Dict[str, McpServerConfig]] = None
```

**Step 2: 创建 response.py**

```python
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
```

**Step 3: 更新 __init__.py**

```python
"""Anthropic-compatible data models."""

from .content import (
    Base64DocumentSource,
    Base64ImageSource,
    ContentBlock,
    DocumentBlock,
    DocumentSource,
    ImageBlock,
    TextBlock,
    TextDocumentSource,
    UrlDocumentSource,
    UrlImageSource,
)
from .request import (
    Message,
    MessagesRequest,
    ThinkingConfig,
)
from .response import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    MessagesResponse,
    TextDelta,
    Usage,
)

__all__ = [
    # Content
    "Base64DocumentSource",
    "Base64ImageSource",
    "ContentBlock",
    "DocumentBlock",
    "DocumentSource",
    "ImageBlock",
    "TextBlock",
    "TextDocumentSource",
    "UrlDocumentSource",
    "UrlImageSource",
    # Request
    "Message",
    "MessagesRequest",
    "ThinkingConfig",
    # Response
    "ContentBlockDelta",
    "ContentBlockStart",
    "ContentBlockStop",
    "MessageDelta",
    "MessageStart",
    "MessageStop",
    "MessagesResponse",
    "TextDelta",
    "Usage",
]
```

**Step 4: 验证导入**

Run: `python -c "from src.models.anthropic import MessagesRequest, MessagesResponse, Message; print('OK')"`
Expected: OK

**Step 5: Commit**

```bash
git add src/models/anthropic/
git commit -m "feat(models): add Anthropic request/response models"
```

---

## Task 6: 更新 models/__init__.py

**Files:**
- Modify: `src/models/__init__.py`

**Step 1: 更新 __init__.py**

```python
"""Data models for OpenAI and Anthropic APIs."""

from . import anthropic, openai
from .common import (
    McpHttpServerConfig,
    McpServerConfig,
    McpSSEServerConfig,
    McpStdioServerConfig,
    TokenUsage,
)

__all__ = [
    # Submodules
    "anthropic",
    "openai",
    # Common
    "McpHttpServerConfig",
    "McpServerConfig",
    "McpSSEServerConfig",
    "McpStdioServerConfig",
    "TokenUsage",
]
```

**Step 2: 验证导入**

Run: `python -c "from src.models import openai, anthropic; from src.models.openai import ChatCompletionRequest; from src.models.anthropic import MessagesRequest; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/models/__init__.py
git commit -m "feat(models): update models __init__ for new structure"
```

---

## Task 7: 创建基础适配器工具 (adapters/base.py)

**Files:**
- Create: `src/adapters/base.py`

**Step 1: 创建 base.py**

```python
"""Base adapter utilities for file handling and caching."""

import base64
import hashlib
import logging
import mimetypes
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class FileCache:
    """File cache manager for media and documents."""

    # File extension mapping for MIME types
    MIME_TO_EXT = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
        "application/json": ".json",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "text/html": ".html",
        "text/csv": ".csv",
    }

    _cache_dir: Optional[Path] = None

    @classmethod
    def get_cache_dir(cls, cwd: Optional[str] = None) -> Path:
        """Get or create cache directory for media files."""
        if cls._cache_dir is None:
            if cwd:
                cls._cache_dir = Path(cwd) / ".claude_media_cache"
            else:
                cls._cache_dir = Path(tempfile.gettempdir()) / "claude_media_cache"
            cls._cache_dir.mkdir(parents=True, exist_ok=True)
        return cls._cache_dir

    @classmethod
    def set_cache_dir(cls, path: str) -> None:
        """Set custom cache directory."""
        cls._cache_dir = Path(path)
        cls._cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_file_hash(data: bytes) -> str:
        """Generate short hash for file content."""
        return hashlib.sha256(data).hexdigest()[:16]

    @classmethod
    async def save(
        cls,
        data: bytes,
        media_type: str,
        filename: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> Path:
        """Save binary data to cache and return file path."""
        cache_dir = cls.get_cache_dir(cwd)
        file_hash = cls.get_file_hash(data)

        # Determine extension
        ext = cls.MIME_TO_EXT.get(media_type, "")
        if not ext and filename:
            ext = Path(filename).suffix

        # Create filename
        if filename:
            safe_name = re.sub(r"[^\w\-.]", "_", Path(filename).stem)[:32]
            cache_filename = f"{safe_name}_{file_hash}{ext}"
        else:
            cache_filename = f"file_{file_hash}{ext}"

        cache_path = cache_dir / cache_filename

        # Write if not exists
        if not cache_path.exists():
            cache_path.write_bytes(data)
            logger.debug(f"Cached file: {cache_path}")

        return cache_path


def parse_data_url(url: str) -> Optional[Tuple[str, str]]:
    """Parse a data URL into (mime_type, base64_data)."""
    match = re.match(r"^data:([^;]+);base64,(.+)$", url)
    return (match.group(1), match.group(2)) if match else None


async def fetch_url(url: str, cwd: Optional[str] = None) -> Optional[Path]:
    """Fetch file from URL and save to cache."""
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").split(";")[0].strip()

            # Try to get filename from URL or Content-Disposition
            filename = None
            cd = response.headers.get("content-disposition", "")
            if "filename=" in cd:
                filename = cd.split("filename=")[-1].strip("\"'")
            if not filename:
                filename = url.split("/")[-1].split("?")[0]

            return await FileCache.save(response.content, content_type, filename, cwd)
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def guess_mime_type(filename: str) -> str:
    """Guess MIME type from filename."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def estimate_tokens(text: str) -> int:
    """Roughly estimate token count (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)
```

**Step 2: 验证导入**

Run: `python -c "from src.adapters.base import FileCache, parse_data_url, fetch_url; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/adapters/base.py
git commit -m "feat(adapters): add base utilities for file caching"
```

---

## Task 8: 创建 OpenAI 适配器 (adapters/openai_adapter.py)

**Files:**
- Create: `src/adapters/openai_adapter.py`

**Step 1: 创建 openai_adapter.py**

```python
"""OpenAI format adapter - converts OpenAI messages to Claude format."""

import base64
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models.openai import Message
from .base import FileCache, fetch_url, guess_mime_type, parse_data_url

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """Adapter for converting OpenAI format to Claude internal format."""

    @classmethod
    async def process_image_url(
        cls, image_url: Dict[str, Any], cwd: Optional[str] = None
    ) -> Optional[str]:
        """Process image_url content block."""
        url = image_url.get("url", "")

        if url.startswith("data:"):
            parsed = parse_data_url(url)
            if parsed:
                mime_type, b64_data = parsed
                data = base64.b64decode(b64_data)
                path = await FileCache.save(data, mime_type, None, cwd)
                return f"[Image: {path}]"
        else:
            path = await fetch_url(url, cwd)
            if path:
                return f"[Image: {path}]"

        return "[Failed to process image]"

    @classmethod
    async def process_file(
        cls, file_data: Dict[str, Any], cwd: Optional[str] = None
    ) -> Optional[str]:
        """Process file content block (OpenAI standard format)."""
        filename = file_data.get("filename", "")
        b64_data = file_data.get("file_data", "")

        if not b64_data:
            return "[Empty file]"

        try:
            data = base64.b64decode(b64_data)
            mime_type = guess_mime_type(filename)
            path = await FileCache.save(data, mime_type, filename, cwd)
            return f"[File: {path}]"
        except Exception as e:
            logger.warning(f"Failed to process file {filename}: {e}")
            return f"[Failed to process file: {filename}]"

    @classmethod
    async def process_content_block(
        cls, block: Dict[str, Any], cwd: Optional[str] = None
    ) -> Optional[str]:
        """Process a content block and return text representation."""
        block_type = block.get("type")

        if block_type == "text":
            return block.get("text", "")

        elif block_type == "image_url":
            image_url = block.get("image_url", {})
            return await cls.process_image_url(image_url, cwd)

        elif block_type == "file":
            file_data = block.get("file", {})
            return await cls.process_file(file_data, cwd)

        return None

    @classmethod
    async def to_claude_prompt(
        cls,
        messages: List[Message],
        cwd: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Convert OpenAI messages to Claude prompt format.

        Files are saved to cache and referenced by path.
        """
        system_prompt = None
        conversation_parts = []

        for msg in messages:
            if isinstance(msg.content, str):
                content_text = msg.content
            else:
                content_parts = []
                for block in msg.content:
                    if isinstance(block, dict):
                        text = await cls.process_content_block(block, cwd)
                        if text:
                            content_parts.append(text)
                content_text = "\n".join(content_parts)

            if msg.role == "system":
                if system_prompt:
                    system_prompt = f"{system_prompt}\n\n{content_text}"
                else:
                    system_prompt = content_text
            elif msg.role == "user":
                conversation_parts.append(f"Human: {content_text}")
            elif msg.role == "assistant":
                conversation_parts.append(f"Assistant: {content_text}")

        prompt = "\n\n".join(conversation_parts)

        if messages and messages[-1].role == "assistant":
            prompt += "\n\nHuman: Please continue."

        return prompt, system_prompt

    @staticmethod
    def filter_response(content: str) -> str:
        """Filter Claude response to remove tool usage tags and thinking blocks."""
        if not content:
            return ""

        # Remove <thinking> blocks
        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)

        # Extract content from <attempt_completion> blocks
        attempt_pattern = r"<attempt_completion>(.*?)</attempt_completion>"
        attempt_matches = re.findall(attempt_pattern, content, flags=re.DOTALL)
        if attempt_matches:
            extracted = attempt_matches[-1].strip()
            result_pattern = r"<result>(.*?)</result>"
            result_matches = re.findall(result_pattern, extracted, flags=re.DOTALL)
            if result_matches:
                return result_matches[-1].strip()
            return extracted

        # Remove tool usage blocks
        tool_tags = [
            "read_file", "write_file", "write_to_file", "bash", "execute_command",
            "edit", "edit_file", "glob", "grep", "search_files", "list_files", "tool_use",
        ]
        for tag in tool_tags:
            content = re.sub(rf"<{tag}>.*?</{tag}>", "", content, flags=re.DOTALL)

        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

        return content.strip()

    @staticmethod
    def extract_text_from_content(content: List) -> str:
        """Extract text from Claude content blocks."""
        texts = []
        for block in content:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif "text" in block:
                    texts.append(block["text"])
        return "\n".join(texts)
```

**Step 2: 验证导入**

Run: `python -c "from src.adapters.openai_adapter import OpenAIAdapter; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/adapters/openai_adapter.py
git commit -m "feat(adapters): add OpenAI adapter with file support"
```

---

## Task 9: 创建 Anthropic 适配器 (adapters/anthropic_adapter.py)

**Files:**
- Create: `src/adapters/anthropic_adapter.py`

**Step 1: 创建 anthropic_adapter.py**

```python
"""Anthropic format adapter - converts Anthropic messages to Claude format."""

import base64
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models.anthropic import Message
from .base import FileCache, fetch_url, parse_data_url

logger = logging.getLogger(__name__)


class AnthropicAdapter:
    """Adapter for converting Anthropic format to Claude internal format."""

    @classmethod
    async def process_image_block(
        cls, block: Dict[str, Any], cwd: Optional[str] = None
    ) -> Optional[str]:
        """Process Anthropic image block."""
        source = block.get("source", {})
        source_type = source.get("type")

        if source_type == "base64":
            media_type = source.get("media_type", "image/png")
            try:
                data = base64.b64decode(source.get("data", ""))
                path = await FileCache.save(data, media_type, None, cwd)
                return f"[Image: {path}]"
            except Exception as e:
                logger.warning(f"Failed to decode base64 image: {e}")
                return "[Failed to process image]"

        elif source_type == "url":
            url = source.get("url", "")
            path = await fetch_url(url, cwd)
            if path:
                return f"[Image: {path}]"
            return "[Failed to fetch image]"

        return None

    @classmethod
    async def process_document_block(
        cls, block: Dict[str, Any], cwd: Optional[str] = None
    ) -> Optional[str]:
        """Process Anthropic document block."""
        source = block.get("source", {})
        source_type = source.get("type")
        title = block.get("title")
        context = block.get("context")

        file_path = None

        if source_type == "text":
            data = source.get("data", "").encode("utf-8")
            media_type = source.get("media_type", "text/plain")
            file_path = await FileCache.save(data, media_type, title, cwd)

        elif source_type == "base64":
            media_type = source.get("media_type", "application/octet-stream")
            try:
                data = base64.b64decode(source.get("data", ""))
                file_path = await FileCache.save(data, media_type, title, cwd)
            except Exception as e:
                logger.warning(f"Failed to decode base64 document: {e}")

        elif source_type == "url":
            url = source.get("url", "")
            file_path = await fetch_url(url, cwd)

        if file_path:
            parts = []
            if title:
                parts.append(f"Title: {title}")
            if context:
                parts.append(f"Context: {context}")
            parts.append(f"File: {file_path}")
            return f"[Document - {', '.join(parts)}]"

        return "[Failed to process document]"

    @classmethod
    async def process_content_block(
        cls, block: Dict[str, Any], cwd: Optional[str] = None
    ) -> Optional[str]:
        """Process a content block and return text representation."""
        block_type = block.get("type")

        if block_type == "text":
            return block.get("text", "")

        elif block_type == "image":
            return await cls.process_image_block(block, cwd)

        elif block_type == "document":
            return await cls.process_document_block(block, cwd)

        return None

    @classmethod
    async def to_claude_prompt(
        cls,
        messages: List[Message],
        cwd: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Convert Anthropic messages to Claude prompt format.

        Files are saved to cache and referenced by path.
        """
        system_prompt = system
        conversation_parts = []

        for msg in messages:
            if isinstance(msg.content, str):
                content_text = msg.content
            else:
                content_parts = []
                for block in msg.content:
                    if isinstance(block, dict):
                        text = await cls.process_content_block(block, cwd)
                        if text:
                            content_parts.append(text)
                content_text = "\n".join(content_parts)

            if msg.role == "user":
                conversation_parts.append(f"Human: {content_text}")
            elif msg.role == "assistant":
                conversation_parts.append(f"Assistant: {content_text}")

        prompt = "\n\n".join(conversation_parts)

        if messages and messages[-1].role == "assistant":
            prompt += "\n\nHuman: Please continue."

        return prompt, system_prompt

    @staticmethod
    def filter_response(content: str) -> str:
        """Filter Claude response to remove tool usage tags and thinking blocks."""
        if not content:
            return ""

        # Remove <thinking> blocks
        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)

        # Extract content from <attempt_completion> blocks
        attempt_pattern = r"<attempt_completion>(.*?)</attempt_completion>"
        attempt_matches = re.findall(attempt_pattern, content, flags=re.DOTALL)
        if attempt_matches:
            extracted = attempt_matches[-1].strip()
            result_pattern = r"<result>(.*?)</result>"
            result_matches = re.findall(result_pattern, extracted, flags=re.DOTALL)
            if result_matches:
                return result_matches[-1].strip()
            return extracted

        # Remove tool usage blocks
        tool_tags = [
            "read_file", "write_file", "write_to_file", "bash", "execute_command",
            "edit", "edit_file", "glob", "grep", "search_files", "list_files", "tool_use",
        ]
        for tag in tool_tags:
            content = re.sub(rf"<{tag}>.*?</{tag}>", "", content, flags=re.DOTALL)

        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

        return content.strip()

    @staticmethod
    def extract_text_from_content(content: List) -> str:
        """Extract text from Claude content blocks."""
        texts = []
        for block in content:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif "text" in block:
                    texts.append(block["text"])
        return "\n".join(texts)
```

**Step 2: 验证导入**

Run: `python -c "from src.adapters.anthropic_adapter import AnthropicAdapter; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/adapters/anthropic_adapter.py
git commit -m "feat(adapters): add Anthropic adapter"
```

---

## Task 10: 更新 adapters/__init__.py

**Files:**
- Modify: `src/adapters/__init__.py`

**Step 1: 更新 __init__.py**

```python
"""Adapters for format conversion."""

from .anthropic_adapter import AnthropicAdapter
from .base import FileCache, estimate_tokens, fetch_url, guess_mime_type, parse_data_url
from .openai_adapter import OpenAIAdapter

__all__ = [
    "AnthropicAdapter",
    "FileCache",
    "OpenAIAdapter",
    "estimate_tokens",
    "fetch_url",
    "guess_mime_type",
    "parse_data_url",
]
```

**Step 2: 验证导入**

Run: `python -c "from src.adapters import OpenAIAdapter, AnthropicAdapter; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/adapters/__init__.py
git commit -m "feat(adapters): update adapters __init__ for new structure"
```

---

## Task 11: 更新 routes/chat.py

**Files:**
- Modify: `src/routes/chat.py`

**Step 1: 更新 imports**

替换文件开头的 imports:

```python
"""Chat completion routes."""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..adapters.openai_adapter import OpenAIAdapter
from ..middleware.auth import AuthResult, verify_api_key
from ..models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    Message,
    ResponseMessage,
    StreamChoice,
    Usage,
)
from ..services.claude import ClaudeService
from ..services.session import SessionManager
```

**Step 2: 更新 _build_completion_context 中的适配器调用**

将 `MessageAdapter.to_claude_prompt` 替换为 `OpenAIAdapter.to_claude_prompt`:

```python
    # Convert messages to Claude format (files saved to cache in cwd)
    cwd = claude_service.cwd if claude_service else None
    prompt, system_prompt = await OpenAIAdapter.to_claude_prompt(request.messages, cwd)
```

**Step 3: 更新 generate_stream 中的 filter_response 调用**

将 `MessageAdapter.filter_response` 替换为 `OpenAIAdapter.filter_response`:

```python
            filtered = OpenAIAdapter.filter_response(content)
```

**Step 4: 更新 generate_response 中的 filter_response 调用**

```python
        content = OpenAIAdapter.filter_response(raw_content or "")
```

**Step 5: 更新 _extract_text 函数中的调用**

```python
def _extract_text(content: Any) -> Optional[str]:
    """Extract text from content (string or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return OpenAIAdapter.extract_text_from_content(content)
    return None
```

**Step 6: 更新 Choice 构造中的 Message 为 ResponseMessage**

在 generate_response 的返回语句中:

```python
        return ChatCompletionResponse(
            id=ctx.request_id,
            model=request.model,
            choices=[Choice(message=ResponseMessage(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(**usage_data),
        )
```

**Step 7: 验证导入**

Run: `python -c "from src.routes.chat import router; print('OK')"`
Expected: OK

**Step 8: Commit**

```bash
git add src/routes/chat.py
git commit -m "refactor(routes): update chat.py to use OpenAIAdapter"
```

---

## Task 12: 更新 routes/anthropic.py

**Files:**
- Modify: `src/routes/anthropic.py`

**Step 1: 更新 imports**

替换文件开头的 imports:

```python
"""Anthropic Messages API routes."""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..adapters.anthropic_adapter import AnthropicAdapter
from ..middleware.auth import AuthResult, verify_api_key
from ..models.anthropic import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    Message,
    MessageDelta,
    MessageStart,
    MessageStop,
    MessagesRequest,
    MessagesResponse,
    TextBlock,
    TextDelta,
    Usage,
)
from ..services.claude import ClaudeService
from ..services.session import SessionManager
```

**Step 2: 更新 MessagesContext 中的 request 类型**

```python
@dataclass
class MessagesContext:
    """Shared context for messages processing."""

    request: MessagesRequest
    ...
```

**Step 3: 更新 _build_messages_context 函数签名和内部逻辑**

```python
async def _build_messages_context(
    request: MessagesRequest,
    auth: AuthResult,
) -> MessagesContext:
    """Build shared context for messages processing."""
    request_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Get Claude session ID if resuming
    claude_session_id = None
    if request.session_id:
        claude_session_id = session_manager.get_claude_session(request.session_id)

    # Convert to Claude format (files saved to cache in cwd)
    cwd = claude_service.cwd if claude_service else None

    # Convert Anthropic messages to internal format for prompt
    internal_messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            internal_messages.append({"role": msg.role, "content": msg.content})
        else:
            # Convert content blocks to dict format
            content_blocks = []
            for block in msg.content:
                if hasattr(block, "model_dump"):
                    content_blocks.append(block.model_dump())
                elif isinstance(block, dict):
                    content_blocks.append(block)
            internal_messages.append({"role": msg.role, "content": content_blocks})

    prompt, system_prompt = await AnthropicAdapter.to_claude_prompt(
        [Message(**m) for m in internal_messages],
        cwd,
        request.system
    )
    ...
```

**Step 4: 更新 filter_response 调用**

将所有 `MessageAdapter.filter_response` 替换为 `AnthropicAdapter.filter_response`:

```python
            filtered = AnthropicAdapter.filter_response(content)
```

和

```python
        content = AnthropicAdapter.filter_response(raw_content or "")
```

**Step 5: 更新 _extract_text 函数**

```python
def _extract_text(content: Any) -> Optional[str]:
    """Extract text from content (string or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return AnthropicAdapter.extract_text_from_content(content)
    return None
```

**Step 6: 更新 endpoint 函数签名**

```python
@router.post("/messages", response_model=MessagesResponse)
async def anthropic_messages(
    request: MessagesRequest,
    ...
```

**Step 7: 更新 thinking 配置访问**

```python
    # Extract max_thinking_tokens from native Anthropic thinking config
    max_thinking_tokens = None
    if request.thinking is not None and request.thinking.type == "enabled":
        max_thinking_tokens = request.thinking.budget_tokens
```

**Step 8: 验证导入**

Run: `python -c "from src.routes.anthropic import router; print('OK')"`
Expected: OK

**Step 9: Commit**

```bash
git add src/routes/anthropic.py
git commit -m "refactor(routes): update anthropic.py to use AnthropicAdapter"
```

---

## Task 13: 更新 routes/models.py

**Files:**
- Modify: `src/routes/models.py`

**Step 1: 检查并更新 imports**

```python
from ..models.openai import ModelInfo, ModelListResponse
```

**Step 2: 验证导入**

Run: `python -c "from src.routes.models import router; print('OK')"`
Expected: OK

**Step 3: Commit (如有改动)**

```bash
git add src/routes/models.py
git commit -m "refactor(routes): update models.py imports"
```

---

## Task 14: 更新 routes/__init__.py

**Files:**
- Modify: `src/routes/__init__.py`

**Step 1: 更新 __init__.py**

```python
"""API routes for Claude OpenAI Wrapper."""

from .anthropic import router as anthropic_router
from .chat import router as chat_router
from .models import router as models_router
from .sessions import router as sessions_router

__all__ = ["anthropic_router", "chat_router", "models_router", "sessions_router"]
```

**Step 2: 验证导入**

Run: `python -c "from src.routes import chat_router, anthropic_router; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/routes/__init__.py
git commit -m "refactor(routes): update routes __init__"
```

---

## Task 15: 删除旧文件

**Files:**
- Delete: `src/models/openai.py`
- Delete: `src/adapters/message.py`

**Step 1: 确认所有导入正常**

Run: `python -c "from src.main import app; print('OK')"`
Expected: OK

**Step 2: 删除旧文件**

```bash
rm src/models/openai.py
rm src/adapters/message.py
```

**Step 3: 验证应用仍可启动**

Run: `python -c "from src.main import app; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove old openai.py and message.py"
```

---

## Task 16: 最终验证

**Step 1: 运行完整导入测试**

```bash
python -c "
from src.models import openai, anthropic
from src.models.openai import ChatCompletionRequest, FileContent, Message
from src.models.anthropic import MessagesRequest, DocumentBlock
from src.adapters import OpenAIAdapter, AnthropicAdapter
from src.routes import chat_router, anthropic_router
from src.main import app
print('All imports OK')
"
```

**Step 2: 启动服务器测试**

```bash
cd /home/farewell/data/docker/services/dev-sandbox/wrapper
timeout 5 python -m src.main || true
```

Expected: 服务器启动无错误 (5秒后超时退出)

**Step 3: Final Commit**

```bash
git add -A
git commit -m "feat: complete OpenAI/Anthropic model separation with file support"
```

---

## 完成检查清单

- [ ] `models/common.py` - 共享类型
- [ ] `models/openai/` - OpenAI 模型目录
- [ ] `models/anthropic/` - Anthropic 模型目录
- [ ] `adapters/base.py` - 基础工具
- [ ] `adapters/openai_adapter.py` - OpenAI 适配器
- [ ] `adapters/anthropic_adapter.py` - Anthropic 适配器
- [ ] `routes/chat.py` - 更新导入
- [ ] `routes/anthropic.py` - 更新导入
- [ ] 删除旧文件
- [ ] 最终验证通过
