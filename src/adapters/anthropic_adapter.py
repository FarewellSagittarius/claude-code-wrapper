"""Anthropic format adapter - converts Anthropic messages to Claude format."""

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models.anthropic import Message
from .base import FileCache, fetch_url

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

        elif block_type == "tool_use":
            # Tool use block from assistant - format as text
            tool_name = block.get("name", "unknown_tool")
            tool_id = block.get("id", "")
            tool_input = block.get("input", {})
            return f"[Tool Call: {tool_name} (id: {tool_id})]\nInput: {json.dumps(tool_input)}"

        elif block_type == "tool_result":
            # Tool result block from user - format as text
            tool_use_id = block.get("tool_use_id", "")
            is_error = block.get("is_error", False)
            content = block.get("content", "")
            if isinstance(content, list):
                # Content can be a list of content blocks
                content_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_parts.append(part.get("text", ""))
                content = "\n".join(content_parts)
            status = "Error" if is_error else "Result"
            return f"[Tool {status} (id: {tool_use_id})]: {content}"

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
                    # Convert Pydantic model to dict if needed
                    if hasattr(block, "model_dump"):
                        block_dict = block.model_dump()
                    elif isinstance(block, dict):
                        block_dict = block
                    else:
                        continue
                    text = await cls.process_content_block(block_dict, cwd)
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
    def filter_response(content: str, strip: bool = True) -> str:
        """Filter Claude response to remove tool usage tags and thinking blocks.

        Args:
            content: The content to filter
            strip: Whether to strip leading/trailing whitespace (False for streaming)
        """
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

        return content.strip() if strip else content

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
