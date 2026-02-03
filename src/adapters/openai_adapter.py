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
