"""Claude Agent SDK service wrapper."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ..adapters.base import estimate_tokens
from ..adapters.openai_adapter import OpenAIAdapter
from ..config import settings

logger = logging.getLogger(__name__)


def load_user_mcp_servers(cwd: str) -> Dict[str, Any]:
    """
    Load mcpServers from user's .claude.json config for the given project.

    SDK's setting_sources loads plugins but NOT mcpServers, so we need to
    manually read and pass them.
    """
    home = Path.home()
    config_path = home / ".claude.json"

    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read config {config_path}: {e}")
        return {}

    projects = config.get("projects", {})
    cwd_normalized = str(Path(cwd).resolve())

    # Try exact match first
    if cwd_normalized in projects:
        mcp_servers = projects[cwd_normalized].get("mcpServers", {})
        if mcp_servers:
            logger.info(f"Loaded {len(mcp_servers)} MCP servers: {list(mcp_servers.keys())}")
            return mcp_servers

    # Try parent path match
    for project_path, project_config in projects.items():
        if cwd_normalized.startswith(project_path):
            mcp_servers = project_config.get("mcpServers", {})
            if mcp_servers:
                logger.info(f"Loaded {len(mcp_servers)} MCP servers from {project_path}: {list(mcp_servers.keys())}")
                return mcp_servers

    return {}


class ClaudeService:
    """Service for interacting with Claude Agent SDK."""

    def __init__(
        self,
        cwd: Optional[str] = None,
        timeout: int = 600,
    ):
        """
        Initialize Claude service.

        Args:
            cwd: Working directory for Claude operations
            timeout: Timeout in seconds
        """
        self.timeout = timeout

        # Set working directory
        if cwd:
            self.cwd = cwd
        elif settings.CLAUDE_CWD:
            self.cwd = settings.CLAUDE_CWD
        else:
            # Create isolated temp directory
            self.temp_dir = tempfile.mkdtemp(prefix="claude_workspace_")
            self.cwd = self.temp_dir

        # Ensure directory exists
        os.makedirs(self.cwd, exist_ok=True)

    async def run_completion(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        enable_internal_tools: bool = False,
        enable_internal_mcp: bool = False,
        max_turns: int = 10,
        session_id: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        max_thinking_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute Claude query and yield message chunks.

        Args:
            prompt: The prompt to send (string for text-only, list for multimodal)
            system_prompt: Optional system prompt
            model: Model to use
            enable_internal_tools: Whether to enable internal/built-in tools (API key controlled)
            enable_internal_mcp: Whether to enable internal MCP servers (API key controlled)
            max_turns: Maximum conversation turns
            session_id: Session ID for resumption
            allowed_tools: External tools from request (always enabled if provided)
            disallowed_tools: List of disallowed tools
            mcp_servers: External MCP servers from request (always enabled if provided)
            stream: Whether to enable streaming (partial messages)
            max_thinking_tokens: Extended thinking token budget (None = disabled)

        Yields:
            Message dictionaries from Claude
        """
        try:
            # Import here to avoid issues if SDK not installed
            from claude_agent_sdk import query, ClaudeAgentOptions
        except ImportError:
            logger.error("claude-agent-sdk package not installed")
            raise ImportError(
                "claude-agent-sdk package is required. Install with: pip install claude-agent-sdk"
            )

        # Determine if any tools/MCP will be used
        has_any_tools = enable_internal_tools or allowed_tools or mcp_servers or enable_internal_mcp

        # Build options
        options = ClaudeAgentOptions(
            max_turns=max_turns if has_any_tools else 1,
            cwd=self.cwd,
            include_partial_messages=stream,
        )

        # Extended thinking configuration
        if max_thinking_tokens and max_thinking_tokens >= 1024:
            options.max_thinking_tokens = max_thinking_tokens
            logger.info(f"Extended thinking enabled with budget: {max_thinking_tokens} tokens")

        # System prompt
        if system_prompt:
            options.system_prompt = system_prompt

        # Tool configuration
        # 1. External tools (from request.allowed_tools) - always enabled if provided
        # 2. Internal tools (from settings.DEFAULT_ALLOWED_TOOLS) - controlled by API key
        combined_tools = []

        # Add external tools (always)
        if allowed_tools:
            combined_tools.extend(allowed_tools)

        # Add internal tools (if enabled by API key)
        if enable_internal_tools:
            combined_tools.extend(settings.CLAUDE_TOOLS)

        if combined_tools:
            # Deduplicate while preserving order
            options.allowed_tools = list(dict.fromkeys(combined_tools))
            options.permission_mode = "bypassPermissions"
        else:
            # No tools enabled - disable all
            options.disallowed_tools = settings.CLAUDE_TOOLS

        # Additional disallowed tools
        if disallowed_tools:
            existing = options.disallowed_tools or []
            options.disallowed_tools = list(set(existing + disallowed_tools))

        # MCP servers configuration
        # External MCP (from request) - always enabled if provided
        if mcp_servers:
            options.mcp_servers = mcp_servers
            options.permission_mode = "bypassPermissions"
            logger.info(f"External MCP servers configured: {list(mcp_servers.keys())}")

        # Internal MCP (from user's settings) - controlled by API key
        if enable_internal_mcp:
            # Heavy mode: enable SDK to load user-configured MCP servers and plugins
            options.setting_sources = ["user", "project"]
            options.permission_mode = "bypassPermissions"

            # SDK's setting_sources loads plugins but NOT mcpServers
            # We need to manually load and merge them
            user_mcp_servers = load_user_mcp_servers(self.cwd)
            if user_mcp_servers:
                # Merge with any existing mcp_servers (external takes precedence)
                existing = options.mcp_servers or {}
                options.mcp_servers = {**user_mcp_servers, **existing}
                logger.info(f"Internal MCP mode enabled with servers: {list(options.mcp_servers.keys())}")
            else:
                logger.info("Internal MCP mode enabled (loading user settings with plugins)")

        # Session resumption
        if session_id:
            options.resume = session_id

        # Model selection - resolve alias to official model name
        options.model = settings.resolve_model(model or settings.DEFAULT_MODEL)

        # Handle multimodal prompts
        # The claude-agent-sdk currently doesn't support vision/images directly
        # Extract text content for now
        if isinstance(prompt, list):
            # Multimodal: extract text from the last user message
            text_parts = []
            has_images = False
            for msg in prompt:
                content = msg.get("content", [])
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "image":
                            has_images = True

            prompt_text = "\n\n".join(text_parts) if text_parts else "Hello"
            if has_images:
                logger.warning("Images in request will be ignored - SDK does not support vision")
        else:
            prompt_text = prompt

        # Log prompt (truncate for display)
        prompt_preview = prompt_text[:100] if len(prompt_text) > 100 else prompt_text
        logger.debug(f"Running Claude completion with prompt: {prompt_preview}...")

        # Execute query
        async for message in query(prompt=prompt_text, options=options):
            yield self._normalize_message(message)

    def _normalize_message(self, message: Any) -> Dict[str, Any]:
        """Convert message object to dictionary."""
        if isinstance(message, dict):
            return message
        elif hasattr(message, "model_dump"):
            return message.model_dump()
        elif hasattr(message, "__dict__"):
            return vars(message)
        else:
            return {"raw": str(message)}

    def _extract_text_content(self, content: Any) -> Optional[str]:
        """Extract text from content (string or list of blocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text = OpenAIAdapter.extract_text_from_content(content)
            return text if text else None
        return None

    def extract_response(self, messages: List[Dict]) -> Optional[str]:
        """Extract final response text from message stream."""
        # Priority 1: ResultMessage (success subtype)
        for msg in messages:
            if msg.get("subtype") == "success" and "result" in msg:
                return msg["result"]

        # Priority 2: Direct content field
        for msg in reversed(messages):
            if "content" in msg:
                text = self._extract_text_content(msg["content"])
                if text:
                    return text

        # Priority 3: Nested message.content field
        for msg in reversed(messages):
            if "message" in msg:
                message = msg["message"]
                if isinstance(message, str):
                    return message
                if isinstance(message, dict) and "content" in message:
                    return self._extract_text_content(message["content"])

        return None

    def get_session_id(self, messages: List[Dict]) -> Optional[str]:
        """
        Extract session ID from initialization message.

        Args:
            messages: List of message dictionaries

        Returns:
            Session ID or None
        """
        for msg in messages:
            if msg.get("subtype") == "init":
                data = msg.get("data", {})
                if isinstance(data, dict):
                    return data.get("session_id")
            # Also check for sessionId at message level
            if "sessionId" in msg:
                return msg["sessionId"]
            if "session_id" in msg:
                return msg["session_id"]
        return None

    def extract_usage(self, messages: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Extract real token usage from result message.

        Args:
            messages: List of message dictionaries

        Returns:
            Usage dictionary with prompt_tokens, completion_tokens, total_tokens,
            and additional fields for OpenWebUI compatibility
        """
        for msg in messages:
            # Check for result message (subtype=success) with usage data
            if msg.get("subtype") == "success" and "usage" in msg:
                usage = msg["usage"]
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cache_creation = usage.get("cache_creation_input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)

                # Calculate total prompt tokens (input + cache)
                prompt_tokens = input_tokens + cache_creation + cache_read

                # Import models here to avoid circular import
                from ..models.openai import PromptTokensDetails, CompletionTokensDetails

                return {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": prompt_tokens + output_tokens,
                    # Additional details for OpenWebUI
                    "prompt_tokens_details": PromptTokensDetails(
                        cached_tokens=cache_read,
                        audio_tokens=0,
                    ),
                    "completion_tokens_details": CompletionTokensDetails(
                        reasoning_tokens=0,
                        audio_tokens=0,
                        accepted_prediction_tokens=0,
                        rejected_prediction_tokens=0,
                    ),
                }
        return None

    def estimate_usage(
        self, prompt: Union[str, List[Dict[str, Any]]], completion: str, model: str
    ) -> Dict[str, int]:
        """
        Estimate token usage (fallback when real usage not available).

        Args:
            prompt: Input prompt (string or multimodal list)
            completion: Output completion
            model: Model name (for future model-specific estimation)

        Returns:
            Dictionary with token counts
        """
        # Handle multimodal prompts
        if isinstance(prompt, list):
            # Extract text from multimodal content for estimation
            text_parts = []
            for msg in prompt:
                content = msg.get("content", [])
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "image":
                            # Images use ~85 tokens per tile (rough estimate)
                            text_parts.append("[IMAGE]" * 85)
            prompt_text = " ".join(text_parts)
        else:
            prompt_text = prompt

        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(completion)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
