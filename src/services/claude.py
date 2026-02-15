"""Claude Agent SDK service wrapper."""

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ..adapters.anthropic_adapter import AnthropicAdapter
from ..adapters.base import estimate_tokens
from ..config import settings
from ..utils.debug_logger import log_internal_request, log_internal_response, log_sdk_chunk

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
    home_str = str(home)

    # Merge all scopes: user scope → project scope (project wins on conflict)
    merged: Dict[str, Any] = {}

    # 1. Top-level mcpServers (user scope)
    merged.update(config.get("mcpServers", {}))

    # 2. Project-level mcpServers (exact cwd match)
    if cwd_normalized in projects:
        merged.update(projects[cwd_normalized].get("mcpServers", {}))

    if merged:
        logger.info(f"Loaded {len(merged)} MCP servers (merged): {list(merged.keys())}")

    return merged


class ClaudeService:
    """Service for interacting with Claude Agent SDK."""

    def __init__(self, cwd: Optional[str] = None):
        """
        Initialize Claude service.

        Args:
            cwd: Working directory for Claude operations
        """
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
        prompt: str,
        system_prompt: Optional[Any] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute Claude query and yield message chunks.

        Args:
            prompt: Text prompt (images referenced as [Image: path])
            system_prompt: Optional system prompt
            model: Model name (opus/sonnet/haiku sets SDK model)
            session_id: Session ID for resumption
            mcp_servers: External MCP servers from request
            stream: Whether to enable streaming (partial messages)

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

        # Build options
        options = ClaudeAgentOptions(
            cwd=self.cwd,
            include_partial_messages=stream,
        )

        # System prompt
        if system_prompt:
            options.system_prompt = system_prompt

        # Tools: pass-through from env var to SDK
        if settings.TOOLS is not None:
            if settings.TOOLS.strip() == "":
                options.tools = []
            else:
                options.tools = [t.strip() for t in settings.TOOLS.split(",")]

        options.permission_mode = "bypassPermissions"
        options.thinking = {"type": "adaptive"}
        options.effort = "high"
        options.setting_sources = ["user", "project"] if settings.LOAD_USER_MCP else []

        # User MCP servers
        if settings.LOAD_USER_MCP:
            user_mcp_servers = load_user_mcp_servers(self.cwd)
            if user_mcp_servers:
                options.mcp_servers = {**user_mcp_servers, **(options.mcp_servers or {})}
                logger.info(f"User MCP servers: {list(user_mcp_servers.keys())}")

        # External MCP servers from request (always loaded)
        if mcp_servers:
            options.mcp_servers = {**(options.mcp_servers or {}), **mcp_servers}
            logger.info(f"External MCP servers: {list(mcp_servers.keys())}")

        # Session resumption
        if session_id:
            options.resume = session_id

        # Model selection: check for known model aliases
        if model:
            for alias in ("opus", "sonnet", "haiku"):
                if alias in model:
                    options.model = alias
                    break

        # Prompt is already converted to text by AnthropicAdapter.to_claude_prompt()
        # Images are saved to .claude_media_cache/ and referenced as [Image: /path]
        # SDK can read these files via the Read tool
        prompt_text = prompt

        # Log prompt and options (truncate for display)
        prompt_preview = prompt_text[:100] if len(prompt_text) > 100 else prompt_text
        logger.debug(f"Running Claude completion with prompt: {prompt_preview}...")
        logger.info(f"SDK options: tools={getattr(options, 'tools', None)}, permission_mode={getattr(options, 'permission_mode', None)}")

        # Generate internal request ID for logging
        internal_request_id = f"sdk_{uuid.uuid4().hex[:12]}"

        # Log internal request to SDK
        log_internal_request(
            request_id=internal_request_id,
            prompt=prompt_text,
            system_prompt=system_prompt,
            model=options.model,
            options={
                "tools": getattr(options, "tools", None),
                "mcp_servers": list(options.mcp_servers.keys()) if options.mcp_servers else None,
                "include_partial_messages": options.include_partial_messages,
                "permission_mode": getattr(options, "permission_mode", None),
            },
        )

        # Execute query and collect messages for logging
        messages_for_log = []

        async for message in query(prompt=prompt_text, options=options):
            normalized = self._normalize_message(message)
            messages_for_log.append(normalized)
            log_sdk_chunk(internal_request_id, normalized)
            yield normalized

        # Log internal response summary
        final_content = self.extract_response(messages_for_log)
        log_internal_response(
            request_id=internal_request_id,
            messages=messages_for_log,
            final_content=final_content,
        )

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
            text = AnthropicAdapter.extract_text_from_content(content)
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
            Usage dictionary with input_tokens and output_tokens (Anthropic format)
        """
        for msg in messages:
            # Check for result message (subtype=success) with usage data
            if msg.get("subtype") == "success" and "usage" in msg:
                usage = msg["usage"]
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cache_creation = usage.get("cache_creation_input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)

                return {
                    "input_tokens": input_tokens + cache_creation + cache_read,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                }
        return None

    def estimate_usage(
        self, prompt: str, completion: str, model: str
    ) -> Dict[str, int]:
        """
        Estimate token usage (fallback when real usage not available).

        Args:
            prompt: Input prompt text
            completion: Output completion
            model: Model name (for future model-specific estimation)

        Returns:
            Dictionary with input_tokens and output_tokens (Anthropic format)
        """
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(completion)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
