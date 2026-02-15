"""Configuration management for Claude OpenAI Wrapper."""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    # Server
    PORT: int = int(os.getenv("PORT", "8790"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Claude
    CLAUDE_CWD: Optional[str] = os.getenv("CLAUDE_CWD")

    # Internal API key for backend authentication
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "sk-claude-code-wrapper")

    # Debug
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    # Detailed payload logging (requests/responses)
    DEBUG_LOG_PAYLOADS: bool = os.getenv("DEBUG_LOG_PAYLOADS", "false").lower() == "true"
    # Maximum length for logged payloads (0 = unlimited)
    DEBUG_LOG_MAX_LENGTH: int = int(os.getenv("DEBUG_LOG_MAX_LENGTH", "0"))
    # Log file directory (relative to project root or absolute path)
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    # Log file name
    LOG_FILE: str = os.getenv("LOG_FILE", "wrapper.log")
    # Enable file logging
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"

    # Direct pass-through to SDK options.tools
    # Not set or "None" → SDK default (all tools); empty "" → []; "Task,Bash,Read" → ["Task","Bash","Read"]
    TOOLS: Optional[str] = None if os.getenv("TOOLS", "None").lower() == "none" else os.getenv("TOOLS")
    LOAD_USER_MCP: bool = os.getenv("LOAD_USER_MCP", "true").lower() == "true"

    # Thinking: expose thinking blocks to client (default: filtered out)
    EXPOSE_THINKING: bool = os.getenv("EXPOSE_THINKING", "false").lower() == "true"


settings = Settings()
