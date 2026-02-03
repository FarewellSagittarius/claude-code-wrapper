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
    MAX_TIMEOUT: int = int(os.getenv("MAX_TIMEOUT", "600000"))

    # Authentication
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_AUTH_METHOD: Optional[str] = os.getenv("CLAUDE_AUTH_METHOD")

    # Multi-key authentication with different tool permissions
    # tool_mode:
    #   "light" = no tools (chat only)
    #   "basic" = all built-in tools
    #   "heavy" = all tools + plugins + user MCP servers
    #   "custom" = use request settings
    API_KEYS: dict[str, dict] = {
        "sk-light": {
            "key": os.getenv("API_KEY_LIGHT", "sk-light-dev"),
            "tools": "light",  # No tools, chat only
        },
        "sk-basic": {
            "key": os.getenv("API_KEY_BASIC", "sk-basic-dev"),
            "tools": "basic",  # All built-in tools
        },
        "sk-heavy": {
            "key": os.getenv("API_KEY_HEAVY", "sk-heavy-dev"),
            "tools": "heavy",  # All tools + plugins + MCP servers
        },
        "sk-custom": {
            "key": os.getenv("API_KEY_CUSTOM", "sk-custom-dev"),
            "tools": "custom",  # Use request settings
        },
    }

    # Reasoning effort to max_thinking_tokens mapping
    # Compatible with OpenAI o1/o3 reasoning_effort parameter
    REASONING_EFFORT_MAP: dict[str, int | None] = {
        "none": None,    # Disable extended thinking
        "low": 8000,     # Light reasoning
        "medium": 16000, # Balanced reasoning
        "high": 31999,   # Maximum reasoning (Claude Code default max)
    }

    # Debug
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # Models - alias to official model mapping
    MODEL_ALIASES: dict[str, str] = {
        "claude-code-opus": "claude-opus-4-5-20251101",
        "claude-code-sonnet": "claude-sonnet-4-5-20250929",
        "claude-code-haiku": "claude-haiku-4-5-20251001",
    }
    DEFAULT_MODEL: str = "claude-code-opus"
    SUPPORTED_MODELS: list[str] = list(MODEL_ALIASES.keys())

    @classmethod
    def resolve_model(cls, model: str) -> str:
        """Resolve model alias to official model name."""
        return cls.MODEL_ALIASES.get(model, model)

    # All Claude Code built-in tools
    # Reference: https://code.claude.com/docs/en/skills
    CLAUDE_TOOLS: list[str] = [
        "Task",
        "Bash",
        "Glob",
        "Grep",
        "Read",
        "Edit",
        "Write",
        "NotebookEdit",
        "WebFetch",
        "TodoWrite",
        "WebSearch",
        "Skill",
    ]


settings = Settings()
