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
