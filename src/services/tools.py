"""Tool classification and management for external tool support."""

import logging
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


# Internal tools provided by Claude Agent SDK
# These are handled by the SDK automatically and should not be passed to clients
INTERNAL_TOOLS: Set[str] = {
    # File operations
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    # Shell operations
    "Bash",
    # Search operations
    "Glob",
    "Grep",
    "LS",
    # Web operations
    "WebFetch",
    "WebSearch",
    # Task management
    "TodoRead",
    "TodoWrite",
    "Task",
    # Jupyter
    "Jupyter",
    "NotebookEdit",
}


def classify_tools(
    tools: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Classify tools into internal (SDK-handled) and external (client-handled).

    Args:
        tools: List of tool definitions in Anthropic format
            Each tool should have 'name', 'description', and 'input_schema'

    Returns:
        Tuple of (internal_tool_names, external_tool_definitions)
        - internal_tool_names: List of tool names to enable in SDK
        - external_tool_definitions: List of full tool defs for external tools
    """
    internal: List[str] = []
    external: List[Dict[str, Any]] = []

    for tool in tools:
        name = tool.get("name", "")
        if name in INTERNAL_TOOLS:
            internal.append(name)
            logger.debug(f"Tool '{name}' classified as internal")
        else:
            external.append(tool)
            logger.debug(f"Tool '{name}' classified as external")

    logger.info(
        f"Classified {len(tools)} tools: "
        f"{len(internal)} internal, {len(external)} external"
    )

    return internal, external


def is_internal_tool(name: str) -> bool:
    """Check if a tool name is an internal SDK tool."""
    return name in INTERNAL_TOOLS
