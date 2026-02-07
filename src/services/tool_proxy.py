"""Transparent MCP proxy for external tool support.

Converts Anthropic API tool definitions into a stdio MCP server subprocess.
The SDK communicates with the subprocess via MCP protocol (stdin/stdout).
The subprocess communicates with the wrapper via TCP IPC for tool call
blocking — the handler sends a tool_call over TCP, wrapper pushes it to
the event queue and blocks on a Future until the client submits a tool_result.
"""

import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Path to the standalone MCP stdio proxy script
_PROXY_SCRIPT = str(Path(__file__).parent / "mcp_stdio_proxy.py")


@dataclass
class ToolProxy:
    """Per-session proxy state for external tool calls."""

    session_id: str
    tools: List[Dict[str, Any]]  # Original Anthropic tool definitions
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pending_calls: Dict[str, asyncio.Future] = field(default_factory=dict)
    mcp_server_config: Optional[Dict[str, Any]] = None
    _sdk_task: Optional[asyncio.Task] = None
    _ipc_server: Optional[asyncio.Server] = None
    _ipc_port: int = 0

    @property
    def has_pending_calls(self) -> bool:
        """Check if there are unresolved tool calls."""
        return any(not f.done() for f in self.pending_calls.values())

    def submit_tool_result(
        self,
        tool_use_id: str,
        content: Any,
        is_error: bool = False,
    ) -> bool:
        """Resolve a pending tool call Future with client-provided result.

        Returns True if the Future was found and resolved.
        """
        future = self.pending_calls.get(tool_use_id)
        if future is None or future.done():
            logger.warning(f"No pending future for tool_use_id={tool_use_id}")
            return False

        future.set_result({"content": content, "is_error": is_error})
        logger.info(f"Resolved tool_use_id={tool_use_id} is_error={is_error}")
        return True

    def cancel_all(self) -> None:
        """Cancel all pending futures and the SDK task."""
        for tid, future in self.pending_calls.items():
            if not future.done():
                future.cancel()
                logger.debug(f"Cancelled pending future {tid}")
        if self._sdk_task and not self._sdk_task.done():
            self._sdk_task.cancel()
            logger.debug(f"Cancelled SDK task for session {self.session_id}")
        if self._ipc_server:
            self._ipc_server.close()
            logger.debug(f"Closed IPC server for session {self.session_id}")


async def _handle_ipc_connection(
    proxy: ToolProxy,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Handle a single IPC connection from the MCP subprocess.

    Each tool call arrives as a JSON line. We push it to the event queue
    (so the SSE generator can emit tool_use), block on a Future until
    the client provides a tool_result, then send the result back.
    """
    loop = asyncio.get_event_loop()

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            try:
                call = json.loads(line.decode())
            except json.JSONDecodeError:
                logger.warning(f"Invalid IPC message: {line!r}")
                continue

            tool_name = call.get("name", "unknown")
            arguments = call.get("arguments", {})
            tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
            future = loop.create_future()
            proxy.pending_calls[tool_use_id] = future

            # Push tool_use event to SSE generator
            await proxy.event_queue.put({
                "type": "tool_use_pending",
                "tool_use_id": tool_use_id,
                "name": tool_name,
                "input": arguments,
            })

            logger.info(f"IPC tool call: {tool_name} id={tool_use_id}, blocking...")

            try:
                result = await asyncio.wait_for(future, timeout=300)
            except asyncio.TimeoutError:
                result = {"content": "Tool call timed out (5 min)", "is_error": True}
            except asyncio.CancelledError:
                result = {"content": "Tool call cancelled", "is_error": True}

            logger.info(f"IPC tool result: {tool_name} id={tool_use_id}")

            # Send result back to subprocess
            response = json.dumps(result) + "\n"
            writer.write(response.encode())
            await writer.drain()

    except Exception as e:
        logger.error(f"IPC handler error: {e}", exc_info=True)
    finally:
        writer.close()


async def create_tool_proxy(
    session_id: str,
    tools: List[Dict[str, Any]],
) -> ToolProxy:
    """Create a ToolProxy with a stdio MCP subprocess connected via TCP IPC.

    1. Starts a TCP IPC server on a random port
    2. Builds mcp_servers config pointing to the subprocess script
    3. SDK will spawn the subprocess when query() starts
    """
    proxy = ToolProxy(session_id=session_id, tools=tools)

    # Start TCP IPC server on a random port
    ipc_server = await asyncio.start_server(
        lambda r, w: _handle_ipc_connection(proxy, r, w),
        "127.0.0.1",
        0,  # random port
    )
    proxy._ipc_server = ipc_server
    proxy._ipc_port = ipc_server.sockets[0].getsockname()[1]

    logger.info(f"IPC server listening on port {proxy._ipc_port}")

    # Build stdio MCP server config for the SDK
    # Use direct script path (not -m) because SDK may run subprocess
    # in a different cwd where module resolution would fail
    tools_json = json.dumps(tools, ensure_ascii=False)
    server_name = f"external_tools_{session_id[:8]}"
    proxy.mcp_server_config = {
        server_name: {
            "type": "stdio",
            "command": sys.executable,
            "args": [
                _PROXY_SCRIPT,
                str(proxy._ipc_port),
                tools_json,
            ],
        }
    }

    logger.info(
        f"Created stdio MCP proxy '{server_name}' with {len(tools)} tools, "
        f"IPC port {proxy._ipc_port}"
    )
    return proxy


# =============================================================================
# Global Proxy Registry
# =============================================================================

_active_proxies: Dict[str, ToolProxy] = {}


def register_proxy(session_id: str, proxy: ToolProxy) -> None:
    """Register a proxy in the global registry."""
    _active_proxies[session_id] = proxy
    logger.debug(f"Registered proxy for session {session_id}")


def get_proxy(session_id: str) -> Optional[ToolProxy]:
    """Get a proxy by session ID."""
    return _active_proxies.get(session_id)


def remove_proxy(session_id: str) -> Optional[ToolProxy]:
    """Remove and return a proxy from the registry."""
    proxy = _active_proxies.pop(session_id, None)
    if proxy:
        logger.debug(f"Removed proxy for session {session_id}")
    return proxy


def cleanup_all_proxies() -> None:
    """Cancel and remove all active proxies. Called on shutdown."""
    for sid, proxy in list(_active_proxies.items()):
        proxy.cancel_all()
    _active_proxies.clear()
    logger.info("Cleaned up all active proxies")
