#!/usr/bin/env python3
"""Standalone MCP stdio server that proxies tool calls via TCP to the wrapper.

Started as a subprocess by ToolProxy. Communicates with the SDK via
stdin/stdout (MCP protocol) and with the wrapper via a TCP socket (JSON lines).

Usage:
    python -m src.services.mcp_stdio_proxy <ipc_port> <tools_json>
"""

import asyncio
import json
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


async def main():
    ipc_port = int(sys.argv[1])
    tools = json.loads(sys.argv[2])

    # Connect to wrapper's IPC server
    reader, writer = await asyncio.open_connection("127.0.0.1", ipc_port)

    server = Server("tool_proxy")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t.get("description", ""),
                inputSchema=t.get("input_schema", {"type": "object", "properties": {}}),
            )
            for t in tools
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        # Send tool call to wrapper via TCP
        request = json.dumps({"name": name, "arguments": arguments}) + "\n"
        writer.write(request.encode())
        await writer.drain()

        # Wait for result from wrapper
        line = await asyncio.wait_for(reader.readline(), timeout=300)
        if not line:
            return [TextContent(type="text", text="IPC connection closed")]

        result = json.loads(line.decode())
        content = result.get("content", "")
        is_error = result.get("is_error", False)

        # Normalize content to text
        if isinstance(content, list):
            # Content blocks - extract text
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            text = "\n".join(texts) if texts else str(content)
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        if is_error:
            return [TextContent(type="text", text=f"[Error] {text}")]
        return [TextContent(type="text", text=text)]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

    writer.close()


if __name__ == "__main__":
    asyncio.run(main())
