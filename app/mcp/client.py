"""
MCPClient — connect to an MCP server and call its tools.

Wraps the MCP stdio transport and ClientSession into an async context manager
with convenience methods, so callers can discover and invoke tools without
dealing with the protocol plumbing:

    async with MCPClient() as client:
        names = await client.list_tools()
        result = await client.call_tool("parse_jd", {"raw_text": "..."})

By default it launches this project's own server (``python -m app.mcp.server``),
but any command can be supplied to talk to a different MCP server.
"""

import json

from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPToolError(RuntimeError):
    """Raised when an MCP tool call returns an error result."""


def _parse_tool_result(result: Any) -> Any:
    """Turn a CallToolResult into a Python value.

    Concatenates the text content blocks and parses them as JSON when
    possible; otherwise returns the raw text.

    Args:
        result: A CallToolResult-like object with a ``content`` list.

    Returns:
        A dict/list (if the content was JSON) or the raw text string.
    """
    texts = [c.text for c in result.content if hasattr(c, "text")]
    text = "\n".join(texts)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


class MCPClient:
    """Async client for an MCP server over stdio."""

    def __init__(self, command: str = "python", args: list[str] | None = None):
        """Configure how to launch/connect to the server.

        Args:
            command: Executable to run the server (default: "python").
            args: Arguments to it (default: ["-m", "app.mcp.server"]).
        """
        self.params = StdioServerParameters(
            command=command,
            args=args if args is not None else ["-m", "app.mcp.server"],
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "MCPClient":
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(self.params))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected; use 'async with'.")
        return self._session

    async def list_tools(self) -> list[str]:
        """Return the names of tools the server exposes."""
        result = await self._require_session().list_tools()
        return [tool.name for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        """Call a tool by name and return its parsed result.

        Args:
            name: Tool name (as listed by ``list_tools``).
            arguments: JSON-serializable arguments for the tool.

        Returns:
            The tool output, JSON-decoded to a dict/list when possible.

        Raises:
            MCPToolError: If the server reports the call as an error.
        """
        result = await self._require_session().call_tool(name, arguments or {})
        if getattr(result, "isError", False):
            detail = _parse_tool_result(result)
            raise MCPToolError(f"Tool {name!r} failed: {detail}")
        return _parse_tool_result(result)
