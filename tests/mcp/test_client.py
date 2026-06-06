"""Tests for MCPClient — unit (parsing) and integration (live server)."""
import asyncio

import pytest

from app.mcp.client import MCPClient, MCPToolError, _parse_tool_result


class _Text:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Result:
    def __init__(self, content, is_error=False):
        self.content = content
        self.isError = is_error


class TestParseToolResult:
    def test_parses_json_object(self):
        out = _parse_tool_result(_Result([_Text('{"a": 1}')]))
        assert out == {"a": 1}

    def test_parses_json_array(self):
        out = _parse_tool_result(_Result([_Text('[1, 2, 3]')]))
        assert out == [1, 2, 3]

    def test_returns_plain_text_when_not_json(self):
        out = _parse_tool_result(_Result([_Text("just text")]))
        assert out == "just text"

    def test_joins_multiple_text_blocks(self):
        out = _parse_tool_result(_Result([_Text("not"), _Text("json")]))
        assert out == "not\njson"

    def test_ignores_non_text_content(self):
        class _Blob:
            type = "image"

        out = _parse_tool_result(_Result([_Blob(), _Text('{"ok": true}')]))
        assert out == {"ok": True}


class TestNotConnected:
    def test_call_before_connect_raises(self):
        client = MCPClient()
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(client.call_tool("parse_jd", {"raw_text": "x"}))

    def test_default_args_target_project_server(self):
        client = MCPClient()
        assert client.params.args == ["-m", "app.mcp.server"]


class TestAgainstLiveServer:
    """Spawns the real MCP server subprocess and talks to it."""

    def test_list_call_and_error(self):
        async def run():
            async with MCPClient() as client:
                names = await client.list_tools()
                assert {
                    "parse_jd", "parse_resume", "match_resume",
                    "audit_resume", "rank_projects",
                } <= set(names)

                # audit_resume needs no embedding model → fast roundtrip.
                out = await client.call_tool(
                    "audit_resume",
                    {"resume": {"raw_text": "x", "skills": ["mcp"]}},
                )
                assert "findings" in out
                assert "risk_score" in out

                # Invalid input (missing required raw_text) → server error.
                with pytest.raises(MCPToolError):
                    await client.call_tool("audit_resume", {"resume": {}})

        asyncio.run(run())
