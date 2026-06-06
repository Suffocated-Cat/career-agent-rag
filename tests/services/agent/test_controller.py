"""Tests for AgentController — rule-based tool routing."""
import pytest

from app.services.agent.controller import (
    AgentContext,
    AgentController,
    Tool,
    ToolResult,
    _matched_keywords,
)


def _echo_tool(name: str, keywords: tuple[str, ...]) -> Tool:
    return Tool(
        name=name,
        description=f"{name} tool",
        keywords=keywords,
        handler=lambda ctx: f"{name}:{ctx.task}",
    )


def _controller() -> AgentController:
    return AgentController(
        [
            _echo_tool("parser", ("parse", "jd", "job description")),
            _echo_tool("matcher", ("match", "fit", "compare")),
            _echo_tool("auditor", ("audit", "risk")),
        ]
    )


class TestMatchedKeywords:
    def test_finds_substring_keywords(self):
        tool = _echo_tool("t", ("job description", "match"))
        assert _matched_keywords(tool, "analyze this job description") == ["job description"]


class TestSelectTool:
    def test_selects_by_keyword(self):
        c = _controller()
        assert c.select_tool("please audit my resume for risk").name == "auditor"

    def test_selects_highest_scoring(self):
        c = _controller()
        # "match" and "fit" both hit matcher (2) vs parser's "jd" (1).
        assert c.select_tool("match and check fit for this jd").name == "matcher"

    def test_no_match_returns_none(self):
        c = _controller()
        assert c.select_tool("write me a poem") is None

    def test_tie_breaks_by_registration_order(self):
        c = AgentController(
            [
                _echo_tool("first", ("alpha",)),
                _echo_tool("second", ("alpha",)),
            ]
        )
        assert c.select_tool("alpha").name == "first"

    def test_case_insensitive(self):
        c = _controller()
        assert c.select_tool("AUDIT THE RISK").name == "auditor"


class TestRegister:
    def test_register_adds_tool(self):
        c = AgentController()
        c.register(_echo_tool("x", ("xkw",)))
        assert "x" in c.tools

    def test_register_replaces_by_name(self):
        c = AgentController()
        c.register(_echo_tool("x", ("a",)))
        c.register(_echo_tool("x", ("b",)))
        assert c.tools["x"].keywords == ("b",)


class TestRun:
    def test_runs_selected_tool(self):
        c = _controller()
        result = c.run(AgentContext(task="audit my resume risk"))
        assert isinstance(result, ToolResult)
        assert result.tool == "auditor"
        assert result.output == "auditor:audit my resume risk"

    def test_reason_lists_matched_keywords(self):
        c = _controller()
        result = c.run(AgentContext(task="audit risk"))
        assert "audit" in result.reason
        assert "risk" in result.reason

    def test_raises_when_no_tool_matches(self):
        c = _controller()
        with pytest.raises(LookupError, match="No tool matched"):
            c.run(AgentContext(task="unrelated request"))


class TestDescribeTools:
    def test_lists_name_and_description(self):
        c = _controller()
        described = c.describe_tools()
        assert {d["name"] for d in described} == {"parser", "matcher", "auditor"}
        assert all("description" in d for d in described)
