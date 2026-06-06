"""Tests for LLMToolSelector — LLM-based tool routing with fallback."""
from app.services.agent.controller import (
    KeywordToolSelector,
    Selection,
    Tool,
    ToolSelector,
)
from app.services.agent.selectors import LLMToolSelector, _parse_tool_name


def _tool(name, keywords):
    return Tool(name=name, description=f"{name} tool", keywords=keywords,
                handler=lambda ctx: name)


TOOLS = [
    _tool("jd_parser", ("jd",)),
    _tool("resume_matcher", ("match",)),
    _tool("project_auditor", ("audit",)),
]


class FakeLLM:
    """Configurable stand-in for LLMClient."""

    def __init__(self, reply="", configured=True, raises=False):
        self.reply = reply
        self.configured = configured
        self.raises = raises
        self.prompts = []

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        self.prompts.append((prompt, system))
        if self.raises:
            raise RuntimeError("api down")
        return self.reply


class TestParseToolName:
    def test_strict_json(self):
        name, reason = _parse_tool_name('{"tool": "jd_parser", "reason": "it is a jd"}', TOOLS)
        assert name == "jd_parser"
        assert reason == "it is a jd"

    def test_json_with_unknown_name_falls_through(self):
        name, _ = _parse_tool_name('{"tool": "nonexistent"}', TOOLS)
        assert name is None

    def test_substring_fallback(self):
        name, reason = _parse_tool_name("I think project_auditor fits best", TOOLS)
        assert name == "project_auditor"
        assert reason

    def test_no_match(self):
        name, _ = _parse_tool_name("none of these apply", TOOLS)
        assert name is None

    def test_malformed_json(self):
        name, _ = _parse_tool_name("{not json", TOOLS)
        assert name is None


class TestLLMToolSelector:
    def test_is_a_tool_selector(self):
        assert isinstance(LLMToolSelector(FakeLLM()), ToolSelector)

    def test_selects_tool_from_json(self):
        sel = LLMToolSelector(FakeLLM(reply='{"tool": "resume_matcher", "reason": "fit"}'))
        result = sel.select("how well do I fit?", TOOLS)
        assert result.tool.name == "resume_matcher"
        assert "LLM selected" in result.reason

    def test_prompt_includes_tools_and_task(self):
        llm = FakeLLM(reply='{"tool": "jd_parser"}')
        LLMToolSelector(llm).select("parse the posting", TOOLS)
        prompt, system = llm.prompts[0]
        assert "jd_parser" in prompt
        assert "parse the posting" in prompt
        assert system is not None

    def test_falls_back_when_not_configured(self):
        # Unconfigured LLM → keyword fallback handles it.
        sel = LLMToolSelector(FakeLLM(configured=False))
        result = sel.select("please audit this", TOOLS)
        assert result.tool.name == "project_auditor"

    def test_falls_back_on_llm_error(self):
        sel = LLMToolSelector(FakeLLM(raises=True))
        result = sel.select("match my resume", TOOLS)
        assert result.tool.name == "resume_matcher"

    def test_falls_back_on_unparseable_reply(self):
        sel = LLMToolSelector(FakeLLM(reply="I have no idea"))
        result = sel.select("audit my resume", TOOLS)
        assert result.tool.name == "project_auditor"  # keyword fallback

    def test_falls_back_with_no_tools(self):
        sel = LLMToolSelector(FakeLLM(reply='{"tool": "x"}'))
        assert sel.select("anything", []).tool is None

    def test_custom_fallback_used(self):
        called = {}

        class _FB:
            def select(self, task, tools):
                called["yes"] = True
                return Selection(tool=None, reason="fb")

        sel = LLMToolSelector(FakeLLM(configured=False), fallback=_FB())
        sel.select("task", TOOLS)
        assert called.get("yes")
