"""Tests for the tool-selection evaluation harness."""
from pathlib import Path

from app.eval.selector_eval import (
    CaseResult,
    SelectorCase,
    SelectorReport,
    evaluate_selector,
    load_selector_cases,
)
from app.services.agent.controller import (
    KeywordToolSelector,
    Selection,
    Tool,
)
from app.services.agent.tools import default_tools

FIXTURE = Path(__file__).parent.parent / "fixtures" / "tool_selection.json"


class _FakeSelector:
    """Selector that returns the tool named by a task→name mapping."""

    def __init__(self, mapping: dict[str, str | None]):
        self.mapping = mapping

    def select(self, task, tools):
        name = self.mapping.get(task)
        tool = next((t for t in tools if t.name == name), None)
        return Selection(tool=tool, reason="fake")


def _tool(name):
    return Tool(name=name, description=name, keywords=(name,), handler=lambda ctx: name)


class TestLoadCases:
    def test_loads_fixture(self):
        cases = load_selector_cases(FIXTURE)
        assert len(cases) == 16
        assert all(isinstance(c, SelectorCase) for c in cases)
        # Includes "no tool" cases.
        assert any(c.expected is None for c in cases)


class TestEvaluateSelector:
    def test_perfect_selector(self):
        tools = [_tool("a"), _tool("b")]
        cases = [SelectorCase("do a", "a"), SelectorCase("do b", "b")]
        report = evaluate_selector(
            _FakeSelector({"do a": "a", "do b": "b"}), tools, cases
        )
        assert isinstance(report, SelectorReport)
        assert report.accuracy == 1.0
        assert report.n == 2
        assert all(isinstance(r, CaseResult) for r in report.results)

    def test_handles_none_expected(self):
        tools = [_tool("a")]
        cases = [SelectorCase("unrelated", None)]
        report = evaluate_selector(_FakeSelector({"unrelated": None}), tools, cases)
        assert report.accuracy == 1.0
        assert report.results[0].predicted is None

    def test_counts_wrong_predictions(self):
        tools = [_tool("a"), _tool("b")]
        cases = [SelectorCase("do a", "a")]
        report = evaluate_selector(_FakeSelector({"do a": "b"}), tools, cases)
        assert report.accuracy == 0.0
        assert report.results[0].correct is False

    def test_empty_cases(self):
        report = evaluate_selector(_FakeSelector({}), [_tool("a")], [])
        assert report.accuracy == 0.0
        assert report.n == 0


class TestKeywordSelectorOnDataset:
    def test_reasonable_but_imperfect(self):
        cases = load_selector_cases(FIXTURE)
        report = evaluate_selector(KeywordToolSelector(), default_tools(), cases)
        # Strong baseline, but ambiguous phrasings keep it below perfect —
        # which is exactly the headroom an LLM selector can close.
        assert report.accuracy >= 0.75
        assert report.accuracy < 1.0

    def test_ambiguous_match_case_is_misrouted(self):
        cases = load_selector_cases(FIXTURE)
        report = evaluate_selector(KeywordToolSelector(), default_tools(), cases)
        by_task = {r.task: r for r in report.results}
        # "How well does my resume match this job?" — 'resume' and 'match'
        # tie, so registration order wins and it routes to the parser.
        ambiguous = by_task["How well does my resume match this job?"]
        assert ambiguous.expected == "resume_matcher"
        assert not ambiguous.correct
