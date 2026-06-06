"""
Evaluate tool selectors against a labeled task→tool dataset.

Loads tasks paired with the tool that should handle them, runs a selector over
each, and reports accuracy. This lets the keyword and LLM selectors be compared
on the same cases (the tool-routing analogue of the retrieval eval harness).

A label of ``None`` means "no tool should match"; the selector is expected to
return no tool for those.
"""

import json

from dataclasses import dataclass
from pathlib import Path

from app.services.agent.controller import Tool, ToolSelector


@dataclass
class SelectorCase:
    """One labeled routing example."""

    task: str
    expected: str | None  # expected tool name, or None for "no tool"


@dataclass
class CaseResult:
    """The outcome of a selector on a single case."""

    task: str
    expected: str | None
    predicted: str | None
    correct: bool


@dataclass
class SelectorReport:
    """Aggregate accuracy of a selector over a dataset."""

    accuracy: float
    n: int
    results: list[CaseResult]


def load_selector_cases(path: str | Path) -> list[SelectorCase]:
    """Load labeled tool-selection cases from a JSON file."""
    data = json.loads(Path(path).read_text())
    return [SelectorCase(task=c["task"], expected=c["expected"]) for c in data]


def evaluate_selector(
    selector: ToolSelector,
    tools: list[Tool],
    cases: list[SelectorCase],
) -> SelectorReport:
    """Run *selector* over *cases* and report accuracy.

    Args:
        selector: The selector under test.
        tools: The tool set the selector chooses from.
        cases: Labeled task→expected-tool examples.

    Returns:
        A SelectorReport with per-case results and overall accuracy.
    """
    results: list[CaseResult] = []
    for case in cases:
        chosen = selector.select(case.task, tools).tool
        predicted = chosen.name if chosen is not None else None
        results.append(
            CaseResult(
                task=case.task,
                expected=case.expected,
                predicted=predicted,
                correct=predicted == case.expected,
            )
        )

    n = len(results)
    accuracy = sum(r.correct for r in results) / n if n else 0.0
    return SelectorReport(accuracy=accuracy, n=n, results=results)
