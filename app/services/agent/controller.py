"""
AgentController — route a task to the right tool and run it.

This turns CareerAgent from a fixed pipeline into a multi-tool agent: given a
natural-language task ("analyze this JD", "check my resume for risks"), the
controller picks the most appropriate tool and executes it against a shared
context.

Selection is pluggable via the ``ToolSelector`` protocol. The default
``KeywordToolSelector`` is rule-based — transparent and offline. An
LLM-backed selector (see ``selectors.py``) can be passed in instead without
touching the tools or the run loop.
"""

import time

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from app.models.jd import JobDescription
from app.models.resume import Resume
from app.services.agent.trace import (
    STATUS_ERROR,
    STATUS_NO_TOOL,
    STATUS_OK,
    TraceEntry,
    Tracer,
)


@dataclass
class AgentContext:
    """Inputs available to any tool during a run.

    A tool reads what it needs and ignores the rest. ``task`` drives tool
    selection; the remaining fields carry the data tools operate on.
    """

    task: str
    jd: JobDescription | None = None
    resume: Resume | None = None
    jd_text: str | None = None
    resume_text: str | None = None
    embedding_service: Any | None = None


@dataclass
class Tool:
    """A capability the agent can invoke.

    Attributes:
        name: Unique tool identifier.
        description: What the tool does (also used by future LLM selection).
        keywords: Lowercase phrases that signal this tool fits a task.
        handler: Callable run against the AgentContext, returning any output.
    """

    name: str
    description: str
    keywords: tuple[str, ...]
    handler: Callable[[AgentContext], Any]


@dataclass
class ToolResult:
    """The outcome of running a tool."""

    tool: str
    output: Any
    reason: str = ""  # why this tool was selected
    latency_ms: float = 0.0  # wall-clock time to select + execute


@dataclass
class Selection:
    """A selector's choice of tool, with an explanation."""

    tool: Tool | None
    reason: str = ""


@runtime_checkable
class ToolSelector(Protocol):
    """Strategy for choosing a tool given a task and the available tools."""

    def select(self, task: str, tools: list[Tool]) -> Selection:
        """Return the chosen tool (or Selection with tool=None if no fit)."""
        ...


def _matched_keywords(tool: Tool, task_lower: str) -> list[str]:
    """Keywords of *tool* that appear in the (lowercased) task."""
    return [kw for kw in tool.keywords if kw in task_lower]


class KeywordToolSelector:
    """Rule-based selector: score tools by keyword overlap with the task."""

    def select(self, task: str, tools: list[Tool]) -> Selection:
        task_lower = task.lower()
        best: Tool | None = None
        best_matched: list[str] = []
        for tool in tools:
            matched = _matched_keywords(tool, task_lower)
            if len(matched) > len(best_matched):
                best = tool
                best_matched = matched
        if best is None:
            return Selection(tool=None, reason="no keywords matched")
        return Selection(
            tool=best, reason=f"matched keywords: {', '.join(best_matched)}"
        )


class AgentController:
    """Holds a set of tools and dispatches tasks to them."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        selector: ToolSelector | None = None,
        tracer: Tracer | None = None,
    ):
        self.tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)
        self.selector: ToolSelector = selector or KeywordToolSelector()
        self.tracer = tracer

    def register(self, tool: Tool) -> None:
        """Add (or replace) a tool by name."""
        self.tools[tool.name] = tool

    def select_tool(self, task: str) -> Tool | None:
        """Pick the best-matching tool for *task*, or None if nothing fits."""
        return self.selector.select(task, list(self.tools.values())).tool

    def _trace(self, entry: TraceEntry) -> None:
        """Record a trace entry if a tracer is attached."""
        if self.tracer is not None:
            self.tracer.record(entry)

    def run(self, context: AgentContext) -> ToolResult:
        """Select a tool for ``context.task`` and execute it.

        Each call is traced (task, tool, latency, status) when a tracer is
        attached, including selection misses and tool errors.

        Args:
            context: The task plus any data tools may need.

        Returns:
            A ToolResult with the tool name, its output, the selection reason,
            and the elapsed time in milliseconds.

        Raises:
            LookupError: If no tool is selected for the task.
        """
        start = time.perf_counter()
        selection = self.selector.select(context.task, list(self.tools.values()))

        if selection.tool is None:
            latency_ms = (time.perf_counter() - start) * 1000
            self._trace(
                TraceEntry(
                    task=context.task, tool=None, status=STATUS_NO_TOOL,
                    latency_ms=latency_ms, reason=selection.reason,
                )
            )
            raise LookupError(f"No tool matched task: {context.task!r}")

        try:
            output = selection.tool.handler(context)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            self._trace(
                TraceEntry(
                    task=context.task, tool=selection.tool.name, status=STATUS_ERROR,
                    latency_ms=latency_ms, reason=selection.reason, error=str(exc),
                )
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        self._trace(
            TraceEntry(
                task=context.task, tool=selection.tool.name, status=STATUS_OK,
                latency_ms=latency_ms, reason=selection.reason,
                output_type=type(output).__name__,
            )
        )
        return ToolResult(
            tool=selection.tool.name,
            output=output,
            reason=selection.reason,
            latency_ms=latency_ms,
        )

    def describe_tools(self) -> list[dict[str, str]]:
        """List tool names and descriptions (for inspection / LLM selection)."""
        return [
            {"name": t.name, "description": t.description}
            for t in self.tools.values()
        ]
