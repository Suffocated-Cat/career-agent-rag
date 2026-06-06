"""
LLMToolSelector — let an LLM choose which tool fits a task.

Where KeywordToolSelector matches surface words, this asks the model to read
the task and the tool descriptions and name the best tool. It degrades
gracefully: if the LLM is not configured, errors, or returns an unknown name,
it falls back to the keyword selector — so the agent never hard-fails on the
model.
"""

import json

from app.services.agent.controller import (
    KeywordToolSelector,
    Selection,
    Tool,
    ToolSelector,
)
from app.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a tool router for a career-analysis agent. Given a user task and "
    "a list of available tools, choose the single best tool for the task. "
    'Respond with strict JSON: {"tool": "<tool_name>", "reason": "<short reason>"}. '
    "Use exactly one of the provided tool names, or null if none fit."
)


def _build_prompt(task: str, tools: list[Tool]) -> str:
    """Render the task and tool catalog into a user prompt."""
    lines = ["Available tools:"]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description}")
    lines.append("")
    lines.append(f"Task: {task}")
    return "\n".join(lines)


def _parse_tool_name(raw: str, tools: list[Tool]) -> tuple[str | None, str]:
    """Extract (tool_name, reason) from the model's reply.

    Tries strict JSON first, then falls back to finding any known tool name as
    a substring of the reply.
    """
    names = {t.name for t in tools}

    try:
        data = json.loads(raw)
        name = data.get("tool")
        reason = data.get("reason") or ""
        if name in names:
            return name, reason
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    lowered = raw.lower()
    for name in names:
        if name in lowered:
            return name, "name found in model response"
    return None, ""


class LLMToolSelector:
    """Selector that delegates the choice to an LLM, with keyword fallback."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        fallback: ToolSelector | None = None,
    ):
        """Configure the selector.

        Args:
            llm_client: The LLM client to use (default: a new LLMClient).
            fallback: Selector used when the LLM is unavailable or unhelpful
                (default: KeywordToolSelector).
        """
        self.llm = llm_client or LLMClient()
        self.fallback: ToolSelector = fallback or KeywordToolSelector()

    def select(self, task: str, tools: list[Tool]) -> Selection:
        if not tools or not self.llm.is_configured():
            return self.fallback.select(task, tools)

        try:
            raw = self.llm.complete(_build_prompt(task, tools), system=_SYSTEM_PROMPT)
        except Exception:
            return self.fallback.select(task, tools)

        name, reason = _parse_tool_name(raw, tools)
        if name is None:
            return self.fallback.select(task, tools)

        tool = next(t for t in tools if t.name == name)
        return Selection(tool=tool, reason=f"LLM selected '{name}': {reason}".strip())
