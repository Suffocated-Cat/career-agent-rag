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
    "You are a deterministic tool router for a career-analysis agent. "
    "Choose exactly one available tool for the user's task, or choose null "
    "when the task is outside the listed tool capabilities. Do not answer the "
    "task itself. Return only strict JSON with this schema: "
    '{"tool": "<tool_name_or_null>", "reason": "<short reason>"}.'
)


def _build_prompt(task: str, tools: list[Tool]) -> str:
    """Render the task and tool catalog into a user prompt."""
    lines = [
        "Available tools:",
    ]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description}")
    lines.extend(
        [
            "",
            "Routing rules:",
            "- Use jd_parser when the task asks to parse or analyze a job description.",
            "- Use resume_parser when the task asks to parse or extract fields from a resume/CV.",
            "- Use resume_matcher when the task asks for overall resume-to-job fit, match, score, gaps, or comparison.",
            "- Use project_auditor when the task asks whether resume claims are credible, supported, exaggerated, vague, or risky.",
            "- Use project_ranker when the task asks which projects or experiences are most relevant to a job.",
            "- Use null when none of the listed tools can handle the task.",
            "",
            f"Task: {task}",
        ]
    )
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
