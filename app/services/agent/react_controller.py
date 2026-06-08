"""
ReactAgent — the Reason + Act loop over CareerAgent tools.

Unlike a one-shot router, the agent iterates: at each step the LLM produces a
thought and either an action (a tool call) or a final answer. The tool runs,
its observation is fed back, and the loop continues until the LLM finishes or a
step budget is hit:

    Thought → Action → Observation → Thought → ... → Final Answer

Tools operate on a shared ``ReactState``, so the model passes only small inputs
and reads concise observations rather than echoing large objects between steps.
Tool preconditions surface as error observations the model can recover from. The
agent is LLM-driven and requires a configured LLM.
"""

from typing import TYPE_CHECKING

from app.services.agent.schemas import (
    ReactDecision,
    ReactResult,
    ReactState,
    ReactStep,
    ReactTool,
)
from app.services.agent.trace import format_scratchpad
from app.services.llm_support import extract_json

if TYPE_CHECKING:
    from app.services.llm_client import LLMClient


_SYSTEM = (
    "You are a career-analysis ReAct agent. Solve the task by reasoning step by "
    "step and calling tools. At each step respond with STRICT JSON, either:\n"
    '  {"thought": "...", "action": "<tool_name>", "action_input": {...}}\n'
    "to call a tool,\n"
    '  {"thought": "...", "action": "ask_user", "action_input": {"question": "..."}}\n'
    "to ask the user a clarifying question when you genuinely need information "
    "only they can provide (e.g. missing metrics for a resume bullet, or their "
    "answer in a mock interview) — don't use it for things the tools can find, "
    "or:\n"
    '  {"thought": "...", "final_answer": "..."}\n'
    "when the task is complete. Use only the listed tools. If an observation "
    "reports an error, reason about it and recover. The JD(s) and resume are "
    "already loaded in working memory — call the tools (which read it) to parse "
    "and analyze them; do NOT ask the user for inputs that are already loaded. "
    "Output JSON only."
)


def _state_summary(state: ReactState) -> str:
    """Describe what's already in working memory, so the model uses the tools
    instead of asking the user for inputs that are already loaded."""
    parts: list[str] = []
    if state.jd is not None:
        parts.append("JD parsed")
    elif state.jd_text or state.jd_inputs:
        parts.append("raw JD loaded (call parse_jd to parse it)")
    if len(state.jd_inputs) > 1:
        parts.append(
            f"{len(state.jd_inputs)} candidate JDs loaded for comparison "
            "(use compare_jds, then select_jd)"
        )
    if state.resume is not None:
        parts.append("resume parsed")
    elif state.resume_text:
        parts.append("raw resume loaded (call parse_resume to parse it)")
    if state.match is not None:
        parts.append("match computed")
    return "; ".join(parts)


class ReactAgent:
    """Runs a ReAct loop over a set of tools using an LLM."""

    def __init__(
        self,
        llm: "LLMClient",
        tools: list[ReactTool],
        max_steps: int = 8,
    ):
        self.llm = llm
        self.tools: dict[str, ReactTool] = {t.name: t for t in tools}
        self.max_steps = max_steps

    def _prompt(self, task: str, steps: list[ReactStep], state: ReactState) -> str:
        """Render task, working-memory summary, tool catalog, and scratchpad."""
        lines = [f"Task: {task}"]
        summary = _state_summary(state)
        if summary:
            lines += ["", f"Working memory: {summary}."]
        lines += ["", "Available tools:"]
        for tool in self.tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        scratchpad = format_scratchpad(steps)
        if scratchpad:
            lines += ["", scratchpad]
        lines += ["", "Respond with the next step as JSON."]
        return "\n".join(lines)

    def run(
        self,
        task: str,
        state: ReactState,
        steps: list[ReactStep] | None = None,
    ) -> ReactResult:
        """Run the ReAct loop until a final answer, an ask_user pause, or budget.

        Args:
            task: The natural-language task.
            state: Working memory (seed jd_text/resume_text and the embedding
                service / llm the tools should use).
            steps: Prior steps to resume from (for multi-turn). To resume after
                an ask_user pause, set the last step's ``observation`` to the
                user's reply, then pass the steps back here.

        Returns:
            A ReactResult that either completed (final answer), paused
            (``pending_question`` set), or ran out of budget.

        Raises:
            RuntimeError: If the LLM is not configured.
        """
        if not self.llm.is_configured():
            raise RuntimeError("ReactAgent requires a configured LLM.")

        steps = steps if steps is not None else []
        while len(steps) < self.max_steps:
            raw = self.llm.complete(self._prompt(task, steps, state), system=_SYSTEM)

            try:
                data = extract_json(raw)
            except Exception:
                steps.append(ReactStep("", None, {}, "Error: reply was not valid JSON."))
                continue
            try:
                decision = ReactDecision.model_validate(data)
            except Exception:
                steps.append(
                    ReactStep("", None, {}, "Error: expected a JSON object with the required fields.")
                )
                continue

            if decision.final_answer is not None:
                return ReactResult(
                    answer=str(decision.final_answer), steps=steps, completed=True
                )

            action_input = (
                decision.action_input if isinstance(decision.action_input, dict) else {}
            )

            # ask_user is a control action, not a tool: pause and hand back the
            # question. The caller resumes by filling this step's observation
            # with the user's reply and calling run() again with these steps.
            if decision.action == "ask_user":
                question = str(action_input.get("question", "")).strip()
                if not question:
                    steps.append(
                        ReactStep(decision.thought, "ask_user", action_input,
                                  "Error: ask_user requires action_input.question.")
                    )
                    continue
                steps.append(ReactStep(decision.thought, "ask_user", action_input, ""))
                return ReactResult(
                    answer="", steps=steps, completed=False, pending_question=question
                )

            if decision.action is None:
                steps.append(
                    ReactStep(decision.thought, None, {}, "Error: provide 'action' or 'final_answer'.")
                )
                continue

            tool = self.tools.get(decision.action)
            if tool is None:
                observation = (
                    f"Error: unknown tool {decision.action!r}. "
                    f"Available: {', '.join(self.tools)}."
                )
            else:
                try:
                    observation = tool.handler(state, action_input)
                except Exception as exc:
                    observation = f"Error: {exc}"

            steps.append(
                ReactStep(decision.thought, decision.action, action_input, observation)
            )

        return ReactResult(answer="", steps=steps, completed=False)
