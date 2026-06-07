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
    "to call a tool, or:\n"
    '  {"thought": "...", "final_answer": "..."}\n'
    "when the task is complete. Use only the listed tools. If an observation "
    "reports an error, reason about it and recover. Output JSON only."
)


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

    def _prompt(self, task: str, steps: list[ReactStep]) -> str:
        """Render task, tool catalog, and the scratchpad so far."""
        lines = [f"Task: {task}", "", "Available tools:"]
        for tool in self.tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        scratchpad = format_scratchpad(steps)
        if scratchpad:
            lines += ["", scratchpad]
        lines += ["", "Respond with the next step as JSON."]
        return "\n".join(lines)

    def run(self, task: str, state: ReactState) -> ReactResult:
        """Run the ReAct loop until a final answer or the step budget.

        Args:
            task: The natural-language task.
            state: Working memory (seed jd_text/resume_text and the embedding
                service / llm the tools should use).

        Returns:
            A ReactResult with the final answer, the recorded steps, and whether
            it completed within the step budget.

        Raises:
            RuntimeError: If the LLM is not configured.
        """
        if not self.llm.is_configured():
            raise RuntimeError("ReactAgent requires a configured LLM.")

        steps: list[ReactStep] = []
        for _ in range(self.max_steps):
            raw = self.llm.complete(self._prompt(task, steps), system=_SYSTEM)

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
