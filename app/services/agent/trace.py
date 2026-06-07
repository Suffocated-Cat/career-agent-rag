"""
Trace rendering for the ReAct agent.

The trajectory itself is just the ``list[ReactStep]`` the controller records.
These helpers render it — for the model's scratchpad (fed back each step) and
for serialization (logs / an API).
"""

from dataclasses import asdict

from app.services.agent.schemas import ReactStep


def format_scratchpad(steps: list[ReactStep]) -> str:
    """Render prior steps as a Thought/Action/Observation scratchpad."""
    if not steps:
        return ""
    lines: list[str] = ["Steps so far:"]
    for step in steps:
        if step.action:
            lines.append(f"Thought: {step.thought}")
            lines.append(f"Action: {step.action} {step.action_input}")
        lines.append(f"Observation: {step.observation}")
    return "\n".join(lines)


def steps_as_dicts(steps: list[ReactStep]) -> list[dict]:
    """Serialize steps to plain dicts (e.g. for logging or an API)."""
    return [asdict(step) for step in steps]
