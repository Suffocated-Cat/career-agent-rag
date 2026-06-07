"""Tests for ReAct trace rendering helpers."""
from app.services.agent.schemas import ReactStep
from app.services.agent.trace import format_scratchpad, steps_as_dicts


def _step(action="match", obs="ok"):
    return ReactStep(thought="t", action=action, action_input={"x": 1}, observation=obs)


class TestFormatScratchpad:
    def test_empty(self):
        assert format_scratchpad([]) == ""

    def test_renders_thought_action_observation(self):
        text = format_scratchpad([_step()])
        assert "Steps so far:" in text
        assert "Thought: t" in text
        assert "Action: match" in text
        assert "Observation: ok" in text

    def test_step_without_action_shows_only_observation(self):
        text = format_scratchpad([ReactStep("", None, {}, "Error: bad JSON")])
        assert "Action:" not in text
        assert "Observation: Error: bad JSON" in text


class TestStepsAsDicts:
    def test_serializes(self):
        dicts = steps_as_dicts([_step(obs="done")])
        assert dicts == [
            {"thought": "t", "action": "match", "action_input": {"x": 1}, "observation": "done"}
        ]

    def test_empty(self):
        assert steps_as_dicts([]) == []
