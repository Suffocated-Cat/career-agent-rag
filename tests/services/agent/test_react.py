"""Tests for the ReAct agent loop."""
import json

import pytest

from app.services.agent.react_controller import ReactAgent
from app.services.agent.schemas import ReactResult, ReactState, ReactTool


class ScriptedLLM:
    """Scripts the agent's decisions (ReAct-system prompts); every other call —
    tool-internal extraction and the final-answer composition — returns a fixed
    ``answer`` so it doesn't consume the decision script."""

    def __init__(self, replies, configured=True, answer="FINAL ANSWER"):
        self.replies = list(replies)
        self.configured = configured
        self.answer = answer
        self.calls = 0
        self.prompts = []

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        self.prompts.append(prompt)
        if system and "ReAct agent" in system:
            reply = self.replies[min(self.calls, len(self.replies) - 1)]
            self.calls += 1
            return reply
        return self.answer  # compose / tool-internal

    def stream(self, prompt, system=None, **kwargs):
        self.prompts.append(prompt)
        yield self.answer


def _act(action, **args):
    return json.dumps({"thought": "go", "action": action, "action_input": args})


def _final(answer="done"):
    # Finishing is a control action now; the answer is composed separately, so
    # the argument is ignored (kept for call-site readability).
    return json.dumps({"thought": "done", "action": "finish"})


def _ask(question):
    return json.dumps({"thought": "need info", "action": "ask_user",
                       "action_input": {"question": question}})


def _echo_tool():
    return ReactTool("echo", "echo back", lambda state, args: f"echoed {args.get('x')}")


def _state():
    return ReactState()


class TestReactLoop:
    def test_final_answer_immediately(self):
        agent = ReactAgent(ScriptedLLM([_final()], answer="all done"), [_echo_tool()])
        result = agent.run("task", _state())
        assert isinstance(result, ReactResult)
        assert result.completed is True
        assert result.answer == "all done"  # composed after the finish action
        assert result.steps == []

    def test_tool_call_then_finish(self):
        llm = ScriptedLLM([_act("echo", x="hi"), _final("done")])
        agent = ReactAgent(llm, [_echo_tool()])
        result = agent.run("task", _state())
        assert result.completed is True
        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.action == "echo"
        assert step.observation == "echoed hi"

    def test_observation_fed_back_into_prompt(self):
        llm = ScriptedLLM([_act("echo", x="hi"), _final("done")])
        ReactAgent(llm, [_echo_tool()]).run("task", _state())
        # Second prompt must contain the first step's observation.
        assert "echoed hi" in llm.prompts[1]

    def test_unknown_tool_observation(self):
        llm = ScriptedLLM([_act("nope"), _final("x")])
        result = ReactAgent(llm, [_echo_tool()]).run("task", _state())
        assert "unknown tool" in result.steps[0].observation

    def test_tool_error_becomes_observation(self):
        def _boom(state, args):
            raise ValueError("kaboom")

        llm = ScriptedLLM([_act("boom"), _final("x")])
        agent = ReactAgent(llm, [ReactTool("boom", "boom", _boom)])
        result = agent.run("task", _state())
        assert "Error: kaboom" in result.steps[0].observation

    def test_self_correction_after_precondition_error(self):
        ready = {"v": False}

        def setup(state, args):
            ready["v"] = True
            return "setup done"

        def need(state, args):
            return "ok" if ready["v"] else "Error: run setup first"

        tools = [ReactTool("setup", "s", setup), ReactTool("need", "n", need)]
        # need (error) → setup → need (ok) → final
        llm = ScriptedLLM([_act("need"), _act("setup"), _act("need"), _final("done")])
        result = ReactAgent(llm, tools).run("task", _state())
        assert result.completed is True
        assert "Error: run setup first" in result.steps[0].observation
        assert result.steps[2].observation == "ok"

    def test_invalid_json_observation(self):
        llm = ScriptedLLM(["not json at all", _final("recovered")])
        result = ReactAgent(llm, [_echo_tool()]).run("task", _state())
        assert "not valid JSON" in result.steps[0].observation
        assert result.completed is True

    def test_non_object_json(self):
        llm = ScriptedLLM(["[1, 2, 3]", _final("ok")])
        result = ReactAgent(llm, [_echo_tool()]).run("task", _state())
        assert "expected a JSON object" in result.steps[0].observation

    def test_missing_action(self):
        llm = ScriptedLLM([json.dumps({"thought": "hmm"}), _final("ok")])
        result = ReactAgent(llm, [_echo_tool()]).run("task", _state())
        assert "provide an 'action'" in result.steps[0].observation

    def test_max_steps_exhausted(self):
        # Never finishes → loop hits the budget.
        llm = ScriptedLLM([_act("echo", x="loop")])
        agent = ReactAgent(llm, [_echo_tool()], max_steps=3)
        result = agent.run("task", _state())
        assert result.completed is False
        assert len(result.steps) == 3
        # An incomplete run still composes a best-effort answer.
        assert result.answer != ""

    def test_requires_configured_llm(self):
        agent = ReactAgent(ScriptedLLM([_final("x")], configured=False), [_echo_tool()])
        with pytest.raises(RuntimeError, match="requires a configured LLM"):
            agent.run("task", _state())

    def test_prompt_lists_tools_and_task(self):
        llm = ScriptedLLM([_final("x")])
        ReactAgent(llm, [_echo_tool()]).run("analyze this", _state())
        assert "analyze this" in llm.prompts[0]
        assert "echo" in llm.prompts[0]

    def test_ask_user_pauses(self):
        llm = ScriptedLLM([_act("echo", x="hi"), _ask("What is your impact?"), _final("x")])
        result = ReactAgent(llm, [_echo_tool()]).run("task", _state())
        assert result.completed is False
        assert result.pending_question == "What is your impact?"
        # The last step is the pending ask_user with an empty observation.
        assert result.steps[-1].action == "ask_user"
        assert result.steps[-1].observation == ""

    def test_ask_user_resume(self):
        # First run pauses on ask_user.
        llm = ScriptedLLM([_ask("impact?"), _final()], answer="done with 40% speedup")
        agent = ReactAgent(llm, [_echo_tool()])
        first = agent.run("task", _state())
        assert first.pending_question == "impact?"
        # Caller fills the user's reply into the pending step, then resumes.
        first.steps[-1].observation = "cut latency 40%"
        resumed = agent.run("task", _state(), steps=first.steps)
        assert resumed.completed is True
        assert resumed.answer == "done with 40% speedup"  # composed answer
        # The reply is visible in the scratchpad the answer is composed from.
        assert "cut latency 40%" in llm.prompts[-1]

    def test_ask_user_without_question_is_error(self):
        bad = json.dumps({"thought": "?", "action": "ask_user", "action_input": {}})
        llm = ScriptedLLM([bad, _final("ok")])
        result = ReactAgent(llm, [_echo_tool()]).run("task", _state())
        assert "ask_user requires" in result.steps[0].observation
        assert result.completed is True

    def test_iter_run_yields_each_step(self):
        llm = ScriptedLLM([_act("echo", x="hi"), _final("done")])
        gen = ReactAgent(llm, [_echo_tool()]).iter_run("task", _state())
        yielded, result = [], None
        try:
            while True:
                yielded.append(next(gen))
        except StopIteration as stop:
            result = stop.value
        assert len(yielded) == 1
        assert yielded[0].action == "echo" and yielded[0].observation == "echoed hi"
        # iter_run signals completion; the answer is composed by run()/the endpoint.
        assert result.completed is True and result.answer == ""

    def test_iter_run_yields_pending_ask_user(self):
        gen = ReactAgent(ScriptedLLM([_ask("impact?")]), [_echo_tool()]).iter_run("t", _state())
        yielded, result = [], None
        try:
            while True:
                yielded.append(next(gen))
        except StopIteration as stop:
            result = stop.value
        assert yielded[-1].action == "ask_user"
        assert result.pending_question == "impact?"

    def test_conversation_injected_into_prompt(self):
        llm = ScriptedLLM([_final("x")])
        state = ReactState(conversation="User: hi\nAssistant: hello")
        ReactAgent(llm, [_echo_tool()]).run("task", state)
        assert "Conversation so far:" in llm.prompts[0]
        assert "User: hi" in llm.prompts[0]

    def test_non_dict_action_input_ignored(self):
        # action_input not a dict → treated as empty, tool still runs.
        bad = json.dumps({"action": "echo", "action_input": "oops"})
        llm = ScriptedLLM([bad, _final("done")])
        result = ReactAgent(llm, [_echo_tool()]).run("task", _state())
        assert result.steps[0].observation == "echoed None"
