"""Tests for the Tracer and AgentController tool-call tracing."""
import pytest

from app.services.agent.controller import AgentContext, AgentController, Tool
from app.services.agent.trace import (
    STATUS_ERROR,
    STATUS_NO_TOOL,
    STATUS_OK,
    TraceEntry,
    Tracer,
)


def _tool(name, keywords, handler=None):
    return Tool(
        name=name,
        description=f"{name} tool",
        keywords=keywords,
        handler=handler or (lambda ctx: f"{name}-out"),
    )


class TestTracer:
    def test_record_and_last(self):
        tr = Tracer()
        assert tr.last is None
        e = TraceEntry(task="t", tool="x", status=STATUS_OK, latency_ms=1.0)
        tr.record(e)
        assert tr.entries == [e]
        assert tr.last is e

    def test_clear(self):
        tr = Tracer()
        tr.record(TraceEntry(task="t", tool="x", status=STATUS_OK, latency_ms=1.0))
        tr.clear()
        assert tr.entries == []

    def test_as_dicts(self):
        tr = Tracer()
        tr.record(TraceEntry(task="t", tool="x", status=STATUS_OK, latency_ms=2.5))
        dicts = tr.as_dicts()
        assert dicts[0]["task"] == "t"
        assert dicts[0]["status"] == STATUS_OK
        assert "timestamp" in dicts[0]


class TestControllerTracing:
    def test_records_ok_entry(self):
        tr = Tracer()
        c = AgentController([_tool("echo", ("go",))], tracer=tr)
        result = c.run(AgentContext(task="go now"))
        assert result.latency_ms >= 0.0
        entry = tr.last
        assert entry.status == STATUS_OK
        assert entry.tool == "echo"
        assert entry.output_type == "str"
        assert entry.latency_ms >= 0.0

    def test_records_error_entry_and_reraises(self):
        def _boom(ctx):
            raise ValueError("kaboom")

        tr = Tracer()
        c = AgentController([_tool("bad", ("go",), handler=_boom)], tracer=tr)
        with pytest.raises(ValueError, match="kaboom"):
            c.run(AgentContext(task="go now"))
        entry = tr.last
        assert entry.status == STATUS_ERROR
        assert entry.tool == "bad"
        assert "kaboom" in entry.error

    def test_records_no_tool_entry_and_raises(self):
        tr = Tracer()
        c = AgentController([_tool("echo", ("go",))], tracer=tr)
        with pytest.raises(LookupError):
            c.run(AgentContext(task="nothing relevant"))
        entry = tr.last
        assert entry.status == STATUS_NO_TOOL
        assert entry.tool is None

    def test_multiple_runs_accumulate(self):
        tr = Tracer()
        c = AgentController([_tool("echo", ("go",))], tracer=tr)
        c.run(AgentContext(task="go 1"))
        c.run(AgentContext(task="go 2"))
        assert len(tr.entries) == 2

    def test_no_tracer_still_runs(self):
        c = AgentController([_tool("echo", ("go",))])  # no tracer
        result = c.run(AgentContext(task="go now"))
        assert result.output == "echo-out"
        assert result.latency_ms >= 0.0
