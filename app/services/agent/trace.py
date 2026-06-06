"""
Tool-call tracing for the agent.

Records one TraceEntry per AgentController.run — what task came in, which tool
was selected (and why), how long it took, and whether it succeeded, failed, or
matched no tool. This is the audit trail for debugging wrong tool selection and
tool failures.
"""

import time

from dataclasses import asdict, dataclass, field

# Status values a trace entry can take.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_NO_TOOL = "no_tool"


@dataclass
class TraceEntry:
    """A single recorded tool invocation."""

    task: str
    tool: str | None
    status: str  # STATUS_OK | STATUS_ERROR | STATUS_NO_TOOL
    latency_ms: float
    reason: str = ""
    error: str = ""
    output_type: str = ""
    timestamp: float = field(default_factory=time.time)


class Tracer:
    """Collects TraceEntry records in memory."""

    def __init__(self) -> None:
        self.entries: list[TraceEntry] = []

    def record(self, entry: TraceEntry) -> None:
        """Append a trace entry."""
        self.entries.append(entry)

    def clear(self) -> None:
        """Drop all recorded entries."""
        self.entries.clear()

    @property
    def last(self) -> TraceEntry | None:
        """The most recent entry, or None if nothing has been recorded."""
        return self.entries[-1] if self.entries else None

    def as_dicts(self) -> list[dict]:
        """Serialize all entries to plain dicts (e.g. for an API or log)."""
        return [asdict(entry) for entry in self.entries]
