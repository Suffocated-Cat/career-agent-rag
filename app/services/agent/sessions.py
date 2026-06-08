"""
In-memory chat sessions for the conversational agent.

A chat turn needs to remember more than a single ``/career/ask`` call: the
parsed JD/resume/match (so later turns don't re-parse), the running
conversation, and — when the agent paused on ``ask_user`` — enough to resume
the same agent run with the user's reply.

This store is a process-local dict, matching the project's prototype boundary
(no auth, no persistence). Restarting the server clears sessions.
"""

import uuid

from dataclasses import dataclass, field

from app.services.agent.schemas import ReactState, ReactStep
from app.services.llm_support import generate_text

# Keep this many of the most recent messages verbatim (3 user/assistant rounds);
# anything older is folded into a rolling LLM summary.
RECENT_WINDOW = 6
_MAX_FALLBACK_SUMMARY = 2000  # cap the no-LLM plain-text trail


@dataclass
class ChatMessage:
    """One turn in the visible conversation."""

    role: str  # "user" | "assistant"
    content: str


@dataclass
class ChatSession:
    """Server-side state for one conversation."""

    session_id: str
    state: ReactState
    history: list[ChatMessage] = field(default_factory=list)

    # Rolling memory: older turns are condensed into ``summary``; ``summarized``
    # is how many leading history messages it already covers.
    summary: str = ""
    summarized: int = 0

    # Set while an agent run is paused waiting for the user's reply.
    pending_question: str | None = None
    pending_task: str | None = None
    pending_steps: list[ReactStep] = field(default_factory=list)

    @property
    def awaiting_user(self) -> bool:
        return self.pending_question is not None

    def clear_pending(self) -> None:
        self.pending_question = None
        self.pending_task = None
        self.pending_steps = []


class SessionStore:
    """Process-local registry of chat sessions, keyed by id."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create(self, state: ReactState) -> ChatSession:
        session_id = uuid.uuid4().hex
        session = ChatSession(session_id=session_id, state=state)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str | None) -> ChatSession | None:
        if not session_id:
            return None
        return self._sessions.get(session_id)


def _render(messages: list[ChatMessage]) -> str:
    """Render messages as a 'User: …/Assistant: …' transcript."""
    return "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in messages
    )


def conversation_context(session: ChatSession) -> str:
    """Context for the agent: the rolling summary plus the recent verbatim turns,
    excluding the just-appended current user message (that's the task)."""
    parts: list[str] = []
    if session.summary:
        parts.append(f"Summary of earlier conversation: {session.summary}")
    recent = session.history[session.summarized : -1]
    if recent:
        parts.append(_render(recent))
    return "\n".join(parts)


def _update_summary(llm, existing: str, messages: list[ChatMessage]) -> str:
    """Fold *messages* into *existing* — via the LLM, or a bounded plain-text
    trail when no LLM is available."""
    rendered = _render(messages)
    fallback = (f"{existing}\n{rendered}".strip())[-_MAX_FALLBACK_SUMMARY:]
    if llm is None or not llm.is_configured():
        return fallback
    prompt = (
        "Maintain a running summary of a career-coaching conversation. Keep the "
        "durable facts, decisions, and context needed to continue helping the "
        "user; drop pleasantries.\n\n"
        f"Existing summary:\n{existing or '(none)'}\n\n"
        f"New messages to fold in:\n{rendered}\n\n"
        "Return the updated summary only."
    )
    return generate_text(
        llm, prompt, system="You compress conversation history faithfully.", fallback=fallback
    )


def fold_old_turns(session: ChatSession, llm) -> None:
    """Move turns older than the recent window into the rolling summary."""
    cutoff = len(session.history) - RECENT_WINDOW
    if cutoff <= session.summarized:
        return
    to_fold = session.history[session.summarized : cutoff]
    session.summary = _update_summary(llm, session.summary, to_fold)
    session.summarized = cutoff
