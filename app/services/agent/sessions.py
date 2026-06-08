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
