import json

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from app.api import deps
from app.services.agent.schemas import JdInput, ReactState
from app.services.agent.sessions import ChatMessage, conversation_context, fold_old_turns
from app.services.agent.slash import is_slash, handle_slash
from app.services.agent.tools import build_default_agent
from app.services.agent.trace import steps_as_dicts
from app.skills.career_match import CareerMatchResult, run_career_match

router = APIRouter()


class CareerMatchRequest(BaseModel):
    """Raw JD + resume text for the end-to-end analysis."""

    jd_text: str = Field(..., min_length=1, description="Raw job description text")
    resume_text: str = Field(..., min_length=1, description="Raw resume text")


class CareerMatchResponse(BaseModel):
    """Response for the end-to-end career-match endpoint."""

    status: str = "success"
    data: CareerMatchResult


@router.post("/career-match", response_model=CareerMatchResponse)
async def career_match_endpoint(request: CareerMatchRequest):
    """Run the full pipeline on raw JD and resume text.

    Parses both, matches, ranks experiences, audits for risks, and generates a
    report — the same flow as the career-match skill, exposed for the frontend.
    """
    result = run_career_match(
        request.jd_text,
        request.resume_text,
        embedding_service=deps.get_embedding_service(),
        llm=deps.get_llm(),
    )
    return CareerMatchResponse(data=result)


class JdInputModel(BaseModel):
    """One candidate JD for multi-JD comparison."""

    text: str = Field(..., min_length=1, description="Raw job description text")
    label: str | None = Field(None, description="Display label, e.g. 'Job A'")


class CareerAskRequest(BaseModel):
    """An open-ended question over a resume + one or more JDs, answered by the
    ReAct agent. Provide a single ``jd_text`` or a list of ``jds`` to compare."""

    question: str = Field(..., min_length=1, description="The user's question")
    resume_text: str = Field(..., min_length=1, description="Raw resume text")
    jd_text: str | None = Field(None, description="Raw JD text (single-JD path)")
    jds: list[JdInputModel] = Field(
        default_factory=list, description="Multiple JDs to compare"
    )
    max_steps: int = Field(12, ge=1, le=20, description="Agent step budget")

    @model_validator(mode="after")
    def _require_a_jd(self) -> "CareerAskRequest":
        if not self.jd_text and not self.jds:
            raise ValueError("provide jd_text or a non-empty jds list")
        return self


class CareerAskResponse(BaseModel):
    """The agent's answer plus its reasoning trace."""

    status: str = "success"
    answer: str
    completed: bool
    steps: list[dict]


def _kb_retriever_or_none():
    """Build the KB retriever, tolerating an unavailable store (no DB, etc.)."""
    try:
        return deps.get_kb_retriever()
    except Exception:
        return None


@router.post("/career/ask", response_model=CareerAskResponse)
async def career_ask_endpoint(request: CareerAskRequest):
    """Answer an open-ended career question by driving the ReAct agent.

    Unlike ``/career-match`` (a fixed pipeline), the agent decides which tools to
    call — diagnosing, retrieving from the KB, advising, rewriting, or comparing
    multiple JDs — based on the question and intermediate results. Returns the
    answer and the full Thought/Action/Observation trace for transparency.
    """
    llm = deps.get_llm()
    if not llm.is_configured():
        raise HTTPException(
            status_code=503,
            detail="The agent requires a configured LLM (set LLM_API_KEY / LLM_MODEL).",
        )

    if request.jds:
        jd_inputs = [
            JdInput(label=j.label or f"Job {chr(65 + i)}", text=j.text)
            for i, j in enumerate(request.jds)
        ]
    else:
        jd_inputs = [JdInput(label="Job A", text=request.jd_text)]

    state = ReactState(
        jd_text=request.jd_text or jd_inputs[0].text,
        resume_text=request.resume_text,
        jd_inputs=jd_inputs,
        embedding_service=deps.get_embedding_service(),
        kb_retriever=_kb_retriever_or_none(),
        llm=llm,
    )
    agent = build_default_agent(llm, max_steps=request.max_steps)
    result = agent.run(request.question, state)
    return CareerAskResponse(
        answer=result.answer,
        completed=result.completed,
        steps=steps_as_dicts(result.steps),
    )


# ── Conversational chat (multi-turn, persistent state, slash commands) ──────


class ChatRequest(BaseModel):
    """One chat turn. ``session_id`` is null on the first turn (a new session is
    created and returned). Optional JD/resume text seeds or updates the session;
    a message starting with ``/`` runs a slash command, otherwise it drives the
    ReAct agent."""

    message: str = Field(..., min_length=1, description="The user's message")
    session_id: str | None = Field(None, description="Returned by the first turn")
    resume_text: str | None = Field(None, description="Seed/replace the resume")
    jd_text: str | None = Field(None, description="Seed/replace the JD (single)")
    jds: list[JdInputModel] = Field(default_factory=list, description="Seed JDs")
    max_steps: int = Field(12, ge=1, le=20, description="Agent step budget")


def _seed_state(request: "ChatRequest") -> ReactState:
    """Build a fresh session state from the request's inputs."""
    jd_inputs = [
        JdInput(label=j.label or f"Job {chr(65 + i)}", text=j.text)
        for i, j in enumerate(request.jds)
    ]
    if not jd_inputs and request.jd_text:
        jd_inputs = [JdInput(label="Job A", text=request.jd_text)]
    return ReactState(
        jd_text=request.jd_text or (jd_inputs[0].text if jd_inputs else None),
        resume_text=request.resume_text,
        jd_inputs=jd_inputs,
        embedding_service=deps.get_embedding_service(),
        kb_retriever=_kb_retriever_or_none(),
        llm=deps.get_llm(),
    )


def _apply_updates(state: ReactState, request: "ChatRequest") -> None:
    """Apply new JD/resume text to an existing session, invalidating stale parses."""
    if request.resume_text and request.resume_text != state.resume_text:
        state.resume_text = request.resume_text
        state.resume = None
        state.match = None
    if request.jds:
        state.jd_inputs = [
            JdInput(label=j.label or f"Job {chr(65 + i)}", text=j.text)
            for i, j in enumerate(request.jds)
        ]
        state.jd_text = state.jd_inputs[0].text
        state.jd = state.match = None
        state.comparison = []
    elif request.jd_text and request.jd_text != state.jd_text:
        state.jd_text = request.jd_text
        state.jd_inputs = [JdInput(label="Job A", text=request.jd_text)]
        state.jd = state.match = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/career/chat/stream")
async def career_chat_stream_endpoint(request: ChatRequest):
    """Multi-turn conversational entry point, streamed over Server-Sent Events.

    Maintains a server-side session (parsed JD/resume/match + rolling history).
    Slash commands (``/match``, ``/report``, ``/prep``, ``/audit``, ``/compare``)
    run the deterministic pipeline. Free text drives the ReAct agent, which may
    pause to ask the user a question (``awaiting_user``) and resume next turn.

    Event sequence: a ``step`` event per ReAct step (live reasoning), then the
    final answer streamed token-by-token as ``token`` events, then a single
    ``done`` event with the full reply, state, session id, and history. Slash
    commands and ask_user pauses emit only ``done``.
    """
    store = deps.get_session_store()
    session = store.get(request.session_id)
    if session is None:
        session = store.create(_seed_state(request))
    else:
        _apply_updates(session.state, request)

    message = request.message.strip()
    session.history.append(ChatMessage(role="user", content=message))
    session.state.conversation = conversation_context(session) or None

    # Decide the path up front so the LLM guard can fail cleanly before streaming.
    is_agent = session.awaiting_user or not is_slash(message)
    llm = deps.get_llm()
    if is_agent and not llm.is_configured():
        raise HTTPException(status_code=503, detail="The agent requires a configured LLM.")
    agent = build_default_agent(llm, max_steps=request.max_steps)

    def run_steps(task, steps):
        """Yield a `step` SSE per ReAct step; return the terminal ReactResult."""
        gen = agent.iter_run(task, session.state, steps)
        try:
            while True:
                yield _sse("step", asdict(next(gen)))
        except StopIteration as stop:
            return stop.value

    def finish(reply, status):
        session.history.append(ChatMessage(role="assistant", content=reply))
        fold_old_turns(session, deps.get_llm())
        yield _sse("done", {
            "session_id": session.session_id,
            "reply": reply,
            "state": status,
            "history": [{"role": m.role, "content": m.content} for m in session.history],
        })

    def events():
        # Slash command: deterministic, no streaming of reasoning or tokens.
        if not session.awaiting_user and is_slash(message):
            session.clear_pending()
            yield from finish(handle_slash(message, session.state), "answered")
            return

        if session.awaiting_user:
            task = session.pending_task
            session.pending_steps[-1].observation = message
            result = yield from run_steps(task, session.pending_steps)
        else:
            task = message
            result = yield from run_steps(task, None)

        # Paused to ask the user: hand back the question, no answer composition.
        if result.pending_question is not None:
            session.pending_question = result.pending_question
            session.pending_task = task
            session.pending_steps = result.steps
            yield from finish(result.pending_question, "awaiting_user")
            return

        # Compose the final answer, streamed token-by-token.
        session.clear_pending()
        parts: list[str] = []
        for token in agent.stream_answer(task, session.state, result.steps):
            parts.append(token)
            yield _sse("token", {"text": token})
        reply = "".join(parts).strip() or "(no answer produced)"
        yield from finish(reply, "answered" if result.completed else "incomplete")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
