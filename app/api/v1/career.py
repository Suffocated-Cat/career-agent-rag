from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from app.api import deps
from app.services.agent.schemas import JdInput, ReactState
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
    max_steps: int = Field(8, ge=1, le=16, description="Agent step budget")

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
        llm=deps.get_llm(),
    )
    agent = build_default_agent(deps.get_llm(), max_steps=request.max_steps)
    result = agent.run(request.question, state)
    return CareerAskResponse(
        answer=result.answer,
        completed=result.completed,
        steps=steps_as_dicts(result.steps),
    )
