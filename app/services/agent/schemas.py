"""
Data models for the ReAct agent.

Pure structures (no behavior): the working memory tools share, the tool
definition, the per-step record, the run result, and the validated shape of one
LLM decision. Behavior lives in ``react_controller.py``; rendering/serialization
in ``trace.py``.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.models.jd import JobDescription
from app.models.match import MatchReport, MatchResult
from app.models.resume import Resume


@dataclass
class JdInput:
    """A raw JD to consider, with a human-readable label (e.g. "Job A")."""

    label: str
    text: str


@dataclass
class JdComparison:
    """One JD's parsed form and its match against the resume, for comparison."""

    label: str
    jd: JobDescription
    match: MatchResult


@dataclass
class ReactState:
    """Working memory shared across tool calls within one run.

    Single-JD flows use ``jd`` / ``match`` (the "active" JD). Multi-JD
    comparison seeds ``jd_inputs`` and populates ``comparison``; ``select_jd``
    then promotes one comparison entry into the active ``jd`` / ``match`` so the
    single-JD tools (rank_projects, advise, …) operate on the chosen role.
    """

    jd_text: str | None = None
    resume_text: str | None = None
    jd: JobDescription | None = None
    resume: Resume | None = None
    match: MatchResult | None = None
    report: MatchReport | None = None
    interview: Any | None = None  # InterviewPrep, kept loose to avoid a service import
    jd_inputs: list[JdInput] = field(default_factory=list)  # candidate JDs to compare
    comparison: list[JdComparison] = field(default_factory=list)  # ranked, best first
    embedding_service: Any | None = None
    kb_retriever: Any | None = None  # knowledge-base retriever for RAG tools
    llm: Any | None = None


@dataclass
class ReactTool:
    """A tool the agent can call.

    ``handler(state, action_input) -> observation`` reads/writes shared state
    and returns a short text observation (including errors).
    """

    name: str
    description: str
    handler: Callable[[ReactState, dict], str]


@dataclass
class ReactStep:
    """One iteration of the loop, recorded for tracing."""

    thought: str
    action: str | None
    action_input: dict
    observation: str


@dataclass
class ReactResult:
    """The outcome of a run.

    Three terminal shapes:
      - completed: the LLM emitted a final answer (``answer`` set).
      - paused: the LLM asked the user a question (``pending_question`` set);
        resume by filling the last step's observation with the user's reply and
        calling ``run`` again with the accumulated ``steps``.
      - incomplete: neither happened within the step budget.
    """

    answer: str
    steps: list[ReactStep] = field(default_factory=list)
    completed: bool = False  # True if the LLM emitted a final answer in budget
    pending_question: str | None = None  # set when the agent paused to ask the user


class ReactDecision(BaseModel):
    """The validated shape of one LLM step.

    A reply is either an action (``action`` + ``action_input``) or a
    ``final_answer``. ``action_input`` is typed loosely so a malformed value
    degrades to "no args" rather than failing validation.
    """

    thought: str = ""
    action: str | None = None
    action_input: Any = Field(default_factory=dict)
    final_answer: str | None = None
