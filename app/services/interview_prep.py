"""
Interview prep — RAG over the interview knowledge base.

This is the real retrieval-augmented generation path: it retrieves relevant
interview questions/notes from the knowledge base for the JD's skills, then has
the LLM write a prep guide **grounded on the retrieved questions** (with the
candidate's skill gaps highlighted). With no LLM it falls back to listing the
retrieved questions plus the gaps — so it degrades gracefully.
"""

from pydantic import BaseModel, Field

from app.models.jd import JobDescription
from app.models.resume import Resume
from app.services.llm_support import generate_text
from app.services.retrieval.base import Retriever

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.llm_client import LLMClient


class InterviewPrep(BaseModel):
    """Interview preparation grounded on retrieved knowledge."""

    skills: list[str] = Field(default_factory=list)  # required skills
    gaps: list[str] = Field(default_factory=list)  # required but not on resume
    questions: list[str] = Field(default_factory=list)  # retrieved from the KB
    guide: str = ""  # narrative prep guide (LLM or deterministic fallback)


_SYSTEM = (
    "You are an interview coach. Using ONLY the provided question bank and the "
    "candidate's skill profile, write a concise interview-prep guide in Markdown: "
    "which questions to expect and what to focus on, with extra attention to the "
    "skill gaps. Do not invent questions beyond those provided."
)


def _query(jd: JobDescription) -> str:
    return " ".join(jd.skills).strip() or jd.raw_text


def _fallback_guide(questions: list[str], gaps: list[str]) -> str:
    """Deterministic guide when no LLM is available."""
    lines: list[str] = []
    if questions:
        lines.append("Likely interview questions:")
        lines += [f"- {q}" for q in questions]
    if gaps:
        lines.append("")
        lines.append("Focus areas (skill gaps): " + ", ".join(gaps))
    return "\n".join(lines)


def _prompt(jd: JobDescription, gaps: list[str], questions: list[str]) -> str:
    bank = "\n".join(f"- {q}" for q in questions) or "(none)"
    return (
        f"Required skills: {', '.join(jd.skills) or '(unspecified)'}\n"
        f"Candidate skill gaps: {', '.join(gaps) or '(none)'}\n\n"
        f"Question bank (retrieved):\n{bank}\n\n"
        "Write the interview-prep guide grounded on these questions."
    )


def generate_interview_prep(
    jd: JobDescription,
    resume: Resume,
    kb_retriever: Retriever,
    llm: "LLMClient | None" = None,
    k: int = 8,
) -> InterviewPrep:
    """Generate grounded interview prep for a JD/resume from the KB.

    Args:
        jd: Parsed job description.
        resume: Parsed resume.
        kb_retriever: A retriever over the interview knowledge base.
        llm: Optional LLM for the narrative guide.
        k: Number of KB documents to retrieve.

    Returns:
        An InterviewPrep with the required skills, gaps, retrieved questions,
        and a guide (LLM-generated or deterministic fallback).
    """
    required = jd.skills
    covered = {s.lower() for s in resume.skills}
    gaps = [s for s in required if s.lower() not in covered]

    results = kb_retriever.search(_query(jd), k=k)
    questions = [r.text for r in results]

    fallback = _fallback_guide(questions, gaps)
    guide = fallback
    if llm is not None:
        guide = generate_text(
            llm, _prompt(jd, gaps, questions), system=_SYSTEM, fallback=fallback
        )

    return InterviewPrep(
        skills=required, gaps=gaps, questions=questions, guide=guide
    )
