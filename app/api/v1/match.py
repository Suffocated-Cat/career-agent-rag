from fastapi import APIRouter

from app.api import deps
from app.models.jd import JobDescription
from app.models.match import (
    MatchRequest,
    MatchResponse,
    MatchResult,
    ReportRequest,
    ReportResponse,
)
from app.models.resume import Resume
from app.services.match_pipeline import analyze_match
from app.services.report_generator import generate_report

router = APIRouter()


def _run_match(jd: JobDescription, resume: Resume) -> MatchResult:
    """Run the full match analysis using the shared orchestrator.

    Audit advice (LLM) is intentionally left off this hot path; the standalone
    ``/audit`` endpoint and the career-match skill supply it.
    """
    return analyze_match(jd, resume, embedding_service=deps.get_embedding_service())


@router.post("/match", response_model=MatchResponse)
async def match_jd_resume_endpoint(request: MatchRequest):
    """Match a parsed job description against a parsed resume.

    Computes skill overlap, semantic similarity, an overall match score, and
    per-experience/project relevance ranking.
    """
    return MatchResponse(data=_run_match(request.jd, request.resume))


@router.post("/match/report", response_model=ReportResponse)
async def generate_match_report(request: ReportRequest):
    """Generate a structured matching report for a JD and resume.

    Runs the full matching pipeline then produces a human-readable report
    with skill analysis, experience alignment, gap analysis, and
    recommendations.
    """
    result = _run_match(request.jd, request.resume)
    report = generate_report(request.jd, request.resume, result, llm=deps.get_llm())
    return ReportResponse(data=report)
