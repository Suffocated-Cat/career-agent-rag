from functools import lru_cache

from fastapi import APIRouter

from app.models.jd import JobDescription
from app.models.match import (
    MatchRequest,
    MatchResponse,
    MatchResult,
    ReportRequest,
    ReportResponse,
)
from app.models.resume import Resume
from app.services.embedding import EmbeddingService
from app.services.keyword_matcher import match as match_jd_resume
from app.services.match_pipeline import rank_resume_projects
from app.services.project_auditor import audit_resume
from app.services.report_generator import generate_report

router = APIRouter()


@lru_cache(maxsize=1)
def _get_embedding_service() -> EmbeddingService | None:
    """Create the embedding service once; fall back to keyword-only matching."""
    try:
        return EmbeddingService()
    except Exception:
        return None


def _run_match(jd: JobDescription, resume: Resume) -> MatchResult:
    """Run the full match: skill/semantic scoring plus project relevance.

    Uses hybrid retrieval for project relevance when an embedding service is
    available, falling back to lexical BM25 when it is not (the vector arm
    needs embeddings).
    """
    embedding_service = _get_embedding_service()
    result = match_jd_resume(jd, resume, embedding_service=embedding_service)

    method = "hybrid" if embedding_service is not None else "bm25"
    result.project_relevance = rank_resume_projects(
        jd, resume, embedding_service=embedding_service, method=method
    )
    result.project_audit = audit_resume(resume)
    return result


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
    report = generate_report(request.jd, request.resume, result)
    return ReportResponse(data=report)
