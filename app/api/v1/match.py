from functools import lru_cache

from fastapi import APIRouter

from app.models.match import MatchRequest, MatchResponse, ReportRequest, ReportResponse
from app.services.embedding import EmbeddingService
from app.services.keyword_matcher import match as match_jd_resume
from app.services.report_generator import generate_report

router = APIRouter()


@lru_cache(maxsize=1)
def _get_embedding_service() -> EmbeddingService | None:
    """Create the embedding service once; fall back to keyword-only matching."""
    try:
        return EmbeddingService()
    except Exception:
        return None


@router.post("/match", response_model=MatchResponse)
async def match_jd_resume_endpoint(request: MatchRequest):
    """Match a parsed job description against a parsed resume.

    Computes skill overlap, semantic similarity, and an overall match score.
    """
    result = match_jd_resume(
        request.jd,
        request.resume,
        embedding_service=_get_embedding_service(),
    )
    return MatchResponse(data=result)


@router.post("/match/report", response_model=ReportResponse)
async def generate_match_report(request: ReportRequest):
    """Generate a structured matching report for a JD and resume.

    Runs the full matching pipeline then produces a human-readable report
    with skill analysis, experience alignment, gap analysis, and
    recommendations.
    """
    result = match_jd_resume(
        request.jd,
        request.resume,
        embedding_service=_get_embedding_service(),
    )
    report = generate_report(request.jd, request.resume, result)
    return ReportResponse(data=report)
