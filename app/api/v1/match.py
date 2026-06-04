from functools import lru_cache

from fastapi import APIRouter

from app.models.match import MatchRequest, MatchResponse
from app.services.embedding import EmbeddingService
from app.services.keyword_matcher import match as match_jd_resume

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
