from fastapi import APIRouter

from app.models.match import MatchRequest, MatchResponse
from app.services.keyword_matcher import match as match_jd_resume

router = APIRouter()


@router.post("/match", response_model=MatchResponse)
async def match_jd_resume_endpoint(request: MatchRequest):
    """Match a parsed job description against a parsed resume.

    Computes skill overlap, match rate, and an overall match score.
    """
    result = match_jd_resume(request.jd, request.resume)
    return MatchResponse(data=result)
