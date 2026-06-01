from fastapi import APIRouter

from app.models.match import MatchRequest, MatchResponse, MatchResult

router = APIRouter()


@router.post("/match", response_model=MatchResponse)
async def match_jd_resume(request: MatchRequest):
    """Match a job description against a resume.

    (Placeholder — full matching logic arrives on Day 4+.)
    """
    result = MatchResult(
        summary="Matching logic not yet implemented. Will be added from Day 4 onward.",
    )
    return MatchResponse(data=result)
