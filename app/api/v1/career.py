from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api import deps
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
