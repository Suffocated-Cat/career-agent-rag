from pydantic import BaseModel, Field

from app.models.jd import JobDescription
from app.models.resume import Resume


class MatchResult(BaseModel):
    """Represents a JD-to-resume match analysis."""

    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    skill_match_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = Field(default="")


class MatchRequest(BaseModel):
    """Request body for matching endpoint."""

    jd: JobDescription
    resume: Resume


class MatchResponse(BaseModel):
    """Response for matching endpoint."""

    status: str = "success"
    data: MatchResult
