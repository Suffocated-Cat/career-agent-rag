from pydantic import BaseModel, Field

from app.models.jd import JobDescription
from app.models.resume import Resume


class SkillMatchDetail(BaseModel):
    """A single semantic skill match between a JD skill and a resume skill."""

    jd_skill: str
    resume_skill: str
    similarity: float = Field(ge=0.0, le=1.0)


class ExperienceMatchDetail(BaseModel):
    """A single semantic match between a JD responsibility and resume experience."""

    jd_responsibility: str
    resume_experience: str
    similarity: float = Field(ge=0.0, le=1.0)


class MatchResult(BaseModel):
    """Represents a JD-to-resume match analysis."""

    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    skill_match_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    summary: str = Field(default="")

    # ── Vector / semantic match details ────────────────────────────────
    semantic_skill_matches: list[SkillMatchDetail] = Field(default_factory=list)
    semantic_skill_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    experience_matches: list[ExperienceMatchDetail] = Field(default_factory=list)
    experience_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class MatchRequest(BaseModel):
    """Request body for matching endpoint."""

    jd: JobDescription
    resume: Resume


class MatchResponse(BaseModel):
    """Response for matching endpoint."""

    status: str = "success"
    data: MatchResult
