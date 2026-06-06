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


class ProjectRelevance(BaseModel):
    """Relevance of a single resume experience/project to the JD query."""

    doc_id: str  # stable id from RetrievalDocument, e.g. "exp:0" / "proj:1"
    source_type: str  # "experience" | "project"
    label: str  # human-readable, e.g. "ML Engineer at Acme"
    score: float  # raw retriever score (scale depends on method)
    normalized_score: float = Field(ge=0.0, le=1.0)  # min-max within results


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

    # ── Retrieval-based project relevance ───────────────────────────────
    project_relevance: list[ProjectRelevance] = Field(default_factory=list)


class MatchRequest(BaseModel):
    """Request body for matching endpoint."""

    jd: JobDescription
    resume: Resume


class MatchResponse(BaseModel):
    """Response for matching endpoint."""

    status: str = "success"
    data: MatchResult


class MatchReport(BaseModel):
    """Structured JD-to-resume matching report."""

    job_title: str = ""
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_rating: str = ""  # Excellent / Good / Fair / Low
    skill_summary: str = ""  # "X/Y skills matched (Z via semantic)"
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    semantic_skill_matches: list[SkillMatchDetail] = Field(default_factory=list)
    experience_alignment: list[ExperienceMatchDetail] = Field(default_factory=list)
    skill_gap_analysis: str = ""  # human-readable gap description
    recommendations: str = ""  # next steps / learning focus
    full_report: str = ""  # complete markdown report


class ReportRequest(BaseModel):
    """Request body for report generation endpoint."""

    jd: JobDescription
    resume: Resume


class ReportResponse(BaseModel):
    """Response for report generation endpoint."""

    status: str = "success"
    data: MatchReport
