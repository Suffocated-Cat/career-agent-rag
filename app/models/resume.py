from pydantic import BaseModel, Field


class ResumeProject(BaseModel):
    """A project listed on the resume."""

    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)


class ResumeEducation(BaseModel):
    """An education entry on the resume."""

    degree: str
    institution: str
    year: str | None = None


class ResumeExperience(BaseModel):
    """A work experience entry on the resume."""

    title: str
    company: str
    duration: str | None = None
    highlights: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    """Represents a parsed resume."""

    raw_text: str = Field(..., description="Original resume text")
    skills: list[str] = Field(default_factory=list, description="Extracted skills")
    projects: list[ResumeProject] = Field(default_factory=list)
    education: list[ResumeEducation] = Field(default_factory=list)
    experience: list[ResumeExperience] = Field(default_factory=list)


class ResumeParseRequest(BaseModel):
    """Request body for resume parsing endpoint."""

    raw_text: str = Field(..., min_length=1, description="Raw resume text")


class ResumeParseResponse(BaseModel):
    """Response for resume parsing endpoint."""

    status: str = "success"
    data: Resume
