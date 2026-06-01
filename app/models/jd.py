from pydantic import BaseModel, Field


class JobDescription(BaseModel):
    """Represents a parsed job description."""

    raw_text: str = Field(..., description="Original JD text")
    title: str | None = Field(default=None, description="Job title")
    company: str | None = Field(default=None, description="Company name")
    skills: list[str] = Field(default_factory=list, description="Required skills")
    responsibilities: list[str] = Field(default_factory=list, description="Key responsibilities")
    nice_to_haves: list[str] = Field(default_factory=list, description="Bonus / nice-to-have skills")


class JdParseRequest(BaseModel):
    """Request body for JD parsing endpoint."""

    raw_text: str = Field(..., min_length=1, description="Raw job description text")


class JdParseResponse(BaseModel):
    """Response for JD parsing endpoint."""

    status: str = "success"
    data: JobDescription
