from pydantic import BaseModel, Field

from app.models.resume import Resume


class RiskFinding(BaseModel):
    """A single authenticity / quality risk detected in a resume."""

    category: str  # "unsupported_skill" | "vague_experience" | "unsupported_project_claim"
    severity: str  # "low" | "medium" | "high"
    subject: str  # the skill name, experience title, or project name
    detail: str  # human-readable explanation of the risk
    evidence: str = ""  # the offending text snippet, when applicable


class ProjectAuditReport(BaseModel):
    """Result of auditing a resume for risky or unsupported claims."""

    findings: list[RiskFinding] = Field(default_factory=list)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)  # higher = riskier
    summary: str = ""
    advice: str = ""  # optional LLM guidance on how to address the findings


class AuditRequest(BaseModel):
    """Request body for the resume audit endpoint."""

    resume: Resume


class AuditResponse(BaseModel):
    """Response for the resume audit endpoint."""

    status: str = "success"
    data: ProjectAuditReport
