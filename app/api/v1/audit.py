from fastapi import APIRouter

from app.models.audit import AuditRequest, AuditResponse
from app.services.project_auditor import audit_resume

router = APIRouter()


@router.post("/audit", response_model=AuditResponse)
async def audit_resume_endpoint(request: AuditRequest):
    """Audit a parsed resume for authenticity and quality risks.

    Runs rule-based checks for unsupported skill claims, vague experience
    descriptions, and unsubstantiated advanced-technology project claims,
    returning findings with an aggregate risk score.
    """
    return AuditResponse(data=audit_resume(request.resume))
