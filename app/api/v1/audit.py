from functools import lru_cache

from fastapi import APIRouter

from app.models.audit import AuditRequest, AuditResponse
from app.services.llm_client import LLMClient
from app.services.project_auditor import audit_resume

router = APIRouter()


@lru_cache(maxsize=1)
def _get_llm() -> LLMClient:
    """Create the LLM client once. Unconfigured → advice is omitted."""
    return LLMClient()


@router.post("/audit", response_model=AuditResponse)
async def audit_resume_endpoint(request: AuditRequest):
    """Audit a parsed resume for authenticity and quality risks.

    Runs rule-based checks for unsupported skill claims, vague experience
    descriptions, and unsubstantiated advanced-technology project claims,
    returning findings with an aggregate risk score. When an LLM is configured,
    it also returns natural-language advice on how to address the findings.
    """
    return AuditResponse(data=audit_resume(request.resume, llm=_get_llm()))
