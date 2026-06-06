"""
MCP tool implementations for CareerAgent.

These are plain functions that take JSON-serializable inputs and return
JSON-serializable dicts — the contract an MCP tool exposes. Keeping the logic
here (rather than inside the FastMCP server) makes it dependency-free and
unit-testable; ``server.py`` simply registers these with FastMCP.

Each function reconstructs the relevant Pydantic models from plain dicts,
calls the underlying service, and dumps the result back to a dict.
"""

from typing import Any

from app.models.jd import JobDescription
from app.models.resume import Resume
from app.services.jd_parser import parse_jd as _parse_jd
from app.services.resume_parser import parse_resume as _parse_resume
from app.services.keyword_matcher import match as _match
from app.services.project_auditor import audit_resume as _audit
from app.services.match_pipeline import rank_resume_projects as _rank


def parse_jd_tool(raw_text: str, embedding_service: Any | None = None) -> dict:
    """Parse raw job-description text into a structured JobDescription dict."""
    return _parse_jd(raw_text, embedding_service=embedding_service).model_dump()


def parse_resume_tool(raw_text: str, embedding_service: Any | None = None) -> dict:
    """Parse raw resume text into a structured Resume dict."""
    return _parse_resume(raw_text, embedding_service=embedding_service).model_dump()


def match_tool(
    jd: dict, resume: dict, embedding_service: Any | None = None
) -> dict:
    """Match a JD against a resume; return the MatchResult as a dict."""
    result = _match(
        JobDescription(**jd), Resume(**resume), embedding_service=embedding_service
    )
    return result.model_dump()


def audit_tool(resume: dict) -> dict:
    """Audit a resume for authenticity risks; return the report as a dict."""
    return _audit(Resume(**resume)).model_dump()


def rank_projects_tool(
    jd: dict, resume: dict, embedding_service: Any | None = None
) -> list[dict]:
    """Rank a resume's experiences/projects by relevance to the JD."""
    method = "hybrid" if embedding_service is not None else "bm25"
    relevances = _rank(
        JobDescription(**jd),
        Resume(**resume),
        embedding_service=embedding_service,
        method=method,
    )
    return [r.model_dump() for r in relevances]
