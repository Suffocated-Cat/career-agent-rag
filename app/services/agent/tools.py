"""
Default tool set for the AgentController.

Each tool wraps an existing CareerAgent service and reads what it needs from
the AgentContext, raising a clear error when a required input is missing.
``build_default_controller`` assembles them into a ready-to-use controller.
"""

from app.services.agent.controller import AgentContext, AgentController, Tool
from app.services.jd_parser import parse_jd
from app.services.resume_parser import parse_resume
from app.services.keyword_matcher import match as match_jd_resume
from app.services.project_auditor import audit_resume
from app.services.match_pipeline import rank_resume_projects


def _require(value, what: str):
    """Return *value* or raise a clear error naming the missing input."""
    if value is None:
        raise ValueError(f"This tool requires {what}.")
    return value


def _jd_parser_handler(ctx: AgentContext):
    text = ctx.jd_text or (ctx.jd.raw_text if ctx.jd else None)
    return parse_jd(_require(text, "jd_text"), embedding_service=ctx.embedding_service)


def _resume_parser_handler(ctx: AgentContext):
    text = ctx.resume_text or (ctx.resume.raw_text if ctx.resume else None)
    return parse_resume(
        _require(text, "resume_text"), embedding_service=ctx.embedding_service
    )


def _resume_matcher_handler(ctx: AgentContext):
    jd = _require(ctx.jd, "a parsed jd")
    resume = _require(ctx.resume, "a parsed resume")
    return match_jd_resume(jd, resume, embedding_service=ctx.embedding_service)


def _project_auditor_handler(ctx: AgentContext):
    return audit_resume(_require(ctx.resume, "a parsed resume"))


def _project_ranker_handler(ctx: AgentContext):
    jd = _require(ctx.jd, "a parsed jd")
    resume = _require(ctx.resume, "a parsed resume")
    method = "hybrid" if ctx.embedding_service is not None else "bm25"
    return rank_resume_projects(
        jd, resume, embedding_service=ctx.embedding_service, method=method
    )


def default_tools() -> list[Tool]:
    """Build the standard CareerAgent tool set."""
    return [
        Tool(
            name="jd_parser",
            description="Parse a job description into skills, responsibilities, "
            "and nice-to-haves.",
            keywords=(
                "jd", "job description", "job posting", "parse jd",
                "analyze jd", "requirements", "responsibilities",
            ),
            handler=_jd_parser_handler,
        ),
        Tool(
            name="resume_parser",
            description="Parse a resume into skills, experience, projects, "
            "and education.",
            keywords=(
                "resume", "cv", "parse resume", "extract resume",
            ),
            handler=_resume_parser_handler,
        ),
        Tool(
            name="resume_matcher",
            description="Match a resume against a job description and score the fit.",
            keywords=(
                "match", "matching", "fit", "compare", "score", "alignment",
                "gap", "suitability",
            ),
            handler=_resume_matcher_handler,
        ),
        Tool(
            name="project_auditor",
            description="Audit a resume for unsupported or vague claims and "
            "authenticity risks.",
            keywords=(
                "audit", "risk", "authenticity", "verify", "fake",
                "unsupported", "trustworth", "credibility",
            ),
            handler=_project_auditor_handler,
        ),
        Tool(
            name="project_ranker",
            description="Rank a resume's experiences and projects by relevance "
            "to the job description.",
            keywords=(
                "rank", "relevance", "most relevant", "ranking",
                "which project", "best experience",
            ),
            handler=_project_ranker_handler,
        ),
    ]


def build_default_controller() -> AgentController:
    """Create an AgentController preloaded with the default tools."""
    return AgentController(default_tools())
