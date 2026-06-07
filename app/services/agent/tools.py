"""
Default ReAct tools — thin wrappers over CareerAgent services.

Each tool reads/writes the shared ``ReactState`` and returns a concise text
observation. Preconditions (e.g. "parse the JD first") are returned as error
observations so the agent can reason about and recover from ordering mistakes —
which is the point of the ReAct loop.
"""

from app.services.agent.react_controller import ReactAgent
from app.services.agent.schemas import ReactState, ReactTool
from app.services.jd_parser import parse_jd
from app.services.resume_parser import parse_resume
from app.services.keyword_matcher import match as match_jd_resume
from app.services.match_pipeline import rank_resume_projects
from app.services.project_auditor import audit_resume
from app.services.report_generator import generate_report


def _parse_jd(state: ReactState, args: dict) -> str:
    text = args.get("text") or state.jd_text
    if not text:
        return "Error: no JD text provided (pass action_input.text or seed jd_text)."
    state.jd = parse_jd(text, embedding_service=state.embedding_service, llm=state.llm)
    return (
        f"Parsed JD: title={state.jd.title!r}, {len(state.jd.skills)} skills, "
        f"{len(state.jd.responsibilities)} responsibilities."
    )


def _parse_resume(state: ReactState, args: dict) -> str:
    text = args.get("text") or state.resume_text
    if not text:
        return "Error: no resume text provided (pass action_input.text or seed resume_text)."
    state.resume = parse_resume(
        text, embedding_service=state.embedding_service, llm=state.llm
    )
    return (
        f"Parsed resume: {len(state.resume.skills)} skills, "
        f"{len(state.resume.experience)} experiences, "
        f"{len(state.resume.projects)} projects."
    )


def _match(state: ReactState, args: dict) -> str:
    if state.jd is None:
        return "Error: parse the JD first (parse_jd)."
    if state.resume is None:
        return "Error: parse the resume first (parse_resume)."
    state.match = match_jd_resume(
        state.jd, state.resume, embedding_service=state.embedding_service
    )
    m = state.match
    return (
        f"Match score {m.overall_score:.2f}: {len(m.matched_skills)} matched, "
        f"{len(m.missing_skills)} missing skills"
        + (f" ({', '.join(m.missing_skills[:5])})." if m.missing_skills else ".")
    )


def _rank_projects(state: ReactState, args: dict) -> str:
    if state.jd is None or state.resume is None:
        return "Error: parse the JD and resume first."
    method = "hybrid" if state.embedding_service is not None else "bm25"
    rels = rank_resume_projects(
        state.jd, state.resume, embedding_service=state.embedding_service, method=method
    )
    if state.match is not None:
        state.match.project_relevance = rels
    if not rels:
        return "No relevant experiences found."
    top = ", ".join(f"{r.label} ({r.normalized_score:.2f})" for r in rels[:3])
    return f"Top experiences by relevance: {top}."


def _audit(state: ReactState, args: dict) -> str:
    if state.resume is None:
        return "Error: parse the resume first (parse_resume)."
    audit = audit_resume(state.resume, llm=state.llm)
    if state.match is not None:
        state.match.project_audit = audit
    return audit.summary


def _generate_report(state: ReactState, args: dict) -> str:
    if state.match is None:
        return "Error: run match first (match)."
    state.report = generate_report(state.jd, state.resume, state.match, llm=state.llm)
    return (
        f"Report generated: {state.report.overall_rating} match, "
        f"score {state.report.overall_score:.2f}."
    )


def default_tools() -> list[ReactTool]:
    """Build the standard CareerAgent ReAct tool set."""
    return [
        ReactTool(
            "parse_jd",
            "Parse the job description into skills/responsibilities. "
            'Optional action_input: {"text": "<jd text>"}.',
            _parse_jd,
        ),
        ReactTool(
            "parse_resume",
            "Parse the resume into skills/experience/projects. "
            'Optional action_input: {"text": "<resume text>"}.',
            _parse_resume,
        ),
        ReactTool(
            "match",
            "Score the resume against the JD. Requires parsed JD + resume. "
            "No action_input.",
            _match,
        ),
        ReactTool(
            "rank_projects",
            "Rank the resume's experiences/projects by relevance to the JD. "
            "Requires parsed JD + resume. No action_input.",
            _rank_projects,
        ),
        ReactTool(
            "audit",
            "Audit the resume for unsupported/exaggerated claims. Requires a "
            "parsed resume. No action_input.",
            _audit,
        ),
        ReactTool(
            "generate_report",
            "Produce the final match report. Requires that match has run. "
            "No action_input.",
            _generate_report,
        ),
    ]


def build_default_agent(llm, max_steps: int = 8) -> ReactAgent:
    """Create a ReactAgent preloaded with the default tools."""
    return ReactAgent(llm, default_tools(), max_steps=max_steps)
