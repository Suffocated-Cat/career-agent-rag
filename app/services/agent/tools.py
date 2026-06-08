"""
Default ReAct tools — thin wrappers over CareerAgent services.

Each tool reads/writes the shared ``ReactState`` and returns a concise text
observation. Preconditions (e.g. "parse the JD first") are returned as error
observations so the agent can reason about and recover from ordering mistakes —
which is the point of the ReAct loop.
"""

from app.services.agent.react_controller import ReactAgent
from app.services.agent.schemas import JdComparison, ReactState, ReactTool
from app.services.interview_prep import generate_interview_prep
from app.services.jd_parser import parse_jd
from app.services.llm_support import generate_text
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


def _kb_search(state: ReactState, args: dict) -> str:
    query = args.get("query")
    if not query:
        return "Error: provide action_input.query (what to look up in the KB)."
    if state.kb_retriever is None:
        return "Error: knowledge base not available."
    filters: dict = {}
    if args.get("role"):
        filters["role"] = args["role"]
    if args.get("difficulty"):
        filters["difficulty"] = args["difficulty"]
    k = args.get("k", 3)
    try:
        hits = state.kb_retriever.search(query, k=k, filters=filters or None)
    except TypeError:
        # Retrievers without filter support fall back to a plain search.
        hits = state.kb_retriever.search(query, k=k)
    if not hits:
        return "No KB results."
    return "\n".join(f"- {h.text[:200]}" for h in hits)


def _interview_prep(state: ReactState, args: dict) -> str:
    if state.jd is None or state.resume is None:
        return "Error: parse the JD and resume first."
    if state.kb_retriever is None:
        return "Error: knowledge base not available."
    state.interview = generate_interview_prep(
        state.jd,
        state.resume,
        state.kb_retriever,
        llm=state.llm,
        role=args.get("role"),
        difficulty=args.get("difficulty"),
    )
    prep = state.interview
    return (
        f"Interview prep ready: {len(prep.questions)} likely questions, "
        f"{len(prep.gaps)} skill gaps"
        + (f" ({', '.join(prep.gaps[:5])})." if prep.gaps else ".")
    )


def _diagnosis_context(state: ReactState) -> str:
    """Summarize whatever the agent has gathered, to ground advice/rewrites."""
    lines: list[str] = []
    if state.jd is not None:
        lines.append(f"Target role: {state.jd.title or '(untitled)'}")
        if state.jd.skills:
            lines.append(f"Required skills: {', '.join(state.jd.skills)}")
    if state.match is not None:
        m = state.match
        lines.append(f"Overall match score: {m.overall_score:.2f}")
        if m.matched_skills:
            lines.append(f"Matched skills: {', '.join(m.matched_skills)}")
        if m.missing_skills:
            lines.append(f"Missing skills: {', '.join(m.missing_skills)}")
        if m.project_relevance:
            top = ", ".join(
                f"{r.label} ({r.normalized_score:.2f})" for r in m.project_relevance[:3]
            )
            lines.append(f"Most relevant experiences: {top}")
        if m.project_audit is not None and m.project_audit.findings:
            risks = "; ".join(
                f"[{f.severity}] {f.subject}: {f.detail}"
                for f in m.project_audit.findings[:5]
            )
            lines.append(f"Audit findings: {risks}")
    return "\n".join(lines)


def _advise(state: ReactState, args: dict) -> str:
    if state.match is None:
        return "Error: run match first so there is a diagnosis to advise on."
    context = _diagnosis_context(state)
    focus = args.get("focus")
    prompt = (
        f"{context}\n\n"
        + (f"Focus the advice on: {focus}\n\n" if focus else "")
        + "Give specific, actionable advice to improve this candidate's fit for the "
        "role: what to prioritize, how to close gaps, and how to better present "
        "existing experience. Be concise and concrete."
    )
    fallback = "Prioritize the missing skills above and surface relevant experience more prominently."
    if state.llm is None:
        return fallback
    return generate_text(
        state.llm,
        prompt,
        system="You are a sharp, practical career advisor.",
        fallback=fallback,
    )


def _rewrite_bullet(state: ReactState, args: dict) -> str:
    text = args.get("text")
    if not text:
        return "Error: provide action_input.text (the resume line to rewrite)."
    focus = args.get("focus")
    jd_line = ""
    if state.jd is not None:
        jd_line = (
            f"Target role: {state.jd.title or '(untitled)'}; "
            f"required skills: {', '.join(state.jd.skills) or '(unspecified)'}.\n"
        )
    prompt = (
        f"{jd_line}"
        + (f"Emphasize: {focus}.\n" if focus else "")
        + f"Rewrite this resume bullet to be stronger and better targeted to the "
        f"role, without inventing facts not implied by the original:\n{text}"
    )
    if state.llm is None:
        return text
    return generate_text(
        state.llm,
        prompt,
        system="You rewrite resume bullets: concrete, results-oriented, honest.",
        fallback=text,
    )


def _compare_jds(state: ReactState, args: dict) -> str:
    if state.resume is None:
        return "Error: parse the resume first (parse_resume)."
    if len(state.jd_inputs) < 2:
        return "Error: need at least two JDs to compare (seed jd_inputs)."

    state.comparison = []
    for item in state.jd_inputs:
        jd = parse_jd(item.text, embedding_service=state.embedding_service, llm=state.llm)
        m = match_jd_resume(jd, state.resume, embedding_service=state.embedding_service)
        state.comparison.append(JdComparison(label=item.label, jd=jd, match=m))
    state.comparison.sort(key=lambda c: c.match.overall_score, reverse=True)

    lines = []
    for c in state.comparison:
        miss = ", ".join(c.match.missing_skills[:3])
        lines.append(
            f"{c.label}: score {c.match.overall_score:.2f}"
            + (f", missing {miss}" if miss else "")
        )
    return f"Best fit: {state.comparison[0].label}.\n" + "\n".join(lines)


def _select_jd(state: ReactState, args: dict) -> str:
    if not state.comparison:
        return "Error: run compare_jds first."
    key = args.get("label", args.get("index"))
    if key is None:
        return "Error: provide action_input.label or action_input.index."

    chosen = None
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        idx = int(key)
        if 0 <= idx < len(state.comparison):
            chosen = state.comparison[idx]
    else:
        for c in state.comparison:
            if c.label.lower() == str(key).lower():
                chosen = c
                break
    if chosen is None:
        labels = ", ".join(c.label for c in state.comparison)
        return f"Error: no JD matching {key!r}. Available: {labels}."

    state.jd = chosen.jd
    state.match = chosen.match
    return (
        f"Active JD set to {chosen.label} (score {chosen.match.overall_score:.2f}). "
        "Now use rank_projects / advise / rewrite_bullet / interview_prep on it."
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
        ReactTool(
            "kb_search",
            "Search the interview/skill knowledge base. "
            'action_input: {"query": "<text>", "role": "<optional>", '
            '"difficulty": "<optional>", "k": <optional int>}. Loop with '
            "different queries (e.g. one per skill gap) to gather context.",
            _kb_search,
        ),
        ReactTool(
            "interview_prep",
            "Generate grounded interview prep (likely questions + focus areas) "
            "from the KB. Requires parsed JD + resume. "
            'Optional action_input: {"role": "<...>", "difficulty": "<...>"}.',
            _interview_prep,
        ),
        ReactTool(
            "advise",
            "Give targeted, actionable advice grounded on the diagnosis so far "
            "(match/audit/ranking). Requires match. "
            'Optional action_input: {"focus": "<what to focus on>"}.',
            _advise,
        ),
        ReactTool(
            "rewrite_bullet",
            "Rewrite a resume line to better target the JD without inventing "
            'facts. action_input: {"text": "<bullet>", "focus": "<optional>"}.',
            _rewrite_bullet,
        ),
        ReactTool(
            "compare_jds",
            "Parse and match every candidate JD (seeded in jd_inputs) against "
            "the resume and rank them best-first. Requires a parsed resume and "
            "at least two JDs. No action_input.",
            _compare_jds,
        ),
        ReactTool(
            "select_jd",
            "After compare_jds, promote one JD to the active role so the other "
            "tools operate on it. "
            'action_input: {"label": "Job A"} or {"index": 0}.',
            _select_jd,
        ),
    ]


def build_default_agent(llm, max_steps: int = 8) -> ReactAgent:
    """Create a ReactAgent preloaded with the default tools."""
    return ReactAgent(llm, default_tools(), max_steps=max_steps)
