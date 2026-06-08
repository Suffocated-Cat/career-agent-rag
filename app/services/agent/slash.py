"""
Slash commands — deterministic shortcuts in the chat.

Free-text messages go to the ReAct agent (flexible, LLM-driven). Slash commands
are the opposite: they call the deterministic pipeline directly, so ``/match``,
``/report``, ``/prep``, ``/audit``, ``/compare`` are cheap, reproducible, and
work even without an LLM. They read/write the session's shared ``ReactState``,
so a later free-text question sees what a slash command computed (and vice
versa).
"""

from app.services.agent.schemas import ReactState
from app.services.agent.tools import _compare_jds
from app.services.interview_prep import generate_interview_prep
from app.services.jd_parser import parse_jd
from app.services.match_pipeline import analyze_match
from app.services.project_auditor import audit_resume
from app.services.report_generator import generate_report
from app.services.resume_parser import parse_resume

COMMANDS = ("help", "match", "report", "prep", "audit", "compare")

_HELP = (
    "**Commands** (free text goes to the agent):\n"
    "- `/match` — score the resume against the JD\n"
    "- `/report` — full match report\n"
    "- `/prep [role] [difficulty]` — interview prep from the knowledge base\n"
    "- `/audit` — flag unsupported / exaggerated resume claims\n"
    "- `/compare` — rank multiple JDs (add JDs first)\n"
    "- `/help` — this list"
)


def is_slash(message: str) -> bool:
    return message.lstrip().startswith("/")


def _ensure_jd(state: ReactState):
    if state.jd is None and state.jd_text:
        state.jd = parse_jd(
            state.jd_text, embedding_service=state.embedding_service, llm=state.llm
        )
    return state.jd


def _ensure_resume(state: ReactState):
    if state.resume is None and state.resume_text:
        state.resume = parse_resume(
            state.resume_text, embedding_service=state.embedding_service, llm=state.llm
        )
    return state.resume


def _match(state: ReactState) -> str:
    jd, resume = _ensure_jd(state), _ensure_resume(state)
    if jd is None or resume is None:
        return "I need both a JD and a resume first."
    state.match = analyze_match(jd, resume, embedding_service=state.embedding_service)
    m = state.match
    missing = ", ".join(m.missing_skills) or "none"
    return (
        f"**Match score: {m.overall_score:.2f}**\n"
        f"- Matched skills: {', '.join(m.matched_skills) or 'none'}\n"
        f"- Missing skills: {missing}"
    )


def _report(state: ReactState) -> str:
    jd, resume = _ensure_jd(state), _ensure_resume(state)
    if jd is None or resume is None:
        return "I need both a JD and a resume first."
    if state.match is None:
        state.match = analyze_match(jd, resume, embedding_service=state.embedding_service)
    state.report = generate_report(jd, resume, state.match, llm=state.llm)
    return state.report.full_report


def _prep(state: ReactState, args: str) -> str:
    jd, resume = _ensure_jd(state), _ensure_resume(state)
    if jd is None or resume is None:
        return "I need both a JD and a resume first."
    if state.kb_retriever is None:
        return "The knowledge base isn't available, so I can't build interview prep."
    parts = args.split()
    role = parts[0] if parts else None
    difficulty = parts[1] if len(parts) > 1 else None
    state.interview = generate_interview_prep(
        jd, resume, state.kb_retriever, llm=state.llm, role=role, difficulty=difficulty
    )
    prep = state.interview
    gaps = ", ".join(prep.gaps) or "none"
    return f"**Interview prep** (gaps: {gaps})\n\n{prep.guide}"


def _audit(state: ReactState) -> str:
    resume = _ensure_resume(state)
    if resume is None:
        return "I need a resume first."
    audit = audit_resume(resume, llm=state.llm)
    if state.match is not None:
        state.match.project_audit = audit
    if not audit.findings:
        return "No authenticity risks detected."
    lines = [f"**{audit.summary}**"]
    for f in audit.findings:
        lines.append(f"- _{f.severity}_ — {f.subject}: {f.detail}")
    return "\n".join(lines)


def _compare(state: ReactState) -> str:
    _ensure_resume(state)
    return _compare_jds(state, {})


def handle_slash(message: str, state: ReactState) -> str:
    """Run a slash command against the session state, returning a reply string."""
    cmd, _, rest = message.lstrip()[1:].partition(" ")
    cmd = cmd.lower().strip()
    args = rest.strip()

    if cmd in ("help", ""):
        return _HELP
    if cmd == "match":
        return _match(state)
    if cmd == "report":
        return _report(state)
    if cmd == "prep":
        return _prep(state, args)
    if cmd == "audit":
        return _audit(state)
    if cmd == "compare":
        return _compare(state)
    return f"Unknown command `/{cmd}`. Try `/help`."
