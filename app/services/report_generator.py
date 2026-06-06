"""
ReportGenerator — Template-based matching report generation.

Takes a JobDescription, Resume, and MatchResult and produces a structured
MatchReport with scores, ratings, skill gap analysis, recommendations,
and a full markdown report — all via template formatting, no LLM needed.

Replaced by LLM-based report generation in Week 3.
"""

import json

from typing import TYPE_CHECKING

from app.models.jd import JobDescription
from app.models.resume import Resume
from app.models.match import MatchResult, MatchReport
from app.services.llm_support import generate_text

if TYPE_CHECKING:
    from app.services.llm_client import LLMClient

_REPORT_SYSTEM = (
    "You are a career coach writing concise, encouraging, evidence-grounded "
    "job-match reports in Markdown. Use ONLY the structured data provided. "
    "Never invent skills, scores, companies, experiences, or facts not present "
    "in the data, and keep all scores and ratings exactly as given."
)


def _report_prompt(report: MatchReport) -> str:
    """Build the grounding prompt from the structured report data."""
    data = report.model_dump(exclude={"full_report"})
    return (
        "Write a professional job-match report in Markdown, grounded strictly "
        "on this structured data. Cover the overall assessment, skill match and "
        "gaps, most relevant experience, any authenticity risks, and concrete "
        "recommendations.\n\n"
        f"Data (JSON):\n{json.dumps(data, indent=2, ensure_ascii=False)}"
    )


def _rating(score: float) -> str:
    """Map overall score to a human-readable rating."""
    if score >= 0.80:
        return "Excellent"
    elif score >= 0.60:
        return "Good"
    elif score >= 0.40:
        return "Fair"
    else:
        return "Low"


def _skill_summary(result: MatchResult) -> str:
    """Build a one-line skill match summary."""
    total = len(result.matched_skills) + len(result.missing_skills)
    matched = len(result.matched_skills)
    semantic = len(result.semantic_skill_matches)

    base = f"{matched}/{total} skills matched"
    if semantic > 0:
        base += f" ({semantic} via semantic matching)"
    return base


def _skill_gap_analysis(missing: list[str]) -> str:
    """Generate a human-readable skill gap description."""
    if not missing:
        return "No skill gaps identified — all required skills are covered."

    skills_list = ", ".join(missing)
    return (
        f"Your resume is missing {len(missing)} required skill(s): "
        f"{skills_list}. Consider building experience in these areas "
        f"through projects, courses, or self-study."
    )


def _recommendations(
    result: MatchResult,
    jd_skills_count: int,
) -> str:
    """Generate next-step recommendations based on match quality."""
    score = result.overall_score
    missing = result.missing_skills

    if score >= 0.80:
        return (
            "Your profile is a strong match for this position. "
            "Highlight the matched skills in your application and "
            "be prepared to discuss your experience with them in interviews."
        )
    elif score >= 0.60:
        rec = (
            "Your profile is a good match, with some gaps. "
            "Focus on strengthening the missing skills and "
            "emphasize your transferable experience."
        )
        if missing:
            rec += f" Priority areas: {', '.join(missing[:5])}."
        return rec
    elif score >= 0.40:
        rec = (
            "Your profile has partial alignment with this position. "
            "Significant skill gaps exist — consider a targeted "
            "learning plan before applying."
        )
        if missing:
            rec += f" Start with: {', '.join(missing[:5])}."
        return rec
    else:
        rec = (
            "Your profile currently has low alignment with this position. "
            "Focus on building foundational skills before targeting "
            "similar roles."
        )
        if missing:
            rec += f" Begin with: {', '.join(missing[:5])}."
        return rec


def _build_full_report(
    jd: JobDescription,
    resume: Resume,
    result: MatchResult,
    report: "MatchReport",
) -> str:
    """Build the complete markdown report text."""
    lines: list[str] = []

    # ── Header ─────────────────────────────────────────────────────
    title = jd.title or "Untitled Position"
    company = jd.company or ""
    header = f"# Job Match Report: {title}"
    if company:
        header += f" at {company}"
    lines.append(header)
    lines.append("")

    # ── Overall Assessment ─────────────────────────────────────────
    lines.append("## Overall Assessment")
    lines.append("")
    bar = _score_bar(report.overall_score)
    lines.append(f"**Score:** {report.overall_score:.2f} / 1.00  {bar}")
    lines.append(f"**Rating:** {report.overall_rating}")
    lines.append(f"**Skill Match:** {report.skill_summary}")
    lines.append("")

    # ── Skill Analysis ─────────────────────────────────────────────
    lines.append("## Skill Analysis")
    lines.append("")

    if report.matched_skills:
        lines.append("### Matched Skills")
        lines.append("")
        for skill in report.matched_skills:
            lines.append(f"- ✅ {skill}")
        lines.append("")

    if report.semantic_skill_matches:
        lines.append("### Semantic Skill Matches")
        lines.append("")
        lines.append("These skills were matched via embedding similarity "
                      "(not exact string match):")
        lines.append("")
        lines.append("| JD Skill | Resume Skill | Similarity |")
        lines.append("|----------|-------------|------------|")
        for m in report.semantic_skill_matches:
            lines.append(f"| {m.jd_skill} | {m.resume_skill} | {m.similarity:.2f} |")
        lines.append("")

    if report.missing_skills:
        lines.append("### Missing Skills")
        lines.append("")
        for skill in report.missing_skills:
            lines.append(f"- ❌ {skill}")
        lines.append("")

    # ── Experience Alignment ───────────────────────────────────────
    if report.experience_alignment:
        lines.append("## Experience Alignment")
        lines.append("")
        lines.append("| JD Responsibility | Best Matching Experience | Similarity |")
        lines.append("|------------------|-------------------------|------------|")
        for m in report.experience_alignment:
            # Truncate long text for table readability
            exp_short = m.resume_experience[:60] + "..." if len(m.resume_experience) > 60 else m.resume_experience
            lines.append(f"| {m.jd_responsibility[:50]} | {exp_short} | {m.similarity:.2f} |")
        lines.append("")

    # ── Skill Gap Analysis ─────────────────────────────────────────
    lines.append("## Skill Gap Analysis")
    lines.append("")
    lines.append(report.skill_gap_analysis)
    lines.append("")

    # ── Recommendations ────────────────────────────────────────────
    lines.append("## Recommendations")
    lines.append("")
    lines.append(report.recommendations)
    lines.append("")

    # ── Project Risk Audit ─────────────────────────────────────────
    if report.project_audit is not None:
        audit = report.project_audit
        lines.append("## Project Risk Audit")
        lines.append("")
        lines.append(audit.summary)
        lines.append("")
        if audit.findings:
            lines.append("| Severity | Category | Subject | Detail |")
            lines.append("|----------|----------|---------|--------|")
            for f in audit.findings:
                lines.append(
                    f"| {f.severity} | {f.category} | {f.subject} | {f.detail} |"
                )
            lines.append("")

    # ── Footer ─────────────────────────────────────────────────────
    lines.append("---")
    lines.append("*Report generated by CareerAgent • Template-based matching engine*")

    return "\n".join(lines)


def _score_bar(score: float, width: int = 20) -> str:
    """Visual score bar: '████████░░░░░░░░░░░░'."""
    filled = int(score * width)
    empty = width - filled
    return f"`{'█' * filled}{'░' * empty}`"


def generate_report(
    jd: JobDescription,
    resume: Resume,
    result: MatchResult,
    llm: "LLMClient | None" = None,
) -> MatchReport:
    """Generate a structured matching report from match results.

    The structured fields are always computed deterministically. The narrative
    ``full_report`` markdown is rendered from a template by default; when an
    *llm* is provided and configured, it is generated by the LLM grounded on
    those structured fields, falling back to the template on any failure.

    Args:
        jd: Parsed JobDescription.
        resume: Parsed Resume.
        result: MatchResult from keyword_matcher.match().
        llm: Optional LLM client for narrative generation.

    Returns:
        A MatchReport with structured fields and full markdown text.
    """
    report = MatchReport(
        job_title=jd.title or "Untitled Position",
        overall_score=result.overall_score,
        overall_rating=_rating(result.overall_score),
        skill_summary=_skill_summary(result),
        matched_skills=result.matched_skills,
        missing_skills=result.missing_skills,
        semantic_skill_matches=result.semantic_skill_matches,
        experience_alignment=result.experience_matches,
        skill_gap_analysis=_skill_gap_analysis(result.missing_skills),
        recommendations=_recommendations(result, len(jd.skills)),
        project_audit=result.project_audit,
    )

    # Deterministic template is both the default and the LLM fallback.
    template = _build_full_report(jd, resume, result, report)
    if llm is not None:
        report.full_report = generate_text(
            llm, _report_prompt(report), system=_REPORT_SYSTEM, fallback=template
        )
    else:
        report.full_report = template

    return report
