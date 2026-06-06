"""
ProjectAuditor — rule-based authenticity / quality checks on a resume.

Retrieval (Week 2) answers "which experience is most relevant"; auditing asks
"is the claim trustworthy". These are deliberately transparent rules, not an
LLM, so every finding is explainable and reproducible:

  - unsupported_skill        — a skill is listed but never appears in any
                               experience or project (claimed without evidence).
                               High severity for advanced/impressive claims.
  - vague_experience         — a highlight claims impact or effort without any
                               quantification (no numbers/percentages).
  - unsupported_project_claim — a project lists advanced technologies its
                               description does not substantiate.

The output feeds the project-risk section of the match report and gives the
LLM (Week 3) a structured starting point instead of auditing from scratch.
"""

import json
import re

from typing import TYPE_CHECKING

from app.models.audit import ProjectAuditReport, RiskFinding
from app.models.resume import Resume
from app.services.llm_support import generate_text
from app.services.retrieval.base import tokenize

if TYPE_CHECKING:
    from app.services.llm_client import LLMClient

_ADVICE_SYSTEM = (
    "You are a resume coach. Given a list of detected risks in a resume, "
    "explain for each why it weakens credibility and give concrete, honest "
    "steps to fix it — what evidence to add or how to rewrite the claim. "
    "Use only the provided findings; do not invent new issues. Respond in "
    "Markdown."
)


def _advice_prompt(findings: list[RiskFinding]) -> str:
    """Build the grounding prompt from the deterministic findings."""
    data = [f.model_dump() for f in findings]
    return (
        "Give concrete advice for addressing these resume risk findings, "
        "grounded strictly on them:\n\n"
        f"{json.dumps(data, indent=2, ensure_ascii=False)}"
    )

# Impressive, high-claim capabilities that demand supporting evidence.
ADVANCED_CLAIMS: set[str] = {
    "rag", "agent", "agents", "mcp", "llm", "llms", "rlhf", "dpo",
    "lora", "qlora", "reranker", "rerank", "fine-tuning", "finetuning",
    "distillation", "quantization", "vllm", "mlops", "rft",
}

# Markers that a highlight is claiming impact/effort it should quantify.
IMPACT_MARKERS: tuple[str, ...] = (
    "worked on", "helped", "improve", "improved", "enhance", "enhanced",
    "optimi", "increase", "increased", "reduce", "reduced", "better",
    "various", "several", "participated", "involved", "assisted",
    "responsible for", "a variety", "contributed to", "a number of",
)

# A project description shorter than this (in words) is treated as thin.
THIN_DESCRIPTION_WORDS: int = 12

_SEVERITY_WEIGHT: dict[str, int] = {"low": 1, "medium": 2, "high": 3}


def _has_metric(text: str) -> bool:
    """True if the text contains a number or percentage (a quantified claim)."""
    return bool(re.search(r"\d", text))


def _is_advanced(term: str) -> bool:
    """True if a technology term counts as an advanced/high-claim capability."""
    lowered = term.lower().strip()
    if lowered in ADVANCED_CLAIMS:
        return True
    return any(tok in ADVANCED_CLAIMS for tok in tokenize(lowered))


class _Evidence:
    """Evidence corpora used to check whether a skill claim is backed up.

    Two tiers, because not all evidence is equally trustworthy:

      - prose: experience titles/highlights and project names/descriptions —
        text describing actual work.
      - full:  prose plus project technology lists.

    Ordinary skills can be supported by either tier, but advanced/high-claim
    skills must appear in *prose* — merely listing an impressive term as a
    "technology" is the same kind of unverified claim as listing it as a skill.
    """

    def __init__(self, resume: Resume):
        prose_parts: list[str] = []
        for exp in resume.experience:
            prose_parts.append(exp.title or "")
            prose_parts.extend(exp.highlights)
        for proj in resume.projects:
            prose_parts.append(proj.name or "")
            prose_parts.append(proj.description or "")

        tech_parts = [t for proj in resume.projects for t in proj.technologies]

        self.prose_text = " ".join(p for p in prose_parts if p).lower()
        self.prose_tokens = set(tokenize(self.prose_text))

        full = self.prose_text + " " + " ".join(tech_parts).lower()
        self.full_text = full
        self.full_tokens = set(tokenize(full))


def _is_supported(skill: str, advanced: bool, evidence: _Evidence) -> bool:
    """True if *skill* is backed by the evidence corpus.

    Advanced skills are checked against prose only; ordinary skills may also be
    supported by a project's technology list. Single-word skills match on token
    equality (avoiding spurious substring hits); multi-word skills match as a
    substring.
    """
    skill_lower = skill.lower().strip()
    if not skill_lower:
        return True  # nothing to verify

    if advanced:
        text, tokens = evidence.prose_text, evidence.prose_tokens
    else:
        text, tokens = evidence.full_text, evidence.full_tokens

    skill_tokens = tokenize(skill_lower)
    if len(skill_tokens) <= 1:
        return skill_lower in tokens
    return skill_lower in text


def _audit_skills(resume: Resume, evidence: _Evidence) -> list[RiskFinding]:
    """Flag skills claimed without any supporting experience/project."""
    findings: list[RiskFinding] = []
    for skill in resume.skills:
        advanced = _is_advanced(skill)
        if _is_supported(skill, advanced, evidence):
            continue
        findings.append(
            RiskFinding(
                category="unsupported_skill",
                severity="high" if advanced else "medium",
                subject=skill,
                detail=(
                    f"'{skill}' is listed as a skill but does not appear in any "
                    "experience or project"
                    + (
                        " — advanced capabilities like this need concrete evidence."
                        if advanced
                        else "."
                    )
                ),
            )
        )
    return findings


def _audit_experiences(resume: Resume) -> list[RiskFinding]:
    """Flag highlights that claim impact/effort without quantification."""
    findings: list[RiskFinding] = []
    for exp in resume.experience:
        for highlight in exp.highlights:
            lowered = highlight.lower()
            if _has_metric(highlight):
                continue
            if any(marker in lowered for marker in IMPACT_MARKERS):
                findings.append(
                    RiskFinding(
                        category="vague_experience",
                        severity="low",
                        subject=exp.title or "(untitled experience)",
                        detail="Claims impact or effort without any quantification.",
                        evidence=highlight,
                    )
                )
    return findings


def _audit_projects(resume: Resume) -> list[RiskFinding]:
    """Flag projects whose advanced tech claims the description doesn't back up."""
    findings: list[RiskFinding] = []
    for proj in resume.projects:
        advanced = [t for t in proj.technologies if _is_advanced(t)]
        if not advanced:
            continue

        description = (proj.description or "").lower()
        word_count = len(description.split())
        mentioned = [t for t in advanced if t.lower() in description]

        thin = word_count < THIN_DESCRIPTION_WORDS
        unmentioned = [t for t in advanced if t not in mentioned]

        if thin or unmentioned:
            unsupported = unmentioned or advanced
            findings.append(
                RiskFinding(
                    category="unsupported_project_claim",
                    severity="high",
                    subject=proj.name or "(unnamed project)",
                    detail=(
                        "Lists advanced technologies the description does not "
                        f"substantiate: {', '.join(unsupported)}."
                    ),
                    evidence=proj.description or "",
                )
            )
    return findings


def audit_resume(
    resume: Resume,
    llm: "LLMClient | None" = None,
) -> ProjectAuditReport:
    """Audit a resume for unsupported or vague claims.

    Findings and the risk score are always computed deterministically. When an
    *llm* is provided and configured, natural-language advice on how to address
    the findings is generated (grounded strictly on them); the numbers are
    never affected.

    Args:
        resume: The parsed resume to audit.
        llm: Optional LLM client for "how to fix" advice.

    Returns:
        A ProjectAuditReport with findings, an aggregate risk score, a one-line
        summary, and optional advice.
    """
    evidence = _Evidence(resume)

    findings: list[RiskFinding] = []
    findings += _audit_skills(resume, evidence)
    findings += _audit_experiences(resume)
    findings += _audit_projects(resume)

    risk_score = _risk_score(resume, findings)

    advice = ""
    if llm is not None and findings:
        advice = generate_text(
            llm, _advice_prompt(findings), system=_ADVICE_SYSTEM, fallback=""
        )

    return ProjectAuditReport(
        advice=advice,
        findings=findings,
        risk_score=risk_score,
        summary=_summary(findings, risk_score),
    )


def _risk_score(resume: Resume, findings: list[RiskFinding]) -> float:
    """Severity-weighted risk, normalized by resume size to stay in [0, 1]."""
    if not findings:
        return 0.0
    total = sum(_SEVERITY_WEIGHT[f.severity] for f in findings)
    n_units = len(resume.skills) + len(resume.experience) + len(resume.projects)
    denom = _SEVERITY_WEIGHT["high"] * max(1, n_units)
    return round(min(1.0, total / denom), 4)


def _summary(findings: list[RiskFinding], risk_score: float) -> str:
    """Build a one-line human-readable summary of the audit."""
    if not findings:
        return "No authenticity risks detected."

    counts = {"high": 0, "medium": 0, "low": 0}
    for f in findings:
        counts[f.severity] += 1
    parts = [f"{counts[s]} {s}" for s in ("high", "medium", "low") if counts[s]]
    return (
        f"Found {len(findings)} risk(s): {', '.join(parts)}. "
        f"Risk score {risk_score:.2f}."
    )
