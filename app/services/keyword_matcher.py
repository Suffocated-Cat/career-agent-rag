"""
KeywordMatcher — Baseline JD-to-resume matching via keyword overlap.

Compares extracted skills from both JD and resume to produce:
  - matched_skills / missing_skills
  - skill_match_rate
  - overall_score
  - a human-readable summary

When embedding_service is provided, also computes a semantic similarity
score between the JD text and resume text as an additional signal.
"""

from app.models.jd import JobDescription
from app.models.resume import Resume
from app.models.match import MatchResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService


def _fuzzy_match(skill: str, candidate_text: str) -> bool:
    """Check if a skill name appears (case-insensitive) in text."""
    return skill.lower() in candidate_text.lower()


def match(
    jd: JobDescription,
    resume: Resume,
    embedding_service: "EmbeddingService | None" = None,
) -> MatchResult:
    """Match a job description against a resume.

    Args:
        jd: Parsed JobDescription.
        resume: Parsed Resume.
        embedding_service: Optional EmbeddingService for semantic score.

    Returns:
        MatchResult with matched_skills, missing_skills, scores, and summary.
    """
    jd_skills = set(jd.skills)
    resume_skills = set(resume.skills)

    # ── Direct skill overlap ────────────────────────────────────────
    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)

    # ── Extended search: look for missing skills in resume text ─────
    # Build a text corpus from all resume fields
    resume_text_corpus = resume.raw_text.lower()

    extended_matches: list[str] = []
    still_missing: list[str] = []
    for skill in missing:
        if _fuzzy_match(skill, resume_text_corpus):
            extended_matches.append(skill)
        else:
            still_missing.append(skill)

    all_matched = sorted(set(matched) | set(extended_matches))

    # ── Scores ──────────────────────────────────────────────────────
    total_jd = len(jd_skills)
    skill_match_rate = len(all_matched) / total_jd if total_jd > 0 else 0.0

    # Overall score: weighted combination
    # - direct skill match (weight 0.7)
    # - extended match bonus (weight 0.2)
    # - semantic similarity bonus (weight 0.1, only if embedding available)
    direct_score = len(matched) / total_jd if total_jd > 0 else 0.0
    extended_bonus = len(extended_matches) / total_jd if total_jd > 0 else 0.0

    semantic_bonus = 0.0
    if embedding_service is not None:
        from app.services.embedding import EmbeddingService

        jd_text = jd.raw_text
        resume_text = resume.raw_text
        if jd_text and resume_text:
            semantic_bonus = embedding_service.similarity(jd_text, resume_text)

    overall_score = 0.7 * direct_score + 0.2 * extended_bonus + 0.1 * semantic_bonus
    overall_score = min(overall_score, 1.0)

    # ── Summary ─────────────────────────────────────────────────────
    summary_parts: list[str] = []

    if all_matched:
        summary_parts.append(
            f"Matched {len(all_matched)}/{total_jd} required skills: "
            + ", ".join(all_matched[:8])
            + ("..." if len(all_matched) > 8 else "")
        )
    else:
        summary_parts.append(
            f"0/{total_jd} required skills matched directly."
        )

    if still_missing:
        summary_parts.append(
            f"Missing skills: {', '.join(still_missing[:8])}"
            + ("..." if len(still_missing) > 8 else "")
        )

    if extended_matches:
        summary_parts.append(
            f"Found {len(extended_matches)} additional skill(s) in resume text: "
            + ", ".join(extended_matches)
        )

    summary_parts.append(f"Skill match rate: {skill_match_rate:.0%}")
    summary_parts.append(f"Overall score: {overall_score:.2f}")

    return MatchResult(
        matched_skills=all_matched,
        missing_skills=still_missing,
        overall_score=round(overall_score, 4),
        skill_match_rate=round(skill_match_rate, 4),
        summary="\n".join(summary_parts),
    )
