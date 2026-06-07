"""
KeywordMatcher — Baseline JD-to-resume matching via keyword + vector overlap.

Compares extracted skills from both JD and resume to produce:
  - matched_skills / missing_skills (exact + extended text + semantic)
  - skill_match_rate
  - overall_score
  - experience-to-responsibility alignment
  - a human-readable summary

When embedding_service is provided, also performs:
  - Semantic skill-to-skill matching (via VectorMatcher)
  - Experience-to-responsibility alignment (via VectorMatcher)
  - Document-level semantic similarity
"""

from app.models.jd import JobDescription
from app.models.resume import Resume
from app.models.match import (
    MatchResult,
    SkillMatchDetail,
    ExperienceMatchDetail,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService

# ── Overall-score weights (skill-coverage-first) ────────────────────────
# Skill coverage is the primary signal; the rest are supporting context.
# Each branch's weights sum to 1.0.
#
# With experience-alignment signal available:
W_SKILL_WITH_EXP: float = 0.75       # skill coverage
W_EXPERIENCE: float = 0.15           # JD-responsibility ↔ experience alignment
W_DOC_SIM_WITH_EXP: float = 0.10     # document-level semantic similarity
# Without experience signal (skills + document similarity only):
W_SKILL_NO_EXP: float = 0.85
W_DOC_SIM_NO_EXP: float = 0.15
# Keyword-only mode (no embeddings): reserve headroom for absent signals.
W_SKILL_KEYWORD_ONLY: float = 0.90


def _find_skills_in_text(skills: list[str], candidate_text: str) -> set[str]:
    """Find missing skills in raw text using the shared skill matcher."""
    if not candidate_text:
        return set()

    from app.services.jd_parser import _match_skills

    text_skills = set(_match_skills(candidate_text))
    return set(skills) & text_skills


def match(
    jd: JobDescription,
    resume: Resume,
    embedding_service: "EmbeddingService | None" = None,
) -> MatchResult:
    """Match a job description against a resume.

    Args:
        jd: Parsed JobDescription.
        resume: Parsed Resume.
        embedding_service: Optional EmbeddingService for semantic scoring.
            When provided, enables semantic skill matching, experience
            alignment, and document-level similarity.

    Returns:
        MatchResult with matched_skills, missing_skills, scores, semantic
        match details, and summary.
    """
    jd_skills = set(jd.skills)
    resume_skills = set(resume.skills)

    # ── Direct skill overlap ────────────────────────────────────────
    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)

    # ── Extended search: look for missing skills in resume text ─────
    # Build a text corpus from all resume fields
    text_matches = _find_skills_in_text(missing, resume.raw_text)
    extended_matches: list[str] = []
    still_missing: list[str] = []
    for skill in missing:
        if skill in text_matches:
            extended_matches.append(skill)
        else:
            still_missing.append(skill)

    all_matched = sorted(set(matched) | set(extended_matches))

    # ── Semantic skill matching (vector) ────────────────────────────
    semantic_skill_matches: list[SkillMatchDetail] = []
    semantic_skill_match_rate: float | None = None
    semantically_found: list[str] = []

    if embedding_service is not None:
        from app.services.vector_matcher import VectorMatcher

        vm = VectorMatcher(embedding_service)

        # Try to match still-missing skills via embedding similarity
        vm_matches = vm.match_skills(still_missing, sorted(resume_skills))
        for jd_skill, resume_skill, sim in vm_matches:
            semantic_skill_matches.append(
                SkillMatchDetail(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=round(sim, 4),
                )
            )
            semantically_found.append(jd_skill)

        # Augment matched / missing lists
        all_matched = sorted(
            set(all_matched) | set(semantically_found)
        )
        still_missing = sorted(set(still_missing) - set(semantically_found))

        # Compute semantic skill match rate
        total_jd = len(jd_skills)
        if total_jd > 0:
            semantic_skill_match_rate = round(
                len(semantically_found) / total_jd, 4
            )

    # ── Experience → responsibility matching (vector) ───────────────
    experience_matches: list[ExperienceMatchDetail] = []
    experience_match_rate: float | None = None

    if embedding_service is not None and jd.responsibilities and resume.experience:
        vm_exp_matches = vm.match_experiences_to_responsibilities(
            jd.responsibilities, resume.experience
        )
        for resp, exp_text, sim in vm_exp_matches:
            experience_matches.append(
                ExperienceMatchDetail(
                    jd_responsibility=resp,
                    resume_experience=exp_text,
                    similarity=round(sim, 4),
                )
            )

        total_resp = len(jd.responsibilities)
        if total_resp > 0:
            experience_match_rate = round(
                len(vm_exp_matches) / total_resp, 4
            )

    # ── Scores ──────────────────────────────────────────────────────
    total_jd = len(jd_skills)
    skill_match_rate = len(all_matched) / total_jd if total_jd > 0 else 0.0

    direct_score = len(matched) / total_jd if total_jd > 0 else 0.0
    extended_bonus = len(extended_matches) / total_jd if total_jd > 0 else 0.0

    semantic_similarity: float | None = None
    doc_semantic_bonus = 0.0

    if embedding_service is not None:
        jd_text = jd.raw_text
        resume_text = resume.raw_text
        if jd_text and resume_text:
            semantic_similarity = embedding_service.similarity(jd_text, resume_text)
            semantic_similarity = max(0.0, min(semantic_similarity, 1.0))
            doc_semantic_bonus = semantic_similarity

    if embedding_service is not None:
        # ── Vector-enhanced scoring formula ─────────────────────────
        # Skill coverage is the primary signal. It already includes exact,
        # raw-text, and semantic skill matches, so perfect skill coverage
        # should not be penalized for having no additional semantic matches.
        has_experience_signal = bool(jd.responsibilities and resume.experience)
        if has_experience_signal:
            experience_score = experience_match_rate or 0.0
            overall_score = (
                W_SKILL_WITH_EXP * skill_match_rate
                + W_EXPERIENCE * experience_score
                + W_DOC_SIM_WITH_EXP * doc_semantic_bonus
            )
        else:
            overall_score = (
                W_SKILL_NO_EXP * skill_match_rate
                + W_DOC_SIM_NO_EXP * doc_semantic_bonus
            )
    else:
        # ── Keyword-only scoring ────────────────────────────────────
        # Keep a small reserve for semantic/context signals that are not
        # available in keyword-only mode, while making full skill coverage
        # score high enough to read as a strong baseline match.
        overall_score = W_SKILL_KEYWORD_ONLY * skill_match_rate

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

    if semantic_skill_matches:
        first = semantic_skill_matches[0]
        summary_parts.append(
            f"Semantic skill matches: {len(semantic_skill_matches)} "
            f"(e.g., '{first.jd_skill}' ↔ '{first.resume_skill}': "
            f"{first.similarity:.2f})"
        )

    if experience_matches:
        summary_parts.append(
            f"Experience matches: {len(experience_matches)}/"
            f"{len(jd.responsibilities)} responsibilities aligned"
        )

    if semantic_similarity is not None:
        summary_parts.append(f"Document semantic similarity: {semantic_similarity:.2f}")

    summary_parts.append(f"Skill match rate: {skill_match_rate:.0%}")
    summary_parts.append(f"Overall score: {overall_score:.2f}")

    return MatchResult(
        matched_skills=all_matched,
        missing_skills=still_missing,
        overall_score=round(overall_score, 4),
        skill_match_rate=round(skill_match_rate, 4),
        semantic_similarity=round(semantic_similarity, 4)
        if semantic_similarity is not None
        else None,
        summary="\n".join(summary_parts),
        semantic_skill_matches=semantic_skill_matches,
        semantic_skill_match_rate=semantic_skill_match_rate,
        experience_matches=experience_matches,
        experience_match_rate=experience_match_rate,
    )
