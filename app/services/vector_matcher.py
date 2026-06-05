"""
VectorMatcher — Embedding-based semantic matching for JD-to-resume analysis.

Provides:
  - match_skills()  — semantic skill-to-skill matching via embedding similarity
  - match_experiences_to_responsibilities() — JD responsibility → resume experience alignment
  - semantic_skill_match_rate() — convenience wrapper, fraction of JD skills matched

All methods accept an EmbeddingService (duck-typed) and work by embedding
all texts in one batch, then computing a cross-similarity matrix. Each query
text is paired with the candidate that has the highest cosine similarity
above a configurable threshold.
"""

import numpy as np

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService

# Default thresholds tuned for all-MiniLM-L6-v2 (384-dim).
# Skills are short (1-3 words) so need a higher bar to avoid false matches.
# Experiences are longer text where moderate cosine is more meaningful.
DEFAULT_SKILL_THRESHOLD: float = 0.55
DEFAULT_EXPERIENCE_THRESHOLD: float = 0.50


def _build_experience_text(exp: Any) -> str:
    """Build a searchable text summary from a ResumeExperience-like object.

    Args:
        exp: An object with title, company, highlights attributes
             (duck-typed — matches ResumeExperience model).

    Returns:
        A single string combining the key fields.
    """
    parts = [getattr(exp, "title", "")]
    company = getattr(exp, "company", "")
    if company:
        parts.append(f"at {company}")
    highlights = getattr(exp, "highlights", [])
    if highlights:
        parts.append(": " + "; ".join(highlights))
    return " ".join(parts).strip()


class VectorMatcher:
    """Semantic matching service powered by text embeddings.

    Wraps an EmbeddingService to provide skill-level and experience-level
    semantic matching. All methods are stateless aside from the stored
    embedding_service reference and default thresholds.

    Usage::

        vm = VectorMatcher(embedding_service)
        matches = vm.match_skills(
            ["machine learning", "docker"],
            ["deep learning", "kubernetes", "python"],
        )
        # → [("machine learning", "deep learning", 0.72)]
    """

    def __init__(
        self,
        embedding_service: "EmbeddingService",
        skill_threshold: float = DEFAULT_SKILL_THRESHOLD,
        experience_threshold: float = DEFAULT_EXPERIENCE_THRESHOLD,
    ):
        self.embedding_service = embedding_service
        self.skill_threshold = skill_threshold
        self.experience_threshold = experience_threshold

    # ── Skill matching ──────────────────────────────────────────────────

    def match_skills(
        self,
        jd_skills: list[str],
        resume_skills: list[str],
        threshold: float | None = None,
    ) -> list[tuple[str, str, float]]:
        """Find semantic matches between JD skills and resume skills.

        For each JD skill, finds the resume skill with the highest cosine
        similarity.  Pairs whose similarity meets the threshold are returned.

        Skills that match exactly (case-insensitive) are skipped — those
        should already be handled by keyword matching.

        Args:
            jd_skills: Skills required by the job description.
            resume_skills: Skills extracted from the resume.
            threshold: Minimum cosine similarity (default: self.skill_threshold).

        Returns:
            List of (jd_skill, best_resume_skill, similarity) tuples,
            sorted by similarity descending.  Only pairs above the threshold.
        """
        if not jd_skills or not resume_skills:
            return []

        thresh = threshold if threshold is not None else self.skill_threshold

        # Skip JD skills that already have an exact (case-insensitive) match
        resume_lower = {s.lower() for s in resume_skills}
        unresolved = [s for s in jd_skills if s.lower() not in resume_lower]

        if not unresolved:
            return []

        # Embed all texts in one batch for efficiency
        all_texts = unresolved + resume_skills
        embeddings = self.embedding_service.encode(all_texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        jd_embs = embeddings[: len(unresolved)]
        resume_embs = embeddings[len(unresolved) :]

        # Cross-similarity matrix: (n_jd, n_resume)
        sim_matrix = np.dot(jd_embs, resume_embs.T)

        results: list[tuple[str, str, float]] = []
        for i, jd_skill in enumerate(unresolved):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i][best_idx])
            if best_score >= thresh:
                results.append((jd_skill, resume_skills[best_idx], best_score))

        # Sort by similarity descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def semantic_skill_match_rate(
        self,
        jd_skills: list[str],
        resume_skills: list[str],
        threshold: float | None = None,
    ) -> float:
        """Fraction of JD skills that have a semantic match in resume skills.

        Args:
            jd_skills: Skills required by the job description.
            resume_skills: Skills extracted from the resume.
            threshold: Minimum cosine similarity (default: self.skill_threshold).

        Returns:
            A float between 0.0 and 1.0.
        """
        total = len(jd_skills)
        if total == 0:
            return 0.0

        matches = self.match_skills(jd_skills, resume_skills, threshold)
        matched_jd_skills = {m[0] for m in matches}
        return len(matched_jd_skills) / total

    # ── Experience → responsibility matching ────────────────────────────

    def match_experiences_to_responsibilities(
        self,
        jd_responsibilities: list[str],
        resume_experiences: list[Any],
        threshold: float | None = None,
    ) -> list[tuple[str, str, float]]:
        """Match JD responsibilities to resume experience entries.

        For each JD responsibility, finds the resume experience with the
        highest cosine similarity.  Experiences are converted to text via
        _build_experience_text() before embedding.

        Args:
            jd_responsibilities: Responsibility strings from the JD.
            resume_experiences: List of ResumeExperience-like objects.
            threshold: Minimum cosine similarity (default: self.experience_threshold).

        Returns:
            List of (jd_responsibility, experience_summary, similarity) tuples,
            sorted by similarity descending.
        """
        if not jd_responsibilities or not resume_experiences:
            return []

        thresh = threshold if threshold is not None else self.experience_threshold

        exp_texts = [_build_experience_text(exp) for exp in resume_experiences]

        # Embed all texts in one batch
        all_texts = jd_responsibilities + exp_texts
        embeddings = self.embedding_service.encode(all_texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        jd_embs = embeddings[: len(jd_responsibilities)]
        exp_embs = embeddings[len(jd_responsibilities) :]

        # Cross-similarity matrix: (n_jd, n_exp)
        sim_matrix = np.dot(jd_embs, exp_embs.T)

        results: list[tuple[str, str, float]] = []
        for i, responsibility in enumerate(jd_responsibilities):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i][best_idx])
            if best_score >= thresh:
                results.append((responsibility, exp_texts[best_idx], best_score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results
