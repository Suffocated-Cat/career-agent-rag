"""
Shared embedding helpers for JD Parser and Resume Parser.

Provides:
  - _classify_sections()  — embedding-based paragraph → section type
  - _discover_semantic_skills() — embedding-based skill discovery
  - _split_paragraphs()   — split text into semantic paragraphs

All functions accept an EmbeddingService and are used as fallbacks
when rule-based approaches don't find enough signal.
"""

import re
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService

# ── Section type descriptions (embedding anchors) ──────────────────────
# Each value describes what text in that section typically looks like.

JD_SECTION_DESCRIPTIONS: dict[str, str] = {
    "responsibilities": (
        "what you will do in this role, your duties, tasks, "
        "day-to-day work, and key responsibilities"
    ),
    "requirements": (
        "required skills, qualifications, experience, technical knowledge, "
        "must-have abilities, what we are looking for"
    ),
    "nice_to_have": (
        "bonus skills, preferred qualifications, nice to have, "
        "good to have, additional plus points"
    ),
    "about": (
        "about the company, team culture, our mission, who we are, "
        "company description and values"
    ),
}

RESUME_SECTION_DESCRIPTIONS: dict[str, str] = {
    "skills": (
        "technical skills, programming languages, frameworks, tools, "
        "technologies, databases, cloud platforms"
    ),
    "experience": (
        "work experience, professional history, job roles, employment, "
        "positions held, career history"
    ),
    "education": (
        "education background, degrees, university, college, academic "
        "qualifications, school"
    ),
    "projects": (
        "personal projects, side projects, open source contributions, "
        "portfolio projects, github projects"
    ),
    "certifications": (
        "certifications, licenses, professional certificates, "
        "aws certified, google certified"
    ),
}

# ── Thresholds ─────────────────────────────────────────────────────────

SECTION_SIMILARITY_THRESHOLD: float = 0.22
SKILL_DISCOVERY_THRESHOLD: float = 0.42

# Keyword hints that boost a section type's score when found in a paragraph
SECTION_KEYWORD_HINTS: dict[str, dict[str, list[str]]] = {
    "jd": {
        "nice_to_have": [
            "bonus", "nice to have", "good to have", "plus", "preferred",
            "even better", "not required", "it's a plus",
        ],
        "requirements": [
            "require", "must have", "qualification", "looking for",
            "need from you", "you have", "bring", "prerequisite",
        ],
        "responsibilities": [
            "you will", "you'll", "day to day", "day-to-day",
            "your role", "what you'll do", "duties", "responsible for",
        ],
    },
    "resume": {
        "skills": [
            "skill", "tech", "tool", "language", "framework",
            "proficient", "familiar with",
        ],
        "experience": [
            "experience", "worked", "employment", "career",
            "position", "role at",
        ],
        "education": [
            "education", "university", "college", "degree",
            "bachelor", "master", "phd", "school",
        ],
        "projects": [
            "project", "built", "created", "developed",
            "github", "personal project", "side project",
        ],
    },
}

KEYWORD_BOOST: float = 0.06


# ── Public helpers ─────────────────────────────────────────────────────


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by blank lines, skipping short ones."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 15]


def _classify_sections(
    text: str,
    embedding_service: "EmbeddingService",
    section_descriptions: dict[str, str],
    keyword_hints: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Classify paragraphs into section types using embedding similarity.

    Each paragraph is compared against section description embeddings.
    The best match above the threshold wins. Keyword hints provide a
    small boost to resolve ambiguous cases.

    Args:
        text: Full text to classify.
        embedding_service: An initialized EmbeddingService.
        section_descriptions: {section_type: description_text} mapping.
        keyword_hints: Optional {section_type: [keyword, ...]} for boosting.

    Returns:
        {section_type: body_text} mapping, plus a "preamble" key for
        unclassified paragraphs.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return {"preamble": text}

    section_types = list(section_descriptions.keys())
    section_descs = [section_descriptions[t] for t in section_types]

    # Pre-compute section-description embeddings once
    section_embs = embedding_service.encode(section_descs)
    section_embs = section_embs / np.linalg.norm(section_embs, axis=1, keepdims=True)

    # Embed all paragraphs in one batch
    para_embs = embedding_service.encode(paragraphs)
    para_embs = para_embs / np.linalg.norm(para_embs, axis=1, keepdims=True)

    # Similarity matrix: (n_paragraphs, n_section_types)
    sim_matrix = np.dot(para_embs, section_embs.T)

    sections: dict[str, str] = {}
    unclassified: list[str] = []

    for i, paragraph in enumerate(paragraphs):
        # Skip very short paragraphs (titles, one-liners)
        if len(paragraph.split()) < 5:
            unclassified.append(paragraph)
            continue

        # Base scores from embedding
        scores = sim_matrix[i].copy()

        # Apply keyword-hint boosts
        if keyword_hints:
            para_lower = paragraph.lower()
            for j, st in enumerate(section_types):
                for hint in keyword_hints.get(st, []):
                    if hint in para_lower:
                        scores[j] += KEYWORD_BOOST

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= SECTION_SIMILARITY_THRESHOLD:
            section_type = section_types[best_idx]
            body = _strip_header_line(paragraph)
            if section_type in sections:
                sections[section_type] += "\n" + body
            else:
                sections[section_type] = body
        else:
            unclassified.append(paragraph)

    if unclassified:
        sections["preamble"] = "\n".join(unclassified)

    return sections


def _discover_semantic_skills(
    text: str,
    existing_skills: list[str],
    embedding_service: "EmbeddingService",
    skill_vocabulary: set[str],
    skill_aliases: dict[str, str],
) -> list[str]:
    """Discover skills via embedding similarity not caught by keywords.

    Compares each sentence in *text* against the embeddings of every skill
    in *skill_vocabulary*. Returns skills whose similarity exceeds
    SKILL_DISCOVERY_THRESHOLD and are not already in *existing_skills*.

    Args:
        text: The text to search for skills.
        existing_skills: Skills already found (won't duplicate).
        embedding_service: An initialized EmbeddingService.
        skill_vocabulary: Set of skill names to check against.
        skill_aliases: Mapping from raw skill name → canonical form.

    Returns:
        List of newly discovered canonical skill names.
    """
    if not text.strip():
        return []

    # Split into candidate sentences (>= 5 words for meaningful signal)
    candidates = [
        s.strip()
        for s in re.split(r"[.\n;•●]", text)
        if len(s.split()) >= 5
    ]
    if not candidates:
        return []

    existing_set = set(existing_skills)
    skills_to_check = sorted(s for s in skill_vocabulary if s not in existing_set)
    if not skills_to_check:
        return []

    # Batch-embed
    cand_embs = embedding_service.encode(candidates)
    cand_embs = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True)

    skill_embs = embedding_service.encode(skills_to_check)
    skill_embs = skill_embs / np.linalg.norm(skill_embs, axis=1, keepdims=True)

    sim_matrix = np.dot(cand_embs, skill_embs.T)  # (n_candidates, n_skills)

    discovered: set[str] = set()
    for i in range(len(candidates)):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])
        if best_score >= SKILL_DISCOVERY_THRESHOLD:
            skill = skills_to_check[best_idx]
            canonical = skill_aliases.get(skill, skill)
            discovered.add(canonical)

    return sorted(discovered)


# ── Internal helper ────────────────────────────────────────────────────


def _strip_header_line(paragraph: str) -> str:
    """Remove the first line if it looks like a section header."""
    lines = paragraph.split("\n")
    if len(lines) < 2:
        return paragraph
    first = lines[0].strip()
    if first.endswith(":") and len(first.split()) <= 15:
        return "\n".join(lines[1:]).strip()
    return paragraph
