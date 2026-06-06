"""
ResumeParser — Rule-based + embedding-enhanced resume parser.

Extracts structured fields from raw resume text:
  - skills (technical skills matched against vocabulary)
  - projects (name, description, technologies)
  - education (degree, institution, year)
  - experience (title, company, duration, highlights)

Pipeline:
  1. Regex-based section splitting (primary)
  2. Embedding-based section classification (fallback)
  3. Keyword vocabulary skill matching (primary)
  4. Embedding-based semantic skill discovery (fallback)
  5. Rule-based entity extraction (experience, education, projects)

When an LLM client is supplied, ``parse_resume`` instead extracts the fields
with the LLM (schema-validated), falling back to this rule-based pipeline on
any failure or when the LLM is not configured.
"""

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from app.models.resume import (
    Resume,
    ResumeProject,
    ResumeEducation,
    ResumeExperience,
)
from app.services.jd_parser import TECH_SKILLS, SKILL_ALIASES, _match_skills
from app.services.llm_support import generate_model

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService
    from app.services.llm_client import LLMClient


class _ResumeExtraction(BaseModel):
    """Fields the LLM extracts from a resume (raw_text is supplied separately)."""

    skills: list[str] = Field(default_factory=list)
    experience: list[ResumeExperience] = Field(default_factory=list)
    education: list[ResumeEducation] = Field(default_factory=list)
    projects: list[ResumeProject] = Field(default_factory=list)


_RESUME_EXTRACT_SYSTEM = (
    "You extract structured fields from a resume. Return strict JSON with keys: "
    "skills (array of short strings), experience (array of objects with title, "
    "company, duration, highlights[]), education (array of objects with degree, "
    "institution, year), projects (array of objects with name, description, "
    "technologies[]). Do not invent information not present in the text."
)


def _resume_extract_prompt(raw_text: str) -> str:
    """Prompt asking the LLM to extract resume fields as JSON."""
    return f"Extract the resume fields as JSON.\n\nResume:\n{raw_text}"

# ── Resume section header patterns ─────────────────────────────────────

SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "experience",
        re.compile(
            r"(?:^|\n)\s*(?:work\s+)?experience\b|"
            r"(?:professional|employment)\s+(?:experience|history|background)|"
            r"(?:work|career)\s+history",
            re.IGNORECASE,
        ),
    ),
    (
        "education",
        re.compile(
            r"(?:^|\n)\s*education\b|"
            r"academic\s+(?:background|history|qualification)",
            re.IGNORECASE,
        ),
    ),
    (
        "skills",
        re.compile(
            r"(?:^|\n)\s*(?:technical\s+)?skills?\b|"
            r"(?:core\s+)?competenc(?:y|ies)|"
            r"technolog(?:y|ies)(?:\s+(?:stack|used))?|"
            r"tools?\s*(?:&|and)\s*technolog(?:y|ies)",
            re.IGNORECASE,
        ),
    ),
    (
        "projects",
        re.compile(
            r"(?:^|\n)\s*(?:personal\s+)?projects?\b|"
            r"(?:key|notable|selected)\s+projects?\b|"
            r"project\s+(?:experience|portfolio)",
            re.IGNORECASE,
        ),
    ),
    (
        "certifications",
        re.compile(
            r"(?:^|\n)\s*certification(?:s)?\b|"
            r"license(?:s)?\b|"
            r"professional\s+(?:certification|development)",
            re.IGNORECASE,
        ),
    ),
]


def _split_sections(text: str) -> dict[str, str]:
    """Split resume text into named sections based on header patterns."""
    markers: list[tuple[int, int, str]] = []
    for section_type, pattern in SECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            markers.append((m.start(), m.end(), section_type))

    markers.sort(key=lambda x: x[0])

    if not markers:
        return {"preamble": text}

    sections: dict[str, str] = {"preamble": text[: markers[0][0]].strip()}

    for i, (hdr_start, hdr_end, section_type) in enumerate(markers):
        body_start = hdr_end
        body_end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        body = text[body_start:body_end].strip()
        if section_type in sections:
            sections[section_type] += "\n" + body
        else:
            sections[section_type] = body

    return sections


# ── Experience parsing ─────────────────────────────────────────────────

# Pattern for job entry header lines:
#   "Senior ML Engineer, AcmeCorp | Jan 2022 – Present"
#   "Software Developer at Google (2020–2023)"
#   "Data Scientist | StartupX"
_EXPERIENCE_HEADER = re.compile(
    r"^(.+?)\s+(?:at|,|\||—|–|-)\s+(.+?)(?:\s*(?:,|\||—|–|-)\s*(.+))?$"
)

# Pattern to extract duration: "Jan 2022 – Present", "2020–2023", "2020 - 2023"
_DURATION_PATTERN = re.compile(
    r"(\w{3,9}\s+\d{4}|"
    r"\d{4})\s*[–—-]\s*"
    r"(\w{3,9}\s+\d{4}|\d{4}|"
    r"[Pp]resent|[Cc]urrent)",
)


def _parse_experience_entries(section_text: str) -> list[ResumeExperience]:
    """Parse work experience entries from the experience section."""
    if not section_text.strip():
        return []

    # Split into entries by blank lines or major separators
    entries = _split_entries(section_text)
    results: list[ResumeExperience] = []

    for entry_text in entries:
        exp = _parse_single_experience(entry_text)
        if exp and exp.title:
            results.append(exp)

    return results


def _parse_single_experience(text: str) -> ResumeExperience | None:
    """Parse a single experience entry."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return None

    first_line = lines[0]

    title = ""
    company = ""
    duration = None
    highlights: list[str] = []

    # ── Step 1: Split off duration if present ─────────────────────
    dur_match = _DURATION_PATTERN.search(first_line)
    if dur_match:
        duration = dur_match.group(0).strip()
        first_line = _DURATION_PATTERN.sub("", first_line).strip().rstrip("|, -–")

    # ── Step 2: Split title from company ──────────────────────────
    # Common patterns:
    #   "Senior ML Engineer, AcmeCorp"
    #   "Software Developer at Google"
    #   "Data Scientist | StartupX"

    # Try " at " first (most reliable separator)
    at_match = re.search(r"\s+at\s+", first_line, re.IGNORECASE)
    if at_match:
        title = first_line[: at_match.start()].strip()
        company = first_line[at_match.end() :].strip()
    else:
        # Try " | " / " – " / " — " (whitespace around) or ", " (no leading ws needed)
        sep_match = re.search(r"\s+[|–—]\s+|,\s+", first_line)
        if sep_match:
            title = first_line[: sep_match.start()].strip()
            company = first_line[sep_match.end() :].strip()
        else:
            title = first_line.rstrip(",|")

    # If company still looks like a duration, reset it
    if company and _DURATION_PATTERN.search(company):
        company = ""

    # ── Step 3: Extract highlights from remaining lines ───────────
    for line in lines[1:]:
        cleaned = re.sub(r"^[-*•▪▸►]\s*", "", line.strip())
        if len(cleaned) > 5:
            highlights.append(cleaned)

    if not title:
        return None

    return ResumeExperience(
        title=title,
        company=company if company else "",
        duration=duration,
        highlights=highlights,
    )


# ── Education parsing ──────────────────────────────────────────────────

# Pattern: "M.S. Computer Science, Stanford University, 2020"
_EDUCATION_ENTRY = re.compile(
    r"^(.+?)\s+(?:,\s*|\s+at\s+|\s+–\s+|\s+-\s+)(.+?)(?:\s*,\s*(.+))?$"
)


def _parse_education_entries(section_text: str) -> list[ResumeEducation]:
    """Parse education entries from the education section."""
    if not section_text.strip():
        return []

    entries = _split_entries(section_text)
    results: list[ResumeEducation] = []

    for entry_text in entries:
        edu = _parse_single_education(entry_text)
        if edu and edu.degree:
            results.append(edu)

    return results


def _parse_single_education(text: str) -> ResumeEducation | None:
    """Parse a single education entry."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return None

    first_line = lines[0]

    degree = ""
    institution = ""
    year = None

    # Try comma-separated: "Degree, Institution, Year"
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) >= 2:
        degree = parts[0]
        institution = parts[1]
        if len(parts) >= 3:
            # Check if the last part looks like a year
            year_match = re.search(r"\b(19|20)\d{2}\b", parts[-1])
            if year_match:
                year = year_match.group(0)
            elif len(parts[-1]) < 20:
                year = parts[-1]
    else:
        degree = first_line
        if len(lines) > 1:
            institution = lines[1]

    if not degree:
        return None

    return ResumeEducation(degree=degree, institution=institution, year=year)


# ── Project parsing ────────────────────────────────────────────────────

def _parse_project_entries(section_text: str) -> list[ResumeProject]:
    """Parse project entries from the projects section."""
    if not section_text.strip():
        return []

    entries = _split_entries(section_text)
    results: list[ResumeProject] = []

    for entry_text in entries:
        proj = _parse_single_project(entry_text)
        if proj and proj.name:
            results.append(proj)

    return results


def _parse_single_project(text: str) -> ResumeProject | None:
    """Parse a single project entry."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return None

    # First line is the project name (may also contain tags like "| Python, React")
    first_line = lines[0]
    name = first_line
    raw_technologies = ""

    # Check for technology list after separator in first line
    tech_sep = re.search(r"\s*[|/]\s*(.+)", first_line)
    if tech_sep:
        name = first_line[: tech_sep.start()].strip()
        raw_technologies = tech_sep.group(1).strip()

    # Remaining lines form the description
    description_lines = []
    for line in lines[1:]:
        # Check for a dedicated "Technologies:" or "Tech stack:" line
        tech_match = re.match(
            r"(?:technolog(?:y|ies)|tech\s*stack|tools?)\s*:?\s*(.+)",
            line,
            re.IGNORECASE,
        )
        if tech_match:
            raw_technologies = raw_technologies + ", " + tech_match.group(1)
        else:
            cleaned = re.sub(r"^[-*•▪▸►]\s*", "", line)
            if len(cleaned) > 3:
                description_lines.append(cleaned)

    description = " ".join(description_lines)

    # Extract technologies from the raw text
    technologies = _match_skills(raw_technologies) if raw_technologies else []
    # Also try to find tech in description
    desc_techs = _match_skills(description)
    all_techs = sorted(set(technologies) | set(desc_techs))

    return ResumeProject(name=name, description=description, technologies=all_techs)


# ── Helpers ────────────────────────────────────────────────────────────


def _split_entries(text: str) -> list[str]:
    """Split section text into individual entries.

    Splits on blank lines, and also splits within a block when a line
    looks like the start of a new entry (contains a duration pattern
    or title-company separator).
    """
    # First, split on blank lines
    raw = re.split(r"\n\s*\n", text.strip())
    entries: list[str] = []

    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Check if this chunk contains multiple entries on separate lines
        lines = chunk.split("\n")
        sub_entries: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect a new entry start: line contains a duration or title separator
            is_new_entry = False
            if current and not stripped.startswith("-"):
                has_duration = bool(_DURATION_PATTERN.search(stripped))
                has_separator = bool(
                    re.search(r"\s+at\s+|\s+[|–—]\s+|,\s+", stripped)
                )
                is_new_entry = has_duration or has_separator

            if is_new_entry:
                sub_entries.append(current)
                current = [stripped]
            else:
                current.append(stripped)

        if current:
            sub_entries.append(current)

        for sub in sub_entries:
            entries.append("\n".join(sub))

    return entries


# ── Public API ─────────────────────────────────────────────────────────


def parse_resume(
    raw_text: str,
    embedding_service: "EmbeddingService | None" = None,
    llm: "LLMClient | None" = None,
) -> Resume:
    """Parse a raw resume text into a structured Resume.

    Args:
        raw_text: The full resume as a plain text string.
        embedding_service: Optional EmbeddingService for semantic fallback.
            When provided:
            - Section classification falls back to embedding similarity
              if regex patterns find no sections.
            - Skill extraction adds semantic discovery for sentences
              that don't match any vocabulary keywords.
        llm: Optional LLM client. When provided and configured, fields are
            extracted by the LLM (validated against the schema, with the
            rule-based parse as the fallback) — useful for messy resumes.

    Returns:
        A Resume with extracted fields populated.
    """
    sections = _split_sections(raw_text)

    # ── Embedding fallback: section classification ─────────────────
    section_types_found = [k for k in sections if k != "preamble"]
    if not section_types_found and embedding_service is not None:
        from app.services._embedding_helpers import (
            _classify_sections,
            RESUME_SECTION_DESCRIPTIONS,
            SECTION_KEYWORD_HINTS,
        )

        sections = _classify_sections(
            raw_text,
            embedding_service,
            RESUME_SECTION_DESCRIPTIONS,
            SECTION_KEYWORD_HINTS.get("resume"),
        )

    skills_text = sections.get("skills", "")
    experience_text = sections.get("experience", "")
    education_text = sections.get("education", "")
    projects_text = sections.get("projects", "")
    preamble = sections.get("preamble", "")

    # ── Keyword skill matching ─────────────────────────────────────
    all_skill_text = f"{skills_text}\n{preamble}"
    skills = _match_skills(all_skill_text)

    # ── Embedding fallback: semantic skill discovery ────────────────
    if embedding_service is not None:
        from app.services._embedding_helpers import _discover_semantic_skills

        all_text = f"{skills_text}\n{preamble}\n{experience_text}\n{projects_text}"
        discovered = _discover_semantic_skills(
            all_text, skills, embedding_service, TECH_SKILLS, SKILL_ALIASES
        )
        skills = sorted(set(skills) | set(discovered))

    # Parse structured sections
    experience = _parse_experience_entries(experience_text)
    education = _parse_education_entries(education_text)
    projects = _parse_project_entries(projects_text)

    rule_result = Resume(
        raw_text=raw_text,
        skills=skills,
        projects=projects,
        education=education,
        experience=experience,
    )

    if llm is None:
        return rule_result

    # LLM extraction, falling back to the rule-based result on any failure.
    fallback = _ResumeExtraction(
        skills=rule_result.skills,
        experience=rule_result.experience,
        education=rule_result.education,
        projects=rule_result.projects,
    )
    extraction = generate_model(
        llm,
        _resume_extract_prompt(raw_text),
        _ResumeExtraction,
        fallback,
        system=_RESUME_EXTRACT_SYSTEM,
    )
    return Resume(
        raw_text=raw_text,
        # Normalize skills to lowercase to match the rule parser's convention.
        skills=[s.lower() for s in extraction.skills],
        projects=extraction.projects,
        education=extraction.education,
        experience=extraction.experience,
    )
