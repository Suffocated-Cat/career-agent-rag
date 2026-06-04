"""
JDParser — Rule-based + embedding-enhanced job description parser.

Extracts structured fields from raw JD text:
  - title (job title)
  - company (company name, if present)
  - skills (required technical/soft skills)
  - responsibilities (key duties)
  - nice_to_haves (bonus / preferred qualifications)

Pipeline:
  1. Regex-based section splitting (primary)
  2. Embedding-based section classification (fallback)
  3. Keyword vocabulary skill matching (primary)
  4. Embedding-based semantic skill discovery (fallback)
  5. Bullet-item extraction for responsibilities / nice-to-haves

Replaced by LLM-based parsing in Week 3.
"""

import re
from typing import TYPE_CHECKING

from app.models.jd import JobDescription

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService

# ── Tech skills vocabulary ──────────────────────────────────────────────
# Covers: languages, frameworks, platforms, tools, concepts
TECH_SKILLS: set[str] = {
    # Languages
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "c++", "c#", "c", "ruby", "scala", "kotlin", "swift", "php", "perl",
    "r", "matlab", "sql", "bash", "shell",
    # Frontend
    "react", "react.js", "reactjs", "vue", "vue.js", "vuejs", "angular",
    "angular.js", "angularjs", "next.js", "nextjs", "nuxt", "svelte",
    "html", "css", "html5", "css3", "sass", "scss", "less", "tailwind",
    "bootstrap", "jquery", "webpack", "vite", "babel", "redux", "mobx",
    # Backend / Frameworks
    "django", "flask", "fastapi", "spring", "spring boot", "express",
    "express.js", "expressjs", "nest.js", "nestjs", "gin", "rails",
    "ruby on rails", "laravel", ".net", "asp.net", "node.js", "nodejs",
    "node", "graphql", "rest", "restful", "grpc", "websocket",
    # ML / Data Science
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "pytorch", "tensorflow", "keras", "jax", "scikit-learn",
    "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "plotly", "xgboost", "lightgbm", "catboost", "mlflow", "kubeflow",
    "mlops", "hugging face", "huggingface", "transformers", "bert", "gpt",
    "llm", "large language model", "rag", "retrieval augmented generation",
    "langchain", "llamaindex", "vector database", "embedding", "fine-tuning",
    # Data Engineering
    "spark", "apache spark", "hadoop", "kafka", "flink", "airflow",
    "dbt", "snowflake", "databricks", "bigquery", "redshift", "etl",
    "data pipeline", "data warehouse", "data lake",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "circleci", "ci/cd", "ci/cd pipeline", "prometheus", "grafana",
    "elk", "elasticsearch", "logstash", "kibana", "datadog", "nginx",
    "linux", "unix", "helm", "istio", "service mesh",
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "cassandra",
    "dynamodb", "neo4j", "graph database", "rabbitmq", "celery",
    # Soft / General
    "agile", "scrum", "kanban", "git", "github", "gitlab", "jira",
    "confluence", "slack", "communication", "teamwork", "leadership",
    "problem solving", "critical thinking", "system design",
}

# Aliases that normalize to the canonical form
SKILL_ALIASES: dict[str, str] = {
    "golang": "go",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "angularjs": "angular",
    "angular.js": "angular",
    "nextjs": "next.js",
    "nest.js": "nestjs",
    "expressjs": "express.js",
    "express": "express.js",
    "k8s": "kubernetes",
    "sklearn": "scikit-learn",
    "huggingface": "hugging face",
    "large language model": "llm",
    "retrieval augmented generation": "rag",
    "ruby on rails": "rails",
}

# Section header patterns — case-insensitive
SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "responsibilities",
        re.compile(
            r"(?:^|\n)\s*(?:key\s+)?(?:responsibilit(?:y|ies)|what\s+you(?:\'ll|\s+will)\s+do"
            r"|your\s+role|the\s+role|duties?|what\s+you\'d\s+do|about\s+the\s+role)"
            r"\s*:?\s*\n",
            re.IGNORECASE,
        ),
    ),
    (
        "requirements",
        re.compile(
            r"(?:^|\n)\s*(?:requirements?|qualifications?|what\s+you(?:\'ll|\s+will)\s+bring"
            r"|what\s+we(?:\'re|\s+are)\s+looking\s+for|who\s+you\s+are"
            r"|you\s+(?:should\s+)?have|skills?\s*(?:&|and)\s+qualifications?"
            r"|must\s+have|required\s+(?:skills?|experience|qualifications?))"
            r"\s*:?\s*\n",
            re.IGNORECASE,
        ),
    ),
    (
        "nice_to_have",
        re.compile(
            r"(?:^|\n)\s*(?:nice[-\s]to[-\s]haves?|bonus\s+(?:points?|skills?)"
            r"|preferred\s+(?:qualifications?|skills?|experience)"
            r"|good[-\s]to[-\s]haves?|plus\s+(?:points?|skills?)"
            r"|it(?:\'s|\s+is)\s+a\s+plus|even\s+better\s+if"
            r"|not\s+required\s+but)"
            r"\s*:?\s*\n",
            re.IGNORECASE,
        ),
    ),
    (
        "about",
        re.compile(
            r"(?:^|\n)\s*(?:about\s+(?:us|the\s+company|the\s+team|the\s+role)"
            r"|company\s+description|who\s+we\s+are|our\s+mission)"
            r"\s*:?\s*\n",
            re.IGNORECASE,
        ),
    ),
]

# Title patterns — lines that look like job titles
TITLE_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(?:senior\s+|junior\s+|lead\s+|principal\s+|staff\s+|associate\s+)?"
        r"(?:software\s+|machine\s+learning\s+|data\s+|devops\s+|cloud\s+|"
        r"frontend\s+|backend\s+|full\s*stack\s+|platform\s+|security\s+|"
        r"ai\s+|nlp\s+|research\s+)?"
        r"engineer|scientist|developer|architect|manager|designer|analyst|"
        r"director|consultant|specialist",
        re.IGNORECASE,
    ),
]


def _extract_title(text: str) -> str | None:
    """Extract job title from the first few lines or a title-like line."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    # Check the first 10 non-empty lines for a title match
    for line in lines[:10]:
        # Skip lines that look like section headers or company descriptions
        if re.match(r"^(about|at\s|we\s|our\s|in\s|over\s|$)", line, re.IGNORECASE):
            continue
        for pattern in TITLE_PATTERNS:
            if pattern.search(line):
                return line.strip().rstrip(",.|")

    # Fallback: return the first short line that looks like a heading
    for line in lines[:5]:
        if 5 < len(line) < 100 and not line.startswith(("http", "#", "-", "*")):
            return line.strip().rstrip(",.|")

    return None


def _extract_company(text: str) -> str | None:
    """Try to extract company name from common patterns."""
    # "at [Company]" or "Company is looking for"
    m = re.search(r"\bat\s+([A-Z][A-Za-z0-9\s&.]+?)(?:\s*(?:is|are|—|–|-|,|\.|\n|$))", text)
    if m:
        name = m.group(1).strip()
        if 2 < len(name) < 60:
            return name

    # "[Company] is (hiring|looking)"
    m = re.match(
        r"^([A-Z][A-Za-z0-9\s&.]+?)\s+(?:is|are)\s+(?:hiring|looking|seeking|a\s)", text
    )
    if m:
        name = m.group(1).strip()
        if 2 < len(name) < 60:
            return name

    return None


def _split_sections(text: str) -> dict[str, str]:
    """Split JD text into named sections based on header patterns.

    Searches the full text for section headers, then partitions the text
    into named bodies.  Unmatched text goes into the 'preamble' key.
    """
    # Find every section-header match across the full text.
    # Each entry: (start_position, end_position_of_header, section_type)
    markers: list[tuple[int, int, str]] = []
    for section_type, pattern in SECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            markers.append((m.start(), m.end(), section_type))

    # Sort by position in the text
    markers.sort(key=lambda x: x[0])

    if not markers:
        return {"preamble": text}

    sections: dict[str, str] = {"preamble": text[: markers[0][0]].strip()}

    for i, (hdr_start, hdr_end, section_type) in enumerate(markers):
        # Body starts right after the header line
        body_start = hdr_end
        body_end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        body = text[body_start:body_end].strip()

        # Merge if the same section type appears more than once
        if section_type in sections:
            sections[section_type] += "\n" + body
        else:
            sections[section_type] = body

    return sections


def _extract_bullet_items(text: str) -> list[str]:
    """Extract items from bullet-point or numbered list text."""
    items: list[str] = []
    current: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                items.append(" ".join(current))
                current = []
            continue

        # Detect bullet markers: -, *, •, 1., 2., etc.
        is_bullet = bool(re.match(r"^[-*•▪▸►]\s+", stripped))
        is_numbered = bool(re.match(r"^\d+[.)]\s+", stripped))

        if is_bullet or is_numbered:
            if current:
                items.append(" ".join(current))
                current = []
            # Remove the marker
            content = re.sub(r"^[-*•▪▸►]\s+", "", stripped)
            content = re.sub(r"^\d+[.)]\s+", "", content)
            current.append(content)
        else:
            # Continuation or non-bullet line — treat standalone lines as items too
            if current:
                current.append(stripped)
            elif len(stripped) > 10:
                items.append(stripped)

    if current:
        items.append(" ".join(current))

    return [item.strip() for item in items if item.strip()]


def _match_skills(text: str) -> list[str]:
    """Extract known tech skills from text using vocabulary matching.

    Single-word skills use word-boundary matching (\b) to avoid false
    positives like "c" matching inside "docker". Multi-word phrases use
    plain substring matching. Skills with special characters (c++, c#)
    use a hybrid approach: word-boundary on the left, whitespace/punctuation
    on the right.

    Returns a deduplicated, canonically-named list.
    """
    text_lower = text.lower()
    found: set[str] = set()

    for skill in TECH_SKILLS:
        if " " in skill:
            matched = skill in text_lower
        elif all(c.isalnum() for c in skill):
            # Pure alphanumeric word — safe for \b
            matched = bool(re.search(r"\b" + re.escape(skill) + r"\b", text_lower))
        else:
            # Contains special chars (c++, c#, .net) — word boundary on left,
            # then literal text, then must be followed by space/punct/end
            matched = bool(
                re.search(
                    r"\b" + re.escape(skill) + r"(?=\s|$|[.,;:!?\)\]}])",
                    text_lower,
                )
            )

        if matched:
            canonical = SKILL_ALIASES.get(skill, skill)
            found.add(canonical)

    return sorted(found)


def _extract_skills_from_sections(
    requirements_text: str, preamble_text: str, nice_to_have_text: str = "",
) -> list[str]:
    """Combine skill matches from all relevant sections."""
    combined = f"{requirements_text}\n{preamble_text}\n{nice_to_have_text}"
    return _match_skills(combined)


# ── Public API ───────────────────────────────────────────────────────────


def parse_jd(
    raw_text: str,
    embedding_service: "EmbeddingService | None" = None,
) -> JobDescription:
    """Parse a raw job description text into a structured JobDescription.

    Args:
        raw_text: The full job description as a plain string.
        embedding_service: Optional EmbeddingService for semantic fallback.
            When provided:
            - Section classification falls back to embedding similarity
              if regex patterns find no sections.
            - Skill extraction adds semantic discovery for sentences
              that don't match any vocabulary keywords.

    Returns:
        A JobDescription with extracted fields populated.
    """
    title = _extract_title(raw_text)
    company = _extract_company(raw_text)
    sections = _split_sections(raw_text)

    # ── Embedding fallback: section classification ─────────────────
    section_types_found = [k for k in sections if k != "preamble"]
    if not section_types_found and embedding_service is not None:
        from app.services._embedding_helpers import (
            _classify_sections,
            JD_SECTION_DESCRIPTIONS,
            SECTION_KEYWORD_HINTS,
        )

        sections = _classify_sections(
            raw_text,
            embedding_service,
            JD_SECTION_DESCRIPTIONS,
            SECTION_KEYWORD_HINTS.get("jd"),
        )

    requirements_text = sections.get("requirements", "")
    responsibilities_text = sections.get("responsibilities", "")
    nice_to_have_text = sections.get("nice_to_have", "")
    preamble = sections.get("preamble", "")

    # ── Keyword skill matching ─────────────────────────────────────
    skills = _extract_skills_from_sections(requirements_text, preamble, nice_to_have_text)

    # ── Embedding fallback: semantic skill discovery ────────────────
    if embedding_service is not None:
        from app.services._embedding_helpers import _discover_semantic_skills

        all_text = f"{requirements_text}\n{preamble}\n{responsibilities_text}\n{nice_to_have_text}"
        discovered = _discover_semantic_skills(
            all_text, skills, embedding_service, TECH_SKILLS, SKILL_ALIASES
        )
        skills = sorted(set(skills) | set(discovered))

    responsibilities = _extract_bullet_items(responsibilities_text)
    nice_to_haves = _extract_bullet_items(nice_to_have_text)

    return JobDescription(
        raw_text=raw_text,
        title=title,
        company=company,
        skills=skills,
        responsibilities=responsibilities,
        nice_to_haves=nice_to_haves,
    )
