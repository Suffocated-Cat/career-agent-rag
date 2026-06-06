"""Tests for shared embedding parser helpers."""

import numpy as np

from app.services._embedding_helpers import (
    _classify_sections,
    _discover_semantic_skills,
    _split_paragraphs,
    _strip_header_line,
)


class KeywordEmbeddingService:
    """Small deterministic embedding service for parser helper tests."""

    FEATURES = [
        "python",
        "docker",
        "node",
        "required",
        "skills",
        "duties",
        "tasks",
        "company",
        "culture",
    ]

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lower = text.lower()
            rows.append(
                [
                    1.0 if feature in lower else 0.0
                    for feature in self.FEATURES
                ]
            )
        return np.array(rows, dtype=np.float64)


class TestSplitParagraphs:
    def test_splits_on_blank_lines_and_skips_short_paragraphs(self):
        text = "Tiny\n\nThis paragraph is long enough.\n\nAnother useful paragraph."

        assert _split_paragraphs(text) == [
            "This paragraph is long enough.",
            "Another useful paragraph.",
        ]


class TestClassifySections:
    def test_returns_preamble_when_no_paragraphs_are_available(self):
        assert _classify_sections("", KeywordEmbeddingService(), {}) == {
            "preamble": ""
        }

    def test_classifies_multiple_paragraphs_by_embedding_similarity(self):
        text = (
            "Candidates must show required skills in Python and Docker.\n\n"
            "Your duties include production tasks and service ownership.\n\n"
            "Our company culture values careful collaboration."
        )
        sections = _classify_sections(
            text,
            KeywordEmbeddingService(),
            {
                "requirements": "required skills qualifications",
                "responsibilities": "duties tasks day to day work",
                "about": "company culture mission values",
            },
        )

        assert "Python and Docker" in sections["requirements"]
        assert "production tasks" in sections["responsibilities"]
        assert "company culture" in sections["about"]

    def test_keyword_hints_resolve_ambiguous_paragraph(self):
        text = (
            "You must have required duties and strong production ownership habits."
        )

        sections = _classify_sections(
            text,
            KeywordEmbeddingService(),
            {
                "responsibilities": "duties tasks day to day work",
                "requirements": "required skills qualifications",
            },
            keyword_hints={"requirements": ["must have"]},
        )

        assert sections["requirements"] == text

    def test_short_classified_paragraph_is_kept_as_preamble(self):
        sections = _classify_sections(
            "Skills: Python",
            KeywordEmbeddingService(),
            {"requirements": "skills python"},
        )

        assert sections == {"preamble": "Skills: Python"}

    def test_strip_header_line_removes_short_colon_header(self):
        paragraph = "Requirements:\nPython and Docker experience required."

        assert _strip_header_line(paragraph) == (
            "Python and Docker experience required."
        )


class TestDiscoverSemanticSkills:
    def test_discovers_multiple_skills_from_one_sentence(self):
        discovered = _discover_semantic_skills(
            "Built reliable platform services with Python and Docker for teams.",
            [],
            KeywordEmbeddingService(),
            {"python", "docker", "nodejs"},
            {"nodejs": "node.js"},
        )

        assert discovered == ["docker", "python"]

    def test_excludes_existing_skills_by_canonical_alias(self):
        discovered = _discover_semantic_skills(
            "Built reliable platform services with Node for internal teams.",
            ["node.js"],
            KeywordEmbeddingService(),
            {"node", "nodejs"},
            {"node": "node.js", "nodejs": "node.js"},
        )

        assert discovered == []

    def test_returns_empty_for_empty_or_too_short_text(self):
        service = KeywordEmbeddingService()

        assert _discover_semantic_skills("", [], service, {"python"}, {}) == []
        assert _discover_semantic_skills("Python Docker", [], service, {"python"}, {}) == []

    def test_returns_empty_when_all_skills_already_exist(self):
        discovered = _discover_semantic_skills(
            "Built reliable platform services with Python and Docker.",
            ["python", "docker"],
            KeywordEmbeddingService(),
            {"python", "docker"},
            {},
        )

        assert discovered == []
