"""Tests for VectorMatcher — embedding-based semantic matching."""
import numpy as np

from app.services.vector_matcher import (
    VectorMatcher,
    _build_experience_text,
    DEFAULT_SKILL_THRESHOLD,
    DEFAULT_EXPERIENCE_THRESHOLD,
)


class FakeEmbeddingService:
    """Returns controlled embeddings for deterministic testing.

    Uses a simple strategy: each unique text gets an embedding based on
    the hash of its (lowercased, stripped) content.  Similar texts get
    embeddings that point in similar directions so semantic matching
    behaves predictably.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.encode_calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encode_calls.append(list(texts))
        rng = np.random.default_rng(42)  # fixed seed for reproducibility
        emb = rng.normal(size=(len(texts), self.dim)).astype(np.float64)
        # Normalize each row
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def similarity(self, text1: str, text2: str) -> float:
        emb = self.encode([text1, text2])
        return float(np.dot(emb[0], emb[1]))


class TestBuildExperienceText:
    """Tests for _build_experience_text helper."""

    def test_full_experience(self):
        class Exp:
            title = "Senior ML Engineer"
            company = "AcmeCorp"
            highlights = ["Built NLP pipeline", "Led team of 5"]

        result = _build_experience_text(Exp())
        assert "Senior ML Engineer" in result
        assert "AcmeCorp" in result
        assert "Built NLP pipeline" in result
        assert "Led team of 5" in result

    def test_no_company(self):
        class Exp:
            title = "Freelancer"
            company = ""
            highlights = []

        result = _build_experience_text(Exp())
        assert result == "Freelancer"

    def test_no_highlights(self):
        class Exp:
            title = "Developer"
            company = "Startup"
            highlights = []

        result = _build_experience_text(Exp())
        assert "Developer" in result
        assert "Startup" in result

    def test_missing_attributes(self):
        class Exp:
            title = "Consultant"

        result = _build_experience_text(Exp())
        assert result == "Consultant"


class TestVectorMatcher:
    """Tests for VectorMatcher semantic matching."""

    def test_match_skills_returns_above_threshold(self):
        """Semantic matches above the threshold should be returned."""
        vm = VectorMatcher(FakeEmbeddingService(), skill_threshold=0.0)
        matches = vm.match_skills(
            ["machine learning", "docker"],
            ["deep learning", "kubernetes", "python"],
        )
        # With threshold=0, all should find some match (argmax per row)
        assert len(matches) == 2
        # Each result should be a (jd_skill, resume_skill, similarity) tuple
        for m in matches:
            assert len(m) == 3
            assert isinstance(m[0], str)
            assert isinstance(m[1], str)
            assert isinstance(m[2], float)
            assert 0.0 <= m[2] <= 1.0

    def test_match_skills_drops_below_threshold(self):
        """Below-threshold pairs should be excluded."""
        vm = VectorMatcher(FakeEmbeddingService(), skill_threshold=0.99)
        matches = vm.match_skills(
            ["machine learning"],
            ["deep learning", "python"],
        )
        # With very high threshold, nothing should match
        assert matches == []

    def test_match_skills_skips_exact_matches(self):
        """JD skills with exact (case-insensitive) match should be skipped."""
        vm = VectorMatcher(FakeEmbeddingService(), skill_threshold=0.0)
        matches = vm.match_skills(
            ["Python", "Docker"],
            ["python", "docker", "react"],
        )
        # Both skills already have exact (case-insensitive) matches
        # so there are no unresolved skills to try
        assert matches == []

    def test_match_skills_empty_inputs(self):
        """Empty input lists should return immediately."""
        vm = VectorMatcher(FakeEmbeddingService())
        assert vm.match_skills([], ["python"]) == []
        assert vm.match_skills(["python"], []) == []
        assert vm.match_skills([], []) == []

    def test_match_skills_single_item(self):
        """Edge case: one JD skill vs one resume skill."""
        vm = VectorMatcher(FakeEmbeddingService(), skill_threshold=0.0)
        matches = vm.match_skills(["pytorch"], ["tensorflow"])
        assert len(matches) == 1
        assert matches[0][0] == "pytorch"
        assert matches[0][1] == "tensorflow"

    def test_match_skills_sorted_by_similarity(self):
        """Results should be sorted by similarity descending."""
        vm = VectorMatcher(FakeEmbeddingService(), skill_threshold=0.0)
        matches = vm.match_skills(
            ["go", "rust", "c++"],
            ["golang", "systems programming", "c"],
        )
        if len(matches) >= 2:
            for i in range(len(matches) - 1):
                assert matches[i][2] >= matches[i + 1][2]

    def test_semantic_skill_match_rate_full(self):
        """All JD skills matched → 1.0."""
        vm = VectorMatcher(FakeEmbeddingService(), skill_threshold=0.0)
        rate = vm.semantic_skill_match_rate(
            ["pytorch", "docker"],
            ["tensorflow", "kubernetes"],
        )
        assert rate == 1.0

    def test_semantic_skill_match_rate_empty(self):
        """Empty JD skills → 0.0."""
        vm = VectorMatcher(FakeEmbeddingService())
        rate = vm.semantic_skill_match_rate([], ["python"])
        assert rate == 0.0

    def test_match_experiences_basic(self):
        """JD responsibility matched to a resume experience."""
        vm = VectorMatcher(FakeEmbeddingService(), experience_threshold=0.0)

        class Exp:
            def __init__(self, title, company, highlights):
                self.title = title
                self.company = company
                self.highlights = highlights

        exps = [
            Exp("ML Engineer", "Acme", ["Built recommendation system"]),
            Exp("Frontend Dev", "WebCo", ["Built React dashboard"]),
        ]

        matches = vm.match_experiences_to_responsibilities(
            ["Build and deploy machine learning models"],
            exps,
        )
        assert len(matches) == 1
        assert "machine learning" in matches[0][0].lower()
        assert isinstance(matches[0][1], str)
        assert isinstance(matches[0][2], float)

    def test_match_experiences_empty_inputs(self):
        """Empty inputs should return empty list."""
        vm = VectorMatcher(FakeEmbeddingService())

        class Exp:
            title = "Dev"
            company = ""
            highlights = []

        assert vm.match_experiences_to_responsibilities([], [Exp()]) == []
        assert vm.match_experiences_to_responsibilities(["Build API"], []) == []

    def test_match_experiences_threshold_behavior(self):
        """High threshold should filter out all matches."""
        vm = VectorMatcher(FakeEmbeddingService(), experience_threshold=0.99)

        class Exp:
            title = "Dev"
            company = "Co"
            highlights = ["Coding"]

        matches = vm.match_experiences_to_responsibilities(
            ["Write production code"],
            [Exp()],
        )
        assert matches == []

    def test_custom_threshold_override(self):
        """Per-call threshold should override the instance default."""
        vm = VectorMatcher(FakeEmbeddingService(), skill_threshold=0.0)
        matches = vm.match_skills(
            ["pytorch"], ["tensorflow"], threshold=0.99
        )
        assert matches == []

    def test_default_thresholds_are_reasonable(self):
        """Default thresholds should be between 0 and 1."""
        assert 0.0 < DEFAULT_SKILL_THRESHOLD < 1.0
        assert 0.0 < DEFAULT_EXPERIENCE_THRESHOLD < 1.0
        assert DEFAULT_SKILL_THRESHOLD > DEFAULT_EXPERIENCE_THRESHOLD
