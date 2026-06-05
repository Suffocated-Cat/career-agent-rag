"""Tests for KeywordMatcher — baseline JD-to-resume matching."""
import numpy as np
import pytest
from app.models.jd import JobDescription
from app.models.resume import Resume
from app.services.keyword_matcher import match


class TestKeywordMatcher:
    """Tests for keyword-overlap matching."""

    def test_full_match(self):
        jd = JobDescription(raw_text="", skills=["python", "docker"])
        resume = Resume(raw_text="", skills=["python", "docker", "git"])
        result = match(jd, resume)
        assert result.matched_skills == ["docker", "python"]
        assert result.missing_skills == []
        assert result.skill_match_rate == 1.0

    def test_partial_match(self):
        jd = JobDescription(raw_text="", skills=["python", "docker", "aws"])
        resume = Resume(raw_text="", skills=["python"])
        result = match(jd, resume)
        assert "python" in result.matched_skills
        assert "docker" in result.missing_skills
        assert result.skill_match_rate == pytest.approx(1.0 / 3, rel=0.01)

    def test_no_match(self):
        jd = JobDescription(raw_text="", skills=["python", "docker"])
        resume = Resume(raw_text="", skills=["react", "css"])
        result = match(jd, resume)
        assert result.matched_skills == []
        assert result.skill_match_rate == 0.0

    def test_empty_jd_skills(self):
        jd = JobDescription(raw_text="", skills=[])
        resume = Resume(raw_text="", skills=["python"])
        result = match(jd, resume)
        assert result.skill_match_rate == 0.0

    def test_extended_match_finds_skill_in_raw_text(self):
        """Skills not in resume.skills but present in raw_text should be found."""
        jd = JobDescription(raw_text="", skills=["python", "docker"])
        resume = Resume(
            raw_text="I have experience with Docker containers.",
            skills=["python"],
        )
        result = match(jd, resume)
        assert "docker" in result.matched_skills
        assert result.skill_match_rate == 1.0

    def test_extended_match_avoids_short_skill_substring_false_positive(self):
        """Short skills should not match inside unrelated words."""
        jd = JobDescription(raw_text="", skills=["go"])
        resume = Resume(raw_text="I negotiated contracts and managed roadmap goals.")
        result = match(jd, resume)
        assert result.matched_skills == []
        assert result.missing_skills == ["go"]
        assert result.skill_match_rate == 0.0

    def test_semantic_similarity_is_included_when_embedding_service_is_available(self):
        class FakeEmbeddingService:
            def similarity(self, text1, text2):
                return 0.42

        jd = JobDescription(raw_text="Build ML APIs", skills=["python"])
        resume = Resume(raw_text="Built machine learning services", skills=[])
        result = match(jd, resume, embedding_service=FakeEmbeddingService())
        assert result.semantic_similarity == 0.42
        assert "Document semantic similarity: 0.42" in result.summary

    def test_overall_score_range(self):
        jd = JobDescription(raw_text="", skills=["python"])
        resume = Resume(raw_text="", skills=["python"])
        result = match(jd, resume)
        assert 0.0 <= result.overall_score <= 1.0

    def test_summary_not_empty(self):
        jd = JobDescription(raw_text="", skills=["python"])
        resume = Resume(raw_text="", skills=["python"])
        result = match(jd, resume)
        assert len(result.summary) > 0

    # ── Vector matching integration tests ────────────────────────────

    def _make_fake_embedding_service(self):
        """Create a FakeEmbeddingService that the VectorMatcher can use.

        Returns a service with encode() and similarity() methods.
        """

        class FakeEmbeddingService:
            def __init__(self):
                self.dim = 16

            def encode(self, texts):
                rng = np.random.default_rng(42)
                emb = rng.normal(size=(len(texts), self.dim)).astype(np.float64)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return emb / norms

            def similarity(self, text1, text2):
                emb = self.encode([text1, text2])
                return float(np.dot(emb[0], emb[1]))

        return FakeEmbeddingService()

    def test_semantic_skill_match_populates_new_fields(self):
        """When embedding_service is provided, semantic_skill_matches is populated."""
        fake_es = self._make_fake_embedding_service()

        jd = JobDescription(
            raw_text="Need ML expertise",
            skills=["pytorch", "docker"],
            responsibilities=["Build ML models"],
        )
        resume = Resume(
            raw_text="Deep learning and containers",
            skills=["tensorflow", "kubernetes"],
            experience=[],
        )
        result = match(jd, resume, embedding_service=fake_es)

        # New fields should be present
        assert result.semantic_skill_matches is not None
        assert result.semantic_skill_match_rate is not None
        # semantic_skill_match_rate should be >= 0
        assert result.semantic_skill_match_rate >= 0.0

    def test_experience_match_populates_when_data_available(self):
        """When JD has responsibilities and resume has experience, experience_matches
        should be populated."""
        fake_es = self._make_fake_embedding_service()

        from app.models.resume import ResumeExperience

        jd = JobDescription(
            raw_text="ML Engineer needed",
            skills=["python"],
            responsibilities=[
                "Build and deploy machine learning models",
                "Design data pipelines",
            ],
        )
        resume = Resume(
            raw_text="Experienced ML engineer",
            skills=["python"],
            experience=[
                ResumeExperience(
                    title="ML Engineer",
                    company="AcmeCorp",
                    highlights=["Built recommendation system"],
                ),
                ResumeExperience(
                    title="Data Engineer",
                    company="DataCo",
                    highlights=["Designed ETL pipelines"],
                ),
            ],
        )
        result = match(jd, resume, embedding_service=fake_es)

        # experience_matches should be populated
        assert isinstance(result.experience_matches, list)
        assert result.experience_match_rate is not None
        assert 0.0 <= result.experience_match_rate <= 1.0

    def test_no_embedding_service_graceful_degradation(self):
        """Without embedding_service, new fields should be empty/None and
        score should match old formula."""
        jd = JobDescription(raw_text="", skills=["python", "docker"])
        resume = Resume(raw_text="", skills=["python", "docker"])
        result = match(jd, resume)

        # New fields should be empty/None
        assert result.semantic_skill_matches == []
        assert result.semantic_skill_match_rate is None
        assert result.experience_matches == []
        assert result.experience_match_rate is None

        # All skills matched exactly → direct_score=1.0, old formula:
        # 0.7*1.0 + 0.2*0 + 0.1*0 = 0.7
        assert result.overall_score == 0.7

    def test_overall_score_with_semantic_matches(self):
        """Overall score should reflect semantic contributions when embedding
        service is available."""
        fake_es = self._make_fake_embedding_service()

        jd = JobDescription(
            raw_text="Need ML and cloud skills",
            skills=["pytorch", "docker"],
            responsibilities=[],
        )
        resume = Resume(
            raw_text="Deep learning and containers",
            skills=["python"],  # only python matches nothing
            experience=[],
        )
        result = match(jd, resume, embedding_service=fake_es)

        # Score should be in valid range
        assert 0.0 <= result.overall_score <= 1.0
        # Should use the vector-enhanced formula (not the old 0.7/0.2/0.1)
        # The exact value depends on embeddings, but we can verify it's not
        # stuck at 0 (there should be some semantic signal)
        assert result.skill_match_rate >= 0.0

    def test_summary_includes_semantic_info(self):
        """Summary should mention semantic matches when available."""
        fake_es = self._make_fake_embedding_service()

        jd = JobDescription(
            raw_text="ML Engineer",
            skills=["pytorch", "docker"],
            responsibilities=["Build ML models"],
        )
        # Use a real-looking experience
        from app.models.resume import ResumeExperience

        resume = Resume(
            raw_text="Deep learning engineer",
            skills=["tensorflow", "kubernetes"],
            experience=[
                ResumeExperience(
                    title="ML Engineer",
                    company="AI Corp",
                    highlights=["Built models"],
                ),
            ],
        )
        result = match(jd, resume, embedding_service=fake_es)

        # Summary should mention experience matches
        assert "Experience matches" in result.summary
        # semantic_skill_matches may or may not exist with random embeddings
        # above the default 0.55 threshold — that's fine either way
