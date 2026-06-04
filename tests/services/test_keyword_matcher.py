"""Tests for KeywordMatcher — baseline JD-to-resume matching."""
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
