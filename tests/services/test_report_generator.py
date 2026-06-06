"""Tests for ReportGenerator — template-based matching report generation."""
import pytest
from app.models.jd import JobDescription
from app.models.resume import Resume, ResumeExperience, ResumeProject
from app.models.match import (
    MatchResult,
    MatchReport,
    SkillMatchDetail,
    ExperienceMatchDetail,
)
from app.services.report_generator import (
    generate_report,
    _rating,
    _skill_summary,
    _skill_gap_analysis,
    _recommendations,
    _score_bar,
)


class TestRating:
    def test_excellent(self):
        assert _rating(0.90) == "Excellent"
        assert _rating(0.80) == "Excellent"

    def test_good(self):
        assert _rating(0.75) == "Good"
        assert _rating(0.60) == "Good"

    def test_fair(self):
        assert _rating(0.55) == "Fair"
        assert _rating(0.40) == "Fair"

    def test_low(self):
        assert _rating(0.30) == "Low"
        assert _rating(0.00) == "Low"


class TestSkillSummary:
    def test_with_semantic(self):
        result = MatchResult(
            matched_skills=["python", "docker"],
            missing_skills=["aws"],
            semantic_skill_matches=[
                SkillMatchDetail(
                    jd_skill="docker", resume_skill="kubernetes", similarity=0.65
                )
            ],
        )
        summary = _skill_summary(result)
        assert "2/3" in summary
        assert "1 via semantic" in summary

    def test_no_semantic(self):
        result = MatchResult(
            matched_skills=["python"],
            missing_skills=["docker"],
        )
        summary = _skill_summary(result)
        assert "1/2" in summary
        assert "semantic" not in summary


class TestSkillGapAnalysis:
    def test_with_gaps(self):
        text = _skill_gap_analysis(["pytorch", "docker"])
        assert "2 required skill" in text
        assert "pytorch" in text
        assert "docker" in text

    def test_no_gaps(self):
        text = _skill_gap_analysis([])
        assert "No skill gaps" in text


class TestRecommendations:
    def test_strong_match(self):
        result = MatchResult(overall_score=0.85, missing_skills=[])
        text = _recommendations(result, 5)
        assert "strong match" in text.lower()

    def test_good_match(self):
        result = MatchResult(overall_score=0.65, missing_skills=["docker"])
        text = _recommendations(result, 5)
        assert "good match" in text.lower()
        assert "docker" in text

    def test_low_match(self):
        result = MatchResult(overall_score=0.25, missing_skills=["python", "docker", "aws"])
        text = _recommendations(result, 5)
        assert "low alignment" in text.lower()


class TestScoreBar:
    def test_full(self):
        bar = _score_bar(1.0, width=10)
        assert bar == "`██████████`"

    def test_half(self):
        bar = _score_bar(0.5, width=10)
        assert bar == "`█████░░░░░`"

    def test_empty(self):
        bar = _score_bar(0.0, width=10)
        assert bar == "`░░░░░░░░░░`"


class TestGenerateReport:
    def test_full_report_generated(self):
        jd = JobDescription(
            raw_text="Looking for ML Engineer",
            title="ML Engineer",
            company="AcmeCorp",
            skills=["pytorch", "docker", "aws"],
            responsibilities=["Build ML models"],
        )
        resume = Resume(
            raw_text="Experienced ML engineer",
            skills=["pytorch", "kubernetes"],
            experience=[
                ResumeExperience(
                    title="ML Engineer",
                    company="AI Corp",
                    highlights=["Built recommendation system"],
                ),
            ],
        )
        result = MatchResult(
            matched_skills=["pytorch", "docker"],  # docker via semantic
            missing_skills=["aws"],
            overall_score=0.65,
            skill_match_rate=0.67,
            semantic_similarity=0.55,
            summary="test summary",
            semantic_skill_matches=[
                SkillMatchDetail(
                    jd_skill="docker", resume_skill="kubernetes", similarity=0.72
                )
            ],
            semantic_skill_match_rate=0.33,
            experience_matches=[
                ExperienceMatchDetail(
                    jd_responsibility="Build ML models",
                    resume_experience="ML Engineer at AI Corp: Built recommendation system",
                    similarity=0.68,
                )
            ],
            experience_match_rate=1.0,
        )

        report = generate_report(jd, resume, result)

        # Check structured fields
        assert report.job_title == "ML Engineer"
        assert report.overall_score == 0.65
        assert report.overall_rating == "Good"
        assert "2/3" in report.skill_summary
        assert report.matched_skills == ["pytorch", "docker"]
        assert report.missing_skills == ["aws"]
        assert len(report.semantic_skill_matches) == 1
        assert len(report.experience_alignment) == 1
        assert "aws" in report.skill_gap_analysis
        assert "good match" in report.recommendations.lower()
        assert len(report.full_report) > 0

    def test_full_report_contains_expected_sections(self):
        jd = JobDescription(
            raw_text="Backend role",
            skills=["python", "docker"],
            responsibilities=[],
        )
        resume = Resume(
            raw_text="Backend developer",
            skills=["python"],
            experience=[],
        )
        result = MatchResult(
            matched_skills=["python"],
            missing_skills=["docker"],
            overall_score=0.50,
            skill_match_rate=0.50,
        )

        report = generate_report(jd, resume, result)
        full = report.full_report

        # Verify key sections are present
        assert "## Overall Assessment" in full
        assert "## Skill Analysis" in full
        assert "## Skill Gap Analysis" in full
        assert "## Recommendations" in full
        assert "✅ python" in full
        assert "❌ docker" in full
        assert "Score:" in full
        assert "Rating:" in full

    def test_empty_data_handled(self):
        jd = JobDescription(raw_text="", skills=[])
        resume = Resume(raw_text="", skills=[])
        result = MatchResult()

        report = generate_report(jd, resume, result)

        assert report.overall_score == 0.0
        assert report.overall_rating == "Low"
        assert "0/0" in report.skill_summary
        assert len(report.full_report) > 0

    def test_experience_section_omitted_when_empty(self):
        jd = JobDescription(raw_text="", skills=["python"])
        resume = Resume(raw_text="", skills=["python"])
        result = MatchResult(matched_skills=["python"])

        report = generate_report(jd, resume, result)
        assert "Experience Alignment" not in report.full_report

    def test_risk_audit_section_omitted_when_absent(self):
        jd = JobDescription(raw_text="", skills=["python"])
        resume = Resume(raw_text="", skills=["python"])
        result = MatchResult(matched_skills=["python"])  # no project_audit

        report = generate_report(jd, resume, result)
        assert report.project_audit is None
        assert "Project Risk Audit" not in report.full_report

    def test_risk_audit_section_rendered_when_present(self):
        from app.services.project_auditor import audit_resume

        jd = JobDescription(raw_text="", skills=["python"])
        resume = Resume(
            raw_text="...",
            skills=["python", "rag", "agent"],
            projects=[
                ResumeProject(
                    name="LLM Assistant",
                    description="A chatbot.",
                    technologies=["rag", "agent"],
                )
            ],
        )
        result = MatchResult(project_audit=audit_resume(resume))

        report = generate_report(jd, resume, result)
        assert report.project_audit is not None
        assert "## Project Risk Audit" in report.full_report
        assert "| Severity | Category | Subject | Detail |" in report.full_report
