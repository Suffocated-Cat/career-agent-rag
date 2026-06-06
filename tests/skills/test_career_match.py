"""Tests for the CareerMatch skill orchestrator."""
from app.skills.career_match import CareerMatchResult, run_career_match

JD_TEXT = """Senior ML Engineer at Acme

Requirements:
- Python
- PyTorch
- Docker

Responsibilities:
- Build and deploy recommendation models
"""

RESUME_TEXT = """Skills: Python, PyTorch

Experience:
ML Engineer at Beta
- Built recommendation models in PyTorch
"""


class TestRunCareerMatch:
    def test_returns_full_result(self):
        out = run_career_match(JD_TEXT, RESUME_TEXT)  # offline: rule + bm25
        assert isinstance(out, CareerMatchResult)
        assert out.jd.raw_text == JD_TEXT
        assert out.resume.raw_text == RESUME_TEXT

    def test_match_populated(self):
        out = run_career_match(JD_TEXT, RESUME_TEXT)
        assert 0.0 <= out.match.overall_score <= 1.0
        # python/pytorch overlap → at least one matched skill.
        assert out.match.matched_skills

    def test_report_present(self):
        out = run_career_match(JD_TEXT, RESUME_TEXT)
        assert out.report.full_report
        assert "## Overall Assessment" in out.report.full_report  # template path

    def test_audit_attached(self):
        out = run_career_match(JD_TEXT, RESUME_TEXT)
        assert out.match.project_audit is not None

    def test_project_relevance_ranked(self):
        out = run_career_match(JD_TEXT, RESUME_TEXT)
        # The single experience should be ranked against the JD query.
        assert isinstance(out.match.project_relevance, list)

    def test_uses_llm_when_provided(self):
        class _FakeLLM:
            def is_configured(self):
                return True

            def complete(self, prompt, system=None, **kwargs):
                return "# LLM Report\nGrounded."

        out = run_career_match(JD_TEXT, RESUME_TEXT, llm=_FakeLLM())
        # Report narrative comes from the LLM; structured fields stay computed.
        assert out.report.full_report == "# LLM Report\nGrounded."
        assert 0.0 <= out.report.overall_score <= 1.0
