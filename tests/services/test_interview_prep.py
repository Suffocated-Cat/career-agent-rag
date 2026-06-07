"""Tests for the interview-prep RAG service (in-memory KB, offline)."""
from app.models.jd import JobDescription
from app.models.resume import Resume
from app.services.interview_prep import InterviewPrep, generate_interview_prep
from app.services.knowledge import build_inmemory_kb_retriever


def _kb():
    return build_inmemory_kb_retriever()  # BM25 over the real KB


class _FakeLLM:
    def __init__(self, reply="", configured=True):
        self.reply = reply
        self.configured = configured

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        return self.reply


class TestGenerateInterviewPrep:
    def test_retrieves_questions_and_computes_gaps(self):
        jd = JobDescription(raw_text="x", skills=["python", "docker", "kubernetes"])
        resume = Resume(raw_text="x", skills=["python"])
        prep = generate_interview_prep(jd, resume, _kb())  # no LLM → fallback

        assert isinstance(prep, InterviewPrep)
        assert prep.gaps == ["docker", "kubernetes"]
        assert prep.questions  # retrieved from the KB
        assert "Focus areas" in prep.guide  # deterministic fallback guide

    def test_uses_llm_guide_grounded_on_questions(self):
        jd = JobDescription(raw_text="x", skills=["docker"])
        resume = Resume(raw_text="x", skills=[])
        prep = generate_interview_prep(
            jd, resume, _kb(), llm=_FakeLLM(reply="# Prep\nFocus on Docker layers.")
        )
        assert prep.guide == "# Prep\nFocus on Docker layers."
        assert prep.questions  # still retrieved

    def test_no_questions_when_off_topic(self):
        jd = JobDescription(raw_text="x", skills=["zzz_nonexistent"])
        resume = Resume(raw_text="x", skills=[])
        prep = generate_interview_prep(jd, resume, _kb())
        assert prep.questions == []
        assert "zzz_nonexistent" in prep.guide  # gap still surfaced

    def test_no_gaps_when_all_covered(self):
        jd = JobDescription(raw_text="x", skills=["python"])
        resume = Resume(raw_text="x", skills=["Python"])  # case-insensitive
        prep = generate_interview_prep(jd, resume, _kb())
        assert prep.gaps == []
