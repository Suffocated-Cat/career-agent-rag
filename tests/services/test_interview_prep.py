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

    def test_no_skills_uses_combined_query(self):
        # With no JD skills, retrieval falls back to a combined raw_text query.
        jd = JobDescription(raw_text="docker image layers and caching", skills=[])
        resume = Resume(raw_text="x", skills=[])
        prep = generate_interview_prep(jd, resume, _kb())
        assert prep.gaps == []
        assert any("docker" in q.lower() for q in prep.questions)

    def test_dedup_across_skills(self):
        from app.services.retrieval.base import RetrievalResult

        class _FakeKb:
            def search(self, query, k=10, filters=None):
                table = {
                    "a": [RetrievalResult(1, "shared", 1.0), RetrievalResult(2, "qa", 0.9)],
                    "b": [RetrievalResult(1, "shared", 1.0), RetrievalResult(3, "qb", 0.9)],
                }
                return table.get(query, [])[:k]

        jd = JobDescription(raw_text="x", skills=["a", "b"])
        resume = Resume(raw_text="x", skills=[])
        prep = generate_interview_prep(jd, resume, _FakeKb(), per_skill=2)
        # doc_id 1 ("shared") retrieved for both skills but appears once.
        assert prep.questions == ["shared", "qa", "qb"]

    def test_per_skill_retrieval_covers_each_skill(self):
        # A combined query lets common skills crowd out RAG; per-skill retrieval
        # must surface both python and rag questions.
        jd = JobDescription(raw_text="x", skills=["python", "docker", "rag"])
        resume = Resume(raw_text="x", skills=[])
        prep = generate_interview_prep(jd, resume, _kb(), per_skill=2)
        texts = " ".join(prep.questions).lower()
        assert "python" in texts
        assert "rag" in texts

    def test_answer_outline_fed_into_prompt(self):
        captured = {}

        class _RecordingLLM:
            def is_configured(self):
                return True

            def complete(self, prompt, system=None, **kwargs):
                captured["prompt"] = prompt
                return "guide"

        jd = JobDescription(raw_text="x", skills=["fastapi"])
        resume = Resume(raw_text="x", skills=[])
        generate_interview_prep(jd, resume, _kb(), llm=_RecordingLLM())
        # The retrieved questions' answer outlines must reach the prompt.
        assert "a strong answer covers:" in captured["prompt"]

    def test_role_difficulty_filter_applied(self):
        jd = JobDescription(raw_text="x", skills=["fastapi", "docker", "rag"])
        resume = Resume(raw_text="x", skills=[])
        prep = generate_interview_prep(
            jd, resume, _kb(), k=10, role="frontend", difficulty="junior"
        )
        # Filtered to frontend/junior, so backend/AI questions are excluded;
        # there are no frontend questions matching these skills → empty bank.
        assert prep.questions == []
        # Gaps are still computed regardless of the filter.
        assert prep.gaps == ["fastapi", "docker", "rag"]
