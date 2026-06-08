"""Tests for the default ReAct tools over shared state."""
from app.models.jd import JobDescription
from app.models.match import MatchReport
from app.services.agent.react_controller import ReactAgent
from app.services.agent.schemas import ReactState
from app.services.agent.tools import build_default_agent, default_tools


class FakeLLM:
    """Configured LLM that echoes a canned reply (for advice/rewrite tools)."""

    def __init__(self, reply="LLM OUTPUT"):
        self.reply = reply

    def is_configured(self):
        return True

    def complete(self, prompt, system=None, **kwargs):
        return self.reply


class FakeHit:
    def __init__(self, text, doc_id=0, metadata=None):
        self.text = text
        self.doc_id = doc_id
        self.metadata = metadata or {}


class FakeKb:
    """Records search calls and returns canned hits."""

    def __init__(self, hits=None):
        self.hits = (
            hits
            if hits is not None
            else [FakeHit("How does X work?", doc_id=1)]
        )
        self.calls = []

    def search(self, query, k=10, filters=None):
        self.calls.append({"query": query, "k": k, "filters": filters})
        return self.hits[:k]

JD_TEXT = "ML Engineer at Acme\n\nRequirements:\n- Python\n- Docker"
RESUME_TEXT = "Skills: Python\n\nExperience:\nML Engineer at Beta\n- Built models in Python"


def _tools():
    return {t.name: t for t in default_tools()}


def _state():
    return ReactState(jd_text=JD_TEXT, resume_text=RESUME_TEXT)


class TestToolSet:
    def test_has_expected_tools(self):
        assert set(_tools()) == {
            "parse_jd", "parse_resume", "match", "rank_projects",
            "audit", "generate_report",
            "kb_search", "interview_prep", "advise", "rewrite_bullet",
        }

    def test_build_default_agent(self):
        agent = build_default_agent(llm=object())
        assert isinstance(agent, ReactAgent)
        assert "parse_jd" in agent.tools


class TestPreconditions:
    def test_match_requires_jd(self):
        obs = _tools()["match"].handler(ReactState(resume_text=RESUME_TEXT), {})
        assert "parse the JD first" in obs

    def test_match_requires_resume(self):
        state = ReactState()
        _tools()["parse_jd"].handler(state, {"text": JD_TEXT})
        obs = _tools()["match"].handler(state, {})
        assert "parse the resume first" in obs

    def test_audit_requires_resume(self):
        assert "parse the resume first" in _tools()["audit"].handler(ReactState(), {})

    def test_report_requires_match(self):
        assert "run match first" in _tools()["generate_report"].handler(ReactState(), {})

    def test_parse_jd_requires_text(self):
        assert "no JD text" in _tools()["parse_jd"].handler(ReactState(), {})

    def test_parse_resume_requires_text(self):
        assert "no resume text" in _tools()["parse_resume"].handler(ReactState(), {})


class TestHappyPath:
    def test_full_sequence(self):
        tools = _tools()
        state = _state()

        assert "Parsed JD" in tools["parse_jd"].handler(state, {})
        assert isinstance(state.jd, JobDescription)

        assert "Parsed resume" in tools["parse_resume"].handler(state, {})
        assert state.resume is not None

        match_obs = tools["match"].handler(state, {})
        assert "Match score" in match_obs
        assert state.match is not None

        rank_obs = tools["rank_projects"].handler(state, {})
        assert "experiences" in rank_obs.lower() or "No relevant" in rank_obs

        audit_obs = tools["audit"].handler(state, {})
        assert audit_obs  # summary string

        report_obs = tools["generate_report"].handler(state, {})
        assert "Report generated" in report_obs
        assert isinstance(state.report, MatchReport)

    def test_parse_jd_with_explicit_text(self):
        state = ReactState()
        obs = _tools()["parse_jd"].handler(state, {"text": JD_TEXT})
        assert "Parsed JD" in obs
        assert "python" in state.jd.skills

    def test_rank_requires_both(self):
        assert "parse the JD and resume first" in _tools()["rank_projects"].handler(
            ReactState(), {}
        )

    def test_rank_no_relevant_experiences(self):
        from app.models.resume import Resume, ResumeExperience

        state = ReactState(jd_text="Role\n\nRequirements:\n- Python")
        tools = _tools()
        tools["parse_jd"].handler(state, {})  # JD skill: python
        state.resume = Resume(
            raw_text="...",
            experience=[
                ResumeExperience(
                    title="Chef", company="Kitchen",
                    highlights=["Cooked pasta and salads"],
                )
            ],
        )
        obs = tools["rank_projects"].handler(state, {})
        assert "No relevant experiences found." in obs


class TestKbSearch:
    def test_requires_query(self):
        obs = _tools()["kb_search"].handler(ReactState(kb_retriever=FakeKb()), {})
        assert "provide action_input.query" in obs

    def test_requires_retriever(self):
        obs = _tools()["kb_search"].handler(ReactState(), {"query": "python"})
        assert "knowledge base not available" in obs

    def test_returns_hits_and_passes_filters(self):
        kb = FakeKb([FakeHit("Explain RAG"), FakeHit("Explain embeddings")])
        state = ReactState(kb_retriever=kb)
        obs = _tools()["kb_search"].handler(
            state, {"query": "rag", "role": "ML", "difficulty": "hard", "k": 2}
        )
        assert "Explain RAG" in obs
        assert kb.calls[0]["filters"] == {"role": "ML", "difficulty": "hard"}
        assert kb.calls[0]["k"] == 2

    def test_no_results(self):
        state = ReactState(kb_retriever=FakeKb(hits=[]))
        assert _tools()["kb_search"].handler(state, {"query": "x"}) == "No KB results."


class TestInterviewPrep:
    def test_requires_jd_and_resume(self):
        obs = _tools()["interview_prep"].handler(ReactState(kb_retriever=FakeKb()), {})
        assert "parse the JD and resume first" in obs

    def test_requires_retriever(self):
        state = _state()
        tools = _tools()
        tools["parse_jd"].handler(state, {})
        tools["parse_resume"].handler(state, {})
        assert "knowledge base not available" in tools["interview_prep"].handler(state, {})

    def test_generates_prep(self):
        state = ReactState(jd_text=JD_TEXT, resume_text=RESUME_TEXT, kb_retriever=FakeKb())
        tools = _tools()
        tools["parse_jd"].handler(state, {})
        tools["parse_resume"].handler(state, {})
        obs = tools["interview_prep"].handler(state, {})
        assert "Interview prep ready" in obs
        assert state.interview is not None


class TestAdvise:
    def test_requires_match(self):
        assert "run match first" in _tools()["advise"].handler(ReactState(), {})

    def test_uses_llm_grounded_on_diagnosis(self):
        state = ReactState(jd_text=JD_TEXT, resume_text=RESUME_TEXT, llm=FakeLLM("ADVICE"))
        tools = _tools()
        tools["parse_jd"].handler(state, {})
        tools["parse_resume"].handler(state, {})
        tools["match"].handler(state, {})
        assert tools["advise"].handler(state, {"focus": "docker"}) == "ADVICE"

    def test_fallback_without_llm(self):
        state = _state()
        tools = _tools()
        tools["parse_jd"].handler(state, {})
        tools["parse_resume"].handler(state, {})
        tools["match"].handler(state, {})
        obs = tools["advise"].handler(state, {})
        assert "Prioritize the missing skills" in obs


class TestRewriteBullet:
    def test_requires_text(self):
        assert "provide action_input.text" in _tools()["rewrite_bullet"].handler(
            ReactState(), {}
        )

    def test_rewrites_with_llm(self):
        state = ReactState(llm=FakeLLM("STRONGER BULLET"))
        obs = _tools()["rewrite_bullet"].handler(state, {"text": "did stuff"})
        assert obs == "STRONGER BULLET"

    def test_fallback_returns_original(self):
        obs = _tools()["rewrite_bullet"].handler(ReactState(), {"text": "did stuff"})
        assert obs == "did stuff"
