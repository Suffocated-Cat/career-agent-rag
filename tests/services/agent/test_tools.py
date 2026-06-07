"""Tests for the default ReAct tools over shared state."""
from app.models.jd import JobDescription
from app.models.match import MatchReport
from app.services.agent.react_controller import ReactAgent
from app.services.agent.schemas import ReactState
from app.services.agent.tools import build_default_agent, default_tools

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
