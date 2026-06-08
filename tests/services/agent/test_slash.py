"""Tests for the deterministic slash commands."""
from app.services.agent.schemas import JdInput, ReactState
from app.services.agent.slash import handle_slash, is_slash

JD_TEXT = "ML Engineer at Acme\n\nRequirements:\n- Python\n- Docker"
JD_TEXT_B = "Frontend Engineer\n\nRequirements:\n- React"
RESUME_TEXT = "Skills: Python\n\nExperience:\nML Engineer at Beta\n- Built models in Python"


def _state(**kw):
    return ReactState(jd_text=JD_TEXT, resume_text=RESUME_TEXT, **kw)


class TestIsSlash:
    def test_detects_slash(self):
        assert is_slash("/match")
        assert is_slash("   /help")
        assert not is_slash("why is my match low?")


class TestSlashCommands:
    def test_help_lists_commands(self):
        out = handle_slash("/help", _state())
        assert "/match" in out and "/compare" in out

    def test_bare_slash_is_help(self):
        assert "/match" in handle_slash("/", _state())

    def test_unknown_command(self):
        assert "Unknown command" in handle_slash("/nope", _state())

    def test_match_parses_and_scores(self):
        state = _state()
        out = handle_slash("/match", state)
        assert "Match score" in out
        assert state.jd is not None and state.resume is not None and state.match is not None

    def test_match_needs_inputs(self):
        out = handle_slash("/match", ReactState(jd_text=JD_TEXT))  # no resume
        assert "need both" in out.lower()

    def test_report_returns_markdown(self):
        out = handle_slash("/report", _state())
        assert out and isinstance(out, str)

    def test_audit_runs(self):
        out = handle_slash("/audit", _state())
        assert out  # either findings or "No authenticity risks detected."

    def test_prep_needs_kb(self):
        out = handle_slash("/prep", _state())  # no kb_retriever
        assert "knowledge base isn't available" in out

    def test_compare_ranks_jds(self):
        state = ReactState(
            resume_text=RESUME_TEXT,
            jd_inputs=[JdInput("Job A", JD_TEXT), JdInput("Job B", JD_TEXT_B)],
        )
        out = handle_slash("/compare", state)
        assert "Best fit" in out
        assert len(state.comparison) == 2

    def test_compare_needs_two(self):
        state = ReactState(resume_text=RESUME_TEXT, jd_inputs=[JdInput("A", JD_TEXT)])
        assert "at least two JDs" in handle_slash("/compare", state)
