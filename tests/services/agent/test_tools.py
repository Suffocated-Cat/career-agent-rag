"""Tests for the default agent tool set wired to real services."""
import pytest

from app.models.jd import JobDescription
from app.models.resume import Resume, ResumeExperience, ResumeProject
from app.models.match import MatchResult
from app.models.audit import ProjectAuditReport
from app.services.agent.controller import AgentContext
from app.services.agent.tools import build_default_controller, default_tools


def _resume() -> Resume:
    return Resume(
        raw_text="...",
        skills=["python", "rag"],
        experience=[
            ResumeExperience(
                title="ML Engineer", company="Acme",
                highlights=["Built a rag pipeline in python"],
            )
        ],
        projects=[
            ResumeProject(name="Search", description="vector search", technologies=["python"])
        ],
    )


def _jd() -> JobDescription:
    return JobDescription(raw_text="ML role", skills=["python", "rag"])


class TestDefaultToolSet:
    def test_has_expected_tools(self):
        names = {t.name for t in default_tools()}
        assert names == {
            "jd_parser", "resume_parser", "resume_matcher",
            "project_auditor", "project_ranker",
        }


class TestRoutingAndExecution:
    def test_jd_parsing_task(self):
        c = build_default_controller()
        result = c.run(AgentContext(task="analyze this job description", jd_text="Need Python and Docker."))
        assert result.tool == "jd_parser"
        assert isinstance(result.output, JobDescription)

    def test_resume_parsing_task(self):
        c = build_default_controller()
        result = c.run(AgentContext(task="parse my resume", resume_text="Skills: Python"))
        assert result.tool == "resume_parser"
        assert isinstance(result.output, Resume)

    def test_matching_task(self):
        c = build_default_controller()
        result = c.run(AgentContext(task="match my resume and score the fit", jd=_jd(), resume=_resume()))
        assert result.tool == "resume_matcher"
        assert isinstance(result.output, MatchResult)

    def test_audit_task(self):
        c = build_default_controller()
        result = c.run(AgentContext(task="audit my resume for authenticity risk", resume=_resume()))
        assert result.tool == "project_auditor"
        assert isinstance(result.output, ProjectAuditReport)

    def test_ranking_task(self):
        c = build_default_controller()
        result = c.run(AgentContext(task="rank my projects by relevance", jd=_jd(), resume=_resume()))
        assert result.tool == "project_ranker"
        assert isinstance(result.output, list)


class TestMissingInputs:
    def test_jd_parser_requires_text(self):
        c = build_default_controller()
        with pytest.raises(ValueError, match="jd_text"):
            c.run(AgentContext(task="analyze this jd"))

    def test_resume_parser_requires_text(self):
        c = build_default_controller()
        with pytest.raises(ValueError, match="resume_text"):
            c.run(AgentContext(task="parse my resume"))

    def test_matcher_requires_jd_and_resume(self):
        c = build_default_controller()
        with pytest.raises(ValueError, match="parsed jd"):
            c.run(AgentContext(task="match and compare the fit", resume=_resume()))

    def test_matcher_requires_resume(self):
        c = build_default_controller()
        with pytest.raises(ValueError, match="parsed resume"):
            c.run(AgentContext(task="match and compare the fit", jd=_jd()))

    def test_auditor_requires_resume(self):
        c = build_default_controller()
        with pytest.raises(ValueError, match="parsed resume"):
            c.run(AgentContext(task="audit for risk"))

    def test_ranker_requires_jd(self):
        c = build_default_controller()
        with pytest.raises(ValueError, match="parsed jd"):
            c.run(AgentContext(task="rank by relevance", resume=_resume()))
