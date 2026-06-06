"""Tests for the CareerAgent FastMCP server."""
import asyncio
import json

import pytest

import app.mcp.server as server
from app.models.jd import JobDescription
from app.models.resume import Resume, ResumeExperience, ResumeProject

EXPECTED_TOOLS = {
    "parse_jd", "parse_resume", "match_resume", "audit_resume", "rank_projects",
}


@pytest.fixture
def _no_embeddings(monkeypatch):
    """Avoid loading the heavy embedding model when executing tool handlers."""
    monkeypatch.setattr(server, "_get_embeddings", lambda: None)


def _jd_dict() -> dict:
    return JobDescription(raw_text="ML role", skills=["python", "rag"]).model_dump()


def _resume_dict() -> dict:
    return Resume(
        raw_text="...",
        skills=["python", "rag"],
        experience=[
            ResumeExperience(title="ML Engineer", company="Acme",
                             highlights=["Built a rag pipeline in python"])
        ],
        projects=[ResumeProject(name="Search", description="vector search",
                                technologies=["python"])],
    ).model_dump()


def _content_list(call_result):
    """Normalize call_tool's return across SDK versions to a content list."""
    return call_result[0] if isinstance(call_result, tuple) else call_result


class TestToolListing:
    def test_lists_expected_tools(self):
        names = {t.name for t in asyncio.run(server.mcp.list_tools())}
        assert EXPECTED_TOOLS <= names

    def test_input_schemas(self):
        tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
        assert "raw_text" in tools["parse_jd"].inputSchema["properties"]
        match_props = tools["match_resume"].inputSchema["properties"]
        assert {"jd", "resume"} <= match_props.keys()


@pytest.mark.usefixtures("_no_embeddings")
class TestDecoratedFunctionsCallable:
    """The @mcp.tool() functions remain directly callable."""

    def test_parse_jd(self):
        out = server.parse_jd("Need Python and Docker.")
        assert isinstance(out, dict)
        assert "python" in out["skills"]

    def test_parse_resume(self):
        out = server.parse_resume("Skills: Python")
        assert isinstance(out, dict) and "skills" in out

    def test_match_resume(self):
        out = server.match_resume(_jd_dict(), _resume_dict())
        assert "overall_score" in out

    def test_audit_resume(self):
        out = server.audit_resume(_resume_dict())
        assert "findings" in out and "risk_score" in out

    def test_rank_projects(self):
        out = server.rank_projects(_jd_dict(), _resume_dict())
        assert isinstance(out, list)


@pytest.mark.usefixtures("_no_embeddings")
class TestCallToolRoundtrip:
    def test_parse_jd_over_protocol(self):
        result = asyncio.run(
            server.mcp.call_tool("parse_jd", {"raw_text": "Need Python and Docker."})
        )
        content = _content_list(result)
        payload = json.loads(content[0].text)
        assert "python" in payload["skills"]


class TestEmbeddingsGetter:
    def test_returns_service_when_available(self, monkeypatch):
        monkeypatch.setattr(server, "EmbeddingService", lambda: "EMB")
        server._get_embeddings.cache_clear()
        assert server._get_embeddings() == "EMB"
        server._get_embeddings.cache_clear()

    def test_returns_none_on_failure(self, monkeypatch):
        def _boom():
            raise RuntimeError("no model")

        monkeypatch.setattr(server, "EmbeddingService", _boom)
        server._get_embeddings.cache_clear()
        assert server._get_embeddings() is None
        server._get_embeddings.cache_clear()


class TestMain:
    def test_main_runs_server(self, monkeypatch):
        called = {}
        monkeypatch.setattr(server.mcp, "run", lambda: called.setdefault("ran", True))
        server.main()
        assert called["ran"]
