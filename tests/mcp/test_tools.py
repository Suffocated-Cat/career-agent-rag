"""Tests for the MCP tool implementations (dict-in, dict-out)."""
from app.models.jd import JobDescription
from app.models.resume import Resume, ResumeExperience, ResumeProject
from app.mcp.tools import (
    parse_jd_tool,
    parse_resume_tool,
    match_tool,
    audit_tool,
    rank_projects_tool,
)


def _jd_dict() -> dict:
    return JobDescription(
        raw_text="ML role", skills=["python", "rag"],
        responsibilities=["Build retrieval pipelines"],
    ).model_dump()


def _resume_dict() -> dict:
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
            ResumeProject(name="Search", description="vector search over docs",
                          technologies=["python"]),
        ],
    ).model_dump()


class TestParseTools:
    def test_parse_jd_returns_dict(self):
        out = parse_jd_tool("Looking for Python and Docker skills.")
        assert isinstance(out, dict)
        assert "skills" in out and "raw_text" in out

    def test_parse_resume_returns_dict(self):
        out = parse_resume_tool("Skills: Python, Docker")
        assert isinstance(out, dict)
        assert "skills" in out


class TestMatchTool:
    def test_returns_match_result_dict(self):
        out = match_tool(_jd_dict(), _resume_dict())
        assert isinstance(out, dict)
        assert "overall_score" in out
        assert "matched_skills" in out


class TestAuditTool:
    def test_returns_audit_dict(self):
        out = audit_tool(_resume_dict())
        assert isinstance(out, dict)
        assert "findings" in out
        assert "risk_score" in out

    def test_flags_unsupported_advanced_skill(self):
        resume = Resume(
            raw_text="...",
            skills=["python", "mcp"],  # mcp unsupported anywhere
            experience=[
                ResumeExperience(title="Dev", company="Co",
                                 highlights=["wrote python scripts"])
            ],
        ).model_dump()
        out = audit_tool(resume)
        subjects = {f["subject"] for f in out["findings"]}
        assert "mcp" in subjects


class TestRankProjectsTool:
    def test_returns_list_of_dicts(self):
        out = rank_projects_tool(_jd_dict(), _resume_dict())
        assert isinstance(out, list)
        assert all(isinstance(r, dict) for r in out)
        if out:
            assert {"doc_id", "label", "score", "normalized_score"} <= out[0].keys()

    def test_bm25_used_without_embeddings(self):
        # No embedding_service → bm25 path; should still rank the relevant item.
        out = rank_projects_tool(_jd_dict(), _resume_dict())
        assert any("ML Engineer" in r["label"] for r in out)
