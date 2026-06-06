"""Tests for ProjectAuditor — rule-based resume authenticity checks."""
from app.models.resume import Resume, ResumeExperience, ResumeProject
from app.models.audit import ProjectAuditReport, RiskFinding
from app.services.project_auditor import (
    audit_resume,
    _is_advanced,
    _is_supported,
    _has_metric,
    _Evidence,
)


class _AuditFakeLLM:
    def __init__(self, reply="", configured=True, raises=False):
        self.reply = reply
        self.configured = configured
        self.raises = raises
        self.called = False

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        self.called = True
        if self.raises:
            raise RuntimeError("api down")
        return self.reply


def _vague_claims_resume() -> Resume:
    """A resume modeled on the 'vague_claims' fixture profile."""
    return Resume(
        raw_text="...",
        skills=["python", "react", "llm", "rag", "agent", "mcp", "fastapi"],
        experience=[
            ResumeExperience(
                title="AI Developer Intern",
                company="BrightApps",
                highlights=[
                    "Worked on several AI features and helped improve user experience",
                    "Participated in meetings about LLM strategy",
                ],
            ),
        ],
        projects=[
            ResumeProject(
                name="LLM Assistant Concept",
                description="A chatbot assistant.",
                technologies=["llm", "rag", "agent", "mcp"],
            ),
            ResumeProject(
                name="Portfolio Website",
                description="Created a personal portfolio with project cards and styling.",
                technologies=["react", "vite", "css"],
            ),
        ],
    )


def _strong_resume() -> Resume:
    """A resume whose claims are substantiated."""
    return Resume(
        raw_text="...",
        skills=["python", "fastapi", "rag", "docker"],
        experience=[
            ResumeExperience(
                title="ML Engineer",
                company="Acme",
                highlights=[
                    "Reduced retrieval latency by 40% using a BM25 + embedding pipeline",
                    "Shipped a FastAPI service with 95% test coverage",
                ],
            ),
        ],
        projects=[
            ResumeProject(
                name="CareerAgent",
                description=(
                    "Built an agentic RAG backend with BM25 retrieval, embedding "
                    "vector search, reranking, and docker deployment for job matching."
                ),
                technologies=["python", "fastapi", "rag", "docker"],
            ),
        ],
    )


class TestHelpers:
    def test_has_metric(self):
        assert _has_metric("reduced latency by 40%")
        assert not _has_metric("helped improve things")

    def test_is_advanced(self):
        assert _is_advanced("RAG")
        assert _is_advanced("tool calling agent")  # token match
        assert not _is_advanced("react")

    def test_is_supported_single_word_token_match(self):
        ev = _Evidence(Resume(
            raw_text="x",
            experience=[ResumeExperience(title="Dev", company="Co",
                                         highlights=["built with python"])],
        ))
        assert _is_supported("python", advanced=False, evidence=ev)
        assert not _is_supported("rust", advanced=False, evidence=ev)

    def test_is_supported_multiword_substring(self):
        ev = _Evidence(Resume(
            raw_text="x",
            experience=[ResumeExperience(title="Dev", company="Co",
                                         highlights=["used tool calling in the agent"])],
        ))
        assert _is_supported("tool calling", advanced=False, evidence=ev)
        assert not _is_supported("vector database", advanced=False, evidence=ev)

    def test_advanced_skill_needs_prose_not_tech_list(self):
        # 'rag' only in a technology list, not in any prose → unsupported.
        ev = _Evidence(Resume(
            raw_text="x",
            projects=[ResumeProject(name="Demo", description="A demo.",
                                    technologies=["rag"])],
        ))
        assert not _is_supported("rag", advanced=True, evidence=ev)
        # ordinary skill from the same tech list is fine.
        assert _is_supported("rag", advanced=False, evidence=ev)

    def test_empty_skill_is_supported(self):
        assert _is_supported("", advanced=False, evidence=_Evidence(Resume(raw_text="x")))


class TestAuditResume:
    def test_returns_report(self):
        report = audit_resume(_vague_claims_resume())
        assert isinstance(report, ProjectAuditReport)
        assert all(isinstance(f, RiskFinding) for f in report.findings)

    def test_flags_unsupported_advanced_skills_as_high(self):
        report = audit_resume(_vague_claims_resume())
        unsupported = {
            f.subject for f in report.findings if f.category == "unsupported_skill"
        }
        # llm/rag/agent/mcp are claimed but absent from experience text.
        assert {"agent", "mcp"} <= unsupported
        advanced = [
            f for f in report.findings
            if f.category == "unsupported_skill" and f.subject in {"rag", "agent", "mcp"}
        ]
        assert advanced and all(f.severity == "high" for f in advanced)

    def test_supported_skill_not_flagged(self):
        report = audit_resume(_vague_claims_resume())
        unsupported = {
            f.subject for f in report.findings if f.category == "unsupported_skill"
        }
        # 'react' appears in the Portfolio project → supported.
        assert "react" not in unsupported

    def test_flags_vague_experience(self):
        report = audit_resume(_vague_claims_resume())
        vague = [f for f in report.findings if f.category == "vague_experience"]
        assert vague
        assert any("helped improve" in f.evidence.lower() for f in vague)

    def test_flags_unsupported_project_claim(self):
        report = audit_resume(_vague_claims_resume())
        proj_findings = [
            f for f in report.findings if f.category == "unsupported_project_claim"
        ]
        assert any(f.subject == "LLM Assistant Concept" for f in proj_findings)
        assert all(f.severity == "high" for f in proj_findings)

    def test_risk_score_positive_for_risky_resume(self):
        report = audit_resume(_vague_claims_resume())
        assert report.risk_score > 0.0
        assert "Risk score" in report.summary

    def test_strong_resume_is_low_risk(self):
        report = audit_resume(_strong_resume())
        # Skills substantiated, highlights quantified, project description rich.
        assert not any(
            f.category == "unsupported_project_claim" for f in report.findings
        )
        assert report.risk_score < 0.2

    def test_quantified_highlight_not_vague(self):
        report = audit_resume(_strong_resume())
        assert not any(f.category == "vague_experience" for f in report.findings)

    def test_empty_resume_is_clean(self):
        report = audit_resume(Resume(raw_text="nothing"))
        assert report.findings == []
        assert report.risk_score == 0.0
        assert report.summary == "No authenticity risks detected."

    def test_no_advice_without_llm(self):
        report = audit_resume(_vague_claims_resume())
        assert report.advice == ""

    def test_advice_generated_with_llm(self):
        report = audit_resume(_vague_claims_resume(), llm=_AuditFakeLLM("Fix it like this."))
        assert report.advice == "Fix it like this."
        # Numbers stay deterministic.
        assert report.findings
        assert report.risk_score > 0.0

    def test_no_advice_when_no_findings(self):
        # Clean resume → no findings → LLM not invoked, advice empty.
        llm = _AuditFakeLLM("should not be called")
        report = audit_resume(Resume(raw_text="nothing"), llm=llm)
        assert report.advice == ""
        assert llm.called is False

    def test_advice_falls_back_on_llm_error(self):
        report = audit_resume(_vague_claims_resume(), llm=_AuditFakeLLM(raises=True))
        assert report.advice == ""  # fallback empty, findings intact
        assert report.findings

    def test_thin_description_with_advanced_tech_flagged(self):
        resume = Resume(
            raw_text="...",
            projects=[
                ResumeProject(
                    name="Quick RAG",
                    description="A rag demo.",  # mentions rag but too thin
                    technologies=["rag"],
                )
            ],
        )
        report = audit_resume(resume)
        assert any(
            f.category == "unsupported_project_claim" for f in report.findings
        )
