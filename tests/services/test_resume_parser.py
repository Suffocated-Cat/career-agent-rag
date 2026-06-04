"""
Tests for ResumeParser — rule-based resume parsing.
"""
from app.services.resume_parser import parse_resume

# ── Sample resumes ─────────────────────────────────────────────────────

FULL_RESUME = """
SKILLS
Python, Docker, Kubernetes, AWS, React, TypeScript, PostgreSQL, Redis, Git

EXPERIENCE
Senior ML Engineer, AcmeCorp | Jan 2022 – Present
- Built RAG systems using LangChain and vector databases
- Deployed ML models to production with Docker and Kubernetes
- Led a team of 4 engineers

Software Developer at Google | Jun 2019 – Dec 2021
- Developed internal tools for data processing
- Built real-time monitoring dashboards with Grafana

EDUCATION
M.S. Computer Science, Stanford University, 2020

B.S. Computer Science, MIT, 2017

PROJECTS
RAG Pipeline | Python, LangChain, Pinecone
Built an end-to-end RAG pipeline for document question answering.
Handled 10k+ documents with sub-second latency.

Sentiment Analyzer | Python, PyTorch, FastAPI
Fine-tuned BERT for multi-language sentiment classification.
Deployed as a REST API serving 1k requests/minute.
"""


MINIMAL_RESUME = """
SKILLS
Python, SQL

EXPERIENCE
Junior Developer at StartupX
- Wrote unit tests
"""


NO_SECTIONS_RESUME = """
Experienced developer with skills in Python and Docker.
Worked at Google as a Software Engineer.
Studied Computer Science at Stanford.
"""


# ── Tests ───────────────────────────────────────────────────────────────


class TestResumeParserFull:
    """Tests against a well-structured resume with all sections."""

    def test_extracts_skills(self):
        resume = parse_resume(FULL_RESUME)
        assert "python" in resume.skills
        assert "docker" in resume.skills
        assert "kubernetes" in resume.skills
        assert "react" in resume.skills
        assert "typescript" in resume.skills
        assert "postgresql" in resume.skills
        assert "redis" in resume.skills

    def test_extracts_experience(self):
        resume = parse_resume(FULL_RESUME)
        assert len(resume.experience) >= 2
        # First experience
        exp1 = resume.experience[0]
        assert "Senior ML Engineer" in exp1.title
        assert exp1.company == "AcmeCorp"
        assert exp1.duration == "Jan 2022 – Present"
        assert any("RAG" in h for h in exp1.highlights)
        assert any("Docker" in h for h in exp1.highlights)

        # Second experience
        exp2 = resume.experience[1]
        assert "Software Developer" in exp2.title
        assert exp2.company == "Google"
        assert any("internal tools" in h for h in exp2.highlights)

    def test_extracts_education(self):
        resume = parse_resume(FULL_RESUME)
        assert len(resume.education) >= 2
        edu1 = resume.education[0]
        assert "M.S." in edu1.degree or "Computer Science" in edu1.degree
        assert "Stanford" in edu1.institution
        assert edu1.year == "2020"

    def test_extracts_projects(self):
        resume = parse_resume(FULL_RESUME)
        assert len(resume.projects) >= 2
        proj1 = resume.projects[0]
        assert "RAG Pipeline" in proj1.name
        assert "python" in proj1.technologies
        assert "langchain" in proj1.technologies

    def test_preserves_raw_text(self):
        resume = parse_resume(FULL_RESUME)
        assert resume.raw_text == FULL_RESUME


class TestResumeParserEdgeCases:
    """Tests for edge cases."""

    def test_minimal_resume(self):
        resume = parse_resume(MINIMAL_RESUME)
        assert "python" in resume.skills
        assert resume.experience[0].title == "Junior Developer"

    def test_no_sections(self):
        resume = parse_resume(NO_SECTIONS_RESUME)
        assert "python" in resume.skills
        assert "docker" in resume.skills

    def test_empty_string(self):
        resume = parse_resume("")
        assert resume.raw_text == ""
        assert resume.skills == []
