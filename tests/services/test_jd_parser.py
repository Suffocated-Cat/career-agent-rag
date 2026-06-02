"""
Tests for JDParser — rule-based job description parsing.
"""
import pytest
from app.services.jd_parser import parse_jd


# ── Sample JDs ──────────────────────────────────────────────────────────

FULL_JD = """
Senior Machine Learning Engineer

at AcmeCorp

We are looking for a talented ML engineer to join our team.

Requirements:
- 5+ years experience in machine learning and deep learning
- Proficient in Python, PyTorch, TensorFlow
- Experience with NLP and transformer models
- Strong understanding of Docker and Kubernetes
- Familiar with AWS or GCP

Responsibilities:
- Design and implement ML pipelines
- Build and deploy RAG-based applications
- Optimize model inference performance
- Collaborate with cross-functional teams

Nice to have:
- Experience with MLflow or Kubeflow
- Knowledge of Go or Rust
- Published research papers
"""

SIMPLE_JD = """
Python Developer

Requirements:
- Python
- Django
- PostgreSQL
"""

NO_SECTIONS_JD = """
We need a backend developer who knows Python and Docker.
You will build APIs and manage deployments.
"""

EMPTY_BULLETS_JD = """
Data Analyst

Responsibilities:
"""


# ── Tests ───────────────────────────────────────────────────────────────


class TestJDParserFull:
    """Tests against a well-structured JD with all sections."""

    def test_extracts_title(self):
        jd = parse_jd(FULL_JD)
        assert jd.title == "Senior Machine Learning Engineer"

    def test_extracts_company(self):
        jd = parse_jd(FULL_JD)
        assert jd.company == "AcmeCorp"

    def test_extracts_skills(self):
        jd = parse_jd(FULL_JD)
        assert "python" in jd.skills
        assert "pytorch" in jd.skills
        assert "tensorflow" in jd.skills
        assert "docker" in jd.skills
        assert "kubernetes" in jd.skills
        assert "aws" in jd.skills
        assert "gcp" in jd.skills
        assert "machine learning" in jd.skills
        assert "deep learning" in jd.skills
        assert "nlp" in jd.skills

    def test_extracts_responsibilities(self):
        jd = parse_jd(FULL_JD)
        assert len(jd.responsibilities) >= 3
        assert any("ML pipeline" in r for r in jd.responsibilities)
        assert any("RAG" in r for r in jd.responsibilities)

    def test_extracts_nice_to_haves(self):
        jd = parse_jd(FULL_JD)
        assert len(jd.nice_to_haves) >= 2
        assert any("MLflow" in n for n in jd.nice_to_haves)
        assert any("Go" in n or "Rust" in n for n in jd.nice_to_haves)

    def test_preserves_raw_text(self):
        jd = parse_jd(FULL_JD)
        assert jd.raw_text == FULL_JD


class TestJDParserSimple:
    """Tests against a minimal JD."""

    def test_extracts_title(self):
        jd = parse_jd(SIMPLE_JD)
        assert jd.title == "Python Developer"

    def test_extracts_skills(self):
        jd = parse_jd(SIMPLE_JD)
        assert "python" in jd.skills
        assert "django" in jd.skills
        assert "postgresql" in jd.skills


class TestJDParserEdgeCases:
    """Tests for edge cases."""

    def test_no_sections_finds_skills_in_body(self):
        jd = parse_jd(NO_SECTIONS_JD)
        assert "python" in jd.skills
        assert "docker" in jd.skills

    def test_empty_bullets_returns_empty_lists(self):
        jd = parse_jd(EMPTY_BULLETS_JD)
        assert jd.responsibilities == []

    def test_minimal_text(self):
        jd = parse_jd("Just some random text without any structure.")
        assert jd.raw_text == "Just some random text without any structure."
        assert isinstance(jd.skills, list)
        assert isinstance(jd.responsibilities, list)

    def test_empty_string(self):
        jd = parse_jd("")
        assert jd.raw_text == ""


class TestSkillAliases:
    """Test that skill aliases normalize correctly."""

    def test_nodejs_normalized(self):
        jd = parse_jd("Senior Dev\n\nRequirements:\n- Node.js\n- React.js")
        assert "node.js" in jd.skills
        assert "react" in jd.skills

    def test_k8s_normalized(self):
        jd = parse_jd("DevOps\n\nRequirements:\n- k8s\n- K8S")
        assert "kubernetes" in jd.skills


class TestSkillWordBoundary:
    """Single-letter skill names must not match inside other words."""

    def test_c_not_matched_in_docker(self):
        """'c' should not match as substring of 'docker'."""
        jd = parse_jd("Dev\n\nRequirements:\n- Docker\n- Python")
        assert "c" not in jd.skills

    def test_r_not_matched_in_python(self):
        """'r' should not match as substring of 'Python'."""
        jd = parse_jd("Dev\n\nRequirements:\n- Python\n- Ruby")
        assert "r" not in jd.skills
        # But "ruby" (and its alias "r") should / would match if standalone
        # In this test, "ruby" is present, "r" as standalone is not

    def test_go_matches_standalone(self):
        """'Go' as a standalone word should match."""
        jd = parse_jd("Dev\n\nRequirements:\n- Go\n- Rust")
        assert "go" in jd.skills

    def test_c_matches_standalone(self):
        """'C' as a standalone word should match."""
        jd = parse_jd("Embedded Dev\n\nRequirements:\n- C\n- C++")
        assert "c" in jd.skills
        assert "c++" in jd.skills
