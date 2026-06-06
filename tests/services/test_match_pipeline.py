"""Tests for the match pipeline — ranking resume items against a JD."""
import numpy as np

from app.models.jd import JobDescription
from app.models.resume import Resume, ResumeExperience, ResumeProject
from app.models.match import ProjectRelevance
from app.services.match_pipeline import build_jd_query, rank_resume_projects
from app.services.retrieval.base import tokenize


class ConceptEmbeddingService:
    """Concept-based embeddings so semantic ranking is deterministic."""

    TOKEN_TO_CONCEPT = {
        "recommendation": "ml", "personalization": "ml", "ranking": "ml",
        "pytorch": "ml", "models": "ml",
        "react": "frontend", "dashboard": "frontend", "typescript": "frontend",
        "docker": "infra", "kubernetes": "infra",
    }

    def __init__(self):
        concepts = sorted(set(self.TOKEN_TO_CONCEPT.values()))
        self.cidx = {c: i for i, c in enumerate(concepts)}

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        mat = np.zeros((len(texts), len(self.cidx)))
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                concept = self.TOKEN_TO_CONCEPT.get(tok)
                if concept is not None:
                    mat[i, self.cidx[concept]] += 1.0
        return mat


def _resume():
    return Resume(
        raw_text="...",
        experience=[
            ResumeExperience(
                title="ML Engineer", company="Acme",
                highlights=["Built recommendation ranking models in pytorch"],
            ),
            ResumeExperience(
                title="Frontend Dev", company="WebCo",
                highlights=["Built a react typescript dashboard"],
            ),
        ],
        projects=[
            ResumeProject(
                name="Infra Tooling", description="docker kubernetes deploys",
                technologies=["docker", "kubernetes"],
            ),
        ],
    )


class TestBuildJdQuery:
    def test_combines_skills_and_responsibilities(self):
        jd = JobDescription(
            raw_text="x", skills=["python", "docker"],
            responsibilities=["Build ML pipelines"],
        )
        q = build_jd_query(jd)
        assert "python" in q and "docker" in q and "Build ML pipelines" in q

    def test_falls_back_to_raw_text(self):
        jd = JobDescription(raw_text="some raw jd text")
        assert build_jd_query(jd) == "some raw jd text"


class TestRankResumeProjects:
    def test_bm25_ranks_relevant_experience_first(self):
        jd = JobDescription(
            raw_text="x", skills=["recommendation", "pytorch", "ranking"],
        )
        rels = rank_resume_projects(jd, _resume(), method="bm25")
        assert all(isinstance(r, ProjectRelevance) for r in rels)
        assert rels[0].label == "ML Engineer at Acme"
        assert rels[0].source_type == "experience"

    def test_normalized_scores_in_range_and_top_is_one(self):
        jd = JobDescription(raw_text="x", skills=["docker", "kubernetes"])
        rels = rank_resume_projects(jd, _resume(), method="bm25")
        assert rels[0].normalized_score == 1.0
        assert all(0.0 <= r.normalized_score <= 1.0 for r in rels)

    def test_doc_ids_are_stable(self):
        jd = JobDescription(raw_text="x", skills=["react", "dashboard"])
        rels = rank_resume_projects(jd, _resume(), method="bm25")
        assert rels[0].doc_id == "exp:1"  # the frontend experience

    def test_vector_method_uses_embeddings(self):
        # Query has no literal corpus words, only a paraphrase concept ("ml").
        jd = JobDescription(raw_text="x", skills=["personalization"])
        rels = rank_resume_projects(
            jd, _resume(), embedding_service=ConceptEmbeddingService(),
            method="vector",
        )
        assert rels[0].doc_id == "exp:0"  # semantic match to ML experience

    def test_project_is_labeled_by_name(self):
        jd = JobDescription(raw_text="x", skills=["docker", "kubernetes"])
        rels = rank_resume_projects(jd, _resume(), method="bm25")
        proj = next(r for r in rels if r.source_type == "project")
        assert proj.label == "Infra Tooling"

    def test_experience_label_without_company(self):
        resume = Resume(
            raw_text="...",
            experience=[
                ResumeExperience(
                    title="Freelance Engineer", company="",
                    highlights=["Shipped a graphql gateway"],
                )
            ],
        )
        jd = JobDescription(raw_text="x", skills=["graphql"])
        rels = rank_resume_projects(jd, resume, method="bm25")
        assert rels[0].label == "Freelance Engineer"

    def test_empty_resume_returns_empty(self):
        jd = JobDescription(raw_text="x", skills=["python"])
        assert rank_resume_projects(jd, Resume(raw_text="empty"), method="bm25") == []

    def test_no_matches_returns_empty(self):
        # BM25 drops zero-score docs; an off-corpus query yields nothing.
        jd = JobDescription(raw_text="x", skills=["rust", "haskell"])
        assert rank_resume_projects(jd, _resume(), method="bm25") == []
