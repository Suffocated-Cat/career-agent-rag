"""Tests for BM25Retriever — lexical keyword retrieval."""
import math

from app.services.retrieval.base import (
    RetrievalResult,
    corpus_from_resume,
    tokenize,
)
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.models.resume import Resume, ResumeExperience, ResumeProject


CORPUS = [
    "Built a real-time recommendation system in PyTorch and deployed it with Docker",
    "Developed a React dashboard with TypeScript for an analytics product",
    "Led migration of services to Kubernetes and set up CI/CD pipelines",
    "Wrote data pipelines in Python using Airflow and Spark",
]


class TestTokenize:
    """Tests for the shared tokenizer."""

    def test_lowercases_and_splits(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_preserves_tech_tokens(self):
        tokens = tokenize("C++ and C# with Node.js")
        assert "c++" in tokens
        assert "c#" in tokens
        assert "node.js" in tokens

    def test_drops_punctuation(self):
        assert tokenize("...") == []
        assert tokenize("a, b; c.") == ["a", "b", "c"]

    def test_empty(self):
        assert tokenize("") == []


class TestBM25Retriever:
    """Tests for BM25 search behavior."""

    def test_returns_retrieval_results(self):
        r = BM25Retriever(CORPUS)
        results = r.search("docker", k=5)
        assert all(isinstance(res, RetrievalResult) for res in results)

    def test_finds_relevant_document(self):
        r = BM25Retriever(CORPUS)
        results = r.search("docker pytorch recommendation", k=1)
        assert len(results) == 1
        assert results[0].doc_id == 0
        assert "PyTorch" in results[0].text

    def test_ranks_by_relevance(self):
        r = BM25Retriever(CORPUS)
        results = r.search("kubernetes ci/cd pipelines", k=4)
        # The Kubernetes/CI-CD document should rank first.
        assert results[0].doc_id == 2

    def test_results_sorted_descending(self):
        r = BM25Retriever(CORPUS)
        results = r.search("python pipelines docker", k=4)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_excludes_zero_score_docs(self):
        r = BM25Retriever(CORPUS)
        # "rust" appears in no document.
        results = r.search("rust", k=4)
        assert results == []

    def test_partial_match_only_returns_matching(self):
        r = BM25Retriever(CORPUS)
        results = r.search("react", k=4)
        assert len(results) == 1
        assert results[0].doc_id == 1

    def test_k_limits_results(self):
        r = BM25Retriever(CORPUS)
        results = r.search("python docker react kubernetes", k=2)
        assert len(results) == 2

    def test_empty_corpus(self):
        r = BM25Retriever([])
        assert r.search("anything", k=5) == []

    def test_empty_query(self):
        r = BM25Retriever(CORPUS)
        assert r.search("", k=5) == []

    def test_query_with_no_known_terms(self):
        r = BM25Retriever(CORPUS)
        assert r.search("!!! ??? ...", k=5) == []

    def test_all_scores_positive(self):
        r = BM25Retriever(CORPUS)
        results = r.search("python", k=4)
        assert all(res.score > 0 for res in results)

    def test_idf_penalizes_common_terms(self):
        # "python" appears in two of these, "spark" in one. A query of the
        # rarer term should reward the document that contains it more.
        corpus = ["python data", "python web", "spark streaming"]
        r = BM25Retriever(corpus)
        results = r.search("spark", k=3)
        assert results[0].doc_id == 2

    def test_case_insensitive(self):
        r = BM25Retriever(CORPUS)
        lower = r.search("docker", k=1)
        upper = r.search("DOCKER", k=1)
        assert lower[0].doc_id == upper[0].doc_id

    def test_avgdl_computed(self):
        r = BM25Retriever(["one two", "three four five six"])
        assert math.isclose(r.avgdl, 3.0)

    def test_custom_params_change_scores(self):
        default = BM25Retriever(CORPUS)
        no_norm = BM25Retriever(CORPUS, b=0.0)
        q = "python pipelines"
        d_score = default.search(q, k=1)[0].score
        n_score = no_norm.search(q, k=1)[0].score
        assert d_score != n_score


class TestCorpusFromResume:
    """Tests for building a corpus from a parsed resume."""

    def test_experiences_and_projects_become_docs(self):
        resume = Resume(
            raw_text="...",
            experience=[
                ResumeExperience(
                    title="ML Engineer",
                    company="Acme",
                    highlights=["Built recommendation system"],
                )
            ],
            projects=[
                ResumeProject(
                    name="Chatbot",
                    description="A RAG chatbot",
                    technologies=["Python", "FAISS"],
                )
            ],
        )
        corpus = corpus_from_resume(resume)
        assert len(corpus) == 2
        assert any("ML Engineer" in doc for doc in corpus)
        assert any("Chatbot" in doc and "FAISS" in doc for doc in corpus)

    def test_empty_resume(self):
        resume = Resume(raw_text="nothing structured")
        assert corpus_from_resume(resume) == []

    def test_searchable_via_bm25(self):
        resume = Resume(
            raw_text="...",
            projects=[
                ResumeProject(
                    name="Vector Search",
                    description="Semantic search over documents",
                    technologies=["Python", "Qdrant"],
                )
            ],
        )
        r = BM25Retriever(corpus_from_resume(resume))
        results = r.search("qdrant semantic search", k=1)
        assert len(results) == 1
        assert "Vector Search" in results[0].text
