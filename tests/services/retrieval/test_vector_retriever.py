"""Tests for VectorRetriever — embedding-based semantic retrieval."""
import numpy as np

from app.services.retrieval.base import RetrievalResult, Retriever, tokenize
from app.services.retrieval.vector_retriever import VectorRetriever, _normalize_rows


class FakeEmbeddingService:
    """Deterministic bag-of-words embeddings for predictable ranking.

    Each text is embedded as a count vector over a fixed vocabulary, so
    cosine similarity reflects word overlap. This lets tests assert exact
    rankings without depending on a real model.
    """

    def __init__(self, vocab: list[str]):
        self.vocab = {w: i for i, w in enumerate(vocab)}
        self.encode_calls: list[list[str]] = []

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        self.encode_calls.append(list(texts))
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                if tok in self.vocab:
                    mat[i, self.vocab[tok]] += 1.0
        return mat


VOCAB = [
    "docker", "pytorch", "recommendation", "react", "typescript",
    "kubernetes", "ci", "cd", "python", "airflow", "spark", "dashboard",
]

CORPUS = [
    "recommendation system in pytorch deployed with docker",
    "react dashboard with typescript",
    "kubernetes ci cd pipelines",
    "python airflow spark pipelines",
]


def _service():
    return FakeEmbeddingService(VOCAB)


class TestNormalizeRows:
    """Tests for the row-normalization helper."""

    def test_unit_norm(self):
        m = _normalize_rows(np.array([[3.0, 4.0]]))
        assert np.isclose(np.linalg.norm(m[0]), 1.0)

    def test_zero_row_safe(self):
        m = _normalize_rows(np.array([[0.0, 0.0]]))
        assert not np.any(np.isnan(m))


class TestVectorRetriever:
    """Tests for semantic search behavior."""

    def test_returns_retrieval_results(self):
        r = VectorRetriever(CORPUS, _service())
        results = r.search("docker", k=5)
        assert all(isinstance(res, RetrievalResult) for res in results)

    def test_satisfies_retriever_protocol(self):
        r = VectorRetriever(CORPUS, _service())
        assert isinstance(r, Retriever)

    def test_finds_relevant_document(self):
        r = VectorRetriever(CORPUS, _service())
        results = r.search("docker pytorch recommendation", k=1)
        assert results[0].doc_id == 0

    def test_ranks_by_similarity(self):
        r = VectorRetriever(CORPUS, _service())
        results = r.search("kubernetes ci cd", k=4)
        assert results[0].doc_id == 2

    def test_results_sorted_descending(self):
        r = VectorRetriever(CORPUS, _service())
        results = r.search("python pipelines docker", k=4)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_corpus_encoded_once_at_construction(self):
        svc = _service()
        VectorRetriever(CORPUS, svc)
        # Single batch encode of the whole corpus, no per-query work yet.
        assert len(svc.encode_calls) == 1
        assert svc.encode_calls[0] == CORPUS

    def test_query_encoded_per_search(self):
        svc = _service()
        r = VectorRetriever(CORPUS, svc)
        r.search("docker", k=1)
        assert len(svc.encode_calls) == 2

    def test_k_limits_results(self):
        r = VectorRetriever(CORPUS, _service())
        results = r.search("docker react kubernetes python", k=2)
        assert len(results) == 2

    def test_min_score_filters(self):
        r = VectorRetriever(CORPUS, _service())
        # An off-vocabulary query yields all-zero similarity; min_score drops it.
        results = r.search("nonsense words here", k=4, min_score=0.01)
        assert results == []

    def test_no_min_score_returns_topk_regardless(self):
        r = VectorRetriever(CORPUS, _service())
        results = r.search("nonsense words here", k=4)
        # Without a floor, top-k are still returned (all zero-score here).
        assert len(results) == 4

    def test_empty_corpus(self):
        r = VectorRetriever([], _service())
        assert r.search("anything", k=5) == []

    def test_empty_query(self):
        r = VectorRetriever(CORPUS, _service())
        assert r.search("   ", k=5) == []

    def test_scores_are_floats(self):
        r = VectorRetriever(CORPUS, _service())
        results = r.search("docker", k=1)
        assert isinstance(results[0].score, float)
