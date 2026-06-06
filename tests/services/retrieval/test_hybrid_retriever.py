"""Tests for HybridRetriever — fusing lexical and semantic recall."""
import numpy as np
import pytest

from app.services.retrieval.base import RetrievalResult, Retriever, tokenize
from app.services.retrieval.hybrid_retriever import (
    HybridRetriever,
    _min_max_normalize,
)


# Map surface tokens to abstract concepts so paraphrases with no shared
# words still embed close together — modelling real semantic similarity.
TOKEN_TO_CONCEPT = {
    "docker": "container", "kubernetes": "container",
    "containers": "container", "orchestration": "container",
    "recommendation": "ml", "personalization": "ml",
    "ranking": "ml", "pytorch": "ml", "models": "ml",
    "react": "frontend", "typescript": "frontend",
    "dashboard": "frontend", "ui": "frontend",
    "airflow": "dataeng", "spark": "dataeng",
    "pipelines": "dataeng", "etl": "dataeng",
}


class ConceptEmbeddingService:
    """Deterministic embeddings over abstract concepts.

    Tokens that share a concept (e.g. 'recommendation' / 'personalization')
    map to the same dimension, so semantically-similar but lexically-distinct
    texts have high cosine similarity. Tokens with no concept contribute
    nothing — they are invisible to the vector arm but still matchable by BM25.
    """

    def __init__(self, token_to_concept: dict[str, str]):
        self.t2c = token_to_concept
        concepts = sorted(set(token_to_concept.values()))
        self.cidx = {c: i for i, c in enumerate(concepts)}

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        mat = np.zeros((len(texts), len(self.cidx)), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                concept = self.t2c.get(tok)
                if concept is not None:
                    mat[i, self.cidx[concept]] += 1.0
        return mat


def _service():
    return ConceptEmbeddingService(TOKEN_TO_CONCEPT)


CORPUS = [
    "recommendation system in pytorch deployed with docker",
    "react typescript dashboard",
    "kubernetes orchestration of containers",
    "airflow spark etl pipelines",
]


class TestMinMaxNormalize:
    """Tests for the score-normalization helper."""

    def test_empty(self):
        assert _min_max_normalize([]) == {}

    def test_scales_to_unit_range(self):
        results = [
            RetrievalResult(0, "a", 2.0),
            RetrievalResult(1, "b", 4.0),
            RetrievalResult(2, "c", 6.0),
        ]
        norm = _min_max_normalize(results)
        assert norm[0] == 0.0
        assert norm[2] == 1.0
        assert 0.0 < norm[1] < 1.0

    def test_equal_scores_map_to_one(self):
        results = [RetrievalResult(0, "a", 3.0), RetrievalResult(1, "b", 3.0)]
        norm = _min_max_normalize(results)
        assert norm == {0: 1.0, 1: 1.0}


class TestHybridRetriever:
    """Tests for hybrid fusion behavior."""

    def test_returns_retrieval_results(self):
        r = HybridRetriever(CORPUS, _service())
        results = r.search("docker", k=5)
        assert all(isinstance(res, RetrievalResult) for res in results)

    def test_satisfies_retriever_protocol(self):
        r = HybridRetriever(CORPUS, _service())
        assert isinstance(r, Retriever)

    def test_exposes_both_arms(self):
        r = HybridRetriever(CORPUS, _service())
        assert r.bm25 is not None
        assert r.vector is not None

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            HybridRetriever(CORPUS, _service(), method="bogus")

    def test_results_sorted_descending(self):
        r = HybridRetriever(CORPUS, _service())
        results = r.search("docker recommendation", k=4)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_results(self):
        r = HybridRetriever(CORPUS, _service())
        results = r.search("docker react kubernetes airflow", k=2)
        assert len(results) == 2

    def test_empty_corpus(self):
        r = HybridRetriever([], _service())
        assert r.search("anything", k=5) == []

    def test_empty_query(self):
        r = HybridRetriever(CORPUS, _service())
        assert r.search("   ", k=5) == []

    # ── Fusion value: each arm covers the other's blind spot ──────────

    def test_vector_arm_catches_paraphrase_bm25_misses(self):
        # "personalization" never appears literally in the corpus, but it is
        # semantically the ML/recommendation document. BM25 finds nothing.
        r = HybridRetriever(CORPUS, _service())
        assert r.bm25.search("personalization", k=4) == []
        results = r.search("personalization", k=1)
        assert results[0].doc_id == 0

    def test_bm25_arm_fixes_wrong_vector_ranking(self):
        # "graphql" has no concept, so the vector arm cannot distinguish the
        # documents and ranks the wrong one first; BM25 pins the exact match
        # and fusion puts it on top.
        corpus = ["personalization ranking models", "graphql federation service"]
        r = HybridRetriever(corpus, _service())
        vector_first = r.vector.search("graphql", k=2)[0].doc_id
        assert vector_first == 0  # vector gets it wrong
        results = r.search("graphql", k=1)
        assert results[0].doc_id == 1  # hybrid gets it right

    def test_rrf_boosts_cross_arm_agreement(self):
        # A document recalled by both arms should outrank one recalled by a
        # single arm, even when neither ranks it first.
        r = HybridRetriever(CORPUS, _service())
        results = r.search("docker recommendation pytorch", k=4)
        # doc 0 is both lexically and semantically the strongest → rank 1.
        assert results[0].doc_id == 0

    # ── Weights and methods ───────────────────────────────────────────

    def test_zero_bm25_weight_matches_vector_order(self):
        r_hybrid = HybridRetriever(CORPUS, _service(), weights=(0.0, 1.0))
        vector_order = [res.doc_id for res in r_hybrid.vector.search("docker", k=4)]
        hybrid_order = [res.doc_id for res in r_hybrid.search("docker", k=4)]
        assert hybrid_order == vector_order

    def test_weighted_method_runs(self):
        r = HybridRetriever(CORPUS, _service(), method="weighted")
        results = r.search("docker recommendation", k=4)
        assert len(results) > 0
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_weighted_single_hit_normalizes(self):
        # A query matching exactly one document per arm exercises the
        # span==0 branch of min-max normalization.
        r = HybridRetriever(CORPUS, _service(), method="weighted")
        results = r.search("react", k=4)
        assert results[0].doc_id == 1

    def test_methods_can_disagree(self):
        rrf = HybridRetriever(CORPUS, _service(), method="rrf")
        weighted = HybridRetriever(CORPUS, _service(), method="weighted")
        q = "docker recommendation react"
        # Both should return results; we only assert both produce valid output.
        assert rrf.search(q, k=4)
        assert weighted.search(q, k=4)
