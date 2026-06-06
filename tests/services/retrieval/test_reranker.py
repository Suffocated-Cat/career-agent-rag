"""Tests for Reranker and RerankingRetriever — cross-encoder re-scoring."""
import pytest

from app.core.config import settings
from app.services.retrieval.base import RetrievalResult, Retriever
from app.services.retrieval.reranker import Reranker, RerankingRetriever


class FakeCrossEncoder:
    """Deterministic stand-in for a sentence-transformers CrossEncoder.

    Scores each (query, document) pair by looking the document text up in a
    provided map, so reranking order is fully controllable in tests.
    """

    def __init__(self, *args, score_map: dict[str, float] | None = None, **kwargs):
        self.score_map = score_map or {}
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.predict_calls.append(list(pairs))
        return [self.score_map.get(doc, 0.0) for _query, doc in pairs]


def _candidates() -> list[RetrievalResult]:
    # Upstream order (by score): doc0, doc1, doc2.
    return [
        RetrievalResult(0, "react dashboard", 0.9),
        RetrievalResult(1, "docker kubernetes pipeline", 0.6),
        RetrievalResult(2, "python data engineering", 0.3),
    ]


class TestReranker:
    """Tests for the Reranker re-scoring stage."""

    def test_returns_retrieval_results(self):
        ce = FakeCrossEncoder(score_map={"react dashboard": 1.0})
        r = Reranker(cross_encoder=ce)
        results = r.rerank("anything", _candidates(), k=3)
        assert all(isinstance(res, RetrievalResult) for res in results)

    def test_reorders_by_cross_encoder_score(self):
        # Upstream ranked doc0 first, but the cross-encoder prefers doc2.
        ce = FakeCrossEncoder(
            score_map={
                "react dashboard": 0.1,
                "docker kubernetes pipeline": 0.4,
                "python data engineering": 0.95,
            }
        )
        r = Reranker(cross_encoder=ce)
        results = r.rerank("data pipelines in python", _candidates(), k=3)
        assert [res.doc_id for res in results] == [2, 1, 0]

    def test_carries_cross_encoder_score(self):
        ce = FakeCrossEncoder(score_map={"react dashboard": 0.77})
        r = Reranker(cross_encoder=ce)
        top = r.rerank("ui work", _candidates(), k=1)[0]
        assert top.doc_id == 0
        assert top.score == pytest.approx(0.77)

    def test_predict_receives_query_doc_pairs(self):
        ce = FakeCrossEncoder()
        r = Reranker(cross_encoder=ce)
        r.rerank("my query", _candidates(), k=3)
        pairs = ce.predict_calls[0]
        assert pairs[0] == ("my query", "react dashboard")
        assert len(pairs) == 3

    def test_k_limits_results(self):
        ce = FakeCrossEncoder()
        r = Reranker(cross_encoder=ce)
        results = r.rerank("q", _candidates(), k=2)
        assert len(results) == 2

    def test_empty_candidates(self):
        ce = FakeCrossEncoder()
        r = Reranker(cross_encoder=ce)
        assert r.rerank("q", [], k=5) == []
        # No model work should happen for an empty candidate set.
        assert ce.predict_calls == []

    def test_ties_broken_by_doc_id(self):
        ce = FakeCrossEncoder()  # everything scores 0.0
        r = Reranker(cross_encoder=ce)
        results = r.rerank("q", _candidates(), k=3)
        assert [res.doc_id for res in results] == [0, 1, 2]

    def test_defaults_from_settings(self):
        r = Reranker()
        assert r.model_name == settings.RERANKER_MODEL_NAME
        assert r.device == settings.RERANKER_DEVICE
        # Model is not loaded until first use.
        assert r._model is None

    def test_injected_encoder_skips_lazy_load(self):
        ce = FakeCrossEncoder()
        r = Reranker(cross_encoder=ce)
        assert r._get_model() is ce

    def test_lazy_load_constructs_model_once(self, monkeypatch):
        import sentence_transformers

        constructed: list[tuple] = []

        class _FakeCE(FakeCrossEncoder):
            def __init__(self, *args, **kwargs):
                constructed.append((args, kwargs))
                super().__init__()

        monkeypatch.setattr(sentence_transformers, "CrossEncoder", _FakeCE)

        r = Reranker(model_name="some/model", device="cpu")
        m1 = r._get_model()
        m2 = r._get_model()
        assert m1 is m2  # cached
        assert len(constructed) == 1  # built only once
        assert constructed[0][0] == ("some/model",)


class FakeBaseRetriever:
    """A base retriever that records the k it was asked for."""

    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.last_k: int | None = None

    def search(self, query: str, k: int = 10) -> list[RetrievalResult]:
        self.last_k = k
        return self.results[:k]


class TestRerankingRetriever:
    """Tests for the base-retriever + reranker composition."""

    def test_satisfies_retriever_protocol(self):
        rr = RerankingRetriever(
            FakeBaseRetriever(_candidates()), Reranker(cross_encoder=FakeCrossEncoder())
        )
        assert isinstance(rr, Retriever)

    def test_recalls_pool_then_reranks(self):
        base = FakeBaseRetriever(_candidates())
        ce = FakeCrossEncoder(
            score_map={"python data engineering": 0.99}
        )
        rr = RerankingRetriever(base, Reranker(cross_encoder=ce), candidate_pool=20)
        results = rr.search("data work", k=2)
        # Base was asked for the candidate pool, not the final k.
        assert base.last_k == 20
        # Reranker promoted doc2 to the top.
        assert results[0].doc_id == 2
        assert len(results) == 2

    def test_passes_query_through(self):
        base = FakeBaseRetriever(_candidates())
        ce = FakeCrossEncoder()
        rr = RerankingRetriever(base, Reranker(cross_encoder=ce))
        rr.search("specific query", k=3)
        assert ce.predict_calls[0][0][0] == "specific query"

    def test_empty_base_results(self):
        base = FakeBaseRetriever([])
        ce = FakeCrossEncoder()
        rr = RerankingRetriever(base, Reranker(cross_encoder=ce))
        assert rr.search("q", k=5) == []
