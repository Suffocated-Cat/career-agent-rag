"""Tests for the retriever factory."""
import numpy as np
import pytest

from app.services.retrieval.base import Retriever, tokenize
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.vector_retriever import VectorRetriever
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import Reranker, RerankingRetriever
from app.services.retrieval.factory import build_retriever, METHODS


class FakeEmbeddingService:
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Trivial but valid embeddings: token-count over a tiny vocab.
        vocab = {"a": 0, "b": 1, "c": 2}
        mat = np.zeros((len(texts), 3))
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                if tok in vocab:
                    mat[i, vocab[tok]] += 1.0
        return mat


CORPUS = ["a b", "b c", "c a"]


class TestBuildRetriever:
    def test_bm25(self):
        r = build_retriever("bm25", CORPUS)
        assert isinstance(r, BM25Retriever)
        assert isinstance(r, Retriever)

    def test_vector(self):
        r = build_retriever("vector", CORPUS, embedding_service=FakeEmbeddingService())
        assert isinstance(r, VectorRetriever)

    def test_hybrid(self):
        r = build_retriever("hybrid", CORPUS, embedding_service=FakeEmbeddingService())
        assert isinstance(r, HybridRetriever)

    def test_hybrid_rerank(self):
        class _CE:
            def predict(self, pairs):
                return [0.0] * len(pairs)

        r = build_retriever(
            "hybrid+rerank",
            CORPUS,
            embedding_service=FakeEmbeddingService(),
            reranker=Reranker(cross_encoder=_CE()),
        )
        assert isinstance(r, RerankingRetriever)

    def test_bm25_needs_no_embedding(self):
        # Should not raise without an embedding service.
        build_retriever("bm25", CORPUS)

    @pytest.mark.parametrize("method", ["vector", "hybrid", "hybrid+rerank"])
    def test_vector_methods_require_embedding(self, method):
        with pytest.raises(ValueError, match="embedding_service"):
            build_retriever(method, CORPUS)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown retrieval method"):
            build_retriever("magic", CORPUS)

    def test_methods_constant_covers_branches(self):
        assert set(METHODS) == {"bm25", "vector", "hybrid", "hybrid+rerank"}
