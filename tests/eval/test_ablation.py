"""Tests for the retrieval ablation harness."""
import numpy as np

from app.eval.ablation import (
    AblationResult,
    format_ablation_table,
    run_ablation,
)
from app.eval.datasets import EvalDataset, RelevanceQuery
from app.eval.runner import EvalReport
from app.services.retrieval.base import RetrievalDocument, tokenize


class FakeEmbeddingService:
    """Concept-ish embeddings so vector/hybrid produce non-trivial rankings."""

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vocab = {"python": 0, "docker": 1, "react": 2, "rag": 3}
        mat = np.zeros((len(texts), len(vocab)))
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                if tok in vocab:
                    mat[i, vocab[tok]] += 1.0
        return mat


def _dataset() -> EvalDataset:
    docs = [
        RetrievalDocument(id="d0", text="python docker service", source_type="x",
                          source_index=0, metadata={}),
        RetrievalDocument(id="d1", text="react frontend", source_type="x",
                          source_index=1, metadata={}),
        RetrievalDocument(id="d2", text="rag retrieval pipeline", source_type="x",
                          source_index=2, metadata={}),
    ]
    queries = [
        RelevanceQuery("q0", "j", "python docker", {"d0": 3}),
        RelevanceQuery("q1", "j", "rag pipeline", {"d2": 3}),
    ]
    return EvalDataset(documents=docs, queries=queries)


class TestRunAblation:
    def test_bm25_only(self):
        results = run_ablation(_dataset(), methods=("bm25",), k=3)
        assert len(results) == 1
        assert isinstance(results[0], AblationResult)
        assert results[0].method == "bm25"
        assert isinstance(results[0].report, EvalReport)

    def test_multiple_methods(self):
        results = run_ablation(
            _dataset(),
            methods=("bm25", "vector", "hybrid"),
            k=3,
            embedding_service=FakeEmbeddingService(),
        )
        assert [r.method for r in results] == ["bm25", "vector", "hybrid"]
        for r in results:
            assert 0.0 <= r.report.mean_ndcg_at_k <= 1.0

    def test_bm25_recovers_exact_terms(self):
        results = run_ablation(_dataset(), methods=("bm25",), k=3)
        # Both queries use exact terms present in exactly one doc → perfect.
        assert results[0].report.mean_recall_at_k == 1.0
        assert results[0].report.mean_mrr == 1.0


class TestFormatTable:
    def test_markdown_table(self):
        results = run_ablation(_dataset(), methods=("bm25",), k=5)
        table = format_ablation_table(results, k=5)
        assert "| Method | Recall@5 | MRR | nDCG@5 |" in table
        assert "| bm25 |" in table
        # One header sep + header + one row.
        assert table.count("\n") == 2
