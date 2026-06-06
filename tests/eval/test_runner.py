"""Tests for the evaluation runner over the labeled fixtures (BM25)."""
from pathlib import Path

from app.eval.datasets import RelevanceQuery, load_eval_dataset
from app.eval.runner import evaluate_retriever, EvalReport
from app.services.retrieval.base import RetrievalDocument, document_texts
from app.services.retrieval.factory import build_retriever

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _dataset():
    return load_eval_dataset(
        FIXTURES / "retrieval_documents.json",
        FIXTURES / "relevance_queries.json",
    )


def _bm25(documents):
    return build_retriever("bm25", document_texts(documents))


class TestEvaluateRetriever:
    def test_returns_report_with_per_query(self):
        ds = _dataset()
        report = evaluate_retriever(_bm25(ds.documents), ds.documents, ds.queries, k=10)
        assert isinstance(report, EvalReport)
        assert len(report.per_query) == len(ds.queries)

    def test_means_in_unit_range(self):
        ds = _dataset()
        report = evaluate_retriever(_bm25(ds.documents), ds.documents, ds.queries, k=10)
        for value in (report.mean_recall_at_k, report.mean_mrr, report.mean_ndcg_at_k):
            assert 0.0 <= value <= 1.0

    def test_bm25_is_a_strong_baseline(self):
        # The fixtures are full of exact tech terms, so lexical search ranks
        # the right documents near the top.
        ds = _dataset()
        report = evaluate_retriever(_bm25(ds.documents), ds.documents, ds.queries, k=10)
        assert report.mean_mrr > 0.8
        assert report.mean_ndcg_at_k > 0.5

    def test_recall_is_monotonic_in_k(self):
        ds = _dataset()
        r3 = evaluate_retriever(_bm25(ds.documents), ds.documents, ds.queries, k=3)
        r10 = evaluate_retriever(_bm25(ds.documents), ds.documents, ds.queries, k=10)
        assert r10.mean_recall_at_k >= r3.mean_recall_at_k

    def test_cv_query_ranks_cv_docs(self):
        ds = _dataset()
        report = evaluate_retriever(_bm25(ds.documents), ds.documents, ds.queries, k=5)
        cv = next(m for m in report.per_query if m.query_id == "q_computer_vision_segmentation")
        assert cv.recall_at_k == 1.0
        assert cv.mrr == 1.0
        assert cv.ndcg_at_k > 0.9

    def test_maps_doc_id_back_to_stable_id(self):
        # A single-document corpus: the result's stable id must come through.
        docs = [
            RetrievalDocument(
                id="exp:0", text="pytorch computer vision segmentation",
                source_type="experience", source_index=0, metadata={},
            )
        ]
        queries = [
            RelevanceQuery("q", "j", "pytorch segmentation", {"exp:0": 3})
        ]
        report = evaluate_retriever(_bm25(docs), docs, queries, k=5)
        assert report.mean_recall_at_k == 1.0
        assert report.mean_mrr == 1.0

    def test_empty_queries(self):
        ds = _dataset()
        report = evaluate_retriever(_bm25(ds.documents), ds.documents, [], k=10)
        assert report.per_query == []
        assert report.mean_recall_at_k == 0.0
