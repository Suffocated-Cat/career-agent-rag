"""Tests for the evaluation dataset loaders against the JSON fixtures."""
from pathlib import Path

from app.eval.datasets import (
    RelevanceQuery,
    load_documents,
    load_eval_dataset,
    load_queries,
)
from app.services.retrieval.base import RetrievalDocument

FIXTURES = Path(__file__).parent.parent / "fixtures"
DOCS_PATH = FIXTURES / "retrieval_documents.json"
QUERIES_PATH = FIXTURES / "relevance_queries.json"


class TestLoaders:
    def test_load_documents(self):
        docs = load_documents(DOCS_PATH)
        assert len(docs) == 25
        assert all(isinstance(d, RetrievalDocument) for d in docs)
        assert len({d.id for d in docs}) == len(docs)  # unique ids

    def test_load_queries(self):
        queries = load_queries(QUERIES_PATH)
        assert len(queries) == 12
        assert all(isinstance(q, RelevanceQuery) for q in queries)

    def test_load_eval_dataset(self):
        ds = load_eval_dataset(DOCS_PATH, QUERIES_PATH)
        assert len(ds.documents) == 25
        assert len(ds.queries) == 12

    def test_relevant_ids_excludes_zero_grades(self):
        q = RelevanceQuery(
            query_id="q",
            job_id="j",
            query="text",
            relevance={"a": 3, "b": 0, "c": 1},
        )
        assert q.relevant_ids == {"a", "c"}

    def test_all_labels_resolve_to_corpus(self):
        ds = load_eval_dataset(DOCS_PATH, QUERIES_PATH)
        corpus_ids = {d.id for d in ds.documents}
        for q in ds.queries:
            assert q.relevance.keys() <= corpus_ids
