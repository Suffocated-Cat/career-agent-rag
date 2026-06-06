"""
Loaders for the retrieval evaluation dataset.

Reads the JSON fixtures into typed objects:

  - retrieval_documents.json → list[RetrievalDocument] (the searchable corpus)
  - relevance_queries.json   → list[RelevanceQuery] (queries + graded labels)

The corpus pools documents from several resumes into one index, so each query
retrieves against realistic distractors — which is what makes the ranking
metrics meaningful.
"""

import json

from dataclasses import dataclass
from pathlib import Path

from app.services.retrieval.base import RetrievalDocument


@dataclass
class RelevanceQuery:
    """A query with graded relevance judgments over corpus documents.

    Attributes:
        query_id: Stable identifier for the query.
        job_id: The job this query was derived from.
        query: The natural-language query text.
        relevance: {doc_id: grade}; grade 0 means judged non-relevant.
            Documents not listed are implicitly grade 0.
    """

    query_id: str
    job_id: str
    query: str
    relevance: dict[str, int]

    @property
    def relevant_ids(self) -> set[str]:
        """Ids with a positive relevance grade (for binary metrics)."""
        return {doc_id for doc_id, grade in self.relevance.items() if grade > 0}


@dataclass
class EvalDataset:
    """A retrieval corpus paired with its labeled queries."""

    documents: list[RetrievalDocument]
    queries: list[RelevanceQuery]


def load_documents(path: str | Path) -> list[RetrievalDocument]:
    """Load the corpus from a retrieval_documents.json file."""
    data = json.loads(Path(path).read_text())
    return [RetrievalDocument(**doc) for doc in data]


def load_queries(path: str | Path) -> list[RelevanceQuery]:
    """Load labeled queries from a relevance_queries.json file."""
    data = json.loads(Path(path).read_text())
    return [RelevanceQuery(**query) for query in data]


def load_eval_dataset(
    documents_path: str | Path,
    queries_path: str | Path,
) -> EvalDataset:
    """Load the full evaluation dataset (corpus + queries).

    Args:
        documents_path: Path to retrieval_documents.json.
        queries_path: Path to relevance_queries.json.

    Returns:
        An EvalDataset.
    """
    return EvalDataset(
        documents=load_documents(documents_path),
        queries=load_queries(queries_path),
    )
