"""
Evaluation runner — score a retriever against a labeled query set.

For each query it runs the retriever over the corpus, maps the integer
``doc_id`` results back to stable document ids, and computes recall@k, MRR,
and nDCG@k against the query's relevance judgments. Per-query results are
averaged into an EvalReport.

This is the harness behind project-relevance evaluation and the Day-14
ablations (run the same dataset through different retrieval methods).
"""

from dataclasses import dataclass

from app.eval.datasets import RelevanceQuery
from app.eval.metrics import mrr, ndcg_at_k, recall_at_k
from app.services.retrieval.base import RetrievalDocument, Retriever


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    query_id: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


@dataclass
class EvalReport:
    """Aggregated metrics over a query set at cutoff k."""

    k: int
    per_query: list[QueryMetrics]
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float


def evaluate_retriever(
    retriever: Retriever,
    documents: list[RetrievalDocument],
    queries: list[RelevanceQuery],
    k: int = 10,
) -> EvalReport:
    """Evaluate *retriever* over *documents* against labeled *queries*.

    The retriever must have been built over ``document_texts(documents)`` so
    that result ``doc_id`` values index into *documents*.

    Args:
        retriever: A retriever to evaluate.
        documents: The corpus, index-aligned with the retriever's texts.
        queries: Labeled queries to score against.
        k: Cutoff rank for recall@k and nDCG@k.

    Returns:
        An EvalReport with per-query and mean metrics.
    """
    id_by_index = [doc.id for doc in documents]

    per_query: list[QueryMetrics] = []
    for query in queries:
        results = retriever.search(query.query, k=len(documents))
        ranked_ids = [id_by_index[r.doc_id] for r in results]
        per_query.append(
            QueryMetrics(
                query_id=query.query_id,
                recall_at_k=recall_at_k(ranked_ids, query.relevant_ids, k),
                mrr=mrr(ranked_ids, query.relevant_ids),
                ndcg_at_k=ndcg_at_k(ranked_ids, query.relevance, k),
            )
        )

    n = len(per_query) or 1
    return EvalReport(
        k=k,
        per_query=per_query,
        mean_recall_at_k=sum(m.recall_at_k for m in per_query) / n,
        mean_mrr=sum(m.mrr for m in per_query) / n,
        mean_ndcg_at_k=sum(m.ndcg_at_k for m in per_query) / n,
    )
