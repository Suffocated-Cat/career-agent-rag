"""
Retriever factory — build any retrieval backend by name over a text corpus.

A single entry point so the rest of the app (and the evaluation harness) can
switch backends with a string, which is what powers ablations:

    "bm25"           lexical only
    "vector"         semantic only
    "hybrid"         BM25 + vector fused (RRF)
    "hybrid+rerank"  hybrid recall, then cross-encoder rerank
"""

from app.services.retrieval.base import Retriever
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.vector_retriever import VectorRetriever
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import Reranker, RerankingRetriever

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService

METHODS = ("bm25", "vector", "hybrid", "hybrid+rerank")


def build_retriever(
    method: str,
    corpus: list[str],
    embedding_service: "EmbeddingService | None" = None,
    reranker: Reranker | None = None,
    candidate_pool: int = 20,
) -> Retriever:
    """Construct a retriever for *method* over *corpus*.

    Args:
        method: One of METHODS.
        corpus: Document texts (e.g. from ``document_texts``).
        embedding_service: Required for any method using the vector arm
            ("vector", "hybrid", "hybrid+rerank").
        reranker: Reranker to use for "hybrid+rerank"; one is created lazily
            if not supplied.
        candidate_pool: Recall depth before reranking, for "hybrid+rerank".

    Returns:
        A retriever implementing the standard ``search(query, k)`` interface.

    Raises:
        ValueError: If *method* is unknown, or a vector-based method is
            requested without an embedding_service.
    """
    if method == "bm25":
        return BM25Retriever(corpus)

    if method in ("vector", "hybrid", "hybrid+rerank") and embedding_service is None:
        raise ValueError(f"method {method!r} requires an embedding_service.")

    if method == "vector":
        return VectorRetriever(corpus, embedding_service)

    if method == "hybrid":
        return HybridRetriever(corpus, embedding_service)

    if method == "hybrid+rerank":
        base = HybridRetriever(corpus, embedding_service)
        return RerankingRetriever(
            base, reranker or Reranker(), candidate_pool=candidate_pool
        )

    raise ValueError(f"Unknown retrieval method {method!r}; expected one of {METHODS}.")
