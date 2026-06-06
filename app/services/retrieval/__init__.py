"""
Retrieval package — search a corpus of resume experiences/projects with a
JD-derived query.

Provides a common Retriever interface so different backends are swappable:

  - BM25Retriever   — keyword / lexical recall
  - VectorRetriever — semantic recall via embeddings
  - HybridRetriever — fusion of lexical and semantic recall
  - Reranker        — cross-encoder re-scoring of top-k

Each retriever takes a corpus (list of strings) at construction time and
exposes ``search(query, k) -> list[RetrievalResult]``.
"""

from app.services.retrieval.base import (
    Retriever,
    RetrievalResult,
    corpus_from_resume,
)
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.vector_retriever import VectorRetriever

__all__ = [
    "Retriever",
    "RetrievalResult",
    "corpus_from_resume",
    "BM25Retriever",
    "VectorRetriever",
]
