"""
Shared API dependencies — lazily-built singletons used across endpoints.

Both the embedding service and the LLM client are expensive to construct, so
they are created once (``lru_cache``) and reused. Endpoints call these instead
of building their own, keeping construction in one place.
"""

from functools import lru_cache

from app.services.embedding import EmbeddingService
from app.services.llm_client import LLMClient


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService | None:
    """Create the embedding service once; fall back to None if unavailable."""
    try:
        return EmbeddingService()
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """Create the LLM client once. Unconfigured → deterministic fallbacks apply."""
    return LLMClient()


@lru_cache(maxsize=1)
def get_kb_retriever():
    """Knowledge-base retriever (pgvector) behind a cross-encoder reranker.

    pgvector recalls a candidate pool by cosine similarity; the cross-encoder
    then re-scores it for precision. The reranker forwards metadata filters
    (role/difficulty) and preserves each candidate's metadata, so downstream
    grounding on ``answer_outline`` still works.
    """
    from app.services.retrieval.pgvector_retriever import PgVectorRetriever
    from app.services.retrieval.reranker import Reranker, RerankingRetriever

    base = PgVectorRetriever(get_embedding_service())
    return RerankingRetriever(base, Reranker(), candidate_pool=30)
