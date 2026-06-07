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
    """Knowledge-base retriever (pgvector), sharing the embedding service."""
    from app.services.retrieval.pgvector_retriever import PgVectorRetriever

    return PgVectorRetriever(get_embedding_service())
