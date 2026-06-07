"""Tests for the shared API dependency singletons."""
from app.api import deps
from app.services.llm_client import LLMClient


def test_get_llm_builds_client():
    """get_llm constructs a real LLMClient (offline; no network on build)."""
    deps.get_llm.cache_clear()
    assert isinstance(deps.get_llm(), LLMClient)
    deps.get_llm.cache_clear()


def test_get_kb_retriever_builds(monkeypatch):
    """get_kb_retriever builds a PgVectorRetriever (no DB at construction)."""
    from app.services.retrieval.pgvector_retriever import PgVectorRetriever

    monkeypatch.setattr(deps, "get_embedding_service", lambda: None)
    deps.get_kb_retriever.cache_clear()
    assert isinstance(deps.get_kb_retriever(), PgVectorRetriever)
    deps.get_kb_retriever.cache_clear()


def test_get_embedding_service_success(monkeypatch):
    monkeypatch.setattr(deps, "EmbeddingService", lambda: "EMB")
    deps.get_embedding_service.cache_clear()
    assert deps.get_embedding_service() == "EMB"
    deps.get_embedding_service.cache_clear()


def test_get_embedding_service_fallback_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(deps, "EmbeddingService", _boom)
    deps.get_embedding_service.cache_clear()
    assert deps.get_embedding_service() is None
    deps.get_embedding_service.cache_clear()
