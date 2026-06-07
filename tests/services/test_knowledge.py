"""Tests for knowledge-base loading and the in-memory KB retriever."""
from app.services.knowledge import build_inmemory_kb_retriever, load_kb_documents
from app.services.retrieval.base import RetrievalDocument


def test_load_kb_documents():
    docs = load_kb_documents()
    assert len(docs) >= 10
    assert all(isinstance(d, RetrievalDocument) for d in docs)
    assert all(d.metadata.get("skill") for d in docs)
    assert all(d.text for d in docs)


def test_kb_retriever_finds_topic():
    retriever = build_inmemory_kb_retriever()  # BM25 (no embeddings)
    results = retriever.search("docker image layers and build caching", k=3)
    assert results
    assert "docker" in results[0].text.lower()


def test_kb_retriever_off_topic_returns_nothing():
    retriever = build_inmemory_kb_retriever()
    assert retriever.search("zzz_nonexistent_topic", k=3) == []


def test_enriched_fields_carried_into_metadata():
    docs = {d.id: d for d in load_kb_documents()}
    fa = docs["fastapi:q1"]
    assert fa.metadata["difficulty"] == "mid"
    assert "backend" in fa.metadata["role"]
    assert "pydantic" in fa.metadata["tags"]
    assert fa.metadata["answer_outline"]


def test_basic_entry_has_only_core_metadata():
    docs = {d.id: d for d in load_kb_documents()}
    basic = docs["python:q1"]  # not enriched
    assert set(basic.metadata) == {"skill", "type"}
