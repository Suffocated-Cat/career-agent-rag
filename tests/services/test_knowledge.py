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


def test_all_entries_enriched():
    docs = load_kb_documents()
    for d in docs:
        assert d.metadata.get("role"), f"{d.id} missing role"
        assert d.metadata.get("difficulty"), f"{d.id} missing difficulty"
        assert d.metadata.get("tags"), f"{d.id} missing tags"
        assert d.metadata.get("answer_outline"), f"{d.id} missing answer_outline"


def test_optional_fields_skipped_when_absent():
    # The loader only adds enriched keys when present in the source entry.
    import json
    import tempfile
    from pathlib import Path

    raw = [{"id": "x:1", "skill": "x", "type": "question", "text": "bare entry"}]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "kb.json"
        p.write_text(json.dumps(raw))
        doc = load_kb_documents(p)[0]
    assert set(doc.metadata) == {"skill", "type"}
