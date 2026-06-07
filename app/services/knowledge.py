"""
Knowledge base loading and retriever construction.

The KB (interview questions + skill notes) is a curated document corpus. It can
be searched two ways behind the same ``Retriever`` interface:

  - in-memory (BM25 or vector) — used for tests and offline runs,
  - pgvector — the persistent store used in production (built in ``deps``).

This module loads the KB documents and builds the in-memory retriever; the
ingestion script embeds and upserts the same documents into pgvector.
"""

import json

from pathlib import Path

from app.services.retrieval.base import (
    RetrievalDocument,
    Retriever,
    document_texts,
)
from app.services.retrieval.factory import build_retriever

DEFAULT_KB_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "knowledge"
    / "interview_questions.json"
)


# Optional enrichment fields carried through into metadata for filtering.
_METADATA_FIELDS = ("role", "difficulty", "tags", "answer_outline")


def load_kb_documents(path: str | Path | None = None) -> list[RetrievalDocument]:
    """Load knowledge-base entries as RetrievalDocuments.

    ``skill`` and ``type`` are always present; ``role`` / ``difficulty`` /
    ``tags`` / ``answer_outline`` are optional and, when present, carried into
    metadata so pgvector can do metadata-filtered semantic search.
    """
    data = json.loads(Path(path or DEFAULT_KB_PATH).read_text())
    docs: list[RetrievalDocument] = []
    for i, entry in enumerate(data):
        metadata = {"skill": entry.get("skill", ""), "type": entry.get("type", "question")}
        for field in _METADATA_FIELDS:
            value = entry.get(field)
            if value not in (None, "", []):
                metadata[field] = value
        docs.append(
            RetrievalDocument(
                id=entry["id"],
                text=entry["text"],
                source_type=entry.get("type", "question"),
                source_index=i,
                metadata=metadata,
            )
        )
    return docs


def build_inmemory_kb_retriever(
    embedding_service=None,
    path: str | Path | None = None,
) -> Retriever:
    """Build an in-memory retriever over the KB (vector if embeddings, else BM25)."""
    docs = load_kb_documents(path)
    method = "vector" if embedding_service is not None else "bm25"
    return build_retriever(method, document_texts(docs), embedding_service=embedding_service)
