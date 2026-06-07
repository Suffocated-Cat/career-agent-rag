"""Tests for PgVectorRetriever — fake-DB unit tests + a self-skipping integration test."""
import numpy as np
import pytest

from app.services.retrieval.base import RetrievalResult
from app.services.retrieval.pgvector_retriever import PgVectorRetriever


class _FakeEmbed:
    def encode(self, texts):
        return np.ones((len(texts), 384), dtype=np.float64)


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params):
        self.executed = (sql, params)

    def fetchall(self):
        return self.rows


class _FakeConn:
    def __init__(self, rows):
        self.cur = _FakeCursor(rows)
        self.closed = False

    def cursor(self):
        return self.cur

    def close(self):
        self.closed = True


class TestSearchUnit:
    def test_returns_results_from_db(self, monkeypatch):
        conn = _FakeConn([(1, "docker question", 0.91), (2, "python question", 0.80)])
        monkeypatch.setattr("app.db.connection.get_connection", lambda dsn=None: conn)

        retriever = PgVectorRetriever(_FakeEmbed())
        results = retriever.search("docker", k=2)

        assert all(isinstance(r, RetrievalResult) for r in results)
        assert [r.doc_id for r in results] == [1, 2]
        assert results[0].text == "docker question"
        assert results[0].score == 0.91
        assert conn.closed  # connection is closed

    def test_no_embedding_service_returns_empty(self):
        assert PgVectorRetriever(None).search("docker") == []

    def test_blank_query_returns_empty(self):
        assert PgVectorRetriever(_FakeEmbed()).search("   ") == []


def test_pgvector_roundtrip_integration():
    """Real pgvector round-trip — skipped unless psycopg + a DB are available."""
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    from app.db.connection import EMBED_DIM, ensure_schema, get_connection

    try:
        conn = get_connection()
    except Exception:
        pytest.skip("database not available")

    ensure_schema(conn)
    vec = np.ones(EMBED_DIM, dtype=np.float64)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO knowledge_doc (doc_id, skill, doc_type, text, embedding) "
            "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (doc_id) DO UPDATE SET embedding = EXCLUDED.embedding",
            ("test:integration", "python", "question", "integration sample", vec),
        )
    conn.commit()
    conn.close()

    class _Emb:
        def encode(self, texts):
            return np.ones((len(texts), EMBED_DIM), dtype=np.float64)

    results = PgVectorRetriever(_Emb()).search("anything", k=5)
    assert any(r.text == "integration sample" for r in results)
