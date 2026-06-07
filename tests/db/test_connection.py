"""Tests for the DB schema helper (no real database)."""
from app.db.connection import EMBED_DIM, ensure_schema


class _FakeCursor:
    def __init__(self):
        self.executed = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql):
        self.executed = sql


class _FakeConn:
    def __init__(self):
        self.cur = _FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True


def test_embed_dim():
    assert EMBED_DIM == 384


def test_ensure_schema_creates_table():
    conn = _FakeConn()
    ensure_schema(conn)
    assert "knowledge_doc" in conn.cur.executed
    assert "vector(384)" in conn.cur.executed
    assert conn.committed
