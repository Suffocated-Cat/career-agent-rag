"""
PostgreSQL + pgvector connection helpers for the knowledge base.

``psycopg`` and ``pgvector`` are imported lazily inside the connection helper so
the rest of the app (and the offline test suite) does not require them or a
running database — only code that actually talks to the DB pulls them in.
"""

from typing import Any

from app.core.config import settings

# Dimension of the embedding model (all-MiniLM-L6-v2).
EMBED_DIM = 384

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS knowledge_doc (
    id serial PRIMARY KEY,
    doc_id text UNIQUE NOT NULL,
    skill text,
    doc_type text,
    text text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{{}}',
    embedding vector({EMBED_DIM})
);
"""


def get_connection(dsn: str | None = None) -> Any:  # pragma: no cover - needs a DB
    """Open a psycopg connection with the pgvector type registered."""
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn or settings.DATABASE_URL)
    register_vector(conn)
    return conn


def ensure_schema(conn: Any) -> None:
    """Create the vector extension and knowledge_doc table if absent."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()
