"""
Ingest the knowledge base into PostgreSQL + pgvector.

Embeds each KB document and upserts it into ``knowledge_doc``. Run after the db
service is up (and the backend image has psycopg/pgvector installed):

    docker compose exec backend python -m scripts.ingest_kb
"""

import json

import numpy as np

from app.db.connection import ensure_schema, get_connection
from app.services.embedding import EmbeddingService
from app.services.knowledge import load_kb_documents


def main() -> None:  # pragma: no cover - requires a database
    docs = load_kb_documents()
    embeddings = EmbeddingService().encode([d.text for d in docs])

    conn = get_connection()
    ensure_schema(conn)
    with conn.cursor() as cur:
        for doc, vector in zip(docs, embeddings):
            cur.execute(
                "INSERT INTO knowledge_doc (doc_id, skill, doc_type, text, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET "
                "skill = EXCLUDED.skill, doc_type = EXCLUDED.doc_type, "
                "text = EXCLUDED.text, metadata = EXCLUDED.metadata, "
                "embedding = EXCLUDED.embedding",
                (
                    doc.id,
                    doc.metadata.get("skill"),
                    doc.metadata.get("type"),
                    doc.text,
                    json.dumps(doc.metadata),
                    np.asarray(vector, dtype=np.float64),
                ),
            )
    conn.commit()
    conn.close()
    print(f"Ingested {len(docs)} knowledge documents.")


if __name__ == "__main__":  # pragma: no cover
    main()
