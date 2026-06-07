"""
PgVectorRetriever — semantic retrieval over the knowledge base in PostgreSQL.

Implements the same ``search(query, k) -> list[RetrievalResult]`` interface as
the in-memory retrievers, but backed by pgvector: the query is embedded and
ranked DB-side with the cosine-distance operator (``<=>``). This is the
persistent, filterable knowledge store for RAG; the in-memory retrievers remain
for resume matching and for the offline test suite.
"""

import numpy as np

from typing import TYPE_CHECKING

from app.core.config import settings
from app.services.retrieval.base import RetrievalResult

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService


class PgVectorRetriever:
    """Retriever backed by a pgvector ``knowledge_doc`` table."""

    def __init__(
        self,
        embedding_service: "EmbeddingService | None",
        dsn: str | None = None,
        table: str = "knowledge_doc",
    ):
        """Configure the retriever.

        Args:
            embedding_service: Used to embed the query (required to search;
                if None, search returns nothing).
            dsn: Database URL (default: settings.DATABASE_URL).
            table: Table name to query.
        """
        self.embedding_service = embedding_service
        self.dsn = dsn or settings.DATABASE_URL
        self.table = table

    def search(
        self,
        query: str,
        k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-k knowledge docs most similar to *query*.

        Embeds the query and ranks rows by cosine similarity (``1 - distance``)
        using pgvector. Optional *filters* apply metadata predicates before
        ranking — this is the vector-search-plus-metadata-filtering that makes
        pgvector worthwhile:

          - scalar value  → equality, e.g. ``{"difficulty": "mid"}``
          - list value    → jsonb overlap, e.g. ``{"role": ["backend"]}``

        Returns an empty list if there is no embedding service or the query is
        blank.
        """
        if self.embedding_service is None or not query.strip():
            return []

        from app.db.connection import get_connection

        vector = np.asarray(self.embedding_service.encode([query]), dtype=np.float64)[0]

        where = ["embedding IS NOT NULL"]
        filter_params: list = []
        for key, value in (filters or {}).items():
            if isinstance(value, (list, tuple)):
                where.append("metadata -> %s ?| %s")  # jsonb array overlap
                filter_params += [key, list(value)]
            else:
                where.append("metadata ->> %s = %s")  # scalar equality
                filter_params += [key, str(value)]

        sql = (
            f"SELECT id, text, 1 - (embedding <=> %s) AS score "
            f"FROM {self.table} WHERE {' AND '.join(where)} "
            f"ORDER BY embedding <=> %s LIMIT %s"
        )
        params = [vector, *filter_params, vector, k]

        conn = get_connection(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.close()

        return [
            RetrievalResult(doc_id=row[0], text=row[1], score=float(row[2]))
            for row in rows
        ]
