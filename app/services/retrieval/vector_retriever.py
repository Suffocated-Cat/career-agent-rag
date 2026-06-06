"""
VectorRetriever — semantic (embedding) retrieval over a fixed corpus.

Where BM25 matches surface words, this backend matches meaning: the corpus
is embedded once at construction, and each query is embedded and compared by
cosine similarity. This recalls experiences that are phrased differently from
the JD ("recommendation engine" vs. "personalization system") but mean the
same thing.

Implements the same ``search(query, k) -> list[RetrievalResult]`` interface
as BM25Retriever so the two can be swapped or fused.
"""

import numpy as np

from typing import TYPE_CHECKING

from app.services.retrieval.base import RetrievalResult

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so dot products give cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorRetriever:
    """Embedding-based semantic retriever over a list of documents.

    The corpus is embedded and normalized once at construction; each
    ``search`` call embeds the query and ranks documents by cosine
    similarity.

    Usage::

        r = VectorRetriever(["built a recommendation engine"], embedding_service)
        r.search("personalization system", k=1)
        # → [RetrievalResult(doc_id=0, text=..., score=0.71)]
    """

    def __init__(
        self,
        corpus: list[str],
        embedding_service: "EmbeddingService",
    ):
        """Embed and index *corpus* for cosine-similarity search.

        Args:
            corpus: Documents to search over.
            embedding_service: Service used to embed the corpus and queries.
        """
        self.corpus = corpus
        self.embedding_service = embedding_service
        self.n_docs: int = len(corpus)

        if self.n_docs:
            doc_embs = embedding_service.encode(corpus)
            self._doc_embs: np.ndarray | None = _normalize_rows(
                np.asarray(doc_embs, dtype=np.float64)
            )
        else:
            self._doc_embs = None

    def search(
        self,
        query: str,
        k: int = 10,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-k documents most similar to *query*.

        Args:
            query: The search query text.
            k: Maximum number of results to return.
            min_score: Optional minimum cosine similarity. Results below it
                are dropped. When None, the top-k are returned regardless of
                score.

        Returns:
            Up to *k* RetrievalResult objects, sorted by score descending.
        """
        if self._doc_embs is None or not query.strip():
            return []

        q_emb = self.embedding_service.encode([query])
        q_emb = _normalize_rows(np.asarray(q_emb, dtype=np.float64))[0]

        sims = self._doc_embs @ q_emb  # (n_docs,)

        scored = [(doc_id, float(sims[doc_id])) for doc_id in range(self.n_docs)]
        if min_score is not None:
            scored = [(doc_id, s) for doc_id, s in scored if s >= min_score]
        scored.sort(key=lambda x: (-x[1], x[0]))

        return [
            RetrievalResult(doc_id=doc_id, text=self.corpus[doc_id], score=score)
            for doc_id, score in scored[:k]
        ]
