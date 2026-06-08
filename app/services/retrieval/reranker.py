"""
Reranker — cross-encoder re-scoring of retrieval candidates.

The BM25 / vector / hybrid retrievers are *bi-encoders*: query and document
are embedded independently, so scoring is cheap but coarse. A *cross-encoder*
instead feeds the (query, document) pair through the model together, letting
every query token attend to every document token. That is far more accurate
but far more expensive — so it is used only to re-rank a small candidate pool
already recalled by a cheaper retriever:

    cheap retriever recalls top ~20  →  cross-encoder rescores  →  top-k

RerankingRetriever wires a base retriever and a Reranker together behind the
standard ``search(query, k)`` interface, so reranking is a drop-in stage.

The cross-encoder model is loaded lazily on first use (it is heavy), and a
pre-built model can be injected for testing or to share one instance.
"""

from typing import Any, Protocol

from app.core.config import settings
from app.services.retrieval.base import RetrievalResult, Retriever


class CrossEncoderLike(Protocol):
    """Minimal interface a cross-encoder must provide (duck-typed)."""

    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        """Return one relevance score per (query, document) pair."""
        ...


class Reranker:
    """Re-scores retrieval candidates with a cross-encoder.

    Usage::

        reranker = Reranker()  # model loads lazily on first rerank()
        reranked = reranker.rerank(query, candidates, k=5)
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cross_encoder: "CrossEncoderLike | None" = None,
    ):
        """Configure the reranker.

        Args:
            model_name: Cross-encoder model id (default: settings).
            device: Torch device for the model (default: settings).
            cross_encoder: A pre-built cross-encoder to use directly. When
                provided, no model is loaded lazily — useful for tests or
                sharing one instance across rerankers.
        """
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.device = device or settings.RERANKER_DEVICE
        self._model: "CrossEncoderLike | None" = cross_encoder

    def _get_model(self) -> "CrossEncoderLike":
        """Lazily construct (and cache) the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        k: int = 10,
    ) -> list[RetrievalResult]:
        """Re-score *candidates* against *query* and return the top-k.

        The returned results carry the cross-encoder relevance score (which
        is on a different scale than the upstream retriever's score), with
        ``doc_id``, ``text`` and ``metadata`` preserved from the candidates.

        Args:
            query: The search query text.
            candidates: Candidate results from an upstream retriever.
            k: Maximum number of results to return.

        Returns:
            Up to *k* RetrievalResult objects, sorted by relevance descending,
            ties broken by document id.
        """
        if not candidates:
            return []

        model = self._get_model()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)

        rescored = [
            RetrievalResult(
                doc_id=c.doc_id,
                text=c.text,
                score=float(score),
                metadata=c.metadata,
            )
            for c, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda r: (-r.score, r.doc_id))
        return rescored[:k]


class RerankingRetriever:
    """Wraps a base retriever with a reranking stage.

    Recalls a candidate pool from *base* and re-scores it with *reranker*,
    satisfying the same ``search(query, k)`` interface as every other
    backend so it can be swapped in directly.

    Usage::

        base = HybridRetriever(corpus, embedding_service)
        retriever = RerankingRetriever(base, Reranker())
        retriever.search("docker pipeline", k=5)
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        candidate_pool: int = 20,
    ):
        """Compose a base retriever with a reranker.

        Args:
            base: The upstream retriever providing candidates.
            reranker: The Reranker used to re-score them.
            candidate_pool: How many candidates to recall before reranking.
        """
        self.base = base
        self.reranker = reranker
        self.candidate_pool = candidate_pool

    def search(
        self, query: str, k: int = 10, filters: dict | None = None
    ) -> list[RetrievalResult]:
        """Recall a candidate pool, rerank it, and return the top-k.

        *filters* are forwarded to the base retriever when given, so a
        metadata-filtering backend (e.g. pgvector) keeps its filtering
        behaviour behind the reranking stage.
        """
        if filters is None:
            candidates = self.base.search(query, k=self.candidate_pool)
        else:
            candidates = self.base.search(query, k=self.candidate_pool, filters=filters)
        return self.reranker.rerank(query, candidates, k=k)
