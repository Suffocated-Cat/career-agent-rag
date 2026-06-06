"""
HybridRetriever — fuse lexical (BM25) and semantic (vector) recall.

Each arm catches what the other misses: BM25 nails exact tech terms, vector
catches paraphrases. This retriever runs both over the same corpus and merges
their rankings. Two fusion strategies are supported:

  - "rrf" (default) — Reciprocal Rank Fusion. Combines results by rank, not
    raw score, so the wildly different score scales of BM25 (unbounded) and
    cosine similarity (~0–1) never need normalizing:

        score(d) = Σ_r  w_r · 1 / (rrf_k + rank_r(d))

  - "weighted" — min-max normalize each arm's scores onto [0, 1], then take a
    weighted sum. More tunable, but sensitive to the candidate pool.

Both honor per-arm weights, defaulting to equal weighting. Implements the
same ``search(query, k) -> list[RetrievalResult]`` interface as the other
backends.
"""

from collections import defaultdict

from app.services.retrieval.base import RetrievalResult
from app.services.retrieval.bm25_retriever import (
    BM25Retriever,
    DEFAULT_K1,
    DEFAULT_B,
)
from app.services.retrieval.vector_retriever import VectorRetriever

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService

DEFAULT_RRF_K: int = 60
DEFAULT_CANDIDATE_POOL: int = 50


class HybridRetriever:
    """Fuses a BM25Retriever and a VectorRetriever over one corpus.

    Both sub-retrievers are built at construction and exposed as ``.bm25``
    and ``.vector`` so callers can inspect or ablate either arm.

    Usage::

        r = HybridRetriever(corpus, embedding_service)
        r.search("docker pipeline", k=5)
    """

    def __init__(
        self,
        corpus: list[str],
        embedding_service: "EmbeddingService",
        method: str = "rrf",
        weights: tuple[float, float] = (1.0, 1.0),
        rrf_k: int = DEFAULT_RRF_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ):
        """Build both retrieval arms and configure fusion.

        Args:
            corpus: Documents to search over.
            embedding_service: Service for the vector arm.
            method: Fusion strategy, "rrf" or "weighted".
            weights: (bm25_weight, vector_weight) applied during fusion.
            rrf_k: Reciprocal Rank Fusion constant (larger = flatter).
            candidate_pool: How many results to pull from each arm before
                fusing. Larger pools improve recall at some cost.
            k1: BM25 term-frequency saturation.
            b: BM25 length-normalization strength.
        """
        if method not in ("rrf", "weighted"):
            raise ValueError(
                f"Unknown fusion method {method!r}; expected 'rrf' or 'weighted'."
            )

        self.corpus = corpus
        self.method = method
        self.bm25_weight, self.vector_weight = weights
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

        self.bm25 = BM25Retriever(corpus, k1=k1, b=b)
        self.vector = VectorRetriever(corpus, embedding_service)

    def search(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return the top-k documents after fusing both arms.

        Args:
            query: The search query text.
            k: Maximum number of results to return.

        Returns:
            Up to *k* RetrievalResult objects, sorted by fused score
            descending, ties broken by document id.
        """
        bm25_results = self.bm25.search(query, k=self.candidate_pool)
        vector_results = self.vector.search(query, k=self.candidate_pool)

        if not bm25_results and not vector_results:
            return []

        if self.method == "rrf":
            fused = self._fuse_rrf(bm25_results, vector_results)
        else:
            fused = self._fuse_weighted(bm25_results, vector_results)

        # doc_id -> text, from whichever arm surfaced it.
        texts = {r.doc_id: r.text for r in bm25_results}
        texts.update({r.doc_id: r.text for r in vector_results})

        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
        return [
            RetrievalResult(doc_id=doc_id, text=texts[doc_id], score=score)
            for doc_id, score in ranked[:k]
        ]

    def _fuse_rrf(
        self,
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
    ) -> dict[int, float]:
        """Combine rankings by Reciprocal Rank Fusion."""
        scores: dict[int, float] = defaultdict(float)
        for results, weight in (
            (bm25_results, self.bm25_weight),
            (vector_results, self.vector_weight),
        ):
            for rank, res in enumerate(results):
                scores[res.doc_id] += weight / (self.rrf_k + rank + 1)
        return scores

    def _fuse_weighted(
        self,
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
    ) -> dict[int, float]:
        """Combine min-max normalized scores by weighted sum."""
        scores: dict[int, float] = defaultdict(float)
        for results, weight in (
            (bm25_results, self.bm25_weight),
            (vector_results, self.vector_weight),
        ):
            for doc_id, norm in _min_max_normalize(results).items():
                scores[doc_id] += weight * norm
        return scores


def _min_max_normalize(results: list[RetrievalResult]) -> dict[int, float]:
    """Scale a result set's scores onto [0, 1] by min-max.

    When all scores are equal (or there is a single result), every document
    maps to 1.0 — they are equally relevant within this arm.
    """
    if not results:
        return {}

    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo

    if span == 0:
        return {r.doc_id: 1.0 for r in results}

    return {r.doc_id: (r.score - lo) / span for r in results}
