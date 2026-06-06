"""
BM25Retriever — lexical (keyword) retrieval over a fixed corpus.

Implements Okapi BM25 from scratch (no external dependency). BM25 scores a
document against a query by summing, over the query terms, an IDF weight
times a saturating term-frequency factor that is normalized by document
length:

    score(D, Q) = Σ_t  IDF(t) · ( f(t,D) · (k1 + 1) )
                         / ( f(t,D) + k1 · (1 - b + b · |D| / avgdl) )

where
    f(t, D)  term frequency of t in document D
    |D|      length of D in tokens
    avgdl    average document length across the corpus
    IDF(t)   = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )   (always >= 0)
    k1       term-frequency saturation (typical 1.2–2.0)
    b        length-normalization strength (0 = none, 1 = full)

This matches exact and near-exact skill words (PyTorch, Docker, Kubernetes)
that semantic embeddings can blur together.
"""

import math

from collections import Counter

from app.services.retrieval.base import RetrievalResult, tokenize

DEFAULT_K1: float = 1.5
DEFAULT_B: float = 0.75


class BM25Retriever:
    """Okapi BM25 keyword retriever over a list of documents.

    The corpus is tokenized and indexed once at construction; each
    ``search`` call scores every document and returns the top-k.

    Usage::

        r = BM25Retriever(["built a docker pipeline", "react dashboard"])
        r.search("docker", k=1)
        # → [RetrievalResult(doc_id=0, text="built a docker pipeline", score=...)]
    """

    def __init__(
        self,
        corpus: list[str],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ):
        """Index *corpus* for BM25 scoring.

        Args:
            corpus: Documents to search over.
            k1: Term-frequency saturation parameter.
            b: Document-length normalization strength (0–1).
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        self._doc_tokens: list[list[str]] = [tokenize(doc) for doc in corpus]
        self._doc_freqs: list[Counter[str]] = [
            Counter(tokens) for tokens in self._doc_tokens
        ]
        self._doc_len: list[int] = [len(tokens) for tokens in self._doc_tokens]

        self.n_docs: int = len(corpus)
        self.avgdl: float = (
            sum(self._doc_len) / self.n_docs if self.n_docs else 0.0
        )

        # Document frequency: how many documents contain each term.
        df: Counter[str] = Counter()
        for freqs in self._doc_freqs:
            df.update(freqs.keys())

        # Precompute IDF per term using the BM25 "plus-one" variant so that
        # weights stay non-negative even for very common terms.
        self._idf: dict[str, float] = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def _score_doc(self, query_terms: list[str], doc_id: int) -> float:
        """BM25 score of a single document for the given query terms."""
        freqs = self._doc_freqs[doc_id]
        doc_len = self._doc_len[doc_id]
        denom_len = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

        score = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            score += idf * (tf * (self.k1 + 1)) / (tf + denom_len)
        return score

    def search(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return the top-k documents for *query*, scored by BM25.

        Documents with a zero score (no overlapping query terms) are
        excluded. Results are sorted by score descending, ties broken by
        document order.

        Args:
            query: The search query text.
            k: Maximum number of results to return.

        Returns:
            Up to *k* RetrievalResult objects.
        """
        if self.n_docs == 0 or self.avgdl == 0.0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored = [
            (doc_id, self._score_doc(query_terms, doc_id))
            for doc_id in range(self.n_docs)
        ]
        scored = [(doc_id, s) for doc_id, s in scored if s > 0.0]
        scored.sort(key=lambda x: (-x[1], x[0]))

        return [
            RetrievalResult(doc_id=doc_id, text=self.corpus[doc_id], score=score)
            for doc_id, score in scored[:k]
        ]
