"""
Shared types for the retrieval package.

Defines the common Retriever protocol that every backend implements, the
RetrievalResult returned by a search, a default tokenizer, and a helper to
turn a parsed Resume into a searchable corpus.
"""

import re

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.models.resume import Resume
from app.services.vector_matcher import _build_experience_text


@dataclass
class RetrievalResult:
    """A single scored document returned by a retriever.

    Attributes:
        doc_id: Index of the document in the retriever's corpus.
        text: The document text that was scored.
        score: Relevance score for the query (higher is more relevant).
               Scale depends on the backend (BM25 is unbounded, vector
               similarity is roughly 0–1).
    """

    doc_id: int
    text: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    """Common interface for all retrieval backends.

    A retriever is built around a fixed corpus and answers queries against
    it. This lets BM25, vector, hybrid, and reranking backends be swapped
    or composed without callers caring about the implementation.
    """

    def search(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return the top-k most relevant documents for *query*.

        Args:
            query: The search query text.
            k: Maximum number of results to return.

        Returns:
            Up to *k* RetrievalResult objects, sorted by score descending.
        """
        ...


# Token pattern keeps tech terms intact: c++, c#, .net, node.js, ci/cd → ci, cd.
_TOKEN_RE = re.compile(r"[a-z0-9](?:[a-z0-9+.#]*[a-z0-9+#])?")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into lexical tokens.

    Preserves common technical tokens like ``c++``, ``c#`` and ``node.js``
    while dropping standalone punctuation.

    Args:
        text: Raw text to tokenize.

    Returns:
        A list of lowercase token strings (possibly empty).
    """
    return _TOKEN_RE.findall(text.lower())


def corpus_from_resume(resume: Resume) -> list[str]:
    """Build a searchable corpus from a parsed resume.

    Each work experience and each project becomes one document. Experiences
    reuse the same text builder as the vector matcher so lexical and semantic
    retrieval operate over identical units.

    Args:
        resume: A parsed Resume.

    Returns:
        A list of document strings (one per experience and per project).
    """
    docs: list[str] = []

    for exp in resume.experience:
        text = _build_experience_text(exp)
        if text:
            docs.append(text)

    for proj in resume.projects:
        parts = [proj.name]
        if proj.description:
            parts.append(proj.description)
        if proj.technologies:
            parts.append("Technologies: " + ", ".join(proj.technologies))
        text = " ".join(p for p in parts if p).strip()
        if text:
            docs.append(text)

    return docs
