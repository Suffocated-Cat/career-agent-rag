"""
Shared types for the retrieval package.

Defines the common Retriever protocol that every backend implements, the
RetrievalResult returned by a search, a default tokenizer, and helpers to
turn a parsed Resume into a searchable corpus.

Retrievers operate on plain text (``list[str]``) and identify results by
integer index. ``RetrievalDocument`` carries the provenance — which resume
item a piece of text came from — alongside that corpus, so callers can map a
result's ``doc_id`` back to a structured source for reporting and evaluation.
"""

import re

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.models.resume import Resume
from app.services.vector_matcher import _build_experience_text


@dataclass
class RetrievalDocument:
    """A retrievable unit with provenance back to its resume source.

    The retrievers themselves only consume ``text``; the remaining fields let
    callers attribute a retrieved result to a structured resume item.

    Attributes:
        id: Stable identifier, e.g. ``"exp:0"`` or ``"proj:1"``. Survives
            corpus reordering, so evaluation labels can key on it.
        text: The searchable text fed to retrievers.
        source_type: Origin kind, ``"experience"`` or ``"project"``.
        source_index: Index of the item within its resume list.
        metadata: Display/source fields (title, company, name, technologies).
    """

    id: str
    text: str
    source_type: str
    source_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single scored document returned by a retriever.

    Attributes:
        doc_id: Index of the document in the retriever's corpus.
        text: The document text that was scored.
        score: Relevance score for the query (higher is more relevant).
               Scale depends on the backend (BM25 is unbounded, vector
               similarity is roughly 0–1).
        metadata: Source metadata when the backend has it (e.g. the KB's
               skill/role/difficulty/answer_outline); empty otherwise.
    """

    doc_id: int
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


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


def corpus_from_resume(resume: Resume) -> list[RetrievalDocument]:
    """Build a searchable corpus from a parsed resume.

    Each work experience and each project becomes one document, carrying a
    stable id and provenance back to its source item. Experiences reuse the
    same text builder as the vector matcher so lexical and semantic retrieval
    operate over identical units.

    Items whose text is empty are skipped, but ids/indices reflect the
    original position in the resume so provenance stays accurate.

    Args:
        resume: A parsed Resume.

    Returns:
        A list of RetrievalDocument (one per non-empty experience and project).
        Use ``document_texts`` to get the plain-text corpus for a retriever.
    """
    docs: list[RetrievalDocument] = []

    for i, exp in enumerate(resume.experience):
        text = _build_experience_text(exp)
        if not text:
            continue
        docs.append(
            RetrievalDocument(
                id=f"exp:{i}",
                text=text,
                source_type="experience",
                source_index=i,
                metadata={"title": exp.title, "company": exp.company},
            )
        )

    for i, proj in enumerate(resume.projects):
        parts = [proj.name]
        if proj.description:
            parts.append(proj.description)
        if proj.technologies:
            parts.append("Technologies: " + ", ".join(proj.technologies))
        text = " ".join(p for p in parts if p).strip()
        if not text:
            continue
        docs.append(
            RetrievalDocument(
                id=f"proj:{i}",
                text=text,
                source_type="project",
                source_index=i,
                metadata={
                    "name": proj.name,
                    "technologies": list(proj.technologies),
                },
            )
        )

    return docs


def document_texts(docs: list[RetrievalDocument]) -> list[str]:
    """Extract the plain-text corpus from documents, for feeding a retriever.

    The result is index-aligned with *docs*, so a retriever's integer
    ``doc_id`` maps directly back to ``docs[doc_id]``.

    Args:
        docs: Documents from ``corpus_from_resume``.

    Returns:
        A list of document texts in the same order.
    """
    return [d.text for d in docs]
