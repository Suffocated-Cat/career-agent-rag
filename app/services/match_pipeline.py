"""
Match pipeline — rank a resume's experiences/projects against a JD.

Where the keyword matcher compares skill lists, this scores each resume
experience/project by *retrieval relevance* to a JD-derived query. It builds
the resume corpus, forms a query from the JD's skills and responsibilities,
runs a configurable retriever, and returns per-item relevance — attributed
back to the source via RetrievalDocument provenance.

The retrieval method is selectable ("bm25" / "vector" / "hybrid" /
"hybrid+rerank"), which is what enables ablation comparisons.
"""

from app.models.jd import JobDescription
from app.models.resume import Resume
from app.models.match import ProjectRelevance
from app.services.retrieval.base import (
    RetrievalDocument,
    corpus_from_resume,
    document_texts,
)
from app.services.retrieval.factory import build_retriever
from app.services.retrieval.reranker import Reranker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService


def build_jd_query(jd: JobDescription) -> str:
    """Form a single retrieval query from a job description.

    Combines required skills and responsibilities — the parts that describe
    what the role actually does — falling back to the raw text if structured
    fields are empty.

    Args:
        jd: A parsed JobDescription.

    Returns:
        A query string.
    """
    parts: list[str] = []
    if jd.skills:
        parts.append(" ".join(jd.skills))
    if jd.responsibilities:
        parts.extend(jd.responsibilities)
    query = " ".join(parts).strip()
    return query or jd.raw_text


def _label(doc: RetrievalDocument) -> str:
    """Build a human-readable label for a retrieved document."""
    md = doc.metadata
    if doc.source_type == "experience":
        title = md.get("title") or ""
        company = md.get("company") or ""
        if title and company:
            return f"{title} at {company}"
        return title or company or doc.id
    return md.get("name") or doc.id


def rank_resume_projects(
    jd: JobDescription,
    resume: Resume,
    embedding_service: "EmbeddingService | None" = None,
    method: str = "hybrid",
    reranker: Reranker | None = None,
) -> list[ProjectRelevance]:
    """Rank a resume's experiences/projects by relevance to the JD.

    Args:
        jd: The target job description.
        resume: The resume to rank.
        embedding_service: Required for any vector-based method.
        method: Retrieval method ("bm25" / "vector" / "hybrid" /
            "hybrid+rerank").
        reranker: Optional reranker for "hybrid+rerank".

    Returns:
        ProjectRelevance entries sorted by relevance descending. Scores are
        min-max normalized across the returned items (best = 1.0); the raw
        retriever score is preserved too.
    """
    docs = corpus_from_resume(resume)
    if not docs:
        return []

    retriever = build_retriever(
        method,
        document_texts(docs),
        embedding_service=embedding_service,
        reranker=reranker,
    )
    results = retriever.search(build_jd_query(jd), k=len(docs))
    if not results:
        return []

    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo

    relevances: list[ProjectRelevance] = []
    for r in results:
        doc = docs[r.doc_id]
        normalized = 1.0 if span == 0 else (r.score - lo) / span
        relevances.append(
            ProjectRelevance(
                doc_id=doc.id,
                source_type=doc.source_type,
                label=_label(doc),
                score=round(r.score, 4),
                normalized_score=round(normalized, 4),
            )
        )
    return relevances
