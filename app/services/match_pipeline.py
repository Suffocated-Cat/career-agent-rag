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
from app.models.match import MatchResult, ProjectRelevance
from app.services.keyword_matcher import match as match_jd_resume
from app.services.project_auditor import audit_resume
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
    from app.services.llm_client import LLMClient


def analyze_match(
    jd: JobDescription,
    resume: Resume,
    embedding_service: "EmbeddingService | None" = None,
    audit_llm: "LLMClient | None" = None,
) -> MatchResult:
    """Run the full deterministic match analysis for a JD and resume.

    Produces a MatchResult with skill/semantic scoring, project relevance
    ranking, and the authenticity audit attached. This is the single
    orchestration point shared by the ``/match`` endpoint and the career-match
    skill, so they can't drift apart.

    Args:
        jd: Parsed job description.
        resume: Parsed resume.
        embedding_service: Enables semantic matching + hybrid ranking.
        audit_llm: Optional LLM for risk advice in the audit (kept off the
            hot ``/match`` path; supplied by the standalone skill).

    Returns:
        A populated MatchResult.
    """
    result = match_jd_resume(jd, resume, embedding_service=embedding_service)
    method = "hybrid" if embedding_service is not None else "bm25"
    result.project_relevance = rank_resume_projects(
        jd, resume, embedding_service=embedding_service, method=method
    )
    result.project_audit = audit_resume(resume, llm=audit_llm)
    return result


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
