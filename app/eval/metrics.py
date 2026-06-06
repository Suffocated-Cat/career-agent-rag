"""
Ranking metrics for retrieval evaluation.

All functions take a *ranked* list of document ids (best first, as produced by
a retriever) and a relevance judgment, and return a score in [0, 1]:

  - recall_at_k  — did we recall the relevant documents within the top-k?
  - mrr          — how high is the first relevant document? (1 / its rank)
  - ndcg_at_k    — graded ranking quality, rewarding higher-graded documents
                   placed earlier and discounting by log of position.

Binary metrics (recall, MRR) take a set of relevant ids. nDCG takes graded
relevance as a {doc_id: grade} mapping; any id not present is treated as
grade 0.
"""

import math


def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of relevant documents found in the top-k.

    Args:
        ranked_ids: Document ids ordered best-first.
        relevant_ids: The set of ids considered relevant.
        k: Cutoff rank.

    Returns:
        |relevant ∩ top-k| / |relevant|, or 0.0 when nothing is relevant or
        k <= 0.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant document.

    Args:
        ranked_ids: Document ids ordered best-first.
        relevant_ids: The set of ids considered relevant.

    Returns:
        1 / rank of the first relevant id (rank is 1-based), or 0.0 if none
        of the ranked ids are relevant.
    """
    if not relevant_ids:
        return 0.0
    for index, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_ids:
            return 1.0 / (index + 1)
    return 0.0


def dcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    """Discounted cumulative gain of a ranking at cutoff k.

    Args:
        ranked_ids: Document ids ordered best-first.
        grades: {doc_id: relevance grade}; missing ids count as 0.
        k: Cutoff rank.

    Returns:
        Σ grade_i / log2(i + 2) over the top-k positions.
    """
    dcg = 0.0
    for index, doc_id in enumerate(ranked_ids[:k]):
        grade = grades.get(doc_id, 0)
        if grade:
            dcg += grade / math.log2(index + 2)
    return dcg


def ndcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    """Normalized DCG: actual DCG divided by the ideal DCG.

    The ideal DCG is the DCG of the best possible ordering of the graded
    documents, so a perfect ranking scores 1.0 and a ranking that surfaces
    nothing relevant scores 0.0.

    Args:
        ranked_ids: Document ids ordered best-first.
        grades: {doc_id: relevance grade}; missing ids count as 0.
        k: Cutoff rank.

    Returns:
        nDCG@k in [0, 1], or 0.0 when no document has positive grade.
    """
    dcg = dcg_at_k(ranked_ids, grades, k)

    ideal_grades = sorted(grades.values(), reverse=True)
    idcg = 0.0
    for index, grade in enumerate(ideal_grades[:k]):
        if grade:
            idcg += grade / math.log2(index + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
