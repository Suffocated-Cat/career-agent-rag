"""
Retrieval ablation — compare retrieval methods on the same labeled dataset.

Runs each method (bm25 / vector / hybrid / hybrid+rerank) through the eval
runner over one corpus + query set, so the contribution of each stage
(lexical, semantic, fusion, reranking) is measurable side by side.

Run it (loads the embedding + cross-encoder models):

    python -m app.eval.ablation
"""

from dataclasses import dataclass
from pathlib import Path

from app.eval.datasets import EvalDataset, load_eval_dataset
from app.eval.runner import EvalReport, evaluate_retriever
from app.services.retrieval.base import document_texts
from app.services.retrieval.factory import build_retriever

DEFAULT_METHODS: tuple[str, ...] = ("bm25", "vector", "hybrid", "hybrid+rerank")


@dataclass
class AblationResult:
    """An eval report for one retrieval method."""

    method: str
    report: EvalReport


def run_ablation(
    dataset: EvalDataset,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    k: int = 10,
    embedding_service=None,
    reranker=None,
) -> list[AblationResult]:
    """Evaluate each method over *dataset* and collect the reports.

    Args:
        dataset: Corpus + labeled queries.
        methods: Retrieval methods to compare.
        k: Cutoff for recall@k / nDCG@k.
        embedding_service: Required for vector/hybrid methods.
        reranker: Optional reranker for "hybrid+rerank".

    Returns:
        One AblationResult per method, in order.
    """
    texts = document_texts(dataset.documents)
    results: list[AblationResult] = []
    for method in methods:
        retriever = build_retriever(
            method, texts, embedding_service=embedding_service, reranker=reranker
        )
        report = evaluate_retriever(retriever, dataset.documents, dataset.queries, k=k)
        results.append(AblationResult(method=method, report=report))
    return results


def format_ablation_table(results: list[AblationResult], k: int) -> str:
    """Render ablation results as a Markdown table."""
    lines = [
        f"| Method | Recall@{k} | MRR | nDCG@{k} |",
        "|--------|-----------|-----|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r.method} | {r.report.mean_recall_at_k:.3f} | "
            f"{r.report.mean_mrr:.3f} | {r.report.mean_ndcg_at_k:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    """Run the default ablation over the bundled fixtures and print the table."""
    from app.services.embedding import EmbeddingService
    from app.services.retrieval.reranker import Reranker

    fixtures = Path(__file__).parent.parent.parent / "tests" / "fixtures"
    dataset = load_eval_dataset(
        fixtures / "retrieval_documents.json",
        fixtures / "relevance_queries.json",
    )
    k = 10
    results = run_ablation(
        dataset,
        k=k,
        embedding_service=EmbeddingService(),
        reranker=Reranker(),
    )
    print(format_ablation_table(results, k))


if __name__ == "__main__":  # pragma: no cover
    main()
