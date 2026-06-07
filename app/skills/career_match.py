"""
CareerMatch skill — the end-to-end CareerAgent pipeline as one entry point.

Given raw JD and resume text, it runs the whole flow and returns everything:

    parse JD + resume → match (skills/semantic) → rank projects by relevance
                      → audit for risks → generate report

Each stage uses the deterministic core, optionally enhanced by an LLM
(extraction, advice, narrative) and embeddings — all with graceful fallback,
so it works fully offline too. The same capabilities are exposed over MCP
(``python -m app.mcp.server``) for external hosts; this module is the
in-process, batteries-included version a Skill or CLI can call.
"""

from pydantic import BaseModel

from app.models.jd import JobDescription
from app.models.resume import Resume
from app.models.match import MatchResult, MatchReport
from app.services.jd_parser import parse_jd
from app.services.resume_parser import parse_resume
from app.services.match_pipeline import analyze_match
from app.services.report_generator import generate_report

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService
    from app.services.llm_client import LLMClient


class CareerMatchResult(BaseModel):
    """The full output of the career-match pipeline."""

    jd: JobDescription
    resume: Resume
    match: MatchResult
    report: MatchReport


def run_career_match(
    jd_text: str,
    resume_text: str,
    embedding_service: "EmbeddingService | None" = None,
    llm: "LLMClient | None" = None,
) -> CareerMatchResult:
    """Run the complete CareerAgent analysis on raw JD and resume text.

    Args:
        jd_text: Raw job description text.
        resume_text: Raw resume text.
        embedding_service: Optional embeddings (enables semantic matching and
            hybrid project ranking).
        llm: Optional LLM (enables extraction, risk advice, and a generated
            narrative report).

    Returns:
        A CareerMatchResult with the parsed JD/resume, the match result
        (including project relevance and risk audit), and the report.
    """
    jd = parse_jd(jd_text, embedding_service=embedding_service, llm=llm)
    resume = parse_resume(resume_text, embedding_service=embedding_service, llm=llm)

    result = analyze_match(
        jd, resume, embedding_service=embedding_service, audit_llm=llm
    )

    report = generate_report(jd, resume, result, llm=llm)
    return CareerMatchResult(jd=jd, resume=resume, match=result, report=report)


def main() -> None:  # pragma: no cover
    """CLI: `python -m app.skills.career_match --jd JD.txt --resume RESUME.txt`."""
    import argparse

    from pathlib import Path

    from app.services.embedding import EmbeddingService
    from app.services.llm_client import LLMClient

    parser = argparse.ArgumentParser(description="Run the CareerAgent match pipeline.")
    parser.add_argument("--jd", required=True, help="Path to the job description text file.")
    parser.add_argument("--resume", required=True, help="Path to the resume text file.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM enhancement.")
    args = parser.parse_args()

    jd_text = Path(args.jd).read_text(encoding="utf-8")
    resume_text = Path(args.resume).read_text(encoding="utf-8")

    try:
        embeddings = EmbeddingService()
    except Exception:
        embeddings = None
    llm = None if args.no_llm else LLMClient()

    out = run_career_match(jd_text, resume_text, embedding_service=embeddings, llm=llm)
    print(out.report.full_report)
    if out.match.project_audit and out.match.project_audit.findings:
        print("\n---\n")
        print(out.match.project_audit.summary)


if __name__ == "__main__":  # pragma: no cover
    main()
