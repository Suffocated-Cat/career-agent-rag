"""
LLM-as-Judge — automated quality + grounding evaluation of generated text.

A second LLM scores a generated artifact (e.g. the match report) against the
*structured evidence* it was supposed to be based on, judging:

  - groundedness — are the report's claims supported by the evidence?
  - coverage     — does it use the key evidence (skills, gaps, risks, ...)?
  - clarity      — is it clear and professional?

and listing any claims not supported by the evidence (the hallucination check
for the grounded report generator). The judge is told to use only the provided
evidence, not outside knowledge.

It goes through the hardened ``generate_model`` path — schema-validated, with a
corrective retry and a deterministic fallback — so a judge failure never raises;
it returns an "unevaluated" verdict.
"""

import json

from pydantic import BaseModel, Field

from app.services.llm_client import LLMClient
from app.services.llm_support import generate_model


class JudgeVerdict(BaseModel):
    """Scores and findings from the LLM judge (1–5 scales)."""

    groundedness: int | None = Field(default=None, ge=1, le=5)
    coverage: int | None = Field(default=None, ge=1, le=5)
    clarity: int | None = Field(default=None, ge=1, le=5)
    unsupported_claims: list[str] = Field(default_factory=list)
    rationale: str = ""

    @property
    def evaluated(self) -> bool:
        """True if the judge produced all three scores."""
        return None not in (self.groundedness, self.coverage, self.clarity)

    @property
    def overall(self) -> float | None:
        """Mean of the three scores, or None if not evaluated."""
        if not self.evaluated:
            return None
        return round((self.groundedness + self.coverage + self.clarity) / 3, 2)


_JUDGE_SYSTEM = (
    "You are a strict evaluator of job-match reports. Judge ONLY whether the "
    "report is faithful to and covers the provided structured evidence — do not "
    "use outside knowledge or reward fluent writing that adds unsupported "
    "facts. Score groundedness, coverage, and clarity each as an integer 1-5, "
    "and list any claims in the report not supported by the evidence. Respond "
    "with strict JSON."
)


def _judge_prompt(output_text: str, evidence: dict) -> str:
    """Build the judge prompt from the evidence and the text under review."""
    return (
        "Evaluate the report against the evidence and return JSON with keys: "
        "groundedness, coverage, clarity (integers 1-5), unsupported_claims "
        "(array of strings), rationale (string).\n\n"
        f"Structured evidence (JSON):\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"Report to evaluate:\n{output_text}"
    )


def judge_report(
    llm: LLMClient,
    output_text: str,
    evidence: dict,
    fallback: JudgeVerdict | None = None,
) -> JudgeVerdict:
    """Score *output_text* against *evidence* with an LLM judge.

    Args:
        llm: The judge LLM client.
        output_text: The generated artifact to evaluate (e.g. report markdown).
        evidence: The structured data the artifact should be grounded on.
        fallback: Verdict returned if the judge is unavailable/invalid
            (default: an empty, unevaluated verdict).

    Returns:
        A JudgeVerdict. When the LLM is unconfigured or fails, the verdict is
        unevaluated (scores None).
    """
    return generate_model(
        llm,
        _judge_prompt(output_text, evidence),
        JudgeVerdict,
        fallback or JudgeVerdict(),
        system=_JUDGE_SYSTEM,
    )
