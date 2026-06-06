"""Tests for the LLM-as-Judge evaluator."""
from app.eval.llm_judge import JudgeVerdict, judge_report


class FakeLLM:
    def __init__(self, reply="", configured=True, raises=False):
        self.reply = reply
        self.configured = configured
        self.raises = raises
        self.prompts = []

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        self.prompts.append((prompt, system))
        if self.raises:
            raise RuntimeError("api down")
        return self.reply


_EVIDENCE = {"overall_score": 0.7, "matched_skills": ["python"], "missing_skills": []}
_REPORT = "# Report\nStrong match on Python."


class TestJudgeVerdict:
    def test_evaluated_true_when_all_scores(self):
        v = JudgeVerdict(groundedness=4, coverage=5, clarity=3)
        assert v.evaluated is True
        assert v.overall == 4.0

    def test_not_evaluated_when_missing_score(self):
        v = JudgeVerdict(groundedness=4, coverage=5)  # clarity None
        assert v.evaluated is False
        assert v.overall is None

    def test_empty_verdict_unevaluated(self):
        v = JudgeVerdict()
        assert v.evaluated is False
        assert v.overall is None


class TestJudgeReport:
    def test_parses_verdict(self):
        reply = (
            '{"groundedness": 5, "coverage": 4, "clarity": 4, '
            '"unsupported_claims": [], "rationale": "faithful"}'
        )
        v = judge_report(FakeLLM(reply=reply), _REPORT, _EVIDENCE)
        assert v.evaluated
        assert v.groundedness == 5
        assert v.overall == round((5 + 4 + 4) / 3, 2)
        assert v.rationale == "faithful"

    def test_flags_unsupported_claims(self):
        reply = (
            '{"groundedness": 2, "coverage": 3, "clarity": 4, '
            '"unsupported_claims": ["claims 10 years experience"], '
            '"rationale": "added facts"}'
        )
        v = judge_report(FakeLLM(reply=reply), _REPORT, _EVIDENCE)
        assert v.unsupported_claims == ["claims 10 years experience"]

    def test_prompt_contains_evidence_and_report(self):
        llm = FakeLLM(reply='{"groundedness": 5, "coverage": 5, "clarity": 5}')
        judge_report(llm, _REPORT, _EVIDENCE)
        prompt, system = llm.prompts[0]
        assert "python" in prompt          # evidence serialized in
        assert "Strong match on Python" in prompt  # report under review
        assert system is not None

    def test_fallback_when_unconfigured(self):
        v = judge_report(FakeLLM(configured=False), _REPORT, _EVIDENCE)
        assert v.evaluated is False

    def test_fallback_on_error(self):
        v = judge_report(FakeLLM(raises=True), _REPORT, _EVIDENCE)
        assert v.evaluated is False

    def test_fallback_on_out_of_range_scores(self):
        # 7 is out of the 1-5 range → validation fails → unevaluated fallback.
        reply = '{"groundedness": 7, "coverage": 4, "clarity": 4}'
        v = judge_report(FakeLLM(reply=reply), _REPORT, _EVIDENCE)
        assert v.evaluated is False

    def test_custom_fallback(self):
        fb = JudgeVerdict(rationale="custom")
        v = judge_report(FakeLLM(configured=False), _REPORT, _EVIDENCE, fallback=fb)
        assert v.rationale == "custom"
