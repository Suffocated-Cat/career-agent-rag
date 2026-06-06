"""Tests for token usage and cost tracking."""
import pytest

from app.services.usage import TokenUsage, UsageTracker, estimate_cost


class TestEstimateCost:
    def test_basic(self):
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
        # $1/Mtok input, $2/Mtok output → 1*1 + 0.5*2 = 2.0
        assert estimate_cost(usage, 1.0, 2.0) == pytest.approx(2.0)

    def test_zero_usage(self):
        assert estimate_cost(TokenUsage(), 1.0, 2.0) == 0.0


class TestUsageTracker:
    def test_accumulates(self):
        t = UsageTracker()
        t.add(TokenUsage(prompt_tokens=100, completion_tokens=50))
        t.add(TokenUsage(prompt_tokens=200, completion_tokens=80))
        assert t.calls == 2
        assert t.prompt_tokens == 300
        assert t.completion_tokens == 130
        assert t.total_tokens == 430

    def test_total_returns_token_usage(self):
        t = UsageTracker()
        t.add(TokenUsage(prompt_tokens=10, completion_tokens=5))
        total = t.total()
        assert isinstance(total, TokenUsage)
        assert total.total_tokens == 15

    def test_cost(self):
        t = UsageTracker()
        t.add(TokenUsage(prompt_tokens=2_000_000, completion_tokens=1_000_000))
        assert t.cost(1.0, 3.0) == pytest.approx(2.0 + 3.0)

    def test_empty(self):
        t = UsageTracker()
        assert t.total_tokens == 0
        assert t.cost(1.0, 2.0) == 0.0
