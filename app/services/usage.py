"""
Token usage and cost tracking for LLM calls.

OpenAI-compatible responses report token counts in ``response.usage``.
``TokenUsage`` captures one call's counts; ``UsageTracker`` accumulates across
calls; ``estimate_cost`` converts tokens to a dollar figure given per-million
prices (which are provider/model specific).
"""

from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Token counts for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def estimate_cost(
    usage: TokenUsage,
    input_per_mtok: float,
    output_per_mtok: float,
) -> float:
    """Estimate cost in dollars from token usage and per-million-token prices.

    Args:
        usage: The token usage to price.
        input_per_mtok: Price per 1M prompt tokens.
        output_per_mtok: Price per 1M completion tokens.

    Returns:
        Estimated cost in dollars.
    """
    return (
        usage.prompt_tokens * input_per_mtok
        + usage.completion_tokens * output_per_mtok
    ) / 1_000_000


class UsageTracker:
    """Accumulates token usage across many LLM calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage: TokenUsage) -> None:
        """Add one call's usage to the running totals."""
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens

    @property
    def total_tokens(self) -> int:
        """Total prompt + completion tokens across all calls."""
        return self.prompt_tokens + self.completion_tokens

    def total(self) -> TokenUsage:
        """Aggregate usage as a TokenUsage."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
        )

    def cost(self, input_per_mtok: float, output_per_mtok: float) -> float:
        """Estimated total cost across all calls."""
        return estimate_cost(self.total(), input_per_mtok, output_per_mtok)
