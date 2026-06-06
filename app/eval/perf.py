"""
Latency measurement — collect timings and summarize them as percentiles.

p50/p95/p99 describe the experience of most vs. tail requests far better than a
mean. ``LatencyRecorder`` times code blocks via a context manager and reports a
``LatencyStats`` summary.
"""

import math
import time

from contextlib import contextmanager
from dataclasses import dataclass


def percentile(sorted_samples: list[float], p: float) -> float:
    """Return the *p*-th percentile of *sorted_samples* (linear interpolation).

    Args:
        sorted_samples: Samples sorted ascending (non-empty).
        p: Percentile in [0, 100].

    Returns:
        The interpolated percentile value.
    """
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = (p / 100) * (len(sorted_samples) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_samples[lo]
    frac = rank - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac


@dataclass
class LatencyStats:
    """Summary statistics over a set of latency samples (milliseconds)."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class LatencyRecorder:
    """Collects latency samples and summarizes them.

    Usage::

        rec = LatencyRecorder()
        with rec.measure():
            do_work()
        rec.stats()  # → LatencyStats(...)
    """

    def __init__(self) -> None:
        self.samples_ms: list[float] = []

    def record_ms(self, ms: float) -> None:
        """Record a single latency sample in milliseconds."""
        self.samples_ms.append(ms)

    @contextmanager
    def measure(self):
        """Time the wrapped block and record its latency in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.samples_ms.append((time.perf_counter() - start) * 1000)

    def stats(self) -> LatencyStats:
        """Summarize recorded samples as a LatencyStats."""
        if not self.samples_ms:
            return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s = sorted(self.samples_ms)
        return LatencyStats(
            count=len(s),
            mean_ms=sum(s) / len(s),
            p50_ms=percentile(s, 50),
            p95_ms=percentile(s, 95),
            p99_ms=percentile(s, 99),
            max_ms=s[-1],
        )
