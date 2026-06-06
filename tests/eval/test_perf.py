"""Tests for latency measurement utilities."""
from app.eval.perf import LatencyRecorder, LatencyStats, percentile


class TestPercentile:
    def test_single_sample(self):
        assert percentile([5.0], 50) == 5.0
        assert percentile([5.0], 99) == 5.0

    def test_median_odd(self):
        assert percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_p100_is_max(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0

    def test_p0_is_min(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 0) == 1.0

    def test_linear_interpolation(self):
        # rank for p95 over 0..100 (101 samples) is exactly 95.
        samples = [float(i) for i in range(101)]
        assert percentile(samples, 95) == 95.0
        # interpolated midpoint
        assert percentile([0.0, 10.0], 50) == 5.0


class TestLatencyRecorder:
    def test_record_and_stats(self):
        rec = LatencyRecorder()
        for ms in [10, 20, 30, 40, 50]:
            rec.record_ms(ms)
        stats = rec.stats()
        assert isinstance(stats, LatencyStats)
        assert stats.count == 5
        assert stats.mean_ms == 30.0
        assert stats.p50_ms == 30.0
        assert stats.max_ms == 50.0

    def test_stats_sorts_samples(self):
        rec = LatencyRecorder()
        for ms in [50, 10, 30, 20, 40]:
            rec.record_ms(ms)
        assert rec.stats().p50_ms == 30.0

    def test_empty_stats(self):
        stats = LatencyRecorder().stats()
        assert stats.count == 0
        assert stats.p95_ms == 0.0

    def test_measure_context_manager(self):
        rec = LatencyRecorder()
        with rec.measure():
            sum(range(1000))
        assert len(rec.samples_ms) == 1
        assert rec.samples_ms[0] >= 0.0

    def test_measure_records_on_exception(self):
        rec = LatencyRecorder()
        try:
            with rec.measure():
                raise ValueError("boom")
        except ValueError:
            pass
        assert len(rec.samples_ms) == 1  # recorded despite the error
