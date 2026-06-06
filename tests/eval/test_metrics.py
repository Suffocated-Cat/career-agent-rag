"""Tests for ranking metrics — hand-computed expected values."""
import math

import pytest

from app.eval.metrics import recall_at_k, mrr, dcg_at_k, ndcg_at_k


class TestRecallAtK:
    def test_partial_recall(self):
        ranked = ["a", "b", "c", "d"]
        relevant = {"a", "c", "x"}  # x is not retrievable
        # top-2 = {a, b}; only 'a' is relevant → 1 / 3
        assert recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 3)

    def test_full_recall_within_k(self):
        ranked = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(ranked, relevant, 3) == 1.0

    def test_increases_with_k(self):
        ranked = ["x", "a", "y", "b"]
        relevant = {"a", "b"}
        assert recall_at_k(ranked, relevant, 2) == 0.5
        assert recall_at_k(ranked, relevant, 4) == 1.0

    def test_no_relevant(self):
        assert recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_k_zero(self):
        assert recall_at_k(["a"], {"a"}, 0) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b"], {"a"}) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a"], {"a"}) == 0.5

    def test_third_position(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_uses_first_relevant(self):
        # b at rank 2 is the first relevant; a at rank 4 doesn't matter.
        assert mrr(["x", "b", "y", "a"], {"a", "b"}) == 0.5

    def test_none_relevant_found(self):
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert mrr(["a"], set()) == 0.0


class TestDCG:
    def test_known_value(self):
        # grades 3, 2, 0 at ranks 1, 2, 3
        ranked = ["a", "b", "c"]
        grades = {"a": 3, "b": 2, "c": 0}
        expected = 3 / math.log2(2) + 2 / math.log2(3) + 0
        assert dcg_at_k(ranked, grades, 3) == pytest.approx(expected)

    def test_missing_ids_are_zero(self):
        assert dcg_at_k(["z"], {"a": 3}, 3) == 0.0


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        ranked = ["a", "b", "c"]
        grades = {"a": 3, "b": 2, "c": 0}
        assert ndcg_at_k(ranked, grades, 3) == pytest.approx(1.0)

    def test_reversed_ranking(self):
        ranked = ["c", "b", "a"]
        grades = {"a": 3, "b": 2, "c": 0}
        dcg = 0 + 2 / math.log2(3) + 3 / math.log2(4)
        idcg = 3 / math.log2(2) + 2 / math.log2(3)
        assert ndcg_at_k(ranked, grades, 3) == pytest.approx(dcg / idcg)

    def test_no_positive_grades(self):
        assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 2) == 0.0

    def test_empty_grades(self):
        assert ndcg_at_k(["a"], {}, 2) == 0.0

    def test_in_unit_range(self):
        ranked = ["b", "a", "c"]
        grades = {"a": 2, "b": 1, "c": 3}
        val = ndcg_at_k(ranked, grades, 3)
        assert 0.0 <= val <= 1.0
