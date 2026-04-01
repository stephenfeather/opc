"""Unit tests for benchmark metric computation functions."""

from __future__ import annotations

from scripts.benchmarks.run_rerank_benchmark import (
    compute_mrr,
    compute_ndcg,
    compute_precision_at_k,
    compute_rank_displacement,
    identify_dominant_signal,
    identify_promoted_demoted,
    is_relevant,
)


class TestIsRelevant:
    def test_match_by_id(self):
        assert is_relevant("abc", "content", ["abc", "def"], [])

    def test_no_match_by_id(self):
        assert not is_relevant("xyz", "content", ["abc", "def"], [])

    def test_match_by_keyword(self):
        assert is_relevant("x", "hook development", [], ["hook"])

    def test_keyword_case_insensitive(self):
        assert is_relevant("x", "HOOK patterns", [], ["hook"])

    def test_no_match(self):
        assert not is_relevant("x", "unrelated", ["abc"], ["hook"])

    def test_empty_golden(self):
        assert not is_relevant("x", "content", [], [])

    def test_golden_ids_authoritative_over_keywords(self):
        # When golden_ids exist, keyword match is ignored
        assert not is_relevant(
            "x", "hook code", ["abc", "def"], ["hook"]
        )

    def test_golden_ids_present_id_matches(self):
        # ID match still works when both are present
        assert is_relevant(
            "abc", "unrelated", ["abc", "def"], ["hook"]
        )

    def test_partial_golden_ids_hit_and_miss(self):
        # Only IDs in golden set are relevant
        assert is_relevant("a", "x", ["a", "b", "c"], [])
        assert not is_relevant("d", "x", ["a", "b", "c"], [])


class TestPrecisionAtK:
    def test_perfect(self):
        ids = ["a", "b", "c"]
        contents = ["hook code", "hook test", "hook fix"]
        result = compute_precision_at_k(ids, contents, ["a", "b", "c"], [])
        assert result == 1.0

    def test_none_relevant(self):
        ids = ["a", "b", "c"]
        contents = ["foo", "bar", "baz"]
        result = compute_precision_at_k(ids, contents, ["x", "y"], [])
        assert result == 0.0

    def test_partial(self):
        ids = ["a", "b", "c", "d", "e"]
        contents = ["x", "x", "x", "x", "x"]
        result = compute_precision_at_k(
            ids, contents, ["a", "c", "e"], []
        )
        assert result == 3 / 5

    def test_keyword_match(self):
        ids = ["a", "b", "c"]
        contents = ["hook code", "unrelated", "hook test"]
        result = compute_precision_at_k(ids, contents, [], ["hook"])
        assert abs(result - 2 / 3) < 1e-9

    def test_empty_results(self):
        assert compute_precision_at_k([], [], ["a"], ["hook"]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        ids = ["a", "b", "c"]
        contents = ["x", "x", "x"]
        result = compute_ndcg(ids, contents, ["a", "b", "c"], [], 3)
        assert result == 1.0

    def test_no_relevant(self):
        ids = ["a", "b", "c"]
        contents = ["x", "x", "x"]
        result = compute_ndcg(ids, contents, ["x", "y"], [], 3)
        assert result == 0.0

    def test_inverted_ranking(self):
        # Only last item is relevant, so DCG < IDCG
        ids = ["a", "b", "c"]
        contents = ["x", "x", "x"]
        result = compute_ndcg(ids, contents, ["c"], [], 3)
        assert 0.0 < result < 1.0

    def test_empty(self):
        assert compute_ndcg([], [], ["a"], [], 5) == 0.0


class TestMRR:
    def test_first_position(self):
        ids = ["a", "b", "c"]
        contents = ["x", "x", "x"]
        assert compute_mrr(ids, contents, ["a"], []) == 1.0

    def test_second_position(self):
        ids = ["a", "b", "c"]
        contents = ["x", "x", "x"]
        assert compute_mrr(ids, contents, ["b"], []) == 0.5

    def test_last_position(self):
        ids = ["a", "b", "c", "d", "e"]
        contents = ["x", "x", "x", "x", "x"]
        assert compute_mrr(ids, contents, ["e"], []) == 0.2

    def test_not_found(self):
        ids = ["a", "b", "c"]
        contents = ["x", "x", "x"]
        assert compute_mrr(ids, contents, ["z"], []) == 0.0

    def test_keyword_match(self):
        ids = ["a", "b", "c"]
        contents = ["foo", "hook code", "bar"]
        assert compute_mrr(ids, contents, [], ["hook"]) == 0.5


class TestRankDisplacement:
    def test_identical_order(self):
        mean, mx = compute_rank_displacement(
            ["a", "b", "c"], ["a", "b", "c"]
        )
        assert mean == 0.0
        assert mx == 0

    def test_reversed(self):
        mean, mx = compute_rank_displacement(
            ["c", "b", "a"], ["a", "b", "c"]
        )
        assert mx == 2
        assert abs(mean - 4 / 3) < 1e-9  # (2+0+2)/3

    def test_empty(self):
        mean, mx = compute_rank_displacement([], [])
        assert mean == 0.0
        assert mx == 0

    def test_partial_overlap(self):
        mean, mx = compute_rank_displacement(
            ["a", "d", "c"], ["a", "b", "c"]
        )
        # a: 0->0=0, c: 2->2=0, d not in raw
        assert mean == 0.0
        assert mx == 0


class TestPromotedDemoted:
    def test_promotion(self):
        promoted, demoted = identify_promoted_demoted(
            ["c", "b", "a"], ["a", "b", "c"], threshold=2
        )
        assert "c" in promoted  # moved from 2 to 0
        assert "a" in demoted  # moved from 0 to 2

    def test_no_movement(self):
        promoted, demoted = identify_promoted_demoted(
            ["a", "b", "c"], ["a", "b", "c"]
        )
        assert promoted == []
        assert demoted == []


class TestDominantSignal:
    def test_project_dominant(self):
        details = [
            {"project_match": 1.0, "recency": 0.5, "confidence": 0.5,
             "recall": 0.0, "type_match": 0.5, "tag_overlap": 0.0,
             "pattern": 0.0},
            {"project_match": 0.0, "recency": 0.5, "confidence": 0.5,
             "recall": 0.0, "type_match": 0.5, "tag_overlap": 0.0,
             "pattern": 0.0},
        ]
        assert identify_dominant_signal(details) == "project_match"

    def test_none_when_empty(self):
        assert identify_dominant_signal(None) == "none"
        assert identify_dominant_signal([]) == "none"

    def test_all_same_returns_first_with_zero_variance(self):
        details = [
            {"project_match": 0.5, "recency": 0.5, "confidence": 0.5,
             "recall": 0.5, "type_match": 0.5, "tag_overlap": 0.5,
             "pattern": 0.5},
            {"project_match": 0.5, "recency": 0.5, "confidence": 0.5,
             "recall": 0.5, "type_match": 0.5, "tag_overlap": 0.5,
             "pattern": 0.5},
        ]
        # All have zero variance, first wins
        result = identify_dominant_signal(details)
        assert result in [
            "project_match", "recency", "confidence",
            "recall", "type_match", "tag_overlap", "pattern",
        ]
