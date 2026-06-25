"""Unit tests for benchmark metric computation functions."""

from __future__ import annotations

import pytest

from scripts.benchmarks.run_rerank_benchmark import (
    ComparisonMetrics,
    QueryResult,
    assert_llm_arm_ran,
    compute_comparison_metrics,
    compute_mrr,
    compute_ndcg,
    compute_precision_at_k,
    compute_rank_displacement,
    generate_report,
    identify_dominant_signal,
    identify_promoted_demoted,
    is_relevant,
    print_summary,
    run_query,
)


def _make_query_result(
    query_id: str,
    mode: str,
    ids: list[str],
    *,
    contents: list[str] | None = None,
    scores: list[float] | None = None,
    elapsed_ms: float = 1.0,
) -> QueryResult:
    """Build a QueryResult with sensible defaults for metric tests."""
    n = len(ids)
    return QueryResult(
        query_id=query_id,
        query="q",
        mode=mode,
        result_ids=ids,
        result_scores=scores if scores is not None else [1.0] * n,
        result_contents=contents if contents is not None else ["x"] * n,
        rerank_details=None,
        elapsed_ms=elapsed_ms,
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


class TestComputeComparisonMetrics:
    """Pure metric assembly for the two- and three-way arms (Phase E)."""

    def test_two_way_leaves_llm_fields_none(self):
        reranked = _make_query_result(
            "q1", "reranked", ["a", "b", "c", "d", "e"]
        )
        raw = _make_query_result(
            "q1", "raw", ["b", "d", "a", "c", "e"]
        )
        m = compute_comparison_metrics(
            reranked, raw, ["a", "c", "e"], [], k=5,
        )
        assert m.precision_at_k_reranked == 3 / 5
        assert m.precision_at_k_raw == 3 / 5
        # No LLM arm supplied -> llm fields stay None
        assert m.precision_at_k_llm is None
        assert m.ndcg_at_k_llm is None
        assert m.mrr_llm is None
        assert m.llm_elapsed_ms is None

    def test_three_way_computes_llm_arm(self):
        reranked = _make_query_result(
            "q1", "reranked", ["a", "b", "c", "d", "e"], elapsed_ms=12.0,
        )
        raw = _make_query_result(
            "q1", "raw", ["b", "d", "a", "c", "e"], elapsed_ms=8.0,
        )
        # LLM filtered the pool down to only the relevant ids
        llm = _make_query_result(
            "q1", "llm", ["a", "c", "e"], elapsed_ms=3200.0,
        )
        m = compute_comparison_metrics(
            reranked, raw, ["a", "c", "e"], [], k=5, llm=llm,
        )
        assert m.precision_at_k_reranked == 3 / 5
        assert m.precision_at_k_raw == 3 / 5
        # LLM precision is higher because it filtered to relevant-only
        assert m.precision_at_k_llm == 1.0
        assert m.mrr_llm == 1.0  # 'a' relevant at rank 1
        assert m.ndcg_at_k_llm == 1.0  # all 3 relevant, ideal ordering
        assert m.llm_elapsed_ms == 3200.0

    def test_three_way_preserves_two_way_fields(self):
        reranked = _make_query_result("q1", "reranked", ["a", "b"])
        raw = _make_query_result("q1", "raw", ["b", "a"])
        llm = _make_query_result("q1", "llm", ["a"])
        m = compute_comparison_metrics(
            reranked, raw, ["a"], [], k=5, llm=llm,
        )
        # Displacement / promotion still computed from reranked vs raw
        assert m.query_id == "q1"
        assert m.reranked_elapsed_ms == 1.0
        assert m.raw_elapsed_ms == 1.0


class TestGenerateReportThreeWay:
    """generate_report surfaces the LLM arm when metrics carry it."""

    def _metrics(self, llm: QueryResult | None) -> ComparisonMetrics:
        reranked = _make_query_result(
            "q1", "reranked", ["a", "b", "c", "d", "e"], elapsed_ms=12.0,
        )
        raw = _make_query_result(
            "q1", "raw", ["b", "d", "a", "c", "e"], elapsed_ms=8.0,
        )
        return compute_comparison_metrics(
            reranked, raw, ["a", "c", "e"], [], k=5, llm=llm,
        )

    def test_two_way_report_has_no_llm_keys(self):
        m = self._metrics(llm=None)
        reranked = _make_query_result("q1", "reranked", ["a"])
        raw = _make_query_result("q1", "raw", ["a"])
        report = generate_report([(reranked, raw, m)], [{"id": "q1", "k": 5}])
        assert "llm" not in report["summary"]["precision_at_k"]

    def test_three_way_report_surfaces_llm_and_delta(self):
        llm = _make_query_result(
            "q1", "llm", ["a", "c", "e"], elapsed_ms=3200.0,
        )
        m = self._metrics(llm=llm)
        reranked = _make_query_result(
            "q1", "reranked", ["a", "b", "c", "d", "e"]
        )
        raw = _make_query_result("q1", "raw", ["b", "d", "a", "c", "e"])
        report = generate_report([(reranked, raw, m)], [{"id": "q1", "k": 5}])
        p = report["summary"]["precision_at_k"]
        assert p["llm"] == 1.0
        assert p["reranked"] == round(3 / 5, 4)
        # Acceptance signal: precision_at_k(llm) - precision_at_k(reranked)
        assert p["llm_vs_reranked"] == round(1.0 - 3 / 5, 4)
        assert "llm_avg" in report["summary"]["latency_ms"]

    def test_two_way_per_query_has_no_null_llm_keys(self):
        # Default (no LLM arm) per_query entries must NOT carry the new llm
        # fields as null — the two-way report shape stays byte-for-byte stable.
        m = self._metrics(llm=None)
        reranked = _make_query_result("q1", "reranked", ["a"])
        raw = _make_query_result("q1", "raw", ["a"])
        report = generate_report([(reranked, raw, m)], [{"id": "q1", "k": 5}])
        entry = report["per_query"][0]
        for key in (
            "precision_at_k_llm", "ndcg_at_k_llm", "mrr_llm", "llm_elapsed_ms",
        ):
            assert key not in entry

    def test_three_way_per_query_carries_llm_keys(self):
        llm = _make_query_result("q1", "llm", ["a", "c", "e"], elapsed_ms=3200.0)
        m = self._metrics(llm=llm)
        reranked = _make_query_result(
            "q1", "reranked", ["a", "b", "c", "d", "e"]
        )
        raw = _make_query_result("q1", "raw", ["b", "d", "a", "c", "e"])
        report = generate_report([(reranked, raw, m)], [{"id": "q1", "k": 5}])
        entry = report["per_query"][0]
        assert entry["precision_at_k_llm"] == 1.0
        assert entry["llm_elapsed_ms"] == 3200.0

    def test_llm_delta_uses_same_subset_baseline(self):
        # One query has the LLM arm, one does not. The llm_vs_reranked delta
        # must subtract the reranked baseline of ONLY the LLM-arm query, not the
        # full-n reranked average (which would mix populations).
        # Query 1: reranked precision 0.6, llm precision 1.0.
        rr1 = _make_query_result("q1", "reranked", ["a", "b", "c", "d", "e"])
        raw1 = _make_query_result("q1", "raw", ["b", "d", "a", "c", "e"])
        llm1 = _make_query_result("q1", "llm", ["a", "c", "e"])
        m1 = compute_comparison_metrics(
            rr1, raw1, ["a", "c", "e"], [], k=5, llm=llm1,
        )
        # Query 2: reranked precision 0.0, NO llm arm.
        rr2 = _make_query_result("q2", "reranked", ["x", "y"])
        raw2 = _make_query_result("q2", "raw", ["y", "x"])
        m2 = compute_comparison_metrics(rr2, raw2, ["z"], [], k=5)

        report = generate_report(
            [(rr1, raw1, m1), (rr2, raw2, m2)],
            [{"id": "q1", "k": 5}, {"id": "q2", "k": 5}],
        )
        s = report["summary"]
        assert s["llm_arm_queries"] == 1
        # Subset baseline: 1.0 - 0.6 = 0.4 (NOT 1.0 - mean(0.6, 0.0) = 0.7).
        assert s["precision_at_k"]["llm_vs_reranked"] == round(1.0 - 3 / 5, 4)
        assert s["precision_at_k"]["llm"] == 1.0


class TestRunQueryLlmGuard:
    """run_query rejects the impossible llm-without-rerank combination."""

    async def test_llm_rerank_requires_rerank(self):
        # The LLM stage lives on the rerank path; --no-rerank suppresses it,
        # so this combination must fail fast before spawning a subprocess.
        with pytest.raises(ValueError, match="rerank=True"):
            await run_query("q1", "q", 5, rerank=False, llm_rerank=True)


class TestAssertLlmArmRan:
    """The LLM arm must fail closed on a silent reranker fallback."""

    def _llm_row(self, rid: str) -> dict:
        return {
            "id": rid,
            "rerank_details": {
                "source": "llm_selector", "model": "claude-sonnet-4-6",
                "rank": 0,
            },
        }

    def _reranker_row(self, rid: str) -> dict:
        # The deterministic reranker stamps per-signal scores, no "source".
        return {
            "id": rid,
            "rerank_details": {
                "project_match": 0.5, "recency": 0.1, "confidence": 0.2,
            },
        }

    def test_passes_when_all_rows_llm_selected(self):
        assert_llm_arm_ran("q1", [self._llm_row("a"), self._llm_row("b")])

    def test_raises_on_reranker_fallback(self):
        with pytest.raises(RuntimeError, match="fell back"):
            assert_llm_arm_ran("q1", [self._reranker_row("a")])

    def test_raises_on_mixed_rows(self):
        with pytest.raises(RuntimeError, match="llm_selector"):
            assert_llm_arm_ran(
                "q1", [self._llm_row("a"), self._reranker_row("b")]
            )

    def test_raises_when_rerank_details_missing(self):
        with pytest.raises(RuntimeError):
            assert_llm_arm_ran("q1", [{"id": "a"}])

    def test_empty_results_pass(self):
        # An empty pool is ambiguous and shared by all arms; let the metric
        # layer record precision 0 instead of aborting the whole benchmark.
        assert_llm_arm_ran("q1", [])


class TestPrintSummaryThreeWay:
    """print_summary renders the LLM arm without crashing when present."""

    def _three_way_report(self) -> dict:
        llm = _make_query_result(
            "q1", "llm", ["a", "c", "e"], elapsed_ms=3200.0,
        )
        reranked = _make_query_result(
            "q1", "reranked", ["a", "b", "c", "d", "e"], elapsed_ms=12.0,
        )
        raw = _make_query_result(
            "q1", "raw", ["b", "d", "a", "c", "e"], elapsed_ms=8.0,
        )
        m = compute_comparison_metrics(
            reranked, raw, ["a", "c", "e"], [], k=5, llm=llm,
        )
        return generate_report([(reranked, raw, m)], [{"id": "q1", "k": 5}])

    def test_renders_llm_section(self, capsys):
        print_summary(self._three_way_report())
        out = capsys.readouterr().out
        assert "LLM-as-selector arm" in out
        assert "LLM wins vs reranked" in out

    def test_two_way_omits_llm_section(self, capsys):
        reranked = _make_query_result("q1", "reranked", ["a"])
        raw = _make_query_result("q1", "raw", ["a"])
        m = compute_comparison_metrics(reranked, raw, ["a"], [], k=5)
        report = generate_report([(reranked, raw, m)], [{"id": "q1", "k": 5}])
        print_summary(report)
        out = capsys.readouterr().out
        assert "LLM-as-selector arm" not in out


class TestGenerateReportWinners:
    """Winner tallies for the reranked-vs-raw arm (existing two-way path)."""

    def test_raw_beats_reranked_counts_raw_win(self):
        # raw returns only the relevant id (precision 1.0); reranked dilutes it
        # with an irrelevant row (precision 0.5) -> raw wins the tally.
        reranked = _make_query_result("q1", "reranked", ["x", "a"])
        raw = _make_query_result("q1", "raw", ["a"])
        m = compute_comparison_metrics(reranked, raw, ["a"], [], k=5)
        report = generate_report([(reranked, raw, m)], [{"id": "q1", "k": 5}])
        assert report["summary"]["raw_wins"] == 1
        assert report["summary"]["rerank_wins"] == 0
        assert report["summary"]["ties"] == 0


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
