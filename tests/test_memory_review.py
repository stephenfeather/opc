"""Tests for scripts/core/memory_review.py — candidate-detection (issue #63).

Phase 1: read-only detector. Pure routing/formatting logic is unit-tested;
the async DB handlers are tested with mocked pools.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.core.memory_review import (
    DEFAULT_MIN_RECALL,
    DEFAULT_SIMILARITY_THRESHOLD,
    MergeCandidate,
    PromotionCandidate,
    ReviewReport,
    StaleBucket,
    build_review,
    fetch_active_total,
    fetch_merge_candidates,
    fetch_promotion_candidates,
    fetch_stale_summary,
    format_report,
    is_promotable,
    route_destination,
)

# --- route_destination (pure) ---


class TestRouteDestination:
    def test_user_preference_routes_to_rules(self):
        assert route_destination("USER_PREFERENCE") == "rules/"

    def test_codebase_pattern_routes_to_memory_md(self):
        assert route_destination("CODEBASE_PATTERN") == "MEMORY.md"

    def test_architectural_decision_routes_to_claude_md(self):
        assert route_destination("ARCHITECTURAL_DECISION") == "CLAUDE.md"

    @pytest.mark.parametrize(
        "stay_type",
        ["WORKING_SOLUTION", "ERROR_FIX", "FAILED_APPROACH", "OPEN_THREAD"],
    )
    def test_on_demand_types_have_no_destination(self, stay_type):
        # These benefit from recall-on-demand; promoting them floods context.
        assert route_destination(stay_type) is None

    def test_unknown_type_has_no_destination(self):
        assert route_destination("SOMETHING_NEW") is None


class TestIsPromotable:
    def test_promotable_types(self):
        assert is_promotable("USER_PREFERENCE")
        assert is_promotable("CODEBASE_PATTERN")
        assert is_promotable("ARCHITECTURAL_DECISION")

    def test_non_promotable_types(self):
        assert not is_promotable("WORKING_SOLUTION")
        assert not is_promotable("ERROR_FIX")
        assert not is_promotable("OPEN_THREAD")


# --- format_report (pure) ---


class TestFormatReport:
    def _report(self):
        return ReviewReport(
            project="opc",
            total_active=6400,
            promotions=[
                PromotionCandidate(
                    id="a1",
                    content="Never use git commit; use github-agent-commit",
                    recall_count=34,
                    learning_type="USER_PREFERENCE",
                    destination="rules/",
                ),
                PromotionCandidate(
                    id="b2",
                    content="Recall scoping is a soft reranker boost not a hard filter",
                    recall_count=12,
                    learning_type="CODEBASE_PATTERN",
                    destination="MEMORY.md",
                ),
            ],
            merges=[
                MergeCandidate(
                    id_a="c3",
                    id_b="d4",
                    similarity=0.991,
                    preview_a="GitHub issues do not auto-close on reference",
                    preview_b="Issues don't auto-close when a PR references them",
                ),
            ],
            stale_buckets=[
                StaleBucket(label="never recalled, >60d old", count=1138),
                StaleBucket(label="recalled at least once", count=1243),
            ],
            stale_open_threads=11,
        )

    def test_includes_project_and_total(self):
        out = format_report(self._report())
        assert "opc" in out
        assert "6400" in out or "6,400" in out

    def test_groups_present(self):
        out = format_report(self._report())
        assert "Promotions" in out
        assert "Cleanup" in out or "merge" in out.lower()
        assert "Stale" in out or "stale" in out

    def test_promotion_shows_destination_and_recall(self):
        out = format_report(self._report())
        assert "rules/" in out
        assert "34" in out

    def test_merge_shows_similarity(self):
        out = format_report(self._report())
        assert "0.99" in out

    def test_empty_report_states_nothing_to_do(self):
        empty = ReviewReport(
            project="opc",
            total_active=0,
            promotions=[],
            merges=[],
            stale_buckets=[],
            stale_open_threads=0,
        )
        out = format_report(empty)
        assert isinstance(out, str)
        assert "opc" in out


# --- async DB handlers (mocked pool) ---


def _pool_returning(rows):
    """Build a mock pool whose acquired connection returns `rows` from fetch."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    conn.fetchval = AsyncMock(return_value=0)
    conn.execute = AsyncMock()
    txn_ctx = MagicMock()
    txn_ctx.__aenter__ = AsyncMock(return_value=conn)
    txn_ctx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn_ctx)
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_ctx)
    return pool, conn


class TestFetchPromotionCandidates:
    async def test_maps_rows_to_candidates_with_destination(self):
        rows = [
            {
                "id": "a1",
                "content": "x",
                "recall_count": 20,
                "learning_type": "CODEBASE_PATTERN",
            }
        ]
        pool, conn = _pool_returning(rows)
        out = await fetch_promotion_candidates(pool, "opc", DEFAULT_MIN_RECALL)
        assert len(out) == 1
        assert out[0].destination == "MEMORY.md"
        assert out[0].recall_count == 20

    async def test_scopes_query_by_project_and_min_recall(self):
        pool, conn = _pool_returning([])
        await fetch_promotion_candidates(pool, "binbrain", 7)
        args = conn.fetch.call_args.args
        sql = args[0]
        assert "LOWER(project)" in sql
        assert "superseded_by IS NULL" in sql
        # project and min_recall passed as bound params, not interpolated
        assert "binbrain" in args
        assert 7 in args


class TestFetchMergeCandidates:
    async def test_returns_merge_candidates(self):
        rows = [
            {
                "id_a": "c3",
                "id_b": "d4",
                "similarity": 0.97,
                "preview_a": "p",
                "preview_b": "q",
            }
        ]
        pool, conn = _pool_returning(rows)
        out = await fetch_merge_candidates(pool, "opc", 0.90)
        assert len(out) == 1
        assert out[0].similarity == 0.97

    async def test_ef_search_is_integer_not_interpolated(self):
        pool, conn = _pool_returning([])
        # threshold/ef_search must never reach SQL as raw strings
        await fetch_merge_candidates(pool, "opc", 0.90, ef_search=40)
        # a SET command should have run with an int-coerced value
        executed = [c.args[0] for c in conn.execute.call_args_list]
        assert any("hnsw.ef_search" in s for s in executed)
        assert any("40" in s for s in executed)


class TestFetchStaleSummary:
    async def test_returns_buckets_and_open_thread_count(self):
        bucket_rows = [
            {"staleness_bucket": "never recalled, >60d old", "learnings": 1138},
        ]
        pool, conn = _pool_returning(bucket_rows)
        conn.fetchval = AsyncMock(return_value=11)
        buckets, open_threads = await fetch_stale_summary(pool, "opc")
        assert buckets[0].count == 1138
        assert open_threads == 11


class TestFetchActiveTotal:
    async def test_returns_count(self):
        pool, conn = _pool_returning([])
        conn.fetchval = AsyncMock(return_value=6400)
        total = await fetch_active_total(pool, "opc")
        assert total == 6400


class TestBuildReview:
    async def test_assembles_full_report(self):
        pool, conn = _pool_returning([])
        conn.fetchval = AsyncMock(return_value=6400)
        report = await build_review(pool, "opc")
        assert isinstance(report, ReviewReport)
        assert report.project == "opc"
        assert report.total_active == 6400

    async def test_promote_only_skips_cleanup(self):
        pool, conn = _pool_returning([])
        conn.fetchval = AsyncMock(return_value=10)
        report = await build_review(pool, "opc", cleanup=False)
        assert report.merges == []
        assert report.stale_buckets == []

    async def test_merge_timeout_degrades_gracefully(self, monkeypatch):
        pool, conn = _pool_returning([])
        conn.fetchval = AsyncMock(return_value=6400)

        async def _raise(*a, **k):
            raise TimeoutError

        monkeypatch.setattr("scripts.core.memory_review.fetch_merge_candidates", _raise)
        report = await build_review(pool, "opc")
        # Review still completes; merges flagged as timed out, rest intact.
        assert report.merges == []
        assert report.merges_timed_out is True
        assert report.total_active == 6400

    async def test_threads_ef_search_and_timeout_to_merge(self, monkeypatch):
        pool, conn = _pool_returning([])
        conn.fetchval = AsyncMock(return_value=10)
        captured = {}

        async def _capture(_pool, _project, _threshold, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr("scripts.core.memory_review.fetch_merge_candidates", _capture)
        await build_review(pool, "opc", ef_search=20, merge_timeout=30.0)
        assert captured["ef_search"] == 20
        assert captured["timeout"] == 30.0


class TestFormatReportTimeout:
    def test_timeout_note_rendered(self):
        report = ReviewReport(project="opc", total_active=6400, merges_timed_out=True)
        out = format_report(report)
        assert "exceeded its time budget" in out
        assert "--promote-only" in out


def test_defaults_are_sane():
    assert DEFAULT_MIN_RECALL == 10
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.90
