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

    def test_promotion_shows_full_id_for_apply(self):
        # The apply step needs the full uuid; the report must expose it (not just type/recall).
        out = format_report(self._report())
        assert "id=a1" in out  # full id surfaced for `memory-apply --ids`

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
    # Coverage query default: one embedding space, nothing skipped, so build_review
    # resolves a non-None model and proceeds to the merge scan.
    conn.fetchrow = AsyncMock(
        return_value={
            "scanned_model": "voyage-code-3",
            "scanned_rows": 100,
            "total_embedded": 100,
        }
    )
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
        out = await fetch_merge_candidates(pool, "opc", "voyage-code-3", threshold=0.90)
        assert len(out) == 1
        assert out[0].similarity == 0.97

    async def test_ef_search_is_integer_not_interpolated(self):
        pool, conn = _pool_returning([])
        # threshold/ef_search must never reach SQL as raw strings
        await fetch_merge_candidates(pool, "opc", "voyage-code-3", threshold=0.90, ef_search=40)
        # a SET command should have run with an int-coerced value
        executed = [c.args[0] for c in conn.execute.call_args_list]
        assert any("hnsw.ef_search" in s for s in executed)
        assert any("40" in s for s in executed)

    async def test_neighbors_and_model_passed_as_bound_params(self):
        pool, conn = _pool_returning([])
        await fetch_merge_candidates(pool, "opc", "voyage-code-3", threshold=0.90, neighbors=7)
        args = conn.fetch.call_args.args
        assert 7 in args  # top-k neighbors bound, not interpolated
        assert "voyage-code-3" in args  # model bound, not recomputed in SQL

    async def test_statement_timeout_set_server_side(self):
        pool, conn = _pool_returning([])
        await fetch_merge_candidates(pool, "opc", "voyage-code-3", threshold=0.90, timeout=30.0)
        executed = [c.args[0] for c in conn.execute.call_args_list]
        # Server-enforced budget so the backend actually stops, not just the client.
        assert any("statement_timeout" in s for s in executed)
        assert any("30000" in s for s in executed)  # 30s -> ms

    def test_merge_sql_scopes_to_single_embedding_model(self):
        # Cross-model cosine is meaningless; the scan must restrict to one space.
        # The model is resolved once (coverage) and bound as $5 here (round 3).
        from scripts.core.memory_review import _MERGE_COVERAGE_SQL, _MERGE_SQL

        assert "embedding_model = $5" in _MERGE_SQL
        assert "embedding_model" in _MERGE_COVERAGE_SQL

    def test_merge_sql_canonicalizes_pairs_not_directed_filter(self):
        # The buggy "a.id < nn.id" directed filter drops real pairs; the fix uses
        # LEAST/GREATEST canonicalization with DISTINCT ON.
        from scripts.core.memory_review import _MERGE_SQL

        assert "LEAST" in _MERGE_SQL
        assert "GREATEST" in _MERGE_SQL
        assert "DISTINCT ON" in _MERGE_SQL
        assert "a.id < nn.id" not in _MERGE_SQL


class TestFetchMergePairDetails:
    """fetch_merge_pair_details: resolve two ids to MergeRows in ONE batched query."""

    _A = "11111111-1111-1111-1111-111111111111"
    _B = "22222222-2222-2222-2222-222222222222"

    def _rows(self):
        import datetime as _dt

        return [
            {
                "id": self._A,
                "recall_count": 5,
                "created_at": _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
                "superseded_by": None,
            },
            {
                "id": self._B,
                "recall_count": 9,
                "created_at": _dt.datetime(2026, 2, 1, tzinfo=_dt.UTC),
                "superseded_by": None,
            },
        ]

    async def test_returns_both_rows_keyed_by_id(self):
        from scripts.core.memory_review import MergeRow, fetch_merge_pair_details

        pool, conn = _pool_returning(self._rows())
        out = await fetch_merge_pair_details(pool, "opc", self._A, self._B)
        assert isinstance(out[self._A], MergeRow)
        assert isinstance(out[self._B], MergeRow)
        assert out[self._A].recall_count == 5
        assert out[self._B].recall_count == 9
        assert out[self._A].superseded_by is None

    async def test_carries_created_at_and_superseded_by(self):
        from scripts.core.memory_review import fetch_merge_pair_details

        rows = self._rows()
        rows[0]["superseded_by"] = self._B  # row_a already superseded
        pool, conn = _pool_returning(rows)
        out = await fetch_merge_pair_details(pool, "opc", self._A, self._B)
        assert out[self._A].superseded_by == self._B
        assert out[self._A].created_at is not None
        assert out[self._B].created_at is not None

    async def test_single_batched_any_query_not_per_id(self):
        from scripts.core.memory_review import fetch_merge_pair_details

        pool, conn = _pool_returning(self._rows())
        await fetch_merge_pair_details(pool, "opc", self._A, self._B)
        # ONE round-trip, batched via id = ANY (plan note N-1), never two per-id fetches.
        assert conn.fetch.await_count == 1
        sql = conn.fetch.await_args.args[0]
        assert "ANY(" in sql
        assert "= ANY" in sql or "id::text = ANY" in sql

    async def test_scoped_by_project_and_selects_required_columns(self):
        from scripts.core.memory_review import fetch_merge_pair_details

        pool, conn = _pool_returning(self._rows())
        await fetch_merge_pair_details(pool, "binbrain", self._A, self._B)
        args = conn.fetch.await_args.args
        sql = args[0]
        assert "LOWER(project)" in sql
        assert "recall_count" in sql
        assert "created_at" in sql
        assert "superseded_by" in sql
        assert "binbrain" in args  # project bound, not interpolated

    async def test_missing_id_absent_from_result(self):
        # Only one of the two ids resolves (e.g. the other was hard-deleted).
        from scripts.core.memory_review import fetch_merge_pair_details

        pool, conn = _pool_returning([self._rows()[0]])
        out = await fetch_merge_pair_details(pool, "opc", self._A, self._B)
        assert self._A in out
        assert self._B not in out


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


class TestFetchStaleIds:
    """fetch_stale_ids (issue #63 Phase 2b Step 3): read-only — returns the ids
    eligible for stale archival, mirroring the StaleBucket >60d predicate."""

    _A = "11111111-1111-1111-1111-111111111111"
    _B = "22222222-2222-2222-2222-222222222222"

    async def test_returns_ids_for_eligible_rows(self):
        from scripts.core.memory_review import fetch_stale_ids

        pool, conn = _pool_returning([{"id": self._A}, {"id": self._B}])
        out = await fetch_stale_ids(pool, "opc")
        assert out == [self._A, self._B]

    async def test_predicate_matches_stale_bucket(self):
        """Same predicate the >60d stale bucket uses: recall_count = 0,
        created_at older than 60 days, AND active (not superseded, not archived)."""
        from scripts.core.memory_review import fetch_stale_ids

        pool, conn = _pool_returning([])
        await fetch_stale_ids(pool, "opc")
        sql = conn.fetch.await_args.args[0]
        assert "recall_count = 0" in sql
        assert "60 days" in sql or "make_interval" in sql
        assert "superseded_by IS NULL" in sql
        assert "archived_at IS NULL" in sql

    async def test_scoped_by_project(self):
        from scripts.core.memory_review import fetch_stale_ids

        pool, conn = _pool_returning([])
        await fetch_stale_ids(pool, "binbrain")
        args = conn.fetch.await_args.args
        assert "LOWER(project)" in args[0]
        assert "binbrain" in args  # bound, not interpolated

    async def test_read_only_no_writes(self):
        from scripts.core.memory_review import fetch_stale_ids

        pool, conn = _pool_returning([])
        await fetch_stale_ids(pool, "opc")
        # Read-only: no UPDATE/DELETE/INSERT/execute.
        conn.execute.assert_not_called()
        sql = conn.fetch.await_args.args[0].upper()
        assert "UPDATE" not in sql and "DELETE" not in sql and "INSERT" not in sql


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

    async def test_query_canceled_degrades_gracefully(self, monkeypatch):
        # Server-side statement_timeout raises QueryCanceledError; same graceful path.
        from asyncpg.exceptions import QueryCanceledError

        pool, conn = _pool_returning([])
        conn.fetchval = AsyncMock(return_value=6400)

        async def _raise(*a, **k):
            raise QueryCanceledError

        monkeypatch.setattr("scripts.core.memory_review.fetch_merge_candidates", _raise)
        report = await build_review(pool, "opc")
        assert report.merges == []
        assert report.merges_timed_out is True

    async def test_merge_skipped_when_no_embedding_model(self, monkeypatch):
        # Empty/embedding-less corpus: coverage returns no model -> never run the scan.
        pool, conn = _pool_returning([])
        conn.fetchval = AsyncMock(return_value=0)
        conn.fetchrow = AsyncMock(
            return_value={"scanned_model": None, "scanned_rows": 0, "total_embedded": 0}
        )
        called = False

        async def _should_not_run(*a, **k):
            nonlocal called
            called = True
            return []

        monkeypatch.setattr("scripts.core.memory_review.fetch_merge_candidates", _should_not_run)
        report = await build_review(pool, "opc")
        assert called is False
        assert report.merges == []
        assert report.merges_timed_out is False

    async def test_coverage_model_reused_for_merge_no_drift(self, monkeypatch):
        # The model the report discloses must be the exact model the merge scan used.
        pool, conn = _pool_returning([])
        conn.fetchval = AsyncMock(return_value=10)
        conn.fetchrow = AsyncMock(
            return_value={
                "scanned_model": "bge",
                "scanned_rows": 8,
                "total_embedded": 10,
            }
        )
        captured = {}

        async def _capture(_pool, _project, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)
            return []

        monkeypatch.setattr("scripts.core.memory_review.fetch_merge_candidates", _capture)
        report = await build_review(pool, "opc", ef_search=20, merge_timeout=30.0)
        assert captured["model"] == "bge"  # same model coverage reported
        assert report.merge_scanned_model == "bge"
        assert report.merge_skipped_rows == 2
        assert captured["ef_search"] == 20
        assert captured["timeout"] == 30.0


class TestFormatReportTimeout:
    def test_timeout_note_rendered(self):
        report = ReviewReport(project="opc", total_active=6400, merges_timed_out=True)
        out = format_report(report)
        assert "exceeded its time budget" in out
        assert "--promote-only" in out

    def test_timeout_does_not_read_as_zero(self):
        # Regression (round 2): a timed-out scan must not render "(0 ...)" or fall
        # through to the merges "(none)" line.
        report = ReviewReport(project="opc", total_active=6400, merges_timed_out=True)
        out = format_report(report)
        assert "not scanned: timed out" in out
        assert "0 near-duplicate pairs" not in out
        # Isolate the merges section and confirm it has no "(none)" placeholder.
        merges_section = out.split("### 2.")[1].split("### 3.")[0]
        assert "(none)" not in merges_section


class TestFormatReportCoverage:
    def test_partial_scan_disclosed(self):
        report = ReviewReport(
            project="opc",
            total_active=100,
            merge_scanned_model="voyage-code-3",
            merge_skipped_rows=42,
        )
        out = format_report(report)
        assert "partial scan" in out
        assert "voyage-code-3" in out
        assert "42" in out

    def test_full_scan_no_disclosure(self):
        report = ReviewReport(project="opc", total_active=100, merge_skipped_rows=0)
        assert "partial scan" not in format_report(report)

    def test_merge_pair_shows_ids(self):
        report = ReviewReport(
            project="opc",
            total_active=100,
            merges=[
                MergeCandidate(
                    id_a="abcd1234-0000",
                    id_b="efgh5678-0000",
                    similarity=0.95,
                    preview_a="p",
                    preview_b="q",
                )
            ],
        )
        out = format_report(report)
        assert "abcd1234" in out
        assert "efgh5678" in out


class TestMergeSqlPreviewCanonicalization:
    def test_previews_tied_to_canonical_ids(self):
        # Regression (round 2): preview_lo/hi must follow LEAST/GREATEST id, not the
        # directed (a, nn) roles, or the approval flow shows the wrong side's content.
        from scripts.core.memory_review import _MERGE_SQL

        assert "CASE WHEN a.id <= nn.id THEN a.content ELSE nn.content END" in _MERGE_SQL
        assert "CASE WHEN a.id <= nn.id THEN nn.content ELSE a.content END" in _MERGE_SQL

    def test_dominant_model_has_deterministic_tiebreak(self):
        # Model resolution lives in the coverage query (round 3); reused for the scan.
        from scripts.core.memory_review import _MERGE_COVERAGE_SQL

        assert "n DESC, embedding_model ASC" in _MERGE_COVERAGE_SQL

    def test_merge_lateral_reads_base_table_not_materialized_cte(self):
        # Regression (round 3): the lateral must read archival_memory directly so the
        # CTE materialization boundary cannot strip index eligibility.
        from scripts.core.memory_review import _MERGE_SQL

        assert "FROM archival_memory b" in _MERGE_SQL
        assert "embedding_model = $5" in _MERGE_SQL


class TestDefaultProject:
    def test_worktree_path_resolves_to_repo(self, monkeypatch):
        # Regression (round 2): running from a worktree must review the repo, not the
        # branch directory name.
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setattr(
            "scripts.core.memory_review.os.getcwd",
            lambda: "/Users/x/opc/.claude/worktrees/agent-memory-review-detector",
        )
        from scripts.core.memory_review import _default_project

        assert _default_project() == "opc"

    def test_env_project_dir_honored(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/Users/x/binbrain")
        from scripts.core.memory_review import _default_project

        assert _default_project() == "binbrain"


class TestMainProjectResolution:
    async def test_unresolved_project_returns_error_code(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setattr("scripts.core.memory_review.canonicalize_project", lambda _x: None)
        from scripts.core.memory_review import main

        # Explicit but unresolvable project → non-zero exit, no DB call.
        rc = await main(["   "])
        assert rc == 2


class TestArgValidation:
    # Security (MEDIUM-1/LOW-1): reject inputs that would defeat the scan guard or
    # crash the GUC, before they ever reach Postgres.
    @pytest.mark.parametrize(
        "argv",
        [
            ["opc", "--merge-timeout", "0"],
            ["opc", "--merge-timeout", "-5"],
            ["opc", "--ef-search", "0"],
            ["opc", "--ef-search", "-1"],
            ["opc", "--min-recall", "-1"],
            ["opc", "--threshold", "1.5"],
            ["opc", "--threshold", "-0.1"],
        ],
    )
    def test_invalid_values_rejected(self, argv):
        from scripts.core.memory_review import _parse_args

        with pytest.raises(SystemExit):
            _parse_args(argv)

    def test_valid_boundary_values_accepted(self):
        from scripts.core.memory_review import _parse_args

        ns = _parse_args(["opc", "--ef-search", "1", "--merge-timeout", "0.5", "--min-recall", "0"])
        assert ns.ef_search == 1
        assert ns.merge_timeout == 0.5
        assert ns.min_recall == 0

    def test_promote_only_and_cleanup_only_are_mutually_exclusive(self):
        # Passing both would disable promote AND cleanup → an empty report.
        from scripts.core.memory_review import _parse_args

        with pytest.raises(SystemExit):
            _parse_args(["opc", "--promote-only", "--cleanup-only"])


class TestMergeGuardClamps:
    # Defense-in-depth for programmatic callers that bypass argparse.
    async def test_nonpositive_timeout_does_not_set_statement_timeout(self):
        pool, conn = _pool_returning([])
        await fetch_merge_candidates(pool, "opc", "voyage-code-3", threshold=0.9, timeout=0)
        executed = [c.args[0] for c in conn.execute.call_args_list]
        # statement_timeout=0 means "no limit" in Postgres — must never be sent.
        assert not any("statement_timeout" in s for s in executed)
        # asyncpg client timeout must be coerced to None, not passed as 0/negative.
        assert conn.fetch.call_args.kwargs.get("timeout") is None

    async def test_ef_search_clamped_to_minimum_one(self):
        pool, conn = _pool_returning([])
        await fetch_merge_candidates(pool, "opc", "voyage-code-3", threshold=0.9, ef_search=0)
        executed = [c.args[0] for c in conn.execute.call_args_list]
        assert any("hnsw.ef_search = 1" in s for s in executed)


class TestMainDbErrorHandling:
    async def test_db_error_returns_clean_exit_code(self, monkeypatch):
        # Security (LOW-2): DB failures fail cleanly, no traceback/DSN leak.
        from asyncpg.exceptions import PostgresError

        import scripts.core.memory_review as mr

        async def _boom():
            raise PostgresError("connection refused to claude:secret@host")

        monkeypatch.setattr(mr, "_default_project", lambda: "opc")
        monkeypatch.setattr(mr, "get_pool", _boom)
        rc = await mr.main(["opc"])
        assert rc == 1

    async def test_config_error_returns_clean_exit_code(self, monkeypatch):
        # Missing DATABASE_URL surfaces as ValueError from get_pool — must exit clean.
        import scripts.core.memory_review as mr

        async def _no_url():
            raise ValueError("Database URL not set (environment='<unset>').")

        monkeypatch.setattr(mr, "_default_project", lambda: "opc")
        monkeypatch.setattr(mr, "get_pool", _no_url)
        rc = await mr.main(["opc"])
        assert rc == 1


def test_defaults_are_sane():
    assert DEFAULT_MIN_RECALL == 10
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.90
