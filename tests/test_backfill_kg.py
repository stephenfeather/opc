"""Tests for scripts/core/backfill_kg.py — KG backfill for existing memories (#124)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.core.backfill_kg import (
    backfill_one,
    build_fetch_query,
    format_summary,
    mark_no_entities,
    parse_args,
    run_backfill,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_pool(conn: AsyncMock) -> MagicMock:
    """Build a mock asyncpg pool whose acquire() yields the given conn."""
    acm = AsyncMock()
    acm.__aenter__.return_value = conn
    acm.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = acm
    return pool


def _row(content: str = "content") -> dict:
    """Build a fake archival_memory row."""
    return {
        "id": uuid.uuid4(),
        "content": content,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }


# ---------------------------------------------------------------------------
# Pure: build_fetch_query
# ---------------------------------------------------------------------------


class TestBuildFetchQuery:
    def test_default_selects_unindexed_unmarked_memories(self):
        sql, params = build_fetch_query()
        assert "SELECT id, content" in sql
        assert "archival_memory" in sql
        assert "NOT EXISTS" in sql
        assert "kg_entity_mentions" in sql
        # Zero-entity memories are durably marked and excluded on reruns
        assert "kg_backfill" in sql
        # Keyset pagination requires a deterministic order with id tiebreak
        assert "ORDER BY m.created_at, m.id" in sql
        assert params == []

    def test_count_only_uses_count_star(self):
        sql, params = build_fetch_query(count_only=True)
        assert "count(*)" in sql.lower()
        assert "NOT EXISTS" in sql
        assert "kg_backfill" in sql
        assert "ORDER BY" not in sql
        assert params == []

    def test_limit_adds_numbered_param(self):
        sql, params = build_fetch_query(limit=500)
        assert "LIMIT $1" in sql
        assert params == [500]

    def test_since_filters_created_at(self):
        since = datetime(2026, 1, 1, tzinfo=UTC)
        sql, params = build_fetch_query(since=since)
        assert "created_at >= $1" in sql
        assert params == [since]

    def test_memory_id_filters_id_and_bypasses_marker(self):
        mid = str(uuid.uuid4())
        sql, params = build_fetch_query(memory_id=mid)
        assert "id = $1" in sql
        assert params == [mid]
        # Targeted repair: --memory-id must reprocess rows previously
        # marked kg_backfill=no_entities
        assert "kg_backfill" not in sql

    def test_project_filters_project(self):
        sql, params = build_fetch_query(project="opc")
        assert "project = $1" in sql
        assert params == ["opc"]

    def test_recheck_no_entities_includes_marked_rows(self):
        # After an extractor upgrade, bulk runs can revisit rows previously
        # marked no_entities
        sql, params = build_fetch_query(recheck_no_entities=True)
        assert "kg_backfill" not in sql
        assert params == []

    def test_after_adds_keyset_predicate(self):
        after = (datetime(2026, 1, 1, tzinfo=UTC), str(uuid.uuid4()))
        sql, params = build_fetch_query(after=after)
        assert "(m.created_at, m.id) > ($1, $2::uuid)" in sql
        assert params == [after[0], after[1]]

    def test_combined_filters_number_params_consecutively(self):
        since = datetime(2026, 1, 1, tzinfo=UTC)
        after = (datetime(2026, 2, 1, tzinfo=UTC), str(uuid.uuid4()))
        sql, params = build_fetch_query(limit=10, since=since, after=after)
        assert "created_at >= $1" in sql
        assert "(m.created_at, m.id) > ($2, $3::uuid)" in sql
        assert "LIMIT $4" in sql
        assert params == [since, after[0], after[1], 10]


# ---------------------------------------------------------------------------
# Pure: parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.dry_run is False
        assert args.limit is None
        assert args.since is None
        assert args.memory_id is None
        assert args.batch_size == 500

    def test_dry_run_flag(self):
        assert parse_args(["--dry-run"]).dry_run is True

    def test_since_parsed_as_datetime(self):
        args = parse_args(["--since", "2026-01-01"])
        assert isinstance(args.since, datetime)

    def test_naive_since_normalized_to_utc(self):
        args = parse_args(["--since", "2026-01-01"])
        assert args.since.tzinfo is UTC

    def test_aware_since_preserves_offset(self):
        args = parse_args(["--since", "2026-01-01T00:00:00+05:00"])
        assert args.since.utcoffset().total_seconds() == 5 * 3600

    def test_z_suffix_since_accepted_as_utc(self):
        # Python >= 3.11 fromisoformat accepts the Z suffix natively
        # (repo floor is 3.13); regression-documented for PR #132 review
        args = parse_args(["--since", "2026-01-01T00:00:00Z"])
        assert args.since.utcoffset().total_seconds() == 0

    def test_project_flag(self):
        assert parse_args([]).project is None
        assert parse_args(["--project", "opc"]).project == "opc"

    def test_invalid_since_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--since", "not-a-date"])

    def test_memory_id_validated_as_uuid(self):
        mid = str(uuid.uuid4())
        assert parse_args(["--memory-id", mid]).memory_id == mid

    def test_invalid_memory_id_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--memory-id", "not-a-uuid"])

    def test_zero_limit_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--limit", "0"])

    def test_recheck_no_entities_flag(self):
        assert parse_args([]).recheck_no_entities is False
        assert parse_args(["--recheck-no-entities"]).recheck_no_entities is True

    def test_max_consecutive_errors_default_and_override(self):
        # Issue #131 circuit breaker threshold
        assert parse_args([]).max_consecutive_errors == 10
        assert parse_args(["--max-consecutive-errors", "3"]).max_consecutive_errors == 3

    def test_zero_max_consecutive_errors_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--max-consecutive-errors", "0"])

    def test_memory_id_exclusive_with_other_scoping_flags(self):
        mid = str(uuid.uuid4())
        for extra in (
            ["--since", "2026-01-01"],
            ["--project", "opc"],
            ["--limit", "5"],
        ):
            with pytest.raises(SystemExit):
                parse_args(["--memory-id", mid, *extra])


# ---------------------------------------------------------------------------
# Pure: format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_includes_all_counts(self):
        text = format_summary({"processed": 7, "indexed": 4, "no_entities": 2, "errors": 1})
        assert "7" in text
        assert "4" in text
        assert "2" in text
        assert "1" in text


# ---------------------------------------------------------------------------
# Async: backfill_one
# ---------------------------------------------------------------------------


class TestBackfillOne:
    async def test_indexed_path_stores_entities(self):
        mid = str(uuid.uuid4())
        entities = [MagicMock()]
        kg_stats = {"entities": 1, "edges": 0, "mentions": 1}

        with (
            patch("scripts.core.backfill_kg.extract_entities", return_value=entities) as mock_ents,
            patch("scripts.core.backfill_kg.extract_relations", return_value=[]) as mock_rels,
            patch(
                "scripts.core.backfill_kg.store_entities_and_edges",
                new_callable=AsyncMock,
                return_value=kg_stats,
            ) as mock_store,
        ):
            result = await backfill_one(mid, "uses pytest with asyncpg")

        assert result["status"] == "indexed"
        assert result["stats"] == kg_stats
        mock_ents.assert_called_once_with("uses pytest with asyncpg")
        mock_rels.assert_called_once()
        mock_store.assert_called_once_with(mid, entities, [])

    async def test_no_entities_skips_store(self):
        with (
            patch("scripts.core.backfill_kg.extract_entities", return_value=[]),
            patch(
                "scripts.core.backfill_kg.store_entities_and_edges",
                new_callable=AsyncMock,
            ) as mock_store,
        ):
            result = await backfill_one(str(uuid.uuid4()), "plain text")

        assert result["status"] == "no_entities"
        mock_store.assert_not_called()

    async def test_exception_is_nonfatal(self):
        with patch(
            "scripts.core.backfill_kg.extract_entities",
            side_effect=RuntimeError("boom"),
        ):
            result = await backfill_one(str(uuid.uuid4()), "anything")

        assert result["status"] == "error"
        assert "boom" in result["error"]

    async def test_infrastructure_errors_propagate(self):
        # Issue #131: a dead DB mid-run must abort promptly, not get logged
        # once per row across the whole backlog like a bad-content row
        import asyncpg

        infra_errors = [
            asyncpg.PostgresConnectionError("db down"),
            asyncpg.InterfaceError("pool is closing"),
            ConnectionResetError("socket reset"),
            TimeoutError("pool acquire timed out"),
        ]
        for exc in infra_errors:
            with (
                patch(
                    "scripts.core.backfill_kg.extract_entities",
                    return_value=[MagicMock()],
                ),
                patch("scripts.core.backfill_kg.extract_relations", return_value=[]),
                patch(
                    "scripts.core.backfill_kg.store_entities_and_edges",
                    new_callable=AsyncMock,
                    side_effect=exc,
                ),
            ):
                with pytest.raises(type(exc)):
                    await backfill_one(str(uuid.uuid4()), "uses pytest")

    async def test_oversized_content_truncated_before_extraction(self):
        from scripts.core.backfill_kg import MAX_CONTENT_CHARS

        with patch("scripts.core.backfill_kg.extract_entities", return_value=[]) as mock_ents:
            await backfill_one(str(uuid.uuid4()), "x" * (MAX_CONTENT_CHARS + 5000))

        passed = mock_ents.call_args[0][0]
        assert len(passed) == MAX_CONTENT_CHARS


# ---------------------------------------------------------------------------
# Async: mark_no_entities
# ---------------------------------------------------------------------------


class TestMarkNoEntities:
    async def test_marks_ids_in_metadata(self):
        conn = AsyncMock()
        pool = _mock_pool(conn)
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        await mark_no_entities(pool, ids)

        conn.execute.assert_awaited_once()
        sql_arg = conn.execute.await_args[0][0]
        assert "kg_backfill" in sql_arg
        assert "UPDATE archival_memory" in sql_arg
        assert conn.execute.await_args[0][1] == ids

    async def test_empty_ids_is_noop(self):
        conn = AsyncMock()
        pool = _mock_pool(conn)

        await mark_no_entities(pool, [])

        conn.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# Async: run_backfill orchestration
# ---------------------------------------------------------------------------


class TestRunBackfill:
    async def test_non_postgres_backend_exits_without_db(self):
        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="sqlite"),
            patch("scripts.core.backfill_kg.get_pool", new_callable=AsyncMock) as mock_pool,
        ):
            rc = await run_backfill(parse_args([]))

        assert rc == 1
        mock_pool.assert_not_called()

    async def test_dry_run_reports_count_without_writing(self, capsys):
        conn = AsyncMock()
        conn.fetchval.return_value = 42
        pool = _mock_pool(conn)

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.store_entities_and_edges",
                new_callable=AsyncMock,
            ) as mock_store,
        ):
            rc = await run_backfill(parse_args(["--dry-run"]))

        assert rc == 0
        assert "42" in capsys.readouterr().out
        mock_store.assert_not_called()

    async def test_partial_errors_exit_nonzero(self, capsys):
        rows = [_row(f"content {i}") for i in range(3)]
        conn = AsyncMock()
        conn.fetch.side_effect = [rows, []]
        pool = _mock_pool(conn)

        outcomes = [
            {"status": "indexed", "stats": {"entities": 1, "edges": 0, "mentions": 1}},
            {"status": "error", "error": "row boom"},
            {"status": "no_entities"},
        ]

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                side_effect=outcomes,
            ) as mock_one,
        ):
            rc = await run_backfill(parse_args([]))

        # Partial failure must be visible to automation
        assert rc == 2
        assert mock_one.await_count == 3
        out = capsys.readouterr().out
        assert "indexed" in out
        assert "errors" in out

    async def test_infra_error_aborts_run_with_exit_code_3(self, capsys):
        # Issue #131: DB down mid-run → prompt abort with a distinct exit
        # code, remaining rows never attempted
        import asyncpg

        rows = [_row(f"content {i}") for i in range(3)]
        conn = AsyncMock()
        conn.fetch.side_effect = [rows, []]
        pool = _mock_pool(conn)

        outcomes = [
            {"status": "indexed", "stats": {"entities": 1, "edges": 0, "mentions": 1}},
            asyncpg.PostgresConnectionError("db down"),
        ]

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                side_effect=outcomes,
            ) as mock_one,
        ):
            rc = await run_backfill(parse_args([]))

        assert rc == 3
        # Third row never attempted: the run aborted promptly
        assert mock_one.await_count == 2
        out = capsys.readouterr().out
        assert "infrastructure error" in out
        assert "db down" in out

    async def test_pool_creation_infra_error_exits_3(self, capsys):
        # DB down before row processing even begins must still exit 3,
        # not escape with a traceback (adversarial review round 1)
        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                side_effect=ConnectionRefusedError("connect refused"),
            ),
        ):
            rc = await run_backfill(parse_args([]))

        assert rc == 3
        assert "infrastructure error" in capsys.readouterr().out

    async def test_dry_run_infra_error_exits_3(self, capsys):
        # --dry-run hits the DB too; an acquire timeout is a systemic
        # failure, not a crash
        conn = AsyncMock()
        conn.fetchval.side_effect = TimeoutError("pool acquire timed out")
        pool = _mock_pool(conn)

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
        ):
            rc = await run_backfill(parse_args(["--dry-run"]))

        assert rc == 3
        assert "infrastructure error" in capsys.readouterr().out

    async def test_infra_abort_flushes_no_entities_best_effort(self):
        # Rows already classified no_entities in the aborted page keep their
        # durable marker when the pool still answers (e.g. transient acquire
        # timeout raised by a later row)
        import asyncpg

        rows = [_row("plain"), _row("entity-bearing")]
        conn = AsyncMock()
        conn.fetch.side_effect = [rows, []]
        pool = _mock_pool(conn)

        outcomes = [
            {"status": "no_entities"},
            asyncpg.PostgresConnectionError("db down"),
        ]

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                side_effect=outcomes,
            ),
            patch(
                "scripts.core.backfill_kg.mark_no_entities",
                new_callable=AsyncMock,
            ) as mock_mark,
        ):
            rc = await run_backfill(parse_args([]))

        assert rc == 3
        mock_mark.assert_awaited_once()
        assert mock_mark.await_args[0][1] == [str(rows[0]["id"])]

    async def test_infra_abort_survives_failed_flush(self, capsys):
        # The best-effort flush itself hits the dead DB: still exit 3, no
        # secondary traceback
        import asyncpg

        rows = [_row("plain"), _row("entity-bearing")]
        conn = AsyncMock()
        conn.fetch.side_effect = [rows, []]
        pool = _mock_pool(conn)

        outcomes = [
            {"status": "no_entities"},
            asyncpg.PostgresConnectionError("db down"),
        ]

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                side_effect=outcomes,
            ),
            patch(
                "scripts.core.backfill_kg.mark_no_entities",
                new_callable=AsyncMock,
                side_effect=asyncpg.PostgresConnectionError("still down"),
            ),
        ):
            rc = await run_backfill(parse_args([]))

        assert rc == 3
        assert "infrastructure error" in capsys.readouterr().out

    async def test_consecutive_errors_trip_circuit_breaker(self, capsys):
        # Issue #131: schema mismatch etc. errors every row without raising
        # an INFRA_ERRORS type; the breaker stops the run at the threshold
        rows = [_row(f"content {i}") for i in range(4)]
        conn = AsyncMock()
        conn.fetch.side_effect = [rows, []]
        pool = _mock_pool(conn)

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                return_value={"status": "error", "error": "schema mismatch"},
            ) as mock_one,
        ):
            rc = await run_backfill(parse_args(["--max-consecutive-errors", "2"]))

        assert rc == 3
        # Breaker trips at the threshold: rows 3 and 4 never attempted
        assert mock_one.await_count == 2
        out = capsys.readouterr().out
        assert "consecutive errors" in out

    async def test_success_resets_consecutive_error_count(self):
        # Interleaved successes prove the errors are per-row, not systemic:
        # the breaker must not trip and the run finishes with exit 2
        rows = [_row(f"content {i}") for i in range(5)]
        conn = AsyncMock()
        conn.fetch.side_effect = [rows, []]
        pool = _mock_pool(conn)

        outcomes = [
            {"status": "error", "error": "bad row"},
            {"status": "indexed", "stats": {"entities": 1, "edges": 0, "mentions": 1}},
            {"status": "error", "error": "bad row"},
            {"status": "no_entities"},
            {"status": "error", "error": "bad row"},
        ]

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                side_effect=outcomes,
            ) as mock_one,
        ):
            rc = await run_backfill(parse_args(["--max-consecutive-errors", "2"]))

        assert rc == 2
        assert mock_one.await_count == 5

    async def test_error_log_sanitizes_control_characters(self, capsys):
        rows = [_row("adversarial")]
        conn = AsyncMock()
        conn.fetch.side_effect = [rows, []]
        pool = _mock_pool(conn)

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                return_value={
                    "status": "error",
                    "error": "boom\x1b[2J\nFORGED LOG LINE",
                },
            ),
        ):
            rc = await run_backfill(parse_args([]))

        assert rc == 2
        out = capsys.readouterr().out
        # Issue #104 class: no raw escapes or newline-forged lines on the TTY
        assert "\x1b" not in out
        assert "\nFORGED LOG LINE" not in out

    async def test_all_success_exits_zero(self):
        rows = [_row()]
        conn = AsyncMock()
        conn.fetch.side_effect = [rows, []]
        pool = _mock_pool(conn)

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                return_value={
                    "status": "indexed",
                    "stats": {"entities": 1, "edges": 0, "mentions": 1},
                },
            ),
        ):
            rc = await run_backfill(parse_args([]))

        assert rc == 0

    async def test_no_entity_rows_are_marked(self):
        rows = [_row("plain text")]
        conn = AsyncMock()
        conn.fetch.side_effect = [rows, []]
        pool = _mock_pool(conn)

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                return_value={"status": "no_entities"},
            ),
            patch(
                "scripts.core.backfill_kg.mark_no_entities",
                new_callable=AsyncMock,
            ) as mock_mark,
        ):
            rc = await run_backfill(parse_args([]))

        assert rc == 0
        mock_mark.assert_awaited_once()
        marked_ids = mock_mark.await_args[0][1]
        assert marked_ids == [str(rows[0]["id"])]

    async def test_pagination_fetches_in_batch_sized_pages(self):
        pages = [
            [_row("a"), _row("b")],
            [_row("c"), _row("d")],
            [_row("e")],
        ]
        conn = AsyncMock()
        conn.fetch.side_effect = pages
        pool = _mock_pool(conn)

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                return_value={
                    "status": "indexed",
                    "stats": {"entities": 1, "edges": 0, "mentions": 1},
                },
            ) as mock_one,
        ):
            rc = await run_backfill(parse_args(["--batch-size", "2"]))

        assert rc == 0
        assert mock_one.await_count == 5
        # Short final page (1 < 2) terminates the loop without a 4th fetch
        assert conn.fetch.await_count == 3

    async def test_limit_caps_total_processed(self):
        pages = [[_row("a"), _row("b")], [_row("c")]]
        conn = AsyncMock()
        conn.fetch.side_effect = pages
        pool = _mock_pool(conn)

        with (
            patch("scripts.core.backfill_kg.detect_backend", return_value="postgres"),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                return_value={
                    "status": "indexed",
                    "stats": {"entities": 1, "edges": 0, "mentions": 1},
                },
            ) as mock_one,
        ):
            rc = await run_backfill(parse_args(["--batch-size", "2", "--limit", "3"]))

        assert rc == 0
        assert mock_one.await_count == 3


# ---------------------------------------------------------------------------
# Async: _main_async pool cleanup
# ---------------------------------------------------------------------------


class TestMainAsync:
    async def test_returns_run_backfill_code_and_closes_pool(self):
        from scripts.core.backfill_kg import _main_async

        with (
            patch("scripts.core.backfill_kg._bootstrap"),
            patch(
                "scripts.core.backfill_kg.run_backfill",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch("scripts.core.backfill_kg.close_pool", new_callable=AsyncMock) as mock_close,
        ):
            rc = await _main_async(["--dry-run"])

        assert rc == 0
        mock_close.assert_awaited_once()

    async def test_closes_pool_even_when_run_raises(self):
        from scripts.core.backfill_kg import _main_async

        with (
            patch("scripts.core.backfill_kg._bootstrap"),
            patch(
                "scripts.core.backfill_kg.run_backfill",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch("scripts.core.backfill_kg.close_pool", new_callable=AsyncMock) as mock_close,
        ):
            with pytest.raises(RuntimeError):
                await _main_async([])

        mock_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# CLI: direct file invocation
# ---------------------------------------------------------------------------


class TestDirectInvocation:
    def test_script_runs_directly_without_module_flag(self):
        """Issue #124 acceptance uses `python scripts/core/backfill_kg.py`;
        the script needs the project root on sys.path (memory_daemon pattern)."""
        import os
        import subprocess
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent
        # Sanitized env: an inherited PYTHONPATH containing the project root
        # would make scripts.* importable and false-green this test
        child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run(
            [sys.executable, "scripts/core/backfill_kg.py", "--help"],
            cwd=project_root,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--dry-run" in proc.stdout


# ---------------------------------------------------------------------------
# Integration: real PostgreSQL (skipped when unavailable)
# ---------------------------------------------------------------------------


def _db_available() -> bool:
    """Check if PostgreSQL is reachable by attempting a real connection."""
    import socket

    try:
        sock = socket.create_connection(("localhost", 5432), timeout=2)
        sock.close()
        return True
    except (OSError, TimeoutError):
        return False


@pytest.mark.skipif(not _db_available(), reason="PostgreSQL not available")
class TestBackfillIntegration:
    """Seed real memories, backfill twice, assert the second run is a no-op.

    Uses unique file-path entities so seeded KG rows can be cleaned up
    without touching pre-existing entities.
    """

    @pytest.fixture(autouse=True)
    def _reset_pool(self):
        from scripts.core.db.postgres_pool import reset_pool

        reset_pool()
        yield
        reset_pool()

    async def _seed(self, conn, marker: str) -> tuple[list[str], str]:
        """Insert 3 entity-bearing memories plus 1 zero-entity memory.

        Rows carry a sentinel project so the real run_backfill calls below
        can be scoped with --project and never touch unrelated memories.
        """
        ids = []
        for i in range(3):
            row = await conn.fetchrow(
                """
                INSERT INTO archival_memory (session_id, content, content_hash,
                                             project)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                f"test-backfill-kg-{marker}",
                f"Fixed bug in scripts/bf124_{marker}_{i}.py using pytest",
                f"bf124-{marker}-{i}",
                f"test-bf124-{marker}",
            )
            ids.append(str(row["id"]))
        row = await conn.fetchrow(
            """
            INSERT INTO archival_memory (session_id, content, content_hash,
                                         project)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            f"test-backfill-kg-{marker}",
            "the meeting went well and everyone agreed on the plan",
            f"bf124-{marker}-noent",
            f"test-bf124-{marker}",
        )
        no_entity_id = str(row["id"])
        return ids + [no_entity_id], no_entity_id

    async def _cleanup(self, conn, marker: str, ids: list[str]):
        await conn.execute("DELETE FROM kg_edges WHERE memory_id = ANY($1::uuid[])", ids)
        await conn.execute("DELETE FROM kg_entity_mentions WHERE memory_id = ANY($1::uuid[])", ids)
        await conn.execute("DELETE FROM kg_entities WHERE name LIKE $1", f"%bf124_{marker}%")
        await conn.execute("DELETE FROM archival_memory WHERE id = ANY($1::uuid[])", ids)

    async def test_backfill_twice_second_run_is_noop(self):
        import os

        from scripts.core.db.postgres_pool import get_pool

        if not (os.environ.get("CONTINUOUS_CLAUDE_DB_URL") or os.environ.get("DATABASE_URL")):
            pytest.skip(
                "No DB URL in env (CONTINUOUS_CLAUDE_DB_URL / DATABASE_URL); "
                "issue #62 forbids hardcoded fallbacks"
            )

        marker = uuid.uuid4().hex[:8]
        pool = await get_pool()
        async with pool.acquire() as conn:
            ids, no_entity_id = await self._seed(conn, marker)
        try:
            async with pool.acquire() as conn:
                seeded_at = await conn.fetchval(
                    "SELECT min(created_at) FROM archival_memory " "WHERE id = ANY($1::uuid[])",
                    ids,
                )

            since_arg = seeded_at.isoformat()
            scope = ["--since", since_arg, "--project", f"test-bf124-{marker}"]

            rc1 = await run_backfill(parse_args(scope))
            assert rc1 == 0

            async with pool.acquire() as conn:
                mentions_after_first = await conn.fetchval(
                    "SELECT count(*) FROM kg_entity_mentions " "WHERE memory_id = ANY($1::uuid[])",
                    ids,
                )
                no_entity_flag = await conn.fetchval(
                    "SELECT metadata->>'kg_backfill' FROM archival_memory " "WHERE id = $1::uuid",
                    no_entity_id,
                )
            assert mentions_after_first > 0
            assert no_entity_flag == "no_entities"

            rc2 = await run_backfill(parse_args(scope))
            assert rc2 == 0

            async with pool.acquire() as conn:
                mentions_after_second = await conn.fetchval(
                    "SELECT count(*) FROM kg_entity_mentions " "WHERE memory_id = ANY($1::uuid[])",
                    ids,
                )
                sql, params = build_fetch_query(
                    since=seeded_at,
                    project=f"test-bf124-{marker}",
                    count_only=True,
                )
                eligible_after_second = await conn.fetchval(sql, *params)
            assert mentions_after_second == mentions_after_first
            # Zero-entity row is durably marked: nothing left to retry
            assert eligible_after_second == 0

            # Targeted repair: --memory-id bypasses the no_entities marker
            async with pool.acquire() as conn:
                sql, params = build_fetch_query(memory_id=no_entity_id, count_only=True)
                repair_eligible = await conn.fetchval(sql, *params)
            assert repair_eligible == 1
            rc3 = await run_backfill(parse_args(["--memory-id", no_entity_id]))
            assert rc3 == 0
        finally:
            async with pool.acquire() as conn:
                await self._cleanup(conn, marker, ids)
