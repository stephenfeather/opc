"""Tests for scripts/core/backfill_kg.py — KG backfill for existing memories (#124)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.core.backfill_kg import (
    backfill_one,
    build_fetch_query,
    chunked,
    format_summary,
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


# ---------------------------------------------------------------------------
# Pure: build_fetch_query
# ---------------------------------------------------------------------------


class TestBuildFetchQuery:
    def test_default_selects_unindexed_memories(self):
        sql, params = build_fetch_query()
        assert "SELECT id, content" in sql
        assert "archival_memory" in sql
        assert "NOT EXISTS" in sql
        assert "kg_entity_mentions" in sql
        assert "ORDER BY" in sql
        assert params == []

    def test_count_only_uses_count_star(self):
        sql, params = build_fetch_query(count_only=True)
        assert "count(*)" in sql.lower()
        assert "NOT EXISTS" in sql
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

    def test_memory_id_filters_id(self):
        mid = str(uuid.uuid4())
        sql, params = build_fetch_query(memory_id=mid)
        assert "id = $1" in sql
        assert params == [mid]

    def test_combined_filters_number_params_consecutively(self):
        since = datetime(2026, 1, 1, tzinfo=UTC)
        sql, params = build_fetch_query(limit=10, since=since)
        assert "created_at >= $1" in sql
        assert "LIMIT $2" in sql
        assert params == [since, 10]


# ---------------------------------------------------------------------------
# Pure: chunked
# ---------------------------------------------------------------------------


class TestChunked:
    def test_even_split(self):
        assert list(chunked([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_remainder_in_last_chunk(self):
        assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    def test_empty_input(self):
        assert list(chunked([], 3)) == []


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


# ---------------------------------------------------------------------------
# Pure: format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_includes_all_counts(self):
        text = format_summary(
            {"processed": 7, "indexed": 4, "no_entities": 2, "errors": 1}
        )
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
            patch(
                "scripts.core.backfill_kg.extract_entities", return_value=entities
            ) as mock_ents,
            patch(
                "scripts.core.backfill_kg.extract_relations", return_value=[]
            ) as mock_rels,
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


# ---------------------------------------------------------------------------
# Async: run_backfill orchestration
# ---------------------------------------------------------------------------


class TestRunBackfill:
    async def test_non_postgres_backend_exits_without_db(self):
        with (
            patch(
                "scripts.core.backfill_kg.detect_backend", return_value="sqlite"
            ),
            patch(
                "scripts.core.backfill_kg.get_pool", new_callable=AsyncMock
            ) as mock_pool,
        ):
            rc = await run_backfill(parse_args([]))

        assert rc == 1
        mock_pool.assert_not_called()

    async def test_dry_run_reports_count_without_writing(self, capsys):
        conn = AsyncMock()
        conn.fetchval.return_value = 42
        pool = _mock_pool(conn)

        with (
            patch(
                "scripts.core.backfill_kg.detect_backend", return_value="postgres"
            ),
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

    async def test_processes_rows_and_continues_on_error(self, capsys):
        rows = [
            {"id": uuid.uuid4(), "content": f"content {i}"} for i in range(3)
        ]
        conn = AsyncMock()
        conn.fetch.return_value = rows
        pool = _mock_pool(conn)

        outcomes = [
            {"status": "indexed", "stats": {"entities": 1, "edges": 0, "mentions": 1}},
            {"status": "error", "error": "row boom"},
            {"status": "no_entities"},
        ]

        with (
            patch(
                "scripts.core.backfill_kg.detect_backend", return_value="postgres"
            ),
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

        assert rc == 0
        assert mock_one.await_count == 3
        out = capsys.readouterr().out
        # Summary reflects one of each outcome
        assert "indexed" in out
        assert "errors" in out

    async def test_batching_respects_batch_size(self):
        rows = [
            {"id": uuid.uuid4(), "content": f"content {i}"} for i in range(5)
        ]
        conn = AsyncMock()
        conn.fetch.return_value = rows
        pool = _mock_pool(conn)

        with (
            patch(
                "scripts.core.backfill_kg.detect_backend", return_value="postgres"
            ),
            patch(
                "scripts.core.backfill_kg.get_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "scripts.core.backfill_kg.backfill_one",
                new_callable=AsyncMock,
                return_value={"status": "no_entities"},
            ) as mock_one,
        ):
            rc = await run_backfill(parse_args(["--batch-size", "2"]))

        assert rc == 0
        assert mock_one.await_count == 5


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

    async def _seed(self, conn, marker: str, n: int = 3) -> list[str]:
        ids = []
        for i in range(n):
            row = await conn.fetchrow(
                """
                INSERT INTO archival_memory (session_id, content, content_hash)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                f"test-backfill-kg-{marker}",
                f"Fixed bug in scripts/bf124_{marker}_{i}.py using pytest",
                f"bf124-{marker}-{i}",
            )
            ids.append(str(row["id"]))
        return ids

    async def _cleanup(self, conn, marker: str, ids: list[str]):
        await conn.execute(
            "DELETE FROM kg_edges WHERE memory_id = ANY($1::uuid[])", ids
        )
        await conn.execute(
            "DELETE FROM kg_entity_mentions WHERE memory_id = ANY($1::uuid[])", ids
        )
        await conn.execute(
            "DELETE FROM kg_entities WHERE name LIKE $1", f"%bf124_{marker}%"
        )
        await conn.execute(
            "DELETE FROM archival_memory WHERE id = ANY($1::uuid[])", ids
        )

    async def test_backfill_twice_second_run_is_noop(self):
        import os

        from scripts.core.db.postgres_pool import get_pool

        if not (
            os.environ.get("CONTINUOUS_CLAUDE_DB_URL")
            or os.environ.get("DATABASE_URL")
        ):
            pytest.skip(
                "No DB URL in env (CONTINUOUS_CLAUDE_DB_URL / DATABASE_URL); "
                "issue #62 forbids hardcoded fallbacks"
            )

        marker = uuid.uuid4().hex[:8]
        pool = await get_pool()
        async with pool.acquire() as conn:
            ids = await self._seed(conn, marker)
        try:
            seeded_at = None
            async with pool.acquire() as conn:
                seeded_at = await conn.fetchval(
                    "SELECT min(created_at) FROM archival_memory "
                    "WHERE id = ANY($1::uuid[])",
                    ids,
                )

            since_arg = seeded_at.isoformat()

            rc1 = await run_backfill(parse_args(["--since", since_arg]))
            assert rc1 == 0

            async with pool.acquire() as conn:
                mentions_after_first = await conn.fetchval(
                    "SELECT count(*) FROM kg_entity_mentions "
                    "WHERE memory_id = ANY($1::uuid[])",
                    ids,
                )
            assert mentions_after_first > 0

            rc2 = await run_backfill(parse_args(["--since", since_arg]))
            assert rc2 == 0

            async with pool.acquire() as conn:
                mentions_after_second = await conn.fetchval(
                    "SELECT count(*) FROM kg_entity_mentions "
                    "WHERE memory_id = ANY($1::uuid[])",
                    ids,
                )
            assert mentions_after_second == mentions_after_first
        finally:
            async with pool.acquire() as conn:
                await self._cleanup(conn, marker, ids)
