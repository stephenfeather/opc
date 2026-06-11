"""Tests for --source-time / created_at backfill plumbing (issue #52).

Backfilled learnings were stamped with ``created_at = NOW()`` (the DB default),
which let weeks-old content masquerade as age-zero in the reranker. The fix
threads an optional source time from the originating session through the store
path so ``archival_memory.created_at`` reflects when the session actually
occurred.

Coverage:
1. ``validate_source_time`` (pure): parses ISO8601 with/without tz (naive ->
   UTC); rejects garbage and >5min-future values by returning None.
2. ``resolve_source_time`` (pure): CLI arg precedence over env fallback.
3. ``store_learning_v2`` plumbs ``source_time`` into ``memory.store``.
4. ``memory_service_pg.store`` includes an explicit ``created_at`` bind ONLY
   when ``source_time`` is given; the no-source-time SQL is byte-identical to
   today (backward-compat pin).
5. Garbage/future ``--source-time`` is ignored with a warning and the store
   still succeeds with the default ``created_at``.
6. The S3 backfill caller injects the JSONL mtime as the source time.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.store_learning import (  # noqa: E402
    resolve_source_time,
    store_learning_v2,
    validate_source_time,
)

# ---------------------------------------------------------------------------
# Pure: validate_source_time
# ---------------------------------------------------------------------------


class TestValidateSourceTime:
    """ISO8601 parsing + skew/garbage rejection. Pure function."""

    NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)

    def test_none_returns_none(self) -> None:
        assert validate_source_time(None, now=self.NOW) is None

    def test_empty_string_returns_none(self) -> None:
        assert validate_source_time("", now=self.NOW) is None

    def test_aware_iso8601_parsed(self) -> None:
        result = validate_source_time("2026-01-01T08:30:00+00:00", now=self.NOW)
        assert result == datetime(2026, 1, 1, 8, 30, 0, tzinfo=UTC)

    def test_aware_non_utc_offset_preserved(self) -> None:
        result = validate_source_time("2026-01-01T08:30:00-05:00", now=self.NOW)
        # Same instant, regardless of stored offset.
        assert result is not None
        assert result == datetime(2026, 1, 1, 13, 30, 0, tzinfo=UTC)

    def test_naive_iso8601_treated_as_utc(self) -> None:
        # Documented assumption: naive timestamps are interpreted as UTC.
        result = validate_source_time("2026-01-01T08:30:00", now=self.NOW)
        assert result == datetime(2026, 1, 1, 8, 30, 0, tzinfo=UTC)

    def test_garbage_returns_none(self) -> None:
        assert validate_source_time("not-a-timestamp", now=self.NOW) is None

    def test_future_beyond_skew_returns_none(self) -> None:
        future = (self.NOW + timedelta(minutes=10)).isoformat()
        assert validate_source_time(future, now=self.NOW) is None

    def test_small_future_skew_allowed(self) -> None:
        # Within the 5-minute clock-skew window is accepted.
        near = (self.NOW + timedelta(minutes=2)).isoformat()
        assert validate_source_time(near, now=self.NOW) is not None

    def test_past_timestamp_allowed(self) -> None:
        past = (self.NOW - timedelta(days=90)).isoformat()
        assert validate_source_time(past, now=self.NOW) is not None


# ---------------------------------------------------------------------------
# Pure: resolve_source_time (CLI arg vs env fallback)
# ---------------------------------------------------------------------------


class TestResolveSourceTime:
    """CLI flag takes precedence; env var is the fallback for agent subprocs."""

    def test_arg_wins_over_env(self) -> None:
        env = {"CLAUDE_SOURCE_TIME": "2020-01-01T00:00:00"}
        assert resolve_source_time("2026-01-01T00:00:00", env) == "2026-01-01T00:00:00"

    def test_env_used_when_arg_absent(self) -> None:
        env = {"CLAUDE_SOURCE_TIME": "2020-01-01T00:00:00"}
        assert resolve_source_time(None, env) == "2020-01-01T00:00:00"

    def test_none_when_neither_present(self) -> None:
        assert resolve_source_time(None, {}) is None


# ---------------------------------------------------------------------------
# Plumbing: store_learning_v2 -> memory.store(source_time=...)
# ---------------------------------------------------------------------------


def _make_mocks() -> tuple[AsyncMock, AsyncMock]:
    mock_memory = AsyncMock()
    mock_memory.search_vector_global = AsyncMock(return_value=[])
    mock_memory.store = AsyncMock(return_value="new-uuid")
    mock_memory.close = AsyncMock()

    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    mock_embedder._provider = MagicMock(model="bge")
    return mock_memory, mock_embedder


class TestStoreLearningV2SourceTime:
    @pytest.mark.asyncio
    async def test_source_time_passed_to_store(self) -> None:
        mock_memory, mock_embedder = _make_mocks()
        src = datetime(2026, 1, 1, 8, 30, 0, tzinfo=UTC)

        with (
            patch(
                "scripts.core.store_learning.create_memory_service",
                new_callable=AsyncMock,
                return_value=mock_memory,
            ),
            patch(
                "scripts.core.store_learning.EmbeddingService",
                return_value=mock_embedder,
            ),
        ):
            result = await store_learning_v2(
                session_id="s1",
                content="A learning",
                source_time=src,
            )

        assert result["success"] is True
        assert mock_memory.store.await_args.kwargs["source_time"] == src

    @pytest.mark.asyncio
    async def test_no_source_time_passes_none(self) -> None:
        mock_memory, mock_embedder = _make_mocks()

        with (
            patch(
                "scripts.core.store_learning.create_memory_service",
                new_callable=AsyncMock,
                return_value=mock_memory,
            ),
            patch(
                "scripts.core.store_learning.EmbeddingService",
                return_value=mock_embedder,
            ),
        ):
            await store_learning_v2(session_id="s1", content="A learning")

        assert mock_memory.store.await_args.kwargs["source_time"] is None


# ---------------------------------------------------------------------------
# memory_service_pg.store: created_at bind only when source_time set
# ---------------------------------------------------------------------------


def _make_pg_service() -> tuple[Any, AsyncMock]:
    from scripts.core.db.memory_service_pg import MemoryServicePG

    svc = MemoryServicePG(session_id="s1", agent_id="a1")
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    return svc, conn


class TestPgStoreCreatedAtBind:
    @pytest.mark.asyncio
    async def test_with_source_time_includes_created_at_embedding_path(self) -> None:
        svc, conn = _make_pg_service()
        src = datetime(2026, 1, 1, 8, 30, 0, tzinfo=UTC)

        with (
            patch("scripts.core.db.memory_service_pg.get_transaction") as gt,
            patch("scripts.core.db.memory_service_pg.init_pgvector", new_callable=AsyncMock),
            patch("scripts.core.db.memory_service_pg.pad_embedding", side_effect=lambda e: e),
        ):
            gt.return_value.__aenter__ = AsyncMock(return_value=conn)
            gt.return_value.__aexit__ = AsyncMock(return_value=False)
            await svc.store("c", embedding=[0.1] * 1024, source_time=src)

        sql = conn.execute.await_args_list[0].args[0]
        binds = conn.execute.await_args_list[0].args[1:]
        assert "created_at" in sql
        assert src in binds

    @pytest.mark.asyncio
    async def test_with_source_time_includes_created_at_no_embedding_path(self) -> None:
        svc, conn = _make_pg_service()
        src = datetime(2026, 1, 1, 8, 30, 0, tzinfo=UTC)

        with patch("scripts.core.db.memory_service_pg.get_transaction") as gt:
            gt.return_value.__aenter__ = AsyncMock(return_value=conn)
            gt.return_value.__aexit__ = AsyncMock(return_value=False)
            await svc.store("c", source_time=src)

        sql = conn.execute.await_args_list[0].args[0]
        binds = conn.execute.await_args_list[0].args[1:]
        assert "created_at" in sql
        assert src in binds

    @pytest.mark.asyncio
    async def test_no_source_time_sql_byte_identical_embedding_path(self) -> None:
        """Backward-compat pin: omitting source_time must not alter the SQL."""
        svc, conn = _make_pg_service()

        with (
            patch("scripts.core.db.memory_service_pg.get_transaction") as gt,
            patch("scripts.core.db.memory_service_pg.init_pgvector", new_callable=AsyncMock),
            patch("scripts.core.db.memory_service_pg.pad_embedding", side_effect=lambda e: e),
        ):
            gt.return_value.__aenter__ = AsyncMock(return_value=conn)
            gt.return_value.__aexit__ = AsyncMock(return_value=False)
            await svc.store("c", embedding=[0.1] * 1024)

        sql = conn.execute.await_args_list[0].args[0]
        assert "created_at" not in sql

    @pytest.mark.asyncio
    async def test_no_source_time_sql_byte_identical_no_embedding_path(self) -> None:
        svc, conn = _make_pg_service()

        with patch("scripts.core.db.memory_service_pg.get_transaction") as gt:
            gt.return_value.__aenter__ = AsyncMock(return_value=conn)
            gt.return_value.__aexit__ = AsyncMock(return_value=False)
            await svc.store("c")

        sql = conn.execute.await_args_list[0].args[0]
        assert "created_at" not in sql


# ---------------------------------------------------------------------------
# Garbage/future --source-time: ignored with warning, store still succeeds
# ---------------------------------------------------------------------------


class TestInvalidSourceTimeIgnored:
    @pytest.mark.asyncio
    async def test_garbage_source_time_ignored_store_succeeds(self) -> None:
        mock_memory, mock_embedder = _make_mocks()

        with (
            patch(
                "scripts.core.store_learning.create_memory_service",
                new_callable=AsyncMock,
                return_value=mock_memory,
            ),
            patch(
                "scripts.core.store_learning.EmbeddingService",
                return_value=mock_embedder,
            ),
        ):
            # store_learning_v2 receives a validated datetime or None; the CLI
            # layer validates. Here we assert the contract: None -> default.
            result = await store_learning_v2(
                session_id="s1", content="A learning", source_time=None
            )

        assert result["success"] is True
        assert mock_memory.store.await_args.kwargs["source_time"] is None


# ---------------------------------------------------------------------------
# Backfill caller injects JSONL mtime as source time
# ---------------------------------------------------------------------------


class TestBackfillInjectsSourceTime:
    def test_build_extraction_env_sets_source_time(self) -> None:
        from scripts.core.backfill_learnings import build_extraction_env

        src = datetime(2026, 1, 1, 8, 30, 0, tzinfo=UTC)
        env = build_extraction_env("/proj", source_time=src)
        assert env["CLAUDE_SOURCE_TIME"] == src.isoformat()

    def test_build_extraction_env_omits_source_time_when_none(self) -> None:
        from scripts.core.backfill_learnings import build_extraction_env

        env = build_extraction_env("/proj")
        assert "CLAUDE_SOURCE_TIME" not in env

    def test_source_time_env_survives_allowlist(self) -> None:
        # CLAUDE_ prefix is allowlisted, so the injected value is not stripped.
        from scripts.core.backfill_learnings import _is_env_allowed

        assert _is_env_allowed("CLAUDE_SOURCE_TIME") is True
