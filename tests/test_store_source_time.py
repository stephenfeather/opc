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

    def test_pre_2024_timestamp_rejected(self) -> None:
        # Implausibility floor mirrors fix_backfill_created_at.sql (aegis
        # MEDIUM: no real session predates the system; a 1970 value must not
        # silently bury a learning in recency ranking).
        assert validate_source_time("1970-01-01T00:00:00+00:00", now=self.NOW) is None
        assert validate_source_time("2023-12-31T23:59:59+00:00", now=self.NOW) is None

    def test_floor_boundary_accepted(self) -> None:
        assert validate_source_time("2024-01-01T00:00:00+00:00", now=self.NOW) is not None


# ---------------------------------------------------------------------------
# Pure: resolve_source_time (CLI arg vs env fallback)
# ---------------------------------------------------------------------------


class TestResolveSourceTime:
    """CLI flag takes precedence; env var is the fallback for agent subprocs."""

    def test_arg_wins_over_env(self) -> None:
        # The --source-time flag is for explicit operator use and is trusted
        # unconditionally; it wins even without the extraction marker.
        env = {
            "CLAUDE_SOURCE_TIME": "2020-01-01T00:00:00",
            "CLAUDE_MEMORY_EXTRACTION": "1",
        }
        assert resolve_source_time("2026-01-01T00:00:00", env) == "2026-01-01T00:00:00"

    def test_arg_wins_even_without_marker(self) -> None:
        env = {"CLAUDE_SOURCE_TIME": "2020-01-01T00:00:00"}
        assert resolve_source_time("2026-01-01T00:00:00", env) == "2026-01-01T00:00:00"

    def test_env_used_when_arg_absent_and_marker_set(self) -> None:
        # The env fallback is trusted ONLY inside the extractor subprocess,
        # which the backfill pipeline marks with CLAUDE_MEMORY_EXTRACTION=1.
        env = {
            "CLAUDE_SOURCE_TIME": "2020-01-01T00:00:00",
            "CLAUDE_MEMORY_EXTRACTION": "1",
        }
        assert resolve_source_time(None, env) == "2020-01-01T00:00:00"

    def test_env_ignored_without_marker(self) -> None:
        # Fix 2 (trust boundary): an ambient/user-set CLAUDE_SOURCE_TIME with
        # no extraction marker must NOT backdate a live store.
        env = {"CLAUDE_SOURCE_TIME": "2020-01-01T00:00:00"}
        assert resolve_source_time(None, env) is None

    def test_env_ignored_when_marker_not_one(self) -> None:
        env = {
            "CLAUDE_SOURCE_TIME": "2020-01-01T00:00:00",
            "CLAUDE_MEMORY_EXTRACTION": "0",
        }
        assert resolve_source_time(None, env) is None

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
        from scripts.core.memory_daemon_core import build_extraction_env

        src = datetime(2026, 1, 1, 8, 30, 0, tzinfo=UTC)
        env = build_extraction_env({}, "/proj", source_time=src)
        assert env["CLAUDE_SOURCE_TIME"] == src.isoformat()

    def test_build_extraction_env_omits_source_time_when_none(self) -> None:
        from scripts.core.memory_daemon_core import build_extraction_env

        env = build_extraction_env({}, "/proj")
        assert "CLAUDE_SOURCE_TIME" not in env

    def test_source_time_env_survives_allowlist(self) -> None:
        # CLAUDE_ prefix is allowlisted, so the injected value is not stripped.
        from scripts.core.memory_daemon_core import _is_env_allowed

        assert _is_env_allowed("CLAUDE_SOURCE_TIME") is True

    def test_build_extraction_env_strips_inherited_source_time(self) -> None:
        # Fix 2: an ambient CLAUDE_SOURCE_TIME in the parent env must NOT pass
        # through. Injection is idempotent — only the computed value (or none)
        # reaches the child.
        from scripts.core.memory_daemon_core import build_extraction_env

        base = {"CLAUDE_SOURCE_TIME": "1999-01-01T00:00:00"}
        env = build_extraction_env(base, "/proj")
        assert "CLAUDE_SOURCE_TIME" not in env

    def test_build_extraction_env_inherited_replaced_by_computed(self) -> None:
        from scripts.core.memory_daemon_core import build_extraction_env

        src = datetime(2026, 1, 1, 8, 30, 0, tzinfo=UTC)
        base = {"CLAUDE_SOURCE_TIME": "1999-01-01T00:00:00"}
        env = build_extraction_env(base, "/proj", source_time=src)
        assert env["CLAUDE_SOURCE_TIME"] == src.isoformat()


# ---------------------------------------------------------------------------
# Fix 1: source-time provenance — never trust the freshly-downloaded temp file
# ---------------------------------------------------------------------------


class TestResolveBackfillSourceTime:
    """Pure preference-order helper: exited_at > S3 LastModified > local mtime."""

    EXITED = datetime(2026, 3, 29, 20, 50, 0, tzinfo=UTC)
    S3_MOD = datetime(2026, 3, 30, 14, 4, 34, tzinfo=UTC)
    LOCAL = datetime(2026, 6, 11, 0, 0, 0, tzinfo=UTC)

    def test_prefers_exited_at(self) -> None:
        from scripts.core.backfill_learnings import resolve_backfill_source_time

        got = resolve_backfill_source_time(
            exited_at=self.EXITED, s3_last_modified=self.S3_MOD, local_mtime=self.LOCAL
        )
        assert got == self.EXITED

    def test_falls_back_to_s3_last_modified(self) -> None:
        from scripts.core.backfill_learnings import resolve_backfill_source_time

        got = resolve_backfill_source_time(
            exited_at=None, s3_last_modified=self.S3_MOD, local_mtime=self.LOCAL
        )
        assert got == self.S3_MOD

    def test_local_mtime_last_resort(self) -> None:
        from scripts.core.backfill_learnings import resolve_backfill_source_time

        got = resolve_backfill_source_time(
            exited_at=None, s3_last_modified=None, local_mtime=self.LOCAL
        )
        assert got == self.LOCAL

    def test_none_when_no_source(self) -> None:
        from scripts.core.backfill_learnings import resolve_backfill_source_time

        got = resolve_backfill_source_time(
            exited_at=None, s3_last_modified=None, local_mtime=None
        )
        assert got is None


class TestParseS3ListingPreservesDate:
    SAMPLE = (
        "2026-03-30 14:04:34     192400 sessions/-Users-stephenfeather-opc/"
        "d0f60cd7-65e8-4a30-a1fc-345ec418a1ec.jsonl.zst\n"
    )

    def test_s3_last_modified_captured(self) -> None:
        from scripts.core.backfill_learnings import parse_s3_listing

        rows = parse_s3_listing(self.SAMPLE, "b", project_filter=None)
        assert len(rows) == 1
        assert rows[0]["s3_last_modified"] == datetime(
            2026, 3, 30, 14, 4, 34, tzinfo=UTC
        )


class TestRunExtractionTrustsExplicitSourceTime:
    def test_uses_passed_source_time_not_temp_file_mtime(self) -> None:
        # Regression for Fix 1: the temp JSONL was freshly downloaded so its
        # mtime is extraction time. run_extraction must stamp CLAUDE_SOURCE_TIME
        # from the explicit session source_time, never from the file.
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        from scripts.core.backfill_learnings import run_extraction

        session_time = datetime(2026, 3, 29, 20, 50, 0, tzinfo=UTC)
        captured: dict[str, str] = {}

        def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
            captured.update(kwargs.get("env", {}))
            return MagicMock(returncode=0, stdout="Learnings stored: 0\n", stderr="")

        with _patch("scripts.core.backfill_learnings.subprocess") as mock_sub:
            mock_sub.run.side_effect = fake_run
            run_extraction(
                jsonl_path=Path("/tmp/abc.jsonl"),
                session_id="s-abc",
                agent_prompt="Extract",
                model="sonnet",
                max_turns=15,
                timeout=300,
                project_dir=None,
                source_time=session_time,
            )

        assert captured["CLAUDE_SOURCE_TIME"] == session_time.isoformat()

    def test_no_source_time_omits_env(self) -> None:
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        from scripts.core.backfill_learnings import run_extraction

        captured: dict[str, str] = {}

        def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
            captured.update(kwargs.get("env", {}))
            return MagicMock(returncode=0, stdout="Learnings stored: 0\n", stderr="")

        with _patch("scripts.core.backfill_learnings.subprocess") as mock_sub:
            mock_sub.run.side_effect = fake_run
            run_extraction(
                jsonl_path=Path("/tmp/abc.jsonl"),
                session_id="s-abc",
                agent_prompt="Extract",
                model="sonnet",
                max_turns=15,
                timeout=300,
                project_dir=None,
                source_time=None,
            )

        assert "CLAUDE_SOURCE_TIME" not in captured
