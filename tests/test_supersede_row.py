"""Tests for the shared ``supersede_row`` helper (Issue #63 Phase 2b, D1).

The helper is the single owner of the ``superseded_by`` write invariant. It is
policy-neutral: it performs one guarded UPDATE, stamps the ``superseded_via``
provenance marker (D2), and returns the row-count so each caller decides what a
zero-row result means. It does not catch ``UndefinedColumnError`` — callers own
the pre-migration policy.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import asyncpg
import pytest


class TestSupersedeRow:
    """Unit tests for the policy-neutral supersede helper."""

    async def test_returns_one_on_successful_update(self):
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        count = await supersede_row(
            conn,
            loser_id="11111111-1111-1111-1111-111111111111",
            keeper_id="22222222-2222-2222-2222-222222222222",
            reason="merge",
        )

        assert count == 1

    async def test_returns_zero_on_no_match_without_raising(self):
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")

        count = await supersede_row(
            conn,
            loser_id="11111111-1111-1111-1111-111111111111",
            keeper_id="22222222-2222-2222-2222-222222222222",
            reason="merge",
        )

        assert count == 0  # no exception — caller owns the 0-row policy

    async def test_single_update_statement_covers_all_three_fields(self):
        """W-3: superseded_by, superseded_at, and superseded_via in ONE UPDATE."""
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        await supersede_row(
            conn,
            loser_id="11111111-1111-1111-1111-111111111111",
            keeper_id="22222222-2222-2222-2222-222222222222",
            reason="merge",
        )

        assert conn.execute.await_count == 1
        sql = conn.execute.await_args.args[0]
        assert "superseded_by" in sql
        assert "superseded_at" in sql
        assert "superseded_via" in sql
        assert "UPDATE" in sql.upper()

    async def test_guarded_by_superseded_by_is_null(self):
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        await supersede_row(
            conn,
            loser_id="11111111-1111-1111-1111-111111111111",
            keeper_id="22222222-2222-2222-2222-222222222222",
            reason="store",
        )

        sql = conn.execute.await_args.args[0]
        assert "superseded_by IS NULL" in sql

    async def test_reason_threaded_as_parameter(self):
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        await supersede_row(
            conn,
            loser_id="11111111-1111-1111-1111-111111111111",
            keeper_id="22222222-2222-2222-2222-222222222222",
            reason="stale",
        )

        # reason must reach the DB as a bound parameter, not be interpolated
        params = conn.execute.await_args.args[1:]
        assert "stale" in params

    async def test_keeper_and_loser_threaded_as_parameters(self):
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        loser = "11111111-1111-1111-1111-111111111111"
        keeper = "22222222-2222-2222-2222-222222222222"
        await supersede_row(conn, loser_id=loser, keeper_id=keeper, reason="merge")

        params = conn.execute.await_args.args[1:]
        assert loser in params
        assert keeper in params

    async def test_undefined_column_error_propagates(self):
        """Helper is policy-neutral: pre-migration schema raises to the caller."""
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(
            side_effect=asyncpg.UndefinedColumnError("column does not exist")
        )

        with pytest.raises(asyncpg.UndefinedColumnError):
            await supersede_row(
                conn,
                loser_id="11111111-1111-1111-1111-111111111111",
                keeper_id="22222222-2222-2222-2222-222222222222",
                reason="merge",
            )
