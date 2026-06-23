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


class TestSupersedeRowRequireActiveKeeper:
    """Issue #63 Phase 2b correctness fix: the MERGE path must not supersede a loser
    onto a keeper that died (was itself superseded) after the merge was planned. The
    optional ``require_active_keeper`` flag adds a same-statement keeper-liveness guard.
    Default False keeps the store-time hot path byte-for-byte unchanged."""

    _LOSER = "11111111-1111-1111-1111-111111111111"
    _KEEPER = "22222222-2222-2222-2222-222222222222"

    async def test_default_omits_keeper_liveness_subquery(self):
        # store-time hot path: no extra subquery, SQL unchanged.
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        await supersede_row(
            conn, loser_id=self._LOSER, keeper_id=self._KEEPER, reason="store"
        )

        sql = conn.execute.await_args.args[0]
        # No keeper-liveness EXISTS clause on the default path.
        assert "EXISTS" not in sql.upper()

    async def test_require_active_keeper_adds_single_statement_liveness_guard(self):
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        count = await supersede_row(
            conn,
            loser_id=self._LOSER,
            keeper_id=self._KEEPER,
            reason="merge",
            require_active_keeper=True,
        )

        assert count == 1
        # ONE statement only — no second probe query.
        assert conn.execute.await_count == 1
        sql = conn.execute.await_args.args[0]
        # The keeper-liveness term is present and checks the keeper is not superseded.
        assert "EXISTS" in sql.upper()
        assert sql.upper().count("UPDATE") == 1  # still a single UPDATE
        # loser guard still present
        assert "superseded_by IS NULL" in sql

    async def test_require_active_keeper_zero_rows_when_keeper_dead(self):
        # The DB returns "UPDATE 0" because the same-statement keeper-liveness guard
        # matched nothing (keeper was superseded after planning). The loser is NOT
        # updated; the caller treats 0 as an idempotent skip.
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")

        count = await supersede_row(
            conn,
            loser_id=self._LOSER,
            keeper_id=self._KEEPER,
            reason="merge",
            require_active_keeper=True,
        )

        assert count == 0  # no raise; loser untouched

    async def test_keeper_id_bound_into_liveness_guard(self):
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        await supersede_row(
            conn,
            loser_id=self._LOSER,
            keeper_id=self._KEEPER,
            reason="merge",
            require_active_keeper=True,
        )

        # keeper reaches the DB as a bound parameter (used by both the SET and the guard).
        params = conn.execute.await_args.args[1:]
        assert self._KEEPER in params


class TestSupersedeRowProjectGuard:
    """Defense-in-depth (Issue #63 Phase 2b hardening): an optional in-statement
    project guard. ``supersede_row`` retires the loser by GLOBAL uuid with no project
    predicate; it is safe only because callers pre-filter ids through a project-scoped
    fetch. A future caller that skips that pre-fetch could supersede a row in ANOTHER
    project. The optional ``project`` param appends ``AND LOWER(project) = LOWER($N)``
    so the merge path is self-enforcing. Default (None) keeps the store-time path
    byte-for-byte unchanged (no project predicate)."""

    _LOSER = "11111111-1111-1111-1111-111111111111"
    _KEEPER = "22222222-2222-2222-2222-222222222222"

    async def test_default_none_emits_no_project_predicate(self):
        # store-time hot path: SQL is byte-for-byte unchanged — no project term.
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        await supersede_row(
            conn, loser_id=self._LOSER, keeper_id=self._KEEPER, reason="store"
        )

        sql = conn.execute.await_args.args[0]
        assert "project" not in sql.lower()
        # store-time path passes exactly the original 3 params.
        assert conn.execute.await_args.args[1:] == (self._KEEPER, "store", self._LOSER)

    async def test_project_provided_appends_guard_and_binds_param(self):
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        count = await supersede_row(
            conn,
            loser_id=self._LOSER,
            keeper_id=self._KEEPER,
            reason="merge",
            project="proj-a",
        )

        assert count == 1
        assert conn.execute.await_count == 1
        sql = conn.execute.await_args.args[0]
        # case-insensitive project predicate present, bound as a positional param.
        assert "LOWER(project)" in sql
        assert sql.upper().count("UPDATE") == 1  # still a single UPDATE
        assert "proj-a" in conn.execute.await_args.args[1:]

    async def test_project_param_is_positional_four_with_keeper_guard(self):
        # When BOTH guards are present, the keeper guard reuses $1 (no new positional),
        # so the project term must bind $4 and project is appended as the 4th arg.
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        await supersede_row(
            conn,
            loser_id=self._LOSER,
            keeper_id=self._KEEPER,
            reason="merge",
            require_active_keeper=True,
            project="proj-a",
        )

        sql = conn.execute.await_args.args[0]
        assert "$4" in sql  # project binds the next free positional
        assert "LOWER(project) = LOWER($4::text)" in sql
        # arg order: keeper($1), reason($2), loser($3), project($4)
        assert conn.execute.await_args.args[1:] == (
            self._KEEPER,
            "merge",
            self._LOSER,
            "proj-a",
        )

    async def test_project_param_is_positional_four_without_keeper_guard(self):
        # Without the keeper guard, project is still $4 (the next free positional after
        # keeper $1, reason $2, loser $3) — there is no $1-reuse term to skip over.
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")

        await supersede_row(
            conn,
            loser_id=self._LOSER,
            keeper_id=self._KEEPER,
            reason="merge",
            project="proj-a",
        )

        sql = conn.execute.await_args.args[0]
        assert "LOWER(project) = LOWER($4::text)" in sql
        assert conn.execute.await_args.args[1:] == (
            self._KEEPER,
            "merge",
            self._LOSER,
            "proj-a",
        )

    async def test_wrong_project_yields_zero_rows_no_raise(self):
        # Simulates the DB rejecting the loser because its project != bound project.
        from scripts.core.db.memory_service_pg import supersede_row

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")

        count = await supersede_row(
            conn,
            loser_id=self._LOSER,
            keeper_id=self._KEEPER,
            reason="merge",
            project="proj-a",
        )

        assert count == 0  # cross-project loser left untouched; caller owns policy
