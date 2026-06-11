"""Tests for recall event logging (issue #140).

Validates that:
1. resolve_caller_project resolves the caller's canonical project
2. record_recall captures recalled rows' projects via RETURNING and logs a
   single recall_log row with parallel (recalled_ids, recalled_projects) arrays
3. The INSERT is best-effort: failures are swallowed and never break the
   counter UPDATE (pre-migration DB simulation)
4. main() wiring passes caller_project + source, and --json-full still skips
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.recall_learnings import (  # noqa: E402
    _sanitize_source,
    record_recall,
    resolve_caller_project,
)

# ---------------------------------------------------------------------------
# Capturing fixtures (extend the test_recall_project_first.py pattern to also
# capture execute() calls, not just fetch()).
# ---------------------------------------------------------------------------


class _CapturingConn:
    """Records (sql, args) for every fetch/execute and returns canned rows.

    ``fetch_rows`` is the list returned by every ``fetch`` call (the UPDATE ...
    RETURNING result). ``fetch_error`` / ``execute_error`` raise on the
    respective call to simulate a pre-migration DB or transient failure.
    """

    def __init__(
        self,
        fetch_rows=None,
        *,
        fetch_error: Exception | None = None,
        execute_error: Exception | None = None,
    ):
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []
        self._fetch_rows = fetch_rows if fetch_rows is not None else []
        self._fetch_error = fetch_error
        self._execute_error = execute_error

    async def fetch(self, sql: str, *args):
        self.fetch_calls.append((sql, args))
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._fetch_rows

    async def execute(self, sql: str, *args):
        self.execute_calls.append((sql, args))
        if self._execute_error is not None:
            raise self._execute_error
        return "INSERT 0 1"


class _CapturingPool:
    def __init__(self, conn: _CapturingConn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Acquire:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Acquire()


def _patch_pool(monkeypatch, conn: _CapturingConn) -> None:
    async def fake_get_pool():
        return _CapturingPool(conn)

    import scripts.core.db.postgres_pool as pool_mod

    monkeypatch.setattr(pool_mod, "get_pool", fake_get_pool)


def _patch_postgres(monkeypatch) -> None:
    import scripts.core.recall_learnings as rl

    monkeypatch.setattr(rl, "get_backend", lambda: "postgres")


def _row(rid: str, project: str | None) -> dict:
    """Mimic an asyncpg Record from UPDATE ... RETURNING id, project."""
    return {"id": rid, "project": project}


# ---------------------------------------------------------------------------
# resolve_caller_project
# ---------------------------------------------------------------------------


class TestResolveCallerProject:
    """Pure resolver: explicit project wins (canonicalized), else project_dir."""

    def test_explicit_project_wins_and_is_canonicalized(self):
        # Mixed-case explicit arg canonicalizes to lowercase, ignoring dir.
        assert resolve_caller_project("OPC", "/some/other/repo") == "opc"

    def test_blank_falls_back_to_dir(self):
        assert resolve_caller_project("   ", "/Users/me/Development/myrepo") == "myrepo"

    def test_none_falls_back_to_dir(self):
        assert resolve_caller_project(None, "/Users/me/Development/myrepo") == "myrepo"

    def test_worktree_aware_fallback(self):
        # project_from_path resolves '.worktrees/branch' to the repo basename.
        path = "/Users/me/opc/.worktrees/agent-140-recall-log"
        assert resolve_caller_project(None, path) == "opc"

    def test_returns_none_when_nothing_available(self):
        assert resolve_caller_project(None, None) is None
        assert resolve_caller_project("", "") is None


# ---------------------------------------------------------------------------
# record_recall logging
# ---------------------------------------------------------------------------


class TestRecordRecallLogging:
    """record_recall: UPDATE ... RETURNING, then a best-effort recall_log INSERT."""

    async def test_update_uses_returning_id_and_project(self, monkeypatch):
        rid = str(uuid.uuid4())
        conn = _CapturingConn(fetch_rows=[_row(rid, "opc")])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid], caller_project="opc", source="cli")

        assert len(conn.fetch_calls) == 1
        update_sql = conn.fetch_calls[0][0]
        assert "recall_count = recall_count + 1" in update_sql
        assert "last_recalled = NOW()" in update_sql
        assert "RETURNING id, project" in update_sql
        assert conn.fetch_calls[0][1][0] == [rid]

    async def test_insert_binds_caller_project_and_parallel_arrays(self, monkeypatch):
        # Two recalled rows: one attributed, one NULL (unattributed memory).
        rid1, rid2 = str(uuid.uuid4()), str(uuid.uuid4())
        conn = _CapturingConn(fetch_rows=[_row(rid1, "opc"), _row(rid2, None)])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid1, rid2], caller_project="opc", source="hook")

        assert len(conn.execute_calls) == 1
        insert_sql, args = conn.execute_calls[0]
        assert "INSERT INTO recall_log" in insert_sql
        assert "caller_project" in insert_sql
        assert "recalled_ids" in insert_sql
        assert "recalled_projects" in insert_sql
        assert "result_count" in insert_sql
        assert "source" in insert_sql
        # $1 caller_project, $2 recalled_ids, $3 recalled_projects,
        # $4 result_count, $5 source
        assert args[0] == "opc"
        assert args[1] == [rid1, rid2]
        assert args[2] == ["opc", None]  # parallel array preserves NULL element
        assert args[3] == 2
        assert args[4] == "hook"

    async def test_no_insert_when_update_returns_zero_rows(self, monkeypatch):
        # No rows matched (e.g. concurrently deleted ids) -> skip the INSERT.
        conn = _CapturingConn(fetch_rows=[])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([str(uuid.uuid4())], caller_project="opc", source="cli")

        assert len(conn.fetch_calls) == 1
        assert conn.execute_calls == []

    async def test_event_logged_when_caller_project_is_none(self, monkeypatch):
        rid = str(uuid.uuid4())
        conn = _CapturingConn(fetch_rows=[_row(rid, "opc")])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid], caller_project=None, source=None)

        assert len(conn.execute_calls) == 1
        args = conn.execute_calls[0][1]
        assert args[0] is None  # caller_project bound as NULL
        assert args[4] is None  # source bound as NULL

    async def test_insert_failure_swallowed_update_unaffected(self, monkeypatch):
        # Pre-migration DB: recall_log doesn't exist, so the INSERT raises.
        # The counter UPDATE (fetch) must still have run and not re-raise.
        from asyncpg.exceptions import UndefinedTableError

        rid = str(uuid.uuid4())
        conn = _CapturingConn(
            fetch_rows=[_row(rid, "opc")],
            execute_error=UndefinedTableError('relation "recall_log" does not exist'),
        )
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        # Must not raise.
        await record_recall([rid], caller_project="opc", source="cli")

        assert len(conn.fetch_calls) == 1  # UPDATE ran
        assert len(conn.execute_calls) == 1  # INSERT attempted, then swallowed

    async def test_backward_compatible_no_kwargs(self, monkeypatch):
        # Calling with only result_ids stays valid (caller_project/source None).
        rid = str(uuid.uuid4())
        conn = _CapturingConn(fetch_rows=[_row(rid, "opc")])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid])

        assert len(conn.fetch_calls) == 1
        assert len(conn.execute_calls) == 1
        assert conn.execute_calls[0][1][0] is None  # caller_project None

    async def test_sqlite_skips_entirely(self, monkeypatch):
        import scripts.core.recall_learnings as rl

        monkeypatch.setattr(rl, "get_backend", lambda: "sqlite")
        conn = _CapturingConn(fetch_rows=[_row(str(uuid.uuid4()), "opc")])
        _patch_pool(monkeypatch, conn)

        await record_recall([str(uuid.uuid4())], caller_project="opc", source="cli")

        assert conn.fetch_calls == []
        assert conn.execute_calls == []

    async def test_empty_ids_postgres_logs_zero_result_event(self, monkeypatch):
        # A recall that found nothing is precisely the over-restrictive-scoping
        # signature (#130): skip the counter UPDATE (nothing to update) but
        # still log a zero-result recall_log row (issue #140 r2).
        conn = _CapturingConn(fetch_rows=[])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([], caller_project="opc", source="cli")

        # No counter UPDATE (no ids to update)...
        assert conn.fetch_calls == []
        # ...but a zero-result event is logged with empty parallel arrays.
        assert len(conn.execute_calls) == 1
        insert_sql, args = conn.execute_calls[0]
        assert "INSERT INTO recall_log" in insert_sql
        assert args[0] == "opc"  # caller_project
        assert args[1] == []  # recalled_ids
        assert args[2] == []  # recalled_projects
        assert args[3] == 0  # result_count
        assert args[4] == "cli"  # source

    async def test_empty_ids_postgres_sanitizes_source(self, monkeypatch):
        conn = _CapturingConn(fetch_rows=[])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([], caller_project="opc", source="Bad Label")

        assert len(conn.execute_calls) == 1
        assert conn.execute_calls[0][1][4] is None  # invalid source dropped

    async def test_empty_ids_postgres_insert_failure_swallowed(self, monkeypatch):
        # Pre-migration DB without recall_log: the zero-result INSERT raises
        # and must be swallowed without re-raising.
        from asyncpg.exceptions import UndefinedTableError

        conn = _CapturingConn(
            fetch_rows=[],
            execute_error=UndefinedTableError('relation "recall_log" does not exist'),
        )
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([], caller_project="opc", source="cli")  # must not raise

        assert conn.fetch_calls == []
        assert len(conn.execute_calls) == 1

    async def test_empty_ids_sqlite_skips_entirely(self, monkeypatch):
        import scripts.core.recall_learnings as rl

        monkeypatch.setattr(rl, "get_backend", lambda: "sqlite")
        conn = _CapturingConn(fetch_rows=[])
        _patch_pool(monkeypatch, conn)

        await record_recall([], caller_project="opc", source="cli")

        assert conn.fetch_calls == []
        assert conn.execute_calls == []

    async def test_missing_project_column_falls_back_to_counter_update(self, monkeypatch):
        # Version skew: temporal-decay columns exist but add_project_column.sql
        # was never applied, so UPDATE ... RETURNING project raises. The
        # counter UPDATE must still happen via a project-free fallback, and the
        # recall_log INSERT must be skipped. Must not raise (issue #140 r1).
        from asyncpg.exceptions import UndefinedColumnError

        rid = str(uuid.uuid4())
        conn = _CapturingConn(
            fetch_error=UndefinedColumnError('column "project" does not exist'),
        )
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid], caller_project="opc", source="cli")

        # The RETURNING fetch was attempted...
        assert len(conn.fetch_calls) == 1
        assert "RETURNING id, project" in conn.fetch_calls[0][0]
        # ...then the counter-only fallback ran via execute (no RETURNING).
        assert len(conn.execute_calls) == 1
        fallback_sql, fallback_args = conn.execute_calls[0]
        assert "recall_count = recall_count + 1" in fallback_sql
        assert "last_recalled = NOW()" in fallback_sql
        assert "RETURNING" not in fallback_sql
        assert "recall_log" not in fallback_sql
        assert fallback_args[0] == [rid]

    async def test_invalid_source_label_bound_as_null(self, monkeypatch):
        # A non-conforming source label must be dropped to NULL, not stored.
        rid = str(uuid.uuid4())
        conn = _CapturingConn(fetch_rows=[_row(rid, "opc")])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid], caller_project="opc", source="Hook With Spaces")

        assert len(conn.execute_calls) == 1
        insert_args = conn.execute_calls[0][1]
        assert insert_args[4] is None  # invalid source dropped to NULL

    async def test_valid_source_label_preserved(self, monkeypatch):
        rid = str(uuid.uuid4())
        conn = _CapturingConn(fetch_rows=[_row(rid, "opc")])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid], caller_project="opc", source="mcp")

        assert conn.execute_calls[0][1][4] == "mcp"


# ---------------------------------------------------------------------------
# _sanitize_source
# ---------------------------------------------------------------------------


class TestSanitizeSource:
    """Source labels are validated label-only at the writer (no DB CHECK)."""

    def test_valid_labels_pass_through(self):
        for label in ("hook", "mcp", "cli", "a", "session-start", "x_y-9"):
            assert _sanitize_source(label) == label

    def test_none_stays_none(self):
        assert _sanitize_source(None) is None

    def test_uppercase_dropped(self):
        assert _sanitize_source("Hook") is None

    def test_spaces_dropped(self):
        assert _sanitize_source("hook path") is None

    def test_leading_non_alpha_dropped(self):
        assert _sanitize_source("1hook") is None
        assert _sanitize_source("-hook") is None

    def test_too_long_dropped(self):
        # 33 chars (max is 32: leading alpha + up to 31 trailing).
        assert _sanitize_source("a" * 33) is None

    def test_max_length_passes(self):
        assert _sanitize_source("a" * 32) == "a" * 32

    def test_prompt_like_text_dropped(self):
        assert _sanitize_source("how do I reset my password?") is None

    def test_empty_string_dropped(self):
        assert _sanitize_source("") is None


# ---------------------------------------------------------------------------
# main() wiring
# ---------------------------------------------------------------------------


class TestMainWiring:
    """main(): resolve caller_project + source and pass them to record_recall."""

    async def test_project_and_source_passed_to_record_recall(self, monkeypatch):
        import scripts.core.recall_learnings as rl

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setattr(
            rl.sys,
            "argv",
            [
                "recall_learnings.py",
                "--query",
                "x",
                "--project",
                "Foo",
                "--source",
                "hook",
                "--text-only",
                "--json",
                "--no-rerank",
            ],
            raising=False,
        )

        async def fake_dispatch(params, *, project=None):
            return [{"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        captured: dict = {}

        async def fake_record(ids, *, caller_project=None, source=None):
            captured["ids"] = ids
            captured["caller_project"] = caller_project
            captured["source"] = source

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        assert captured["caller_project"] == "foo"  # canonicalized
        assert captured["source"] == "hook"

    async def test_json_full_skips_record_recall(self, monkeypatch):
        import scripts.core.recall_learnings as rl

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setattr(
            rl.sys,
            "argv",
            [
                "recall_learnings.py",
                "--query",
                "x",
                "--project",
                "Foo",
                "--source",
                "hook",
                "--text-only",
                "--json-full",
                "--no-rerank",
            ],
            raising=False,
        )

        async def fake_dispatch(params, *, project=None):
            return [{"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        called = {"record": False}

        async def fake_record(ids, *, caller_project=None, source=None):
            called["record"] = True

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        assert called["record"] is False

    async def test_output_printed_before_record_recall(self, monkeypatch):
        # Best-effort logging must never delay user-visible output under the
        # memory-awareness hook's 5s spawn timeout: format/print first, then
        # record_recall (issue #140 r2).
        import scripts.core.recall_learnings as rl

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setattr(
            rl.sys,
            "argv",
            [
                "recall_learnings.py",
                "--query",
                "x",
                "--source",
                "hook",
                "--text-only",
                "--json",
                "--no-rerank",
            ],
            raising=False,
        )

        async def fake_dispatch(params, *, project=None):
            return [{"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")

        order: list[str] = []

        def fake_format(*a, **k):
            order.append("format")
            return ""

        async def fake_record(ids, *, caller_project=None, source=None):
            order.append("record")

        monkeypatch.setattr(rl, "_format_output", fake_format, raising=False)
        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        assert order == ["format", "record"]

    async def test_record_recall_timeout_does_not_block_main(self, monkeypatch):
        # Hard latency bound: spawnSync waits for process EXIT, so a slow
        # record_recall would burn the hook's 5s budget. main() must bound it
        # with asyncio.wait_for, still print output, and return 0 promptly
        # (issue #140 r3).
        import scripts.core.recall_learnings as rl

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setattr(
            rl.sys,
            "argv",
            [
                "recall_learnings.py",
                "--query",
                "x",
                "--source",
                "hook",
                "--text-only",
                "--json",
                "--no-rerank",
            ],
            raising=False,
        )

        async def fake_dispatch(params, *, project=None):
            return [{"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")
        # Tiny injected bound so the test is fast.
        monkeypatch.setattr(rl, "RECORD_RECALL_TIMEOUT", 0.05)

        printed: list[str] = []

        def fake_format(*a, **k):
            return "formatted-output"

        monkeypatch.setattr(rl, "_format_output", fake_format, raising=False)
        monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(a[0] if a else ""))

        async def slow_record(ids, *, caller_project=None, source=None):
            await asyncio.sleep(5)  # far longer than the injected 0.05 bound

        monkeypatch.setattr(rl, "record_recall", slow_record)

        start = time.monotonic()
        rc = await rl.main()
        elapsed = time.monotonic() - start

        assert rc == 0
        # Generous margin for loaded CI machines; what matters is that the
        # 5s sleep was cancelled, so anything well under 5s proves the bound.
        assert elapsed < 3.0
        assert printed  # output was still produced before the timeout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
