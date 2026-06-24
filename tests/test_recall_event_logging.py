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
        undefined_column: str | None = None,
    ):
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []
        self._fetch_rows = fetch_rows if fetch_rows is not None else []
        self._fetch_error = fetch_error
        self._execute_error = execute_error
        # Issue #228: simulate a pre-migration DB where a column is absent.
        # When set, any execute() whose SQL mentions this column name is
        # recorded (so the failed attempt is observable) THEN raises
        # UndefinedColumnError, exercising the legacy-INSERT fallback.
        self._undefined_column = undefined_column

    async def fetch(self, sql: str, *args):
        self.fetch_calls.append((sql, args))
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._fetch_rows

    async def execute(self, sql: str, *args):
        self.execute_calls.append((sql, args))
        if self._undefined_column is not None and self._undefined_column in sql:
            from asyncpg.exceptions import UndefinedColumnError

            raise UndefinedColumnError(
                f'column "{self._undefined_column}" does not exist'
            )
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
        # Issue #228: INSERT is now 7 columns (pool_size/fetch_k appended).
        assert len(args) == 7
        assert args[5] is None  # pool_size unset here -> NULL
        assert args[6] is None  # fetch_k unset here -> NULL

    async def test_insert_binds_pool_size_and_fetch_k(self, monkeypatch):
        # Issue #228: pool_size/fetch_k are bound as $6/$7 so selection rate
        # (result_count / pool_size) is computable at query time.
        rid = str(uuid.uuid4())
        conn = _CapturingConn(fetch_rows=[_row(rid, "opc")])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall(
            [rid], caller_project="opc", source="hook", pool_size=50, fetch_k=50
        )

        assert len(conn.execute_calls) == 1
        insert_sql, args = conn.execute_calls[0]
        assert "pool_size" in insert_sql
        assert "fetch_k" in insert_sql
        assert args[5] == 50  # pool_size
        assert args[6] == 50  # fetch_k

    async def test_zero_result_logs_pool_size_and_fetch_k(self, monkeypatch):
        # Issue #228: a "ran, picked nothing" recall now carries a denominator
        # (pool_size) so an empty result is distinguishable from a tiny pool.
        conn = _CapturingConn(fetch_rows=[])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall(
            [], caller_project="opc", source="cli", pool_size=12, fetch_k=50
        )

        assert len(conn.execute_calls) == 1
        _insert_sql, args = conn.execute_calls[0]
        assert args[3] == 0  # result_count
        assert args[5] == 12  # pool_size (the denominator)
        assert args[6] == 50  # fetch_k

    async def test_pre_migration_columns_absent_falls_back_to_legacy_insert(
        self, monkeypatch
    ):
        # Issue #228: recall_log exists but add_recall_log_pool_size.sql hasn't
        # run, so pool_size/fetch_k are absent. The 7-col INSERT raises
        # UndefinedColumnError; record_recall must fall back to the legacy
        # 5-col INSERT so the #140 recall event is STILL logged. Must not raise.
        rid = str(uuid.uuid4())
        conn = _CapturingConn(
            fetch_rows=[_row(rid, "opc")], undefined_column="pool_size"
        )
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid], caller_project="opc", pool_size=50, fetch_k=50)

        assert len(conn.execute_calls) == 2
        first_sql, _first_args = conn.execute_calls[0]
        second_sql, second_args = conn.execute_calls[1]
        # First attempt was the 7-col INSERT (mentions pool_size) and failed.
        assert "pool_size" in first_sql
        # Fallback is the legacy 5-col INSERT: no pool_size, exactly 5 args.
        assert "pool_size" not in second_sql
        assert len(second_args) == 5

    async def test_backward_compatible_pool_size_defaults_none(self, monkeypatch):
        # Issue #228: omitting pool_size/fetch_k binds them as NULL ("rate
        # unknown"), keeping pre-#228 callers valid.
        rid = str(uuid.uuid4())
        conn = _CapturingConn(fetch_rows=[_row(rid, "opc")])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid])

        assert len(conn.execute_calls) == 1
        args = conn.execute_calls[0][1]
        assert args[5] is None  # pool_size
        assert args[6] is None  # fetch_k

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

    async def test_trailing_newline_source_label_bound_as_null(self, monkeypatch):
        # re.match with `$` matches before a trailing newline; fullmatch must
        # reject "hook\n" so the stray byte never persists (aegis LOW-1).
        rid = str(uuid.uuid4())
        conn = _CapturingConn(fetch_rows=[_row(rid, "opc")])
        _patch_postgres(monkeypatch)
        _patch_pool(monkeypatch, conn)

        await record_recall([rid], caller_project="opc", source="hook\n")

        assert len(conn.execute_calls) == 1
        insert_args = conn.execute_calls[0][1]
        assert insert_args[4] is None

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

        async def fake_dispatch(params, *, project=None, capture=None):
            return [{"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        captured: dict = {}

        async def fake_record(
            ids, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
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

        async def fake_dispatch(params, *, project=None, capture=None):
            return [{"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        called = {"record": False}

        async def fake_record(
            ids, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
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

        async def fake_dispatch(params, *, project=None, capture=None):
            return [{"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")

        order: list[str] = []

        def fake_format(*a, **k):
            order.append("format")
            return ""

        async def fake_record(
            ids, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
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

        async def fake_dispatch(params, *, project=None, capture=None):
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

        async def slow_record(
            ids, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
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

    async def test_main_captures_pool_size_before_rerank_trim(self, monkeypatch):
        # Issue #228: pool_size must be the RAW backend candidate pool captured
        # BEFORE enrichment/tag-filter/rerank trim it, so selection rate is
        # computable. Rerank is ON here and trims 6 -> k=3.
        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

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
                "--k",
                "3",
            ],
            raising=False,
        )

        async def fake_dispatch(params, *, project=None, capture=None):
            return [
                {"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}
                for _ in range(6)
            ]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        async def identity(results):
            return results

        monkeypatch.setattr(rl, "enrich_with_pattern_strength", identity)
        monkeypatch.setattr(rl, "enrich_with_kg_context", identity)
        # rerank is imported locally inside main() from scripts.core.reranker.
        monkeypatch.setattr(
            reranker_mod, "rerank", lambda results, ctx, k: results[:k]
        )

        captured: dict = {}

        async def fake_record(
            ids, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            captured["ids"] = ids
            captured["pool_size"] = pool_size
            captured["fetch_k"] = fetch_k

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        assert captured["pool_size"] == 6  # raw pool BEFORE trim
        assert len(captured["ids"]) == 3  # trimmed to k
        assert captured["fetch_k"] == max(3 * 3, 50)  # == 50

    async def test_main_passes_fetch_k_no_rerank(self, monkeypatch):
        # Issue #228: with --no-rerank, compute_fetch_k(k)==k, and no trim
        # happens, so pool_size == returned count == k.
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
                "--k",
                "4",
            ],
            raising=False,
        )

        async def fake_dispatch(params, *, project=None, capture=None):
            return [
                {"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}
                for _ in range(4)
            ]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        captured: dict = {}

        async def fake_record(
            ids, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            captured["pool_size"] = pool_size
            captured["fetch_k"] = fetch_k

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        assert captured["pool_size"] == 4
        assert captured["fetch_k"] == 4  # compute_fetch_k(k, no_rerank=True) == k


# ---------------------------------------------------------------------------
# --exclude-ids (issue #228 item 2: already-surfaced filtering)
# ---------------------------------------------------------------------------


class TestExcludeIds:
    """main(): --exclude-ids drops surfaced learnings BEFORE rerank, AFTER the
    pool_size capture (so item 1's selection-rate telemetry is unaffected)."""

    @staticmethod
    def _argv(*extra: str) -> list[str]:
        return [
            "recall_learnings.py",
            "--query",
            "x",
            "--source",
            "hook",
            "--text-only",
            "--json",
            "--k",
            "5",
            *extra,
        ]

    @staticmethod
    def _result(rid, content="x", sim=0.5):
        # A result dict carrying the fields _build_json_result requires so the
        # real _format_output can render it (for capsys-based assertions).
        return {
            "id": rid,
            "content": content,
            "similarity": sim,
            "session_id": "sess",
            "created_at": "2026-06-23T00:00:00+00:00",
        }

    @staticmethod
    def _wire(monkeypatch, rl, reranker_mod, results):
        async def fake_dispatch(params, *, project=None, capture=None):
            return [dict(r) for r in results]

        async def identity(res):
            return res

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")
        monkeypatch.setattr(rl, "enrich_with_pattern_strength", identity)
        monkeypatch.setattr(rl, "enrich_with_kg_context", identity)
        # rerank is a LOCAL import inside main() from scripts.core.reranker --
        # patch the source module, not recall_learnings.
        monkeypatch.setattr(
            reranker_mod, "rerank", lambda res, ctx, k: res[:k]
        )

    async def test_exclude_ids_drops_matching_result(self, monkeypatch, capsys):
        import json as _json

        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        keep1, drop, keep2 = (str(uuid.uuid4()) for _ in range(3))
        results = [
            self._result(keep1, "a", 0.9),
            self._result(drop, "b", 0.8),
            self._result(keep2, "c", 0.7),
        ]
        monkeypatch.setattr(
            rl.sys, "argv", self._argv("--exclude-ids", drop), raising=False
        )
        self._wire(monkeypatch, rl, reranker_mod, results)

        captured: dict = {}

        async def fake_record(
            ids, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            captured["ids"] = list(ids)

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        out = capsys.readouterr().out
        payload = _json.loads(out)
        out_ids = {r["id"] for r in payload["results"]}
        assert drop not in out_ids
        assert {keep1, keep2} <= out_ids
        assert drop not in captured["ids"]

    async def test_pool_size_captured_before_exclusion(self, monkeypatch):
        # Regression guard for item 1: pool_size must be the RAW backend pool
        # (pre-exclusion), since exclusion is a downstream filter applied after
        # the pool_size capture.
        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        ids = [str(uuid.uuid4()) for _ in range(6)]
        results = [
            {"id": i, "content": "x", "similarity": 0.5} for i in ids
        ]
        # Exclude 2 of the 6 raw candidates.
        monkeypatch.setattr(
            rl.sys,
            "argv",
            self._argv("--exclude-ids", ids[0], ids[1]),
            raising=False,
        )
        self._wire(monkeypatch, rl, reranker_mod, results)
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        captured: dict = {}

        async def fake_record(
            ids_arg, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            captured["pool_size"] = pool_size
            captured["ids"] = list(ids_arg)

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        # pool_size is the pre-exclusion raw pool (6), NOT the post-exclusion 4.
        assert captured["pool_size"] == 6
        assert ids[0] not in captured["ids"]
        assert ids[1] not in captured["ids"]
        assert len(captured["ids"]) == 4

    async def test_exclude_ids_bumps_fetch_k_for_backfill(self, monkeypatch):
        # Issue #228 item 2 (round-2): exclusion runs AFTER the backend's fixed
        # over-fetch, so the pool is over-fetched by the exclude-set size. Without
        # this a session that already surfaced the top of the pool would starve
        # (all over-fetched rows filtered out) instead of getting fresh results.
        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        results = [{"id": i, "content": "x", "similarity": 0.5} for i in ids]
        # k=5 -> base fetch_k = max(3*5, 50) = 50; two distinct excludes -> 52.
        monkeypatch.setattr(
            rl.sys, "argv", self._argv("--exclude-ids", ids[0], ids[1]), raising=False
        )
        self._wire(monkeypatch, rl, reranker_mod, results)
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        captured: dict = {}

        async def fake_record(
            ids_arg, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            captured["fetch_k"] = fetch_k

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        assert captured["fetch_k"] == max(3 * 5, 50) + 2

    async def test_no_rerank_exclude_trims_to_k(self, monkeypatch, capsys):
        # Codex P2: with --no-rerank the rerank k-trim is skipped, so the
        # exclude over-fetch must be trimmed back to base_fetch_k (== k on the
        # no-rerank path) or output exceeds the requested --k when excludes are
        # stale / absent from the pool.
        import json as _json

        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        ids = [str(uuid.uuid4()) for _ in range(5)]
        results = [self._result(i, "x", 0.5) for i in ids]
        stale = str(uuid.uuid4())  # not present in the fetched pool
        argv = [
            "recall_learnings.py", "--query", "x", "--source", "hook",
            "--text-only", "--json", "--k", "2", "--no-rerank",
            "--exclude-ids", stale,
        ]
        monkeypatch.setattr(rl.sys, "argv", argv, raising=False)
        self._wire(monkeypatch, rl, reranker_mod, results)

        async def fake_record(
            ids_arg, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            return None

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        payload = _json.loads(capsys.readouterr().out)
        # Output is capped to --k (2), not the full over-fetched pool.
        assert len(payload["results"]) == 2

    async def test_no_exclude_ids_no_regression(self, monkeypatch):
        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        results = [{"id": i, "content": "x", "similarity": 0.5} for i in ids]
        monkeypatch.setattr(rl.sys, "argv", self._argv(), raising=False)
        self._wire(monkeypatch, rl, reranker_mod, results)
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        captured: dict = {}

        async def fake_record(
            ids_arg, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            captured["ids"] = list(ids_arg)
            captured["pool_size"] = pool_size

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        assert set(captured["ids"]) == set(ids)
        assert captured["pool_size"] == 3

    async def test_exclude_all_candidates_graceful_empty(self, monkeypatch, capsys):
        import json as _json

        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        results = [{"id": i, "content": "x", "similarity": 0.5} for i in ids]
        monkeypatch.setattr(
            rl.sys, "argv", self._argv("--exclude-ids", *ids), raising=False
        )
        self._wire(monkeypatch, rl, reranker_mod, results)

        async def fake_record(
            ids_arg, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            return None

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        out = capsys.readouterr().out
        payload = _json.loads(out)  # must be valid JSON, no crash
        assert payload["results"] == []

    async def test_exclude_id_string_matches_uuid_result(self, monkeypatch):
        # str/UUID normalization: a result id typed as uuid.UUID is dropped when
        # the exclude id is passed as a plain string. (DB rows normally carry
        # str ids; this exercises the str() coercion on both sides directly.)
        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        drop_uuid = uuid.uuid4()
        keep_uuid = uuid.uuid4()
        results = [
            self._result(drop_uuid, "b", 0.8),
            self._result(keep_uuid, "c", 0.7),
        ]
        monkeypatch.setattr(
            rl.sys, "argv", self._argv("--exclude-ids", str(drop_uuid)), raising=False
        )
        self._wire(monkeypatch, rl, reranker_mod, results)
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        captured: dict = {}

        async def fake_record(
            ids_arg, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            captured["ids"] = [str(i) for i in ids_arg]

        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        assert str(drop_uuid) not in captured["ids"]
        assert str(keep_uuid) in captured["ids"]


class TestUnionCap:
    """_union_cap: dedupe-union keeping the most-recent tail when over cap."""

    def test_dedupes_union(self):
        import scripts.core.recall_learnings as rl

        assert rl._union_cap(["a", "b"], ["b", "c"], 500) == ["a", "b", "c"]

    def test_keeps_recent_tail_when_over_cap(self):
        import scripts.core.recall_learnings as rl

        prior = [str(n) for n in range(500)]
        fresh = ["x", "y", "z"]
        out = rl._union_cap(prior, fresh, 500)
        assert len(out) == 500
        for f in fresh:
            assert f in out
        # Oldest three evicted from the head.
        assert "0" not in out and "1" not in out and "2" not in out

    def test_handles_empty(self):
        import scripts.core.recall_learnings as rl

        assert rl._union_cap([], ["a"], 500) == ["a"]


class TestSurfacedSession:
    """main(): --surfaced-session reads the session's prior surfaced ids, unions
    them into the exclusion set (in-process, no second subprocess), and upserts
    the surfaced set with the ids returned this run."""

    @staticmethod
    def _argv(*extra: str) -> list[str]:
        return [
            "recall_learnings.py",
            "--query",
            "x",
            "--source",
            "hook",
            "--text-only",
            "--json",
            "--k",
            "5",
            *extra,
        ]

    @staticmethod
    def _wire(monkeypatch, rl, reranker_mod, results):
        async def fake_dispatch(params, *, project=None, capture=None):
            return [dict(r) for r in results]

        async def identity(res):
            return res

        async def fake_record(
            ids_arg, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            return None

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")
        monkeypatch.setattr(rl, "enrich_with_pattern_strength", identity)
        monkeypatch.setattr(rl, "enrich_with_kg_context", identity)
        monkeypatch.setattr(rl, "record_recall", fake_record)
        monkeypatch.setattr(reranker_mod, "rerank", lambda res, ctx, k: res[:k])

    async def test_prior_surfaced_excluded(self, monkeypatch):
        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        keep1, drop, keep2 = (str(uuid.uuid4()) for _ in range(3))
        results = [
            {"id": keep1, "content": "a", "similarity": 0.9},
            {"id": drop, "content": "b", "similarity": 0.8},
            {"id": keep2, "content": "c", "similarity": 0.7},
        ]
        monkeypatch.setattr(
            rl.sys, "argv", self._argv("--surfaced-session", "sess-1"), raising=False
        )
        self._wire(monkeypatch, rl, reranker_mod, results)
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        async def fake_read(session_id):
            return [drop]

        captured: dict = {}

        async def fake_persist(session_id, ids):
            captured["session_id"] = session_id
            captured["persisted"] = list(ids)

        monkeypatch.setattr(rl, "read_surfaced_ids", fake_read)
        monkeypatch.setattr(rl, "persist_surfaced_ids", fake_persist)

        rc = await rl.main()
        assert rc == 0
        # The prior-surfaced id is dropped; the others persist (prior UNION new).
        assert captured["session_id"] == "sess-1"
        assert drop in captured["persisted"]  # prior is retained in the set
        assert keep1 in captured["persisted"] and keep2 in captured["persisted"]

    async def test_read_failure_degrades_to_no_exclusion(self, monkeypatch, capsys):
        import json as _json

        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        results = [
            {
                "id": i,
                "content": "x",
                "similarity": 0.5,
                "session_id": "sess",
                "created_at": "2026-06-23T00:00:00+00:00",
            }
            for i in ids
        ]
        monkeypatch.setattr(
            rl.sys, "argv", self._argv("--surfaced-session", "sess-1"), raising=False
        )
        self._wire(monkeypatch, rl, reranker_mod, results)

        async def boom(session_id):
            raise RuntimeError("db down")

        persist_calls: list = []

        async def fake_persist(session_id, ids):
            persist_calls.append((session_id, list(ids)))

        monkeypatch.setattr(rl, "read_surfaced_ids", boom)
        monkeypatch.setattr(rl, "persist_surfaced_ids", fake_persist)

        rc = await rl.main()
        assert rc == 0
        # Read failure degrades to no exclusion: all results survive, no crash.
        payload = _json.loads(capsys.readouterr().out)
        out_ids = {r["id"] for r in payload["results"]}
        assert out_ids == set(ids)
        # Critical: persist is SKIPPED after a failed read, so the REPLACE write
        # can't erase the existing stored surfaced set (round-4 finding).
        assert persist_calls == []

    async def test_genuine_empty_read_still_persists(self, monkeypatch):
        # A successful read of an empty/absent set (returns []) is NOT a failure:
        # persist must still run so the first turn's picks are stored.
        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        rid = str(uuid.uuid4())
        results = [{"id": rid, "content": "x", "similarity": 0.5}]
        monkeypatch.setattr(
            rl.sys, "argv", self._argv("--surfaced-session", "sess-1"), raising=False
        )
        self._wire(monkeypatch, rl, reranker_mod, results)
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        async def empty_read(session_id):
            return []

        persist_calls: list = []

        async def fake_persist(session_id, ids):
            persist_calls.append(list(ids))

        monkeypatch.setattr(rl, "read_surfaced_ids", empty_read)
        monkeypatch.setattr(rl, "persist_surfaced_ids", fake_persist)

        rc = await rl.main()
        assert rc == 0
        assert persist_calls == [[rid]]

    async def test_session_surfaced_bumps_fetch_k(self, monkeypatch):
        import scripts.core.recall_learnings as rl
        import scripts.core.reranker as reranker_mod

        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        results = [{"id": str(uuid.uuid4()), "content": "x", "similarity": 0.5}]
        monkeypatch.setattr(
            rl.sys, "argv", self._argv("--surfaced-session", "sess-1"), raising=False
        )
        self._wire(monkeypatch, rl, reranker_mod, results)
        monkeypatch.setattr(rl, "_format_output", lambda *a, **k: "", raising=False)

        prior = [str(uuid.uuid4()) for _ in range(2)]

        async def fake_read(session_id):
            return prior

        captured: dict = {}

        async def fake_record(
            ids_arg, *, caller_project=None, source=None, pool_size=None, fetch_k=None
        ):
            captured["fetch_k"] = fetch_k

        async def fake_persist(session_id, ids):
            return None

        monkeypatch.setattr(rl, "read_surfaced_ids", fake_read)
        monkeypatch.setattr(rl, "persist_surfaced_ids", fake_persist)
        monkeypatch.setattr(rl, "record_recall", fake_record)

        rc = await rl.main()
        assert rc == 0
        # k=5 -> base 50; two session-surfaced ids -> 52.
        assert captured["fetch_k"] == max(3 * 5, 50) + len(prior)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
