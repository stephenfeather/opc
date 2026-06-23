"""Round-3 finding 1 (issue #63 Phase 2b): archived_at column capability probe.

On a partial-migration DB (superseded_by present, archived_at absent) the recall
path must NOT pay repeated FAILED archived_at SQL on every call. It must probe
the column ONCE (cached), and when absent build the lifecycle CTE WITHOUT
archived_at FROM THE START — issuing no failed archived query. Mirrors the #139
project-column and #151 embedding_model probes. The pool teardown must reset the
cache so a re-probe runs against a possibly-different DB.
"""

from __future__ import annotations

from typing import Any

import pytest


def _undefined_archived_at_error() -> Exception:
    from asyncpg.exceptions import UndefinedColumnError

    return UndefinedColumnError('column "archived_at" does not exist')


class TestArchivedAtColumnProbe:
    def _make_conn(self, outcome: Exception | None):
        class FakeConn:
            def __init__(self) -> None:
                self.calls = 0
                self.outcome: Exception | None = outcome

            async def fetch(self, _sql: str, *args: Any) -> list[Any]:
                self.calls += 1
                if self.outcome is not None:
                    raise self.outcome
                return []

        return FakeConn()

    async def test_probe_true_when_column_exists(self):
        from scripts.core import recall_backends as rb

        rb.reset_archived_at_column_cache()
        conn = self._make_conn(None)
        assert await rb.archived_at_column_available(conn) is True

    async def test_probe_false_when_column_missing(self):
        from scripts.core import recall_backends as rb

        rb.reset_archived_at_column_cache()
        conn = self._make_conn(_undefined_archived_at_error())
        assert await rb.archived_at_column_available(conn) is False

    async def test_probe_targets_actual_relation(self):
        from scripts.core import recall_backends as rb

        sql = rb._ARCHIVED_AT_COLUMN_PROBE_SQL
        assert "archival_memory" in sql
        assert "archived_at" in sql
        assert "information_schema" not in sql

    async def test_definitive_results_are_cached(self):
        from scripts.core import recall_backends as rb

        rb.reset_archived_at_column_cache()
        conn = self._make_conn(None)
        await rb.archived_at_column_available(conn)
        await rb.archived_at_column_available(conn)
        assert conn.calls == 1

    async def test_transient_failure_not_cached(self):
        from scripts.core import recall_backends as rb

        rb.reset_archived_at_column_cache()
        conn = self._make_conn(RuntimeError("connection reset"))
        assert await rb.archived_at_column_available(conn) is False
        conn.outcome = None
        assert await rb.archived_at_column_available(conn) is True
        assert conn.calls == 2


class _FakeArchivedDb:
    """Pool/conn reporting embeddings present and archived_at present/absent.

    On a column-absent DB, ANY query that references archived_at raises
    UndefinedColumnError. This lets the test assert that NO archived_at SQL is
    ever executed (no failed archived query) when the probe reports absence.
    """

    def __init__(self, *, has_archived_at: bool) -> None:
        self.has_archived_at = has_archived_at
        self.executed: list[str] = []
        self.failed_archived: list[str] = []

    def make_pool(self):
        db = self

        class FakeConn:
            async def fetch(self, sql: str, *args: Any) -> list[Any]:
                from asyncpg.exceptions import UndefinedColumnError

                if not db.has_archived_at and "archived_at" in sql:
                    # Probe (SELECT archived_at ... LIMIT 0) is the ONLY allowed
                    # archived_at reference on a column-absent DB; a CTE that
                    # references archived_at is a failed hot-path query.
                    if "LIMIT 0" not in sql:
                        db.failed_archived.append(sql)
                    raise UndefinedColumnError(
                        'column "archived_at" does not exist'
                    )
                db.executed.append(sql)
                return []

            async def execute(self, sql: str, *args: Any) -> str:
                return "SET"

            async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
                return {"cnt": 5}

        class FakeAcquire:
            async def __aenter__(self) -> FakeConn:
                return FakeConn()

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        class FakePool:
            def acquire(self) -> FakeAcquire:
                return FakeAcquire()

        return FakePool()


class TestRecallBuildsNoArchivedSqlWhenColumnAbsent:
    def _patch(self, monkeypatch, db: _FakeArchivedDb) -> None:
        async def fake_get_pool():
            return db.make_pool()

        async def fake_init_pgvector(_conn: Any) -> None:
            return None

        import scripts.core.db.embedding_service as emb_mod
        import scripts.core.db.postgres_pool as pool_mod

        class FakeEmbedder:
            def __init__(self, *a: Any, **kw: Any) -> None: ...

            @property
            def model_label(self) -> str:
                return "voyage-code-3"

            async def embed(self, *_a: Any, **_kw: Any) -> list[float]:
                return [0.1] * 8

            async def aclose(self) -> None: ...

        monkeypatch.setattr(pool_mod, "get_pool", fake_get_pool)
        monkeypatch.setattr(pool_mod, "init_pgvector", fake_init_pgvector)
        monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbedder)

    def _reset(self):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        rb.reset_embedding_model_column_cache()
        rb.reset_archived_at_column_cache()
        rb.reset_hnsw_iterative_scan_cache()

    async def test_hybrid_rrf_issues_no_failed_archived_sql_when_absent(
        self, monkeypatch
    ):
        from scripts.core import recall_backends as rb

        self._reset()
        db = _FakeArchivedDb(has_archived_at=False)
        self._patch(monkeypatch, db)

        results = await rb.search_learnings_hybrid_rrf(
            "query terms", k=3, expand=False,
        )
        assert results == []
        assert db.executed, "expected degraded SQL to run"
        # The CRITICAL assertion: NO CTE referencing archived_at was ever
        # executed — the probe gated it off from the start.
        assert db.failed_archived == [], (
            "recall issued failed archived_at SQL on a column-absent DB: "
            + repr(db.failed_archived[:1])
        )
        for sql in db.executed:
            assert "archived_at" not in sql, sql[:200]

    async def test_hybrid_rrf_filters_archived_when_column_present(
        self, monkeypatch
    ):
        from scripts.core import recall_backends as rb

        self._reset()
        db = _FakeArchivedDb(has_archived_at=True)
        self._patch(monkeypatch, db)

        results = await rb.search_learnings_hybrid_rrf(
            "query terms", k=3, expand=False,
        )
        assert results == []
        assert any("archived_at" in sql for sql in db.executed), (
            "steady-state recall (column present) must filter archived_at"
        )

    async def test_postgres_vector_issues_no_failed_archived_sql_when_absent(
        self, monkeypatch
    ):
        from scripts.core import recall_backends as rb

        self._reset()
        db = _FakeArchivedDb(has_archived_at=False)
        self._patch(monkeypatch, db)

        results = await rb.search_learnings_postgres("query terms", k=3)
        assert results == []
        assert db.executed, "expected degraded SQL to run"
        assert db.failed_archived == [], (
            "non-RRF recall issued failed archived_at SQL when absent: "
            + repr(db.failed_archived[:1])
        )
        for sql in db.executed:
            assert "archived_at" not in sql, sql[:200]

    async def test_postgres_vector_filters_archived_when_column_present(
        self, monkeypatch
    ):
        from scripts.core import recall_backends as rb

        self._reset()
        db = _FakeArchivedDb(has_archived_at=True)
        self._patch(monkeypatch, db)

        results = await rb.search_learnings_postgres("query terms", k=3)
        assert results == []
        assert any("archived_at" in sql for sql in db.executed), (
            "steady-state non-RRF recall must filter archived_at"
        )


class TestPoolTeardownResetsArchivedCache:
    async def test_close_pool_resets_archived_at_cache(self):
        from scripts.core import recall_backends as rb
        from scripts.core.db import postgres_pool

        # Seed a definitive cached answer, then close the pool.
        rb._set_archived_at_column_cache_for_tests(True)
        assert rb._archived_at_column_cache is True
        await postgres_pool.close_pool()
        assert rb._archived_at_column_cache is None, (
            "close_pool must reset the archived_at probe cache so a re-probe "
            "runs against a possibly-different DB"
        )

    def test_reset_pool_resets_archived_at_cache(self):
        from scripts.core import recall_backends as rb
        from scripts.core.db import postgres_pool

        rb._set_archived_at_column_cache_for_tests(False)
        assert rb._archived_at_column_cache is False
        postgres_pool.reset_pool()
        assert rb._archived_at_column_cache is None, (
            "reset_pool must reset the archived_at probe cache"
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
