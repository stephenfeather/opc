"""Round 1 FIX 1: embedding_model column capability probe (issue #151).

On a pre-migration DB the embedding_model column may be absent. Recall must
NOT raise UndefinedColumnError — it must degrade to the unfiltered (pre-#151)
SQL exactly like the #139 project-column probe. These tests pin the probe and
the recall/store degradation.
"""

from __future__ import annotations

from typing import Any

import pytest


def _undefined_embedding_model_error() -> Exception:
    from asyncpg.exceptions import UndefinedColumnError

    return UndefinedColumnError('column "embedding_model" does not exist')


class TestEmbeddingModelColumnProbe:
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

        rb.reset_embedding_model_column_cache()
        conn = self._make_conn(None)
        assert await rb.embedding_model_column_available(conn) is True

    async def test_probe_false_when_column_missing(self):
        from scripts.core import recall_backends as rb

        rb.reset_embedding_model_column_cache()
        conn = self._make_conn(_undefined_embedding_model_error())
        assert await rb.embedding_model_column_available(conn) is False

    async def test_probe_targets_actual_relation(self):
        from scripts.core import recall_backends as rb

        sql = rb._EMBEDDING_MODEL_COLUMN_PROBE_SQL
        assert "archival_memory" in sql
        assert "embedding_model" in sql
        assert "information_schema" not in sql

    async def test_definitive_results_are_cached(self):
        from scripts.core import recall_backends as rb

        rb.reset_embedding_model_column_cache()
        conn = self._make_conn(None)
        await rb.embedding_model_column_available(conn)
        await rb.embedding_model_column_available(conn)
        assert conn.calls == 1

    async def test_transient_failure_not_cached(self):
        from scripts.core import recall_backends as rb

        rb.reset_embedding_model_column_cache()
        conn = self._make_conn(RuntimeError("connection reset"))
        assert await rb.embedding_model_column_available(conn) is False
        conn.outcome = None
        assert await rb.embedding_model_column_available(conn) is True
        assert conn.calls == 2


class _FakeVectorDb:
    """Pool/conn that reports embeddings present and column missing/present."""

    def __init__(self, *, has_embedding_model: bool) -> None:
        self.has_embedding_model = has_embedding_model
        self.executed: list[str] = []

    def make_pool(self):
        db = self

        class FakeConn:
            async def fetch(self, sql: str, *args: Any) -> list[Any]:
                from asyncpg.exceptions import UndefinedColumnError

                if not db.has_embedding_model and "embedding_model" in sql:
                    raise UndefinedColumnError(
                        'column "embedding_model" does not exist'
                    )
                db.executed.append(sql)
                return []

            async def execute(self, sql: str, *args: Any) -> str:
                # SET LOCAL hnsw.iterative_scan (issue #153): no-op for the fake.
                return "SET"

            def transaction(self):
                # RRF fetch runs inside a transaction now (issue #153 finding 1).
                class _Tx:
                    async def __aenter__(self):
                        return None

                    async def __aexit__(self, *exc: Any) -> bool:
                        return False

                return _Tx()

            async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
                return {"cnt": 5}  # embeddings present -> vector branch

        class FakeAcquire:
            async def __aenter__(self) -> FakeConn:
                return FakeConn()

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        class FakePool:
            def acquire(self) -> FakeAcquire:
                return FakeAcquire()

        return FakePool()


class TestRecallDegradesWhenColumnAbsent:
    def _patch(self, monkeypatch, db: _FakeVectorDb) -> None:
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

    async def test_postgres_vector_no_filter_when_column_absent(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        rb.reset_embedding_model_column_cache()
        db = _FakeVectorDb(has_embedding_model=False)
        self._patch(monkeypatch, db)

        results = await rb.search_learnings_postgres("query terms", k=3)
        assert results == []
        assert db.executed, "expected SQL to be executed (degraded, not error)"
        for sql in db.executed:
            assert "embedding_model" not in sql, sql[:160]

    async def test_postgres_vector_filtered_when_column_present(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        rb.reset_embedding_model_column_cache()
        db = _FakeVectorDb(has_embedding_model=True)
        self._patch(monkeypatch, db)

        results = await rb.search_learnings_postgres("query terms", k=3)
        assert results == []
        assert any("embedding_model" in sql for sql in db.executed), db.executed

    async def test_hybrid_rrf_no_filter_when_column_absent(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        rb.reset_embedding_model_column_cache()
        db = _FakeVectorDb(has_embedding_model=False)
        self._patch(monkeypatch, db)

        results = await rb.search_learnings_hybrid_rrf(
            "query terms", k=3, expand=False,
        )
        assert results == []
        assert db.executed, "expected degraded SQL to run"
        for sql in db.executed:
            assert "embedding_model" not in sql, sql[:160]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
