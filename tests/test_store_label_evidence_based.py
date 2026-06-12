"""Round 2 FIX 3: write-path label is evidence-based, not probe-based.

A read-degradation probe (cached process-lifetime, or a transient probe
error) must NOT govern writes — otherwise the 'bge' column default
permanently mislabels voyage/openai rows. The store path always attempts the
labeled INSERT first and retries the legacy INSERT only on a real
UndefinedColumnError from the DB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import asyncpg
import pytest


class _Savepoint:
    """Async CM standing in for conn.transaction() (asyncpg savepoint)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _StoreConn:
    """Connection double recording executed INSERT SQL/args.

    ``fail_labeled`` makes the FIRST labeled INSERT raise
    UndefinedColumnError (column missing), exercising the legacy retry.
    """

    def __init__(self, *, fail_labeled: bool = False) -> None:
        self.fail_labeled = fail_labeled
        self.executed: list[tuple[str, tuple]] = []

    def transaction(self) -> _Savepoint:
        return _Savepoint()

    async def execute(self, sql: str, *args):
        if self.fail_labeled and "embedding_model" in sql:
            raise asyncpg.exceptions.UndefinedColumnError(
                'column "embedding_model" does not exist'
            )
        self.executed.append((sql, args))
        return "INSERT 0 1"

    @property
    def insert_calls(self) -> list[tuple[str, tuple]]:
        return [c for c in self.executed if "INSERT INTO archival_memory" in c[0]]


class _FakeTxn:
    def __init__(self, conn: _StoreConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _StoreConn:
        return self._conn

    async def __aexit__(self, *exc):
        return False


def _patches(conn: _StoreConn):
    return (
        patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=_FakeTxn(conn),
        ),
        patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            AsyncMock(),
        ),
    )


class TestEvidenceBasedLabelWrite:
    async def test_labeled_insert_attempted_regardless_of_probe(self):
        """Even if the read-probe is cached False, the labeled INSERT is
        attempted — the write path must not consult the probe."""
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = _StoreConn(fail_labeled=False)
        svc = MemoryServicePG(session_id="s1")
        p1, p2 = _patches(conn)
        # Probe cached False (read-degraded) must be irrelevant to the write.
        with p1, p2, patch(
            "scripts.core.db.memory_service_pg."
            "embedding_model_column_available",
            AsyncMock(return_value=False),
        ) as probe:
            await svc.store(
                "fact", embedding=[0.1] * 1024,
                embedding_model="voyage-code-3",
            )

        assert len(conn.insert_calls) == 1
        sql, args = conn.insert_calls[0]
        assert "embedding_model" in sql
        assert "voyage-code-3" in args
        probe.assert_not_called()  # write path never consults the probe

    async def test_undefined_column_falls_back_to_legacy_insert(self):
        """A real UndefinedColumnError on the labeled INSERT triggers one
        legacy (unlabeled) retry and the store still succeeds."""
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = _StoreConn(fail_labeled=True)
        svc = MemoryServicePG(session_id="s1")
        p1, p2 = _patches(conn)
        with p1, p2:
            result = await svc.store(
                "fact", embedding=[0.1] * 1024,
                embedding_model="voyage-code-3",
            )

        assert result  # non-empty id -> success
        # Exactly one successful INSERT recorded, and it is the legacy shape.
        assert len(conn.insert_calls) == 1
        sql, args = conn.insert_calls[0]
        assert "embedding_model" not in sql
        assert "voyage-code-3" not in args

    async def test_self_heals_after_midprocess_migration(self):
        """No stale-cache dependence: a freshly-migrated column is used on the
        very next store even though a prior read-probe cached False."""
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = _StoreConn(fail_labeled=False)  # column now exists
        svc = MemoryServicePG(session_id="s1")
        p1, p2 = _patches(conn)
        with p1, p2, patch(
            "scripts.core.db.memory_service_pg."
            "embedding_model_column_available",
            AsyncMock(return_value=False),  # stale read cache
        ):
            await svc.store(
                "fact", embedding=[0.1] * 1024,
                embedding_model="voyage-code-3",
            )

        sql, args = conn.insert_calls[0]
        assert "embedding_model" in sql
        assert "voyage-code-3" in args

    async def test_no_label_keeps_legacy_insert(self):
        """When no label is supplied, the legacy INSERT shape is used and the
        labeled path is never attempted."""
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = _StoreConn(fail_labeled=False)
        svc = MemoryServicePG(session_id="s1")
        p1, p2 = _patches(conn)
        with p1, p2:
            await svc.store("fact", embedding=[0.1] * 1024)

        sql, _ = conn.insert_calls[0]
        assert "embedding_model" not in sql


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
