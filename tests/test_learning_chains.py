"""Tests for learning chains (superseded_by) in recall and store.

Validates that:
1. Recall queries filter out superseded learnings (WHERE superseded_by IS NULL)
2. store_learning_v2 with supersedes marks old learning as superseded
3. Graceful fallback when superseded_by column doesn't exist
4. RRF hybrid query filters superseded learnings
5. Text-only query filters superseded learnings
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_recall_probe_caches():
    """Reset all module-level recall probe caches before each test (issue #153
    round-2 test-isolation fix).

    project / embedding_model / hnsw.iterative_scan are process-global caches.
    Left warm by an earlier test they silently skip their capability probes,
    so fetch-counting cascade tests pass or fail by suite ORDER. Resetting to a
    cold, known state before every test makes counts deterministic in isolation
    (FIRST: independence).
    """
    from scripts.core import recall_backends as rb

    rb.reset_project_column_cache()
    rb.reset_embedding_model_column_cache()
    rb.reset_hnsw_iterative_scan_cache()
    yield
    rb.reset_project_column_cache()
    rb.reset_embedding_model_column_cache()
    rb.reset_hnsw_iterative_scan_cache()


class FakeAcquire:
    """Fake async context manager for pool.acquire()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


class _NoopTx:
    """No-op async context manager for conn.transaction() (issue #153).

    Round-3 production sets a SESSION-level ``SET hnsw.iterative_scan`` once per
    connection on acquire and runs the RRF cascade as bare ``conn.fetch`` (no
    per-attempt transaction). This no-op CM is retained only so mock conns whose
    ``transaction`` attribute may be exercised by other paths return a real
    async CM; the RRF cascade itself opens no transaction.
    """

    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


def _attach_tx(conn) -> None:
    """Give a mock conn a working transaction() CM (issue #153 finding 1)."""
    conn.transaction = MagicMock(return_value=_NoopTx())


@pytest.fixture
def mock_pool():
    """Mock PostgreSQL connection pool."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="UPDATE 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value={"cnt": 0})
    _attach_tx(conn)

    pool = MagicMock()
    pool.acquire.return_value = FakeAcquire(conn)

    return pool, conn


# ---------------------------------------------------------------------------
# Recall: superseded_by IS NULL in queries
# ---------------------------------------------------------------------------

class TestRecallChainFilter:
    """Tests that recall queries include superseded_by IS NULL."""

    async def test_text_only_query_includes_chain_filter(self):
        """The primary text-only SQL constants should reference superseded_by."""
        from scripts.core.recall_backends import (
            _TEXT_ONLY_FTS_SQL,
            _TEXT_ONLY_ILIKE_SQL,
        )

        assert "superseded_by IS NULL" in _TEXT_ONLY_FTS_SQL
        assert "superseded_by IS NULL" in _TEXT_ONLY_ILIKE_SQL

    async def test_hybrid_rrf_query_includes_chain_filter(self):
        """build_rrf_cte (used by search_learnings_hybrid_rrf) should include superseded_by."""
        from scripts.core.recall_backends import build_rrf_cte

        cte_sql = build_rrf_cte(chain_filter=True, use_tsquery=False)
        assert "superseded_by IS NULL" in cte_sql

    async def test_postgres_vector_query_includes_chain_filter(self):
        """search_learnings_postgres SQL should reference superseded_by."""
        import inspect

        from scripts.core.recall_learnings import search_learnings_postgres

        source = inspect.getsource(search_learnings_postgres)
        assert "superseded_by IS NULL" in source

    async def test_text_only_fallback_without_chain_column(self, mock_pool):
        """Text-only search falls back if superseded_by column missing."""
        pool, conn = mock_pool
        now = datetime.now(UTC)

        call_count = 0

        async def fake_fetch(sql, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and "superseded_by" in sql:
                raise Exception('column "superseded_by" does not exist')
            return [{
                "id": uuid.uuid4(),
                "session_id": "test",
                "content": "test learning",
                "metadata": '{"type": "session_learning"}',
                "created_at": now,
                "similarity": 0.5,
            }]

        conn.fetch = fake_fetch

        # Pin the project cache so call 1 is the real chain query, not a cold
        # project capability probe (issue #153 round-2 test-isolation).
        from scripts.core import recall_backends as rb
        rb._set_project_column_cache_for_tests(False)

        with patch("scripts.core.recall_learnings.get_backend", return_value="postgres"), \
             patch("scripts.core.db.postgres_pool.get_pool", return_value=pool):
            from scripts.core.recall_learnings import search_learnings_text_only_postgres
            results = await search_learnings_text_only_postgres("test query", k=5)

        assert len(results) == 1
        assert call_count == 2  # First try with filter, fallback without

    async def test_rrf_fallback_without_chain_column(self):
        """Hybrid RRF falls back through cascade if chain column missing."""
        now = datetime.now(UTC)

        call_count = 0

        async def fake_fetch(sql, *args):
            nonlocal call_count
            call_count += 1
            if "superseded_by" in sql:
                raise Exception('column "superseded_by" does not exist')
            if "recall_count" in sql:
                raise Exception('column "recall_count" does not exist')
            # Final fallback: plain query without chain or decay
            return [{
                "id": uuid.uuid4(),
                "session_id": "test",
                "content": "test learning",
                "metadata": '{"type": "session_learning"}',
                "created_at": now,
                "rrf_score": 0.023,
                "fts_rank": 1,
                "vec_rank": 2,
            }]

        conn = AsyncMock()
        conn.fetch = fake_fetch
        _attach_tx(conn)

        pool = MagicMock()
        pool.acquire.return_value = FakeAcquire(conn)

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()

        # Pin the capability caches so the counted conn.fetch calls are EXACTLY
        # the cascade attempts (issue #153 round-2 test-isolation). Without
        # pinning, cold project/embedding_model probes add probe fetches and the
        # count becomes suite-order dependent.
        from scripts.core import recall_backends as rb
        rb._set_project_column_cache_for_tests(False)  # no project probe fetch
        rb._set_embedding_model_column_cache_for_tests(False)  # no probe fetch
        rb._set_hnsw_iterative_scan_cache_for_tests(True)  # no probe; SET via execute

        pgvector_patch = "scripts.core.db.postgres_pool.init_pgvector"
        embed_patch = "scripts.core.db.embedding_service.EmbeddingService"
        with patch("scripts.core.db.postgres_pool.get_pool", return_value=pool), \
             patch(pgvector_patch, new_callable=AsyncMock), \
             patch(embed_patch, return_value=mock_embedder):
            from scripts.core.recall_learnings import search_learnings_hybrid_rrf
            results = await search_learnings_hybrid_rrf("test query", k=5, expand=False)

        assert len(results) == 1
        # Issue #63 Phase 2b round-2 finding 2: the superseded-only (no-archive)
        # middle tier runs ONLY on a provable missing-archived_at asyncpg error.
        # This test raises a generic Exception about superseded_by (not an
        # asyncpg UndefinedColumnError for archived_at), so the cascade SKIPS the
        # archived-dropping middle tier and degrades straight to no-chain:
        # boosted+chain(+archived), plain+chain(+archived), then plain (no
        # chain) = 3 fetches. Probe fetches are excluded (caches pinned above).
        assert call_count == 3

    async def test_rrf_plainto_fallback_keeps_archived_predicate(self):
        """Issue #63 Phase 2b round-2 finding 1: when the expanded to_tsquery
        path returns NO rows and the cascade falls back to plainto_tsquery, the
        plainto fallback CTE must still carry ``archived_at IS NULL``. On a
        migrated DB this normal no-results fallback must not silently recall
        archived rows as active.
        """
        now = datetime.now(UTC)
        plainto_chain_sqls: list[str] = []

        async def fake_fetch(sql, *args):
            # Expanded path uses to_tsquery and returns NO rows so the cascade
            # falls back to plainto_tsquery (the normal no-results fallback).
            if "to_tsquery" in sql and "plainto_tsquery" not in sql:
                return []
            if "plainto_tsquery" in sql and "superseded_by" in sql:
                plainto_chain_sqls.append(sql)
            return [{
                "id": uuid.uuid4(),
                "session_id": "test",
                "content": "test learning",
                "metadata": '{"type": "session_learning"}',
                "created_at": now,
                "rrf_score": 0.023,
                "boosted_score": 0.023,
                "raw_rrf_score": 0.023,
                "recall_count": 0,
                "last_recalled": None,
                "fts_rank": 1,
                "vec_rank": 2,
            }]

        conn = AsyncMock()
        conn.fetch = fake_fetch
        _attach_tx(conn)

        pool = MagicMock()
        pool.acquire.return_value = FakeAcquire(conn)

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()

        from scripts.core import recall_backends as rb
        rb._set_project_column_cache_for_tests(False)
        rb._set_embedding_model_column_cache_for_tests(False)
        rb._set_hnsw_iterative_scan_cache_for_tests(True)

        pgvector_patch = "scripts.core.db.postgres_pool.init_pgvector"
        embed_patch = "scripts.core.db.embedding_service.EmbeddingService"
        with patch("scripts.core.db.postgres_pool.get_pool", return_value=pool), \
             patch(pgvector_patch, new_callable=AsyncMock), \
             patch(embed_patch, return_value=mock_embedder):
            from scripts.core.recall_learnings import search_learnings_hybrid_rrf
            # expand=True so the cascade uses to_tsquery first, then the plainto
            # no-results fallback runs.
            results = await search_learnings_hybrid_rrf(
                "test query", k=5, expand=True,
            )

        assert len(results) == 1
        # The plainto fallback's chain CTE must include archived_at IS NULL.
        assert plainto_chain_sqls, "plainto chain fallback never ran"
        assert any(
            "archived_at IS NULL" in sql for sql in plainto_chain_sqls
        ), "plainto fallback dropped the archived_at predicate"

    async def test_rrf_non_schema_error_does_not_drop_archived(self):
        """Issue #63 Phase 2b round-2 finding 2: a NON-schema failure (e.g. a
        lock timeout) on the full lifecycle query must NOT silently degrade to a
        weaker recall that omits ``archived_at IS NULL``. Only a concrete missing
        archived_at column may drop the archived predicate.

        Here every fetch raises a generic (non-UndefinedColumn) error; the
        cascade must propagate rather than return archived-inclusive rows.
        """
        class _LockTimeoutError(Exception):
            pass

        archived_dropped = False

        async def fake_fetch(sql, *args):
            nonlocal archived_dropped
            if "superseded_by" in sql and "archived_at IS NULL" not in sql:
                # A weaker (archived-dropped but superseded-kept) tier ran: this
                # is exactly the silent degradation the fix must prevent for a
                # non-schema error.
                archived_dropped = True
            raise _LockTimeoutError("lock timeout")

        conn = AsyncMock()
        conn.fetch = fake_fetch
        _attach_tx(conn)

        pool = MagicMock()
        pool.acquire.return_value = FakeAcquire(conn)

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()

        from scripts.core import recall_backends as rb
        rb._set_project_column_cache_for_tests(False)
        rb._set_embedding_model_column_cache_for_tests(False)
        rb._set_hnsw_iterative_scan_cache_for_tests(True)

        pgvector_patch = "scripts.core.db.postgres_pool.init_pgvector"
        embed_patch = "scripts.core.db.embedding_service.EmbeddingService"
        with patch("scripts.core.db.postgres_pool.get_pool", return_value=pool), \
             patch(pgvector_patch, new_callable=AsyncMock), \
             patch(embed_patch, return_value=mock_embedder):
            from scripts.core.recall_learnings import search_learnings_hybrid_rrf
            with pytest.raises(_LockTimeoutError):
                await search_learnings_hybrid_rrf(
                    "test query", k=5, expand=False,
                )

        assert not archived_dropped, (
            "non-schema error degraded to an archived-dropped recall tier"
        )

    async def test_rrf_missing_archived_column_degrades_to_superseded_only(self):
        """Issue #63 Phase 2b round-2 finding 2: an UndefinedColumnError naming
        archived_at (the migration gap this cascade exists for) MUST still
        degrade to the superseded-only tier so a partially-migrated DB keeps the
        superseded filter rather than crashing.
        """
        now = datetime.now(UTC)
        try:
            from asyncpg.exceptions import UndefinedColumnError
        except ImportError:  # pragma: no cover
            pytest.skip("asyncpg not installed")

        superseded_only_ran = False

        async def fake_fetch(sql, *args):
            nonlocal superseded_only_ran
            if "archived_at IS NULL" in sql:
                raise UndefinedColumnError(
                    'column "archived_at" does not exist'
                )
            if "superseded_by" in sql:
                superseded_only_ran = True
            return [{
                "id": uuid.uuid4(),
                "session_id": "test",
                "content": "test learning",
                "metadata": '{"type": "session_learning"}',
                "created_at": now,
                "rrf_score": 0.023,
                "fts_rank": 1,
                "vec_rank": 2,
            }]

        conn = AsyncMock()
        conn.fetch = fake_fetch
        _attach_tx(conn)

        pool = MagicMock()
        pool.acquire.return_value = FakeAcquire(conn)

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()

        from scripts.core import recall_backends as rb
        rb._set_project_column_cache_for_tests(False)
        rb._set_embedding_model_column_cache_for_tests(False)
        rb._set_hnsw_iterative_scan_cache_for_tests(True)

        pgvector_patch = "scripts.core.db.postgres_pool.init_pgvector"
        embed_patch = "scripts.core.db.embedding_service.EmbeddingService"
        with patch("scripts.core.db.postgres_pool.get_pool", return_value=pool), \
             patch(pgvector_patch, new_callable=AsyncMock), \
             patch(embed_patch, return_value=mock_embedder):
            from scripts.core.recall_learnings import search_learnings_hybrid_rrf
            results = await search_learnings_hybrid_rrf(
                "test query", k=5, expand=False,
            )

        assert len(results) == 1
        assert superseded_only_ran, (
            "missing archived_at did not degrade to the superseded-only tier"
        )


# ---------------------------------------------------------------------------
# Store: supersedes parameter
# ---------------------------------------------------------------------------

class TestStoreSupersedes:
    """Tests for the supersedes parameter in store_learning_v2.

    The supersede UPDATE now runs inside memory.store() in the same
    transaction as the INSERT (atomic chaining).  These tests verify
    that store_learning_v2 passes ``supersedes`` through correctly.
    """

    async def test_supersedes_passed_to_memory_store(self):
        """store_learning_v2 with supersedes passes it to memory.store()."""
        old_id = str(uuid.uuid4())
        new_id = str(uuid.uuid4())

        mock_memory = AsyncMock()
        mock_memory.store = AsyncMock(return_value=new_id)
        mock_memory.search_vector_global = AsyncMock(return_value=[])
        mock_memory.close = AsyncMock()

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder._provider = MagicMock()
        mock_embedder._provider.model = "bge-large"

        _es = "scripts.core.store_learning.EmbeddingService"
        _cm = "scripts.core.store_learning.create_memory_service"
        with patch(_es, return_value=mock_embedder), \
             patch(_cm, new_callable=AsyncMock, return_value=mock_memory), \
             patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"}):
            from scripts.core.store_learning import store_learning_v2
            result = await store_learning_v2(
                session_id="test-session",
                content="Updated learning content",
                supersedes=old_id,
            )

        assert result["success"] is True
        assert result["memory_id"] == new_id
        assert result["superseded"] == old_id

        # Verify supersedes was forwarded to memory.store()
        mock_memory.store.assert_called_once()
        call_kwargs = mock_memory.store.call_args
        assert call_kwargs.kwargs.get("supersedes") == old_id

    async def test_supersedes_none_not_passed(self):
        """store_learning_v2 without supersedes passes None."""
        new_id = str(uuid.uuid4())

        mock_memory = AsyncMock()
        mock_memory.store = AsyncMock(return_value=new_id)
        mock_memory.search_vector_global = AsyncMock(return_value=[])
        mock_memory.close = AsyncMock()

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder._provider = MagicMock()
        mock_embedder._provider.model = "bge-large"

        _es = "scripts.core.store_learning.EmbeddingService"
        _cm = "scripts.core.store_learning.create_memory_service"
        with patch(_es, return_value=mock_embedder), \
             patch(_cm, new_callable=AsyncMock, return_value=mock_memory), \
             patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"}):
            from scripts.core.store_learning import store_learning_v2
            result = await store_learning_v2(
                session_id="test-session",
                content="New learning without superseding",
            )

        assert result["success"] is True
        assert "superseded" not in result

        # supersedes=None should be passed through
        call_kwargs = mock_memory.store.call_args
        assert call_kwargs.kwargs.get("supersedes") is None

    async def test_supersedes_not_passed_for_sqlite(self):
        """store_learning_v2 passes supersedes=None for non-postgres backends."""
        old_id = str(uuid.uuid4())
        new_id = str(uuid.uuid4())

        mock_memory = AsyncMock()
        mock_memory.store = AsyncMock(return_value=new_id)
        mock_memory.search_vector_global = AsyncMock(return_value=[])
        mock_memory.close = AsyncMock()

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder._provider = MagicMock()
        mock_embedder._provider.model = "bge-large"

        _es = "scripts.core.store_learning.EmbeddingService"
        _cm = "scripts.core.store_learning.create_memory_service"
        _gb = "scripts.core.store_learning.get_default_backend"
        env_clean = {
            k: v for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "CONTINUOUS_CLAUDE_DB_URL")
        }
        with patch(_es, return_value=mock_embedder), \
             patch(_cm, new_callable=AsyncMock, return_value=mock_memory), \
             patch(_gb, return_value="sqlite"), \
             patch.dict("os.environ", env_clean, clear=True):
            from scripts.core.store_learning import store_learning_v2
            result = await store_learning_v2(
                session_id="test-session",
                content="Learning on sqlite",
                supersedes=old_id,
            )

        assert result["success"] is True
        # supersedes should be None for sqlite (not passed through)
        call_kwargs = mock_memory.store.call_args
        assert call_kwargs.kwargs.get("supersedes") is None


# ---------------------------------------------------------------------------
# Schema: init-schema.sql has chain columns
# ---------------------------------------------------------------------------

class TestSchema:
    """Verify schema includes learning chain columns."""

    def test_init_schema_has_superseded_by(self):
        """init-schema.sql should define superseded_by column."""
        schema_path = Path(__file__).parent.parent / "docker" / "init-schema.sql"
        schema = schema_path.read_text()
        assert "superseded_by UUID" in schema
        assert "superseded_at TIMESTAMPTZ" in schema

    def test_migration_exists(self):
        """Migration script for learning chains should exist."""
        migration_path = (
            Path(__file__).parent.parent
            / "scripts"
            / "migrations"
            / "add_learning_chains.sql"
        )
        assert migration_path.exists()
        content = migration_path.read_text()
        assert "superseded_by" in content
        assert "superseded_at" in content
