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

class FakeAcquire:
    """Fake async context manager for pool.acquire()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_pool():
    """Mock PostgreSQL connection pool."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="UPDATE 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value={"cnt": 0})

    pool = MagicMock()
    pool.acquire.return_value = FakeAcquire(conn)

    return pool, conn


# ---------------------------------------------------------------------------
# Recall: superseded_by IS NULL in queries
# ---------------------------------------------------------------------------

class TestRecallChainFilter:
    """Tests that recall queries include superseded_by IS NULL."""

    async def test_text_only_query_includes_chain_filter(self):
        """search_learnings_text_only_postgres SQL should reference superseded_by."""
        import inspect

        from scripts.core.recall_learnings import search_learnings_text_only_postgres

        source = inspect.getsource(search_learnings_text_only_postgres)
        assert "superseded_by IS NULL" in source

    async def test_hybrid_rrf_query_includes_chain_filter(self):
        """search_learnings_hybrid_rrf SQL should reference superseded_by."""
        import inspect

        from scripts.core.recall_learnings import search_learnings_hybrid_rrf

        source = inspect.getsource(search_learnings_hybrid_rrf)
        assert "superseded_by IS NULL" in source

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

        pool = MagicMock()
        pool.acquire.return_value = FakeAcquire(conn)

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()

        pgvector_patch = "scripts.core.db.postgres_pool.init_pgvector"
        embed_patch = "scripts.core.db.embedding_service.EmbeddingService"
        with patch("scripts.core.db.postgres_pool.get_pool", return_value=pool), \
             patch(pgvector_patch, new_callable=AsyncMock), \
             patch(embed_patch, return_value=mock_embedder):
            from scripts.core.recall_learnings import search_learnings_hybrid_rrf
            results = await search_learnings_hybrid_rrf("test query", k=5, expand=False)

        assert len(results) == 1
        # Should have tried: boosted+chain, plain+chain, plain (no chain)
        assert call_count == 3


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

        _es = "scripts.core.db.embedding_service.EmbeddingService"
        _cm = "scripts.core.db.memory_factory.create_memory_service"
        _gb = "scripts.core.db.memory_factory.get_default_backend"
        with patch(_es, return_value=mock_embedder), \
             patch(_cm, return_value=mock_memory), \
             patch(_gb, return_value="postgres"), \
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

        _es = "scripts.core.db.embedding_service.EmbeddingService"
        _cm = "scripts.core.db.memory_factory.create_memory_service"
        _gb = "scripts.core.db.memory_factory.get_default_backend"
        with patch(_es, return_value=mock_embedder), \
             patch(_cm, return_value=mock_memory), \
             patch(_gb, return_value="postgres"), \
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

        _es = "scripts.core.db.embedding_service.EmbeddingService"
        _cm = "scripts.core.db.memory_factory.create_memory_service"
        _gb = "scripts.core.db.memory_factory.get_default_backend"
        env_clean = {
            k: v for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "CONTINUOUS_CLAUDE_DB_URL")
        }
        with patch(_es, return_value=mock_embedder), \
             patch(_cm, return_value=mock_memory), \
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
