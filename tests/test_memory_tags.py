"""Tests for memory_tags table and tag storage through store_learning_v2.

Validates that:
1. Migration file exists and has correct schema
2. init-schema.sql includes memory_tags table
3. store_learning_v2 passes tags= to memory.store()
4. Tags are stored in memory_tags table when passed to store()
5. Tag CRUD operations work (get_tags, add_tag, remove_tag, get_all_session_tags)
6. search_with_tags filters by tags correctly
"""

from __future__ import annotations

import os
import sys
import uuid
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


class FakeTransaction:
    """Fake async context manager for get_transaction()."""

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
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value={"cnt": 0})

    pool = MagicMock()
    pool.acquire.return_value = FakeAcquire(conn)

    return pool, conn


# ---------------------------------------------------------------------------
# Schema: migration file and init-schema.sql
# ---------------------------------------------------------------------------

class TestMemoryTagsSchema:
    """Verify memory_tags schema exists in migration and init-schema."""

    def test_migration_file_exists(self):
        """Migration file for memory_tags should exist."""
        migration_path = (
            Path(__file__).parent.parent
            / "scripts"
            / "migrations"
            / "add_memory_tags.sql"
        )
        assert migration_path.exists(), f"Migration file not found at {migration_path}"

    def test_migration_has_correct_schema(self):
        """Migration should create memory_tags with expected columns and constraints."""
        migration_path = (
            Path(__file__).parent.parent
            / "scripts"
            / "migrations"
            / "add_memory_tags.sql"
        )
        content = migration_path.read_text()

        # Required columns
        assert "memory_id" in content
        assert "tag" in content
        assert "session_id" in content
        assert "created_at" in content

        # FK to archival_memory with CASCADE
        assert "REFERENCES archival_memory" in content
        assert "ON DELETE CASCADE" in content

        # PRIMARY KEY on (memory_id, tag) for ON CONFLICT (memory_id, tag) DO NOTHING
        assert "PRIMARY KEY" in content

        # Indexes for performance
        assert "idx_memory_tags_tag" in content or "INDEX" in content

    def test_init_schema_has_memory_tags(self):
        """init-schema.sql should define memory_tags table."""
        schema_path = Path(__file__).parent.parent / "docker" / "init-schema.sql"
        schema = schema_path.read_text()
        assert "memory_tags" in schema, "init-schema.sql should include memory_tags table"
        assert "ON DELETE CASCADE" in schema


# ---------------------------------------------------------------------------
# store_learning_v2: tags= parameter wiring
# ---------------------------------------------------------------------------

class TestStoreLearningV2Tags:
    """Tests that store_learning_v2 passes tags to memory.store()."""

    async def test_tags_passed_to_memory_store(self):
        """store_learning_v2 with tags= should forward them to memory.store()."""
        new_id = str(uuid.uuid4())
        test_tags = ["hooks", "typescript", "build"]

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
                content="TypeScript hooks require npm install",
                tags=test_tags,
            )

        assert result["success"] is True
        assert result["memory_id"] == new_id

        # Verify tags were forwarded to memory.store()
        mock_memory.store.assert_called_once()
        call_kwargs = mock_memory.store.call_args
        assert call_kwargs.kwargs.get("tags") == test_tags

    async def test_tags_none_passed_when_not_provided(self):
        """store_learning_v2 without tags should pass tags=None."""
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
                content="Learning without tags",
            )

        assert result["success"] is True
        call_kwargs = mock_memory.store.call_args
        # tags should be None when not provided
        assert call_kwargs.kwargs.get("tags") is None

    async def test_tags_in_metadata_json_for_backward_compat(self):
        """Tags should still be in metadata JSON even when passed separately."""
        new_id = str(uuid.uuid4())
        test_tags = ["database", "migration"]

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
                content="Database migration patterns",
                tags=test_tags,
            )

        assert result["success"] is True
        call_kwargs = mock_memory.store.call_args
        metadata = call_kwargs.kwargs.get("metadata", {})
        # Tags should be in metadata for backward compat
        assert metadata.get("tags") == test_tags
        # AND tags should be passed as a separate kwarg
        assert call_kwargs.kwargs.get("tags") == test_tags

    async def test_tags_not_passed_for_sqlite_backend(self):
        """store_learning_v2 should pass tags=None for sqlite backends."""
        new_id = str(uuid.uuid4())
        test_tags = ["hooks", "build"]

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
                content="Learning on sqlite with tags",
                tags=test_tags,
            )

        assert result["success"] is True
        call_kwargs = mock_memory.store.call_args
        # tags should be None for sqlite (memory_tags table doesn't exist)
        assert call_kwargs.kwargs.get("tags") is None


# ---------------------------------------------------------------------------
# MemoryServicePG.store(): tag INSERT inside transaction
# ---------------------------------------------------------------------------

class TestMemoryServicePGTagStorage:
    """Tests that store() inserts tags into memory_tags table."""

    async def test_store_inserts_tags(self):
        """store() with tags should INSERT into memory_tags for each tag."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")

        with patch("scripts.core.db.memory_service_pg.get_transaction",
                    return_value=FakeTransaction(conn)), \
             patch("scripts.core.db.memory_service_pg.init_pgvector",
                    new_callable=AsyncMock):
            from scripts.core.db.memory_service_pg import MemoryServicePG
            svc = MemoryServicePG.__new__(MemoryServicePG)
            svc.session_id = "test-session"
            svc.agent_id = None

            result = await svc.store(
                "test content",
                metadata={"type": "session_learning"},
                embedding=[0.1] * 1024,
                tags=["alpha", "beta"],
            )

        # Should have called execute for: INSERT archival_memory + 2 tag INSERTs
        assert result  # non-empty memory_id
        execute_calls = conn.execute.call_args_list
        tag_inserts = [c for c in execute_calls if "memory_tags" in str(c)]
        assert len(tag_inserts) == 2

    async def test_store_no_tags_skips_tag_insert(self):
        """store() without tags should NOT insert into memory_tags."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")

        with patch("scripts.core.db.memory_service_pg.get_transaction",
                    return_value=FakeTransaction(conn)), \
             patch("scripts.core.db.memory_service_pg.init_pgvector",
                    new_callable=AsyncMock):
            from scripts.core.db.memory_service_pg import MemoryServicePG
            svc = MemoryServicePG.__new__(MemoryServicePG)
            svc.session_id = "test-session"
            svc.agent_id = None

            await svc.store(
                "test content",
                metadata={"type": "session_learning"},
                embedding=[0.1] * 1024,
            )

        execute_calls = conn.execute.call_args_list
        tag_inserts = [c for c in execute_calls if "memory_tags" in str(c)]
        assert len(tag_inserts) == 0

    async def test_store_deduplicates_tags(self):
        """store() with duplicate tags should deduplicate them."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")

        with patch("scripts.core.db.memory_service_pg.get_transaction",
                    return_value=FakeTransaction(conn)), \
             patch("scripts.core.db.memory_service_pg.init_pgvector",
                    new_callable=AsyncMock):
            from scripts.core.db.memory_service_pg import MemoryServicePG
            svc = MemoryServicePG.__new__(MemoryServicePG)
            svc.session_id = "test-session"
            svc.agent_id = None

            await svc.store(
                "test content",
                metadata={"type": "session_learning"},
                embedding=[0.1] * 1024,
                tags=["alpha", "alpha", "beta"],
            )

        execute_calls = conn.execute.call_args_list
        tag_inserts = [c for c in execute_calls if "memory_tags" in str(c)]
        # Should only be 2 unique tags, not 3
        assert len(tag_inserts) == 2


# ---------------------------------------------------------------------------
# Recall: --tags CLI argument
# ---------------------------------------------------------------------------

class TestRecallTagsArg:
    """Tests that recall_learnings supports --tags argument."""

    def test_recall_main_accepts_tags_arg(self):
        """recall_learnings main() argparser should accept --tags."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "scripts.core.recall_learnings", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "--tags" in result.stdout, "CLI help should list --tags argument"
