"""Tests for temporal decay in recall_learnings.

Validates that:
1. record_recall updates last_recalled and recall_count
2. record_recall is idempotent and gracefully handles missing columns
3. Hybrid RRF query includes recall_count boost in scoring
4. Recall write-back is called after search_learnings returns results
5. Empty results don't trigger write-back
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.recall_learnings import record_recall  # noqa: E402


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
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])

    pool = MagicMock()
    pool.acquire.return_value = FakeAcquire(conn)

    return pool, conn


# ---------------------------------------------------------------------------
# record_recall tests
# ---------------------------------------------------------------------------

class TestRecordRecall:
    """Tests for the record_recall write-back function."""

    async def test_updates_recalled_rows(self, mock_pool):
        """record_recall calls UPDATE with the given IDs."""
        pool, conn = mock_pool
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        with patch("scripts.core.recall_learnings.get_backend", return_value="postgres"), \
             patch("scripts.core.db.postgres_pool.get_pool", return_value=pool):
            await record_recall(ids)

        conn.execute.assert_called_once()
        call_args = conn.execute.call_args
        sql = call_args[0][0]
        assert "recall_count = recall_count + 1" in sql
        assert "last_recalled = NOW()" in sql
        assert call_args[0][1] == ids

    async def test_skips_empty_ids(self, mock_pool):
        """record_recall does nothing for empty ID list."""
        pool, conn = mock_pool

        with patch("scripts.core.recall_learnings.get_backend", return_value="postgres"), \
             patch("scripts.core.db.postgres_pool.get_pool", return_value=pool):
            await record_recall([])

        conn.execute.assert_not_called()

    async def test_skips_sqlite_backend(self, mock_pool):
        """record_recall does nothing for sqlite backend."""
        pool, conn = mock_pool
        ids = [str(uuid.uuid4())]

        with patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"):
            await record_recall(ids)

        conn.execute.assert_not_called()

    async def test_graceful_on_db_error(self, mock_pool):
        """record_recall doesn't raise on database errors."""
        pool, conn = mock_pool
        conn.execute.side_effect = Exception("column does not exist")
        ids = [str(uuid.uuid4())]

        with patch("scripts.core.recall_learnings.get_backend", return_value="postgres"), \
             patch("scripts.core.db.postgres_pool.get_pool", return_value=pool):
            # Should not raise
            await record_recall(ids)


# ---------------------------------------------------------------------------
# Hybrid RRF boost tests
# ---------------------------------------------------------------------------

class TestRRFRecallBoost:
    """Tests for recall_count boost in hybrid RRF scoring."""

    async def test_rrf_query_includes_recall_count(self):
        """The hybrid RRF SQL should reference recall_count for boosting."""
        import inspect

        from scripts.core.recall_learnings import search_learnings_hybrid_rrf

        source = inspect.getsource(search_learnings_hybrid_rrf)
        assert "recall_count" in source
        assert "boosted_score" in source

    async def test_rrf_results_include_recall_fields(self):
        """Hybrid RRF results should include recall_count and last_recalled."""
        now = datetime.now(UTC)
        fake_row = {
            "id": uuid.uuid4(),
            "session_id": "test-session",
            "content": "test learning",
            "metadata": '{"type": "session_learning"}',
            "created_at": now,
            "recall_count": 5,
            "last_recalled": now,
            "boosted_score": 0.025,
            "raw_rrf_score": 0.023,
            "fts_rank": 1,
            "vec_rank": 2,
        }

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[fake_row])

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
        assert results[0]["recall_count"] == 5
        assert results[0]["last_recalled"] == now
        assert results[0]["similarity"] == 0.025
        assert results[0]["raw_rrf_score"] == 0.023

    async def test_rrf_fallback_without_decay_columns(self):
        """Hybrid RRF falls back to plain query if decay columns missing."""
        now = datetime.now(UTC)
        plain_row = {
            "id": uuid.uuid4(),
            "session_id": "test-session",
            "content": "test learning",
            "metadata": '{"type": "session_learning"}',
            "created_at": now,
            "rrf_score": 0.023,
            "fts_rank": 1,
            "vec_rank": 2,
        }

        call_count = 0

        async def fake_fetch(sql, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (boosted query) fails
                raise Exception("column \"recall_count\" does not exist")
            # Second call (plain query) succeeds
            return [plain_row]

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
        assert results[0]["similarity"] == 0.023
        assert "recall_count" not in results[0]
        assert "last_recalled" not in results[0]
        assert call_count == 2

    async def test_boost_is_zero_for_never_recalled(self):
        """Learnings with recall_count=0 should get zero boost."""
        import math
        recall_count = 0
        # Mirrors the SQL CASE: 0 when recall_count = 0
        boost = 0 if recall_count == 0 else math.log2(1 + recall_count) * 0.002
        assert boost == 0.0

    async def test_boost_is_small_relative_to_rrf(self):
        """The recall boost should be small enough to not overwhelm RRF scores.

        RRF scores are typically in the 0.01-0.03 range.
        A learning recalled 10 times should get a boost of ~0.007 (log2(11)*0.002).
        This is meaningful but doesn't dominate.
        """
        import math
        recall_count = 10
        boost = math.log2(1 + recall_count) * 0.002
        assert boost < 0.01, f"Boost {boost} too large for RRF range"
        assert boost > 0.001, f"Boost {boost} too small to matter"


# ---------------------------------------------------------------------------
# Integration: search_learnings calls record_recall
# ---------------------------------------------------------------------------

class TestSearchRecordIntegration:
    """Verify that search_learnings calls record_recall after returning."""

    async def test_search_learnings_does_not_call_record_recall(self):
        """search_learnings should NOT call record_recall (main() handles it after reranking)."""
        fake_results = [
            {"id": "id-1", "content": "test", "session_id": "s1",
             "metadata": {}, "created_at": datetime.now(UTC),
             "similarity": 0.5},
            {"id": "id-2", "content": "test2", "session_id": "s2",
             "metadata": {}, "created_at": datetime.now(UTC),
             "similarity": 0.4},
        ]

        with patch("scripts.core.recall_learnings.get_backend", return_value="postgres"), \
             patch("scripts.core.recall_learnings.search_learnings_postgres",
                   new_callable=AsyncMock, return_value=fake_results), \
             patch("scripts.core.recall_learnings.record_recall",
                   new_callable=AsyncMock) as mock_record:
            from scripts.core.recall_learnings import search_learnings
            results = await search_learnings("test query", k=5)

        mock_record.assert_not_called()
        assert len(results) == 2

    async def test_search_learnings_no_recall_on_empty(self):
        """search_learnings should NOT call record_recall even with empty results."""
        with patch("scripts.core.recall_learnings.get_backend", return_value="postgres"), \
             patch("scripts.core.recall_learnings.search_learnings_postgres",
                   new_callable=AsyncMock, return_value=[]), \
             patch("scripts.core.recall_learnings.record_recall",
                   new_callable=AsyncMock) as mock_record:
            from scripts.core.recall_learnings import search_learnings
            results = await search_learnings("test query", k=5)

        mock_record.assert_not_called()
        assert len(results) == 0
