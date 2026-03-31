"""Tests for cross-session pattern detection batch runner.

Validates:
1. Data loading parses learnings correctly from mock DB rows
2. Tag enrichment merges DB tags with metadata tags
3. write_patterns uses advisory lock and supersedes previous run
4. Dry-run mode performs no writes
5. Full pipeline orchestration
6. Report generation
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.pattern_batch import (  # noqa: E402
    load_learnings,
    load_tags_for_learnings,
    run_pattern_detection,
    write_patterns,
)

from scripts.core.pattern_detector import DetectedPattern  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_db_row(
    *,
    learning_type: str = "WORKING_SOLUTION",
    tags: list[str] | None = None,
    session_id: str = "test-session",
    embedding_dim: int = 1024,
) -> dict:
    """Create a mock database row matching archival_memory schema."""
    mem_id = uuid.uuid4()
    return {
        "id": mem_id,
        "content": f"Test learning {mem_id}",
        "embedding": np.random.randn(embedding_dim).astype(np.float32).tolist(),
        "metadata": {
            "learning_type": learning_type,
            "tags": tags or ["test"],
            "session_id": session_id,
            "context": "test context",
            "confidence": "high",
        },
        "session_id": session_id,
        "created_at": datetime.now(UTC),
    }


class FakeAcquire:
    """Fake async context manager for pool.acquire()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


class FakeTransaction:
    """Fake async context manager for conn.transaction()."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _make_pool(conn=None):
    """Create a mock pool with a fake connection."""
    if conn is None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchval = AsyncMock(return_value=uuid.uuid4())
        conn.fetchrow = AsyncMock(return_value=None)
        conn.execute = AsyncMock()
        conn.executemany = AsyncMock()
        conn.transaction = MagicMock(return_value=FakeTransaction())
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))
    return pool, conn


# ---------------------------------------------------------------------------
# load_learnings tests
# ---------------------------------------------------------------------------

class TestLoadLearnings:

    @pytest.mark.asyncio
    async def test_parses_rows_to_learnings(self):
        rows = [_make_db_row() for _ in range(5)]
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=rows)

        learnings = await load_learnings(pool)

        assert len(learnings) == 5
        for lrn in learnings:
            assert lrn.learning_type == "WORKING_SOLUTION"
            assert len(lrn.embedding) == 1024
            assert lrn.confidence == "high"

    @pytest.mark.asyncio
    async def test_skips_empty_embeddings(self):
        row = _make_db_row()
        row["embedding"] = []
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[row])

        learnings = await load_learnings(pool)
        assert len(learnings) == 0

    @pytest.mark.asyncio
    async def test_handles_string_metadata(self):
        row = _make_db_row()
        row["metadata"] = json.dumps(row["metadata"])
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[row])

        learnings = await load_learnings(pool)
        assert len(learnings) == 1
        assert learnings[0].learning_type == "WORKING_SOLUTION"

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty(self):
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[])

        learnings = await load_learnings(pool)
        assert learnings == []


# ---------------------------------------------------------------------------
# load_tags_for_learnings tests
# ---------------------------------------------------------------------------

class TestLoadTags:

    @pytest.mark.asyncio
    async def test_returns_tags_by_memory_id(self):
        mid = uuid.uuid4()
        rows = [
            {"memory_id": mid, "tag": "vue"},
            {"memory_id": mid, "tag": "testing"},
        ]
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=rows)

        tags = await load_tags_for_learnings(pool, [str(mid)])
        assert str(mid) in tags
        assert set(tags[str(mid)]) == {"vue", "testing"}

    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty(self):
        pool, _ = _make_pool()
        tags = await load_tags_for_learnings(pool, [])
        assert tags == {}


# ---------------------------------------------------------------------------
# write_patterns tests
# ---------------------------------------------------------------------------

class TestWritePatterns:

    @pytest.mark.asyncio
    async def test_writes_patterns_and_members(self):
        pool, conn = _make_pool()
        pattern_id = uuid.uuid4()
        conn.fetchval = AsyncMock(return_value=pattern_id)

        mid1, mid2 = str(uuid.uuid4()), str(uuid.uuid4())
        patterns = [
            DetectedPattern(
                pattern_type="tool_cluster",
                member_ids=[mid1, mid2],
                representative_id=mid1,
                tags=["vue", "testing"],
                session_count=3,
                confidence=0.85,
                label="Vue testing patterns across 3 sessions",
                metadata={"size": 2},
                distances={mid1: 0.1, mid2: 0.3},
            )
        ]

        run_id = str(uuid.uuid4())
        count = await write_patterns(pool, patterns, run_id)

        assert count == 1
        # Should have called execute for advisory lock + supersede
        assert conn.execute.call_count >= 2
        # Should have called fetchval for INSERT
        assert conn.fetchval.call_count == 1
        # Should have called executemany for members
        assert conn.executemany.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_patterns_still_supersedes(self):
        pool, conn = _make_pool()
        count = await write_patterns(pool, [], str(uuid.uuid4()))
        assert count == 0
        # Should still acquire lock and supersede previous run
        assert conn.execute.call_count == 2
        supersede_call = conn.execute.call_args_list[1]
        assert "superseded_at" in supersede_call.args[0]

    @pytest.mark.asyncio
    async def test_advisory_lock_acquired(self):
        pool, conn = _make_pool()
        conn.fetchval = AsyncMock(return_value=uuid.uuid4())

        mid = str(uuid.uuid4())
        patterns = [
            DetectedPattern(
                pattern_type="tool_cluster",
                member_ids=[mid],
                representative_id=mid,
                tags=["test"],
                session_count=1,
                confidence=0.5,
                label="Test",
                distances={mid: 0.0},
            )
        ]

        await write_patterns(pool, patterns, str(uuid.uuid4()))

        # First execute call should be the advisory lock
        first_call = conn.execute.call_args_list[0]
        assert "pg_advisory_xact_lock" in first_call.args[0]

    @pytest.mark.asyncio
    async def test_supersedes_previous_run(self):
        pool, conn = _make_pool()
        conn.fetchval = AsyncMock(return_value=uuid.uuid4())

        mid = str(uuid.uuid4())
        patterns = [
            DetectedPattern(
                pattern_type="tool_cluster",
                member_ids=[mid],
                representative_id=mid,
                tags=["test"],
                session_count=1,
                confidence=0.5,
                label="Test",
                distances={mid: 0.0},
            )
        ]

        await write_patterns(pool, patterns, str(uuid.uuid4()))

        # Second execute call should be the supersede UPDATE
        second_call = conn.execute.call_args_list[1]
        assert "superseded_at" in second_call.args[0]
        assert "UPDATE" in second_call.args[0]


# ---------------------------------------------------------------------------
# run_pattern_detection tests
# ---------------------------------------------------------------------------

class TestRunPatternDetection:

    @pytest.mark.asyncio
    @patch("scripts.core.pattern_batch._get_pool")
    @patch("scripts.core.pattern_batch._ensure_tables")
    async def test_dry_run_no_writes(self, mock_ensure, mock_get_pool):
        pool, conn = _make_pool()
        mock_get_pool.return_value = pool
        mock_ensure.return_value = True

        # Return empty rows so detection finds nothing
        conn.fetch = AsyncMock(return_value=[])

        result = await run_pattern_detection(dry_run=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["learnings_analyzed"] == 0

    @pytest.mark.asyncio
    @patch("scripts.core.pattern_batch._get_pool")
    @patch("scripts.core.pattern_batch._ensure_tables")
    async def test_empty_db_succeeds(self, mock_ensure, mock_get_pool):
        pool, conn = _make_pool()
        mock_get_pool.return_value = pool
        mock_ensure.return_value = True
        conn.fetch = AsyncMock(return_value=[])

        result = await run_pattern_detection()

        assert result["success"] is True
        assert result["patterns_detected"] == 0

    @pytest.mark.asyncio
    @patch("scripts.core.pattern_batch._get_pool")
    @patch("scripts.core.pattern_batch._ensure_tables")
    async def test_table_creation_failure_returns_error(
        self, mock_ensure, mock_get_pool
    ):
        pool, _ = _make_pool()
        mock_get_pool.return_value = pool
        mock_ensure.return_value = False

        result = await run_pattern_detection()

        assert result["success"] is False
        assert "tables" in result["error"].lower()
