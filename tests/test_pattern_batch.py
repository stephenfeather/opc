"""Tests for cross-session pattern detection batch runner.

Validates:
1. Data loading parses learnings correctly from mock DB rows
2. Tag enrichment merges DB tags with metadata tags (immutably)
3. write_patterns uses advisory lock and supersedes previous run
4. Dry-run mode performs no writes
5. Full pipeline orchestration
6. Report generation
7. Pure functions: row parsing, tag merging, counting, summaries, formatting
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
    build_run_summary,
    count_by_type,
    format_dry_run_output,
    format_report,
    load_learnings,
    load_tags_for_learnings,
    merge_tags,
    parse_learning_row,
    run_pattern_detection,
    write_patterns,
)
from scripts.core.pattern_detector import DetectedPattern, Learning  # noqa: E402

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


def _make_learning(
    *,
    learning_id: str | None = None,
    tags: list[str] | None = None,
    learning_type: str = "WORKING_SOLUTION",
) -> Learning:
    """Create a Learning instance for pure function tests."""
    return Learning(
        id=learning_id or str(uuid.uuid4()),
        content="Test learning",
        embedding=np.random.randn(1024).astype(np.float32),
        learning_type=learning_type,
        tags=tags or ["test"],
        session_id="test-session",
        context="test context",
        created_at=datetime.now(UTC),
        confidence="high",
    )


def _make_pattern(
    *,
    pattern_type: str = "tool_cluster",
    member_count: int = 2,
    confidence: float = 0.85,
    label: str = "Test pattern",
    tags: list[str] | None = None,
) -> DetectedPattern:
    """Create a DetectedPattern instance for pure function tests."""
    member_ids = [str(uuid.uuid4()) for _ in range(member_count)]
    return DetectedPattern(
        pattern_type=pattern_type,
        member_ids=member_ids,
        representative_id=member_ids[0],
        tags=tags or ["test"],
        session_count=3,
        confidence=confidence,
        label=label,
        metadata={"size": member_count},
        distances={mid: 0.1 * i for i, mid in enumerate(member_ids)},
    )


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
# parse_learning_row tests (pure function)
# ---------------------------------------------------------------------------

class TestParseLearningRow:

    def test_parses_valid_row(self):
        row = _make_db_row()
        result = parse_learning_row(row)

        assert result is not None
        assert result.learning_type == "WORKING_SOLUTION"
        assert len(result.embedding) == 1024
        assert result.confidence == "high"
        assert result.id == str(row["id"])

    def test_returns_none_for_empty_embedding(self):
        row = _make_db_row()
        row["embedding"] = []
        result = parse_learning_row(row)
        assert result is None

    def test_handles_string_metadata(self):
        row = _make_db_row()
        row["metadata"] = json.dumps(row["metadata"])
        result = parse_learning_row(row)
        assert result is not None
        assert result.learning_type == "WORKING_SOLUTION"

    def test_handles_none_metadata(self):
        row = _make_db_row()
        row["metadata"] = None
        result = parse_learning_row(row)
        assert result is not None
        assert result.learning_type == "UNKNOWN"

    def test_rejects_wrong_dimension_embedding(self):
        row = _make_db_row(embedding_dim=512)
        result = parse_learning_row(row)
        assert result is None

    def test_handles_string_embedding(self):
        row = _make_db_row()
        emb_list = row["embedding"]
        row["embedding"] = "[" + ",".join(str(x) for x in emb_list) + "]"
        result = parse_learning_row(row)
        assert result is not None
        assert len(result.embedding) == 1024

    def test_handles_ndarray_embedding(self):
        row = _make_db_row()
        row["embedding"] = np.array(row["embedding"], dtype=np.float64)
        result = parse_learning_row(row)
        assert result is not None
        assert result.embedding.dtype == np.float32

    def test_returns_none_for_unparseable_embedding(self):
        row = _make_db_row()
        row["embedding"] = 42  # not a valid embedding type
        result = parse_learning_row(row)
        assert result is None

    def test_returns_none_for_non_numeric_embedding_list(self):
        row = _make_db_row()
        row["embedding"] = ["not", "numbers"]
        result = parse_learning_row(row)
        assert result is None

    def test_returns_none_for_scalar_metadata(self):
        row = _make_db_row()
        row["metadata"] = 42  # scalar, not dict or string
        result = parse_learning_row(row)
        assert result is None

    def test_returns_none_for_malformed_json_metadata(self):
        row = _make_db_row()
        row["metadata"] = "{invalid json"
        result = parse_learning_row(row)
        assert result is None

    def test_uses_row_session_id_when_metadata_key_missing(self):
        row = _make_db_row()
        del row["metadata"]["session_id"]
        row["session_id"] = "fallback-session"
        result = parse_learning_row(row)
        assert result is not None
        assert result.session_id == "fallback-session"

    def test_uses_fallback_for_missing_content(self):
        row = _make_db_row()
        row["content"] = None
        result = parse_learning_row(row)
        assert result is not None
        assert result.content == ""

    def test_rejects_null_created_at(self):
        row = _make_db_row()
        row["created_at"] = None
        result = parse_learning_row(row)
        assert result is None

    def test_idempotent_across_reruns(self):
        """Same input produces same output regardless of when it runs."""
        row = _make_db_row()
        result1 = parse_learning_row(row)
        result2 = parse_learning_row(row)
        assert result1 is not None
        assert result2 is not None
        assert result1.id == result2.id
        assert result1.created_at == result2.created_at
        assert result1.learning_type == result2.learning_type


# ---------------------------------------------------------------------------
# merge_tags tests (pure function, immutable)
# ---------------------------------------------------------------------------

class TestMergeTags:

    def test_merges_db_tags_with_existing(self):
        lid = str(uuid.uuid4())
        learnings = [_make_learning(learning_id=lid, tags=["python"])]
        db_tags = {lid: ["testing", "vue"]}

        result = merge_tags(learnings, db_tags)

        assert len(result) == 1
        assert set(result[0].tags) == {"python", "testing", "vue"}

    def test_does_not_mutate_original(self):
        lid = str(uuid.uuid4())
        original_tags = ["python"]
        learnings = [_make_learning(learning_id=lid, tags=original_tags)]
        db_tags = {lid: ["new-tag"]}

        result = merge_tags(learnings, db_tags)

        # Original should not be modified
        assert learnings[0].tags == ["python"]
        # Result should have merged tags
        assert "new-tag" in result[0].tags

    def test_no_db_tags_returns_unchanged_copy(self):
        learnings = [_make_learning(tags=["existing"])]
        result = merge_tags(learnings, {})

        assert len(result) == 1
        assert result[0].tags == ["existing"]
        # Should be a different list object
        assert result is not learnings

    def test_deduplicates_tags_with_stable_order(self):
        lid = str(uuid.uuid4())
        learnings = [_make_learning(learning_id=lid, tags=["python", "testing"])]
        db_tags = {lid: ["testing", "python", "new"]}

        result = merge_tags(learnings, db_tags)
        assert len(result[0].tags) == len(set(result[0].tags))
        # Order should be stable: original tags first, then new
        assert result[0].tags == ["python", "testing", "new"]

    def test_stable_order_across_multiple_calls(self):
        lid = str(uuid.uuid4())
        learnings = [_make_learning(learning_id=lid, tags=["c", "a", "b"])]
        db_tags = {lid: ["b", "d", "a"]}

        results = [merge_tags(learnings, db_tags)[0].tags for _ in range(10)]
        # All calls should produce identical ordering
        assert all(r == results[0] for r in results)
        assert results[0] == ["c", "a", "b", "d"]

    def test_handles_empty_learnings(self):
        result = merge_tags([], {"some-id": ["tag"]})
        assert result == []

    def test_else_branch_isolates_tags_list(self):
        """Even without DB tags, the returned tags list is a separate copy."""
        learnings = [_make_learning(tags=["original"])]
        result = merge_tags(learnings, {})

        result[0].tags.append("mutated")
        assert "mutated" not in learnings[0].tags


# ---------------------------------------------------------------------------
# count_by_type tests (pure function)
# ---------------------------------------------------------------------------

class TestCountByType:

    def test_counts_pattern_types(self):
        patterns = [
            _make_pattern(pattern_type="tool_cluster"),
            _make_pattern(pattern_type="tool_cluster"),
            _make_pattern(pattern_type="session_theme"),
        ]
        result = count_by_type(patterns)
        assert result == {"tool_cluster": 2, "session_theme": 1}

    def test_empty_patterns(self):
        assert count_by_type([]) == {}

    def test_single_type(self):
        patterns = [_make_pattern(pattern_type="workflow")]
        assert count_by_type(patterns) == {"workflow": 1}


# ---------------------------------------------------------------------------
# build_run_summary tests (pure function)
# ---------------------------------------------------------------------------

class TestBuildRunSummary:

    def test_builds_success_summary(self):
        run_id = str(uuid.uuid4())
        by_type = {"tool_cluster": 2}

        result = build_run_summary(
            run_id=run_id,
            learnings_count=100,
            patterns_count=2,
            patterns_by_type=by_type,
            written=2,
            duration=1.23,
            dry_run=False,
        )

        assert result["success"] is True
        assert result["run_id"] == run_id
        assert result["learnings_analyzed"] == 100
        assert result["patterns_detected"] == 2
        assert result["patterns_by_type"] == by_type
        assert result["written"] == 2
        assert result["duration_seconds"] == 1.23
        assert result["dry_run"] is False

    def test_dry_run_summary(self):
        result = build_run_summary(
            run_id="test",
            learnings_count=50,
            patterns_count=3,
            patterns_by_type={"a": 3},
            written=0,
            duration=0.5,
            dry_run=True,
        )

        assert result["dry_run"] is True
        assert result["written"] == 0

    def test_zero_learnings_summary(self):
        result = build_run_summary(
            run_id="test",
            learnings_count=0,
            patterns_count=0,
            patterns_by_type={},
            written=0,
            duration=0.1,
            dry_run=False,
        )

        assert result["learnings_analyzed"] == 0
        assert result["patterns_detected"] == 0


# ---------------------------------------------------------------------------
# format_report tests (pure function)
# ---------------------------------------------------------------------------

class TestFormatReport:

    def test_formats_report_with_patterns(self):
        patterns = [
            {
                "pattern_type": "tool_cluster",
                "label": "Vue testing patterns",
                "confidence": 0.85,
                "member_count": 5,
                "session_count": 3,
                "tags": ["vue", "testing"],
            },
        ]
        run_id = uuid.uuid4()
        created_at = datetime.now(UTC)

        report = format_report(run_id, created_at, patterns)

        assert "Pattern Detection Report" in report
        assert str(run_id) in report
        assert "Vue testing patterns" in report
        assert "0.85" in report
        assert "TOOL_CLUSTER" in report

    def test_formats_empty_patterns(self):
        report = format_report(uuid.uuid4(), datetime.now(UTC), [])
        assert "Pattern Detection Report" in report
        assert "Patterns: 0" in report

    def test_groups_by_type(self):
        patterns = [
            {
                "pattern_type": "tool_cluster",
                "label": "Cluster A",
                "confidence": 0.8,
                "member_count": 3,
                "session_count": 2,
                "tags": ["a"],
            },
            {
                "pattern_type": "session_theme",
                "label": "Theme B",
                "confidence": 0.7,
                "member_count": 4,
                "session_count": 3,
                "tags": ["b"],
            },
        ]
        report = format_report(uuid.uuid4(), datetime.now(UTC), patterns)
        assert "TOOL_CLUSTER" in report
        assert "SESSION_THEME" in report

    def test_truncates_tags_to_five(self):
        patterns = [
            {
                "pattern_type": "tool_cluster",
                "label": "Many tags",
                "confidence": 0.5,
                "member_count": 2,
                "session_count": 1,
                "tags": ["alpha", "bravo", "charlie", "delta", "echo",
                         "foxtrot_extra", "golf_extra"],
            },
        ]
        report = format_report(uuid.uuid4(), datetime.now(UTC), patterns)
        # Should only show first 5 tags
        assert "echo" in report
        assert "foxtrot_extra" not in report
        assert "golf_extra" not in report


# ---------------------------------------------------------------------------
# format_dry_run_output tests (pure function)
# ---------------------------------------------------------------------------

class TestFormatDryRunOutput:

    def test_formats_patterns(self):
        patterns = [
            _make_pattern(
                pattern_type="tool_cluster",
                label="Test cluster",
                confidence=0.85,
            ),
        ]
        output = format_dry_run_output(patterns)
        assert "Dry Run Results" in output
        assert "tool_cluster" in output
        assert "Test cluster" in output
        assert "0.85" in output

    def test_empty_patterns_returns_empty(self):
        output = format_dry_run_output([])
        assert output == ""

    def test_includes_member_and_session_counts(self):
        pattern = _make_pattern(member_count=5)
        output = format_dry_run_output([pattern])
        assert "Members: 5" in output
        assert "Sessions: 3" in output  # default session_count from _make_pattern


# ---------------------------------------------------------------------------
# load_learnings tests (I/O handler)
# ---------------------------------------------------------------------------

class TestLoadLearnings:

    @pytest.mark.asyncio
    async def test_parses_rows_to_learnings(self):
        rows = [_make_db_row() for _ in range(5)]
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=rows)

        learnings, rejected = await load_learnings(pool)

        assert len(learnings) == 5
        assert rejected == 0
        for lrn in learnings:
            assert lrn.learning_type == "WORKING_SOLUTION"
            assert len(lrn.embedding) == 1024
            assert lrn.confidence == "high"

    @pytest.mark.asyncio
    async def test_skips_empty_embeddings_and_reports_rejected(self):
        row = _make_db_row()
        row["embedding"] = []
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[row])

        learnings, rejected = await load_learnings(pool)
        assert len(learnings) == 0
        assert rejected == 1

    @pytest.mark.asyncio
    async def test_handles_string_metadata(self):
        row = _make_db_row()
        row["metadata"] = json.dumps(row["metadata"])
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[row])

        learnings, rejected = await load_learnings(pool)
        assert len(learnings) == 1
        assert rejected == 0
        assert learnings[0].learning_type == "WORKING_SOLUTION"

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty(self):
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[])

        learnings, rejected = await load_learnings(pool)
        assert learnings == []
        assert rejected == 0


# ---------------------------------------------------------------------------
# load_tags_for_learnings tests (I/O handler)
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
# write_patterns tests (I/O handler)
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
# run_pattern_detection tests (orchestrator)
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

    @pytest.mark.asyncio
    @patch("scripts.core.pattern_batch._get_pool")
    @patch("scripts.core.pattern_batch._ensure_tables")
    async def test_all_rows_rejected_preserves_snapshot(
        self, mock_ensure, mock_get_pool
    ):
        """When rows are fetched but all fail parsing, don't wipe the snapshot."""
        pool, conn = _make_pool()
        mock_get_pool.return_value = pool
        mock_ensure.return_value = True
        bad_rows = [_make_db_row(embedding_dim=512) for _ in range(3)]
        conn.fetch = AsyncMock(return_value=bad_rows)

        result = await run_pattern_detection()

        assert result["success"] is False
        assert "rejected" in result["error"].lower()
        assert result["rejected_count"] == 3
        # write_patterns should NOT have been called (no supersede)
        assert conn.execute.call_count == 0
        assert conn.fetchval.call_count == 0
        assert conn.executemany.call_count == 0

    @pytest.mark.asyncio
    @patch("scripts.core.pattern_batch._get_pool")
    @patch("scripts.core.pattern_batch._ensure_tables")
    async def test_partial_rejection_preserves_snapshot(
        self, mock_ensure, mock_get_pool
    ):
        """Mixed valid/invalid rows should abort, not publish from truncated input."""
        pool, conn = _make_pool()
        mock_get_pool.return_value = pool
        mock_ensure.return_value = True
        good_rows = [_make_db_row() for _ in range(5)]
        bad_rows = [_make_db_row(embedding_dim=512) for _ in range(2)]
        conn.fetch = AsyncMock(return_value=good_rows + bad_rows)

        result = await run_pattern_detection()

        assert result["success"] is False
        assert result["rejected_count"] == 2
        assert result["learnings_parsed"] == 5
        # write_patterns should NOT have been called
        assert conn.execute.call_count == 0
        assert conn.fetchval.call_count == 0
        assert conn.executemany.call_count == 0
