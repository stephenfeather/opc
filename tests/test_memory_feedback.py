"""Tests for memory feedback system (memory_feedback.py).

Validates that:
1. store_feedback inserts and upserts correctly
2. get_feedback_for_learning returns correct aggregates
3. get_feedback_summary handles empty and populated tables
4. CLI arg parsing works correctly
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from scripts.core.memory_feedback import (
    aggregate_feedback,
    build_parser,
    compute_helpfulness_rate,
    empty_summary,
    format_feedback_row,
    format_summary_result,
    format_top_entries,
    get_feedback_for_learning,
    get_feedback_summary,
    store_feedback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_connection():
    """Build a mock async connection with fetchval/fetchrow/fetch."""
    conn = AsyncMock()
    return conn


def _make_context_manager(conn):
    """Wrap a mock connection in an async context manager."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Pure functions — format_feedback_row
# ---------------------------------------------------------------------------

class TestFormatFeedbackRow:
    def test_formats_complete_row(self):
        row = {
            "id": uuid4(),
            "session_id": "s1",
            "helpful": True,
            "context": "great tip",
            "source": "manual",
            "created_at": MagicMock(isoformat=lambda: "2026-04-01T00:00:00+00:00"),
        }
        result = format_feedback_row(row)
        assert result["session_id"] == "s1"
        assert result["helpful"] is True
        assert result["context"] == "great tip"
        assert result["source"] == "manual"
        assert result["created_at"] == "2026-04-01T00:00:00+00:00"
        assert isinstance(result["id"], str)

    def test_formats_row_with_none_context(self):
        row = {
            "id": uuid4(),
            "session_id": "s2",
            "helpful": False,
            "context": None,
            "source": "hook",
            "created_at": MagicMock(isoformat=lambda: "2026-04-02T00:00:00+00:00"),
        }
        result = format_feedback_row(row)
        assert result["context"] is None
        assert result["helpful"] is False


# ---------------------------------------------------------------------------
# Pure functions — aggregate_feedback
# ---------------------------------------------------------------------------

class TestAggregateFeedback:
    def test_aggregates_multiple_rows(self):
        lid = str(uuid4())
        rows = [
            {"id": uuid4(), "session_id": "s1", "helpful": True,
             "context": "yes", "source": "manual",
             "created_at": MagicMock(isoformat=lambda: "2026-04-01T00:00:00+00:00")},
            {"id": uuid4(), "session_id": "s2", "helpful": False,
             "context": None, "source": "hook",
             "created_at": MagicMock(isoformat=lambda: "2026-04-01T01:00:00+00:00")},
            {"id": uuid4(), "session_id": "s3", "helpful": True,
             "context": "useful", "source": "auto",
             "created_at": MagicMock(isoformat=lambda: "2026-04-01T02:00:00+00:00")},
        ]
        result = aggregate_feedback(rows, lid)
        assert result["learning_id"] == lid
        assert result["total_feedback"] == 3
        assert result["helpful_count"] == 2
        assert result["not_helpful_count"] == 1
        assert len(result["feedback"]) == 3

    def test_aggregates_empty_rows(self):
        lid = str(uuid4())
        result = aggregate_feedback([], lid)
        assert result["total_feedback"] == 0
        assert result["helpful_count"] == 0
        assert result["not_helpful_count"] == 0
        assert result["feedback"] == []


# ---------------------------------------------------------------------------
# Pure functions — compute_helpfulness_rate
# ---------------------------------------------------------------------------

class TestComputeHelpfulnessRate:
    def test_normal_rate(self):
        assert compute_helpfulness_rate(10, 7) == 70.0

    def test_zero_total(self):
        assert compute_helpfulness_rate(0, 0) == 0.0

    def test_all_helpful(self):
        assert compute_helpfulness_rate(5, 5) == 100.0

    def test_none_helpful(self):
        assert compute_helpfulness_rate(8, 0) == 0.0


# ---------------------------------------------------------------------------
# Pure functions — format_top_entries
# ---------------------------------------------------------------------------

class TestFormatTopEntries:
    def test_formats_entries(self):
        rows = [
            {"learning_id": uuid4(), "content": "a" * 200, "helpful_count": 5},
            {"learning_id": uuid4(), "content": "short", "helpful_count": 2},
        ]
        result = format_top_entries(rows, "helpful_count")
        assert len(result) == 2
        assert len(result[0]["content"]) == 120  # truncated
        assert result[0]["helpful_count"] == 5
        assert result[1]["content"] == "short"

    def test_formats_empty(self):
        assert format_top_entries([], "helpful_count") == []


# ---------------------------------------------------------------------------
# Pure functions — format_summary_result
# ---------------------------------------------------------------------------

class TestFormatSummaryResult:
    def test_formats_normal_summary(self):
        totals = {"total": 10, "helpful": 7, "not_helpful": 3, "unique_learnings": 5}
        top_h = [{"learning_id": uuid4(), "content": "good", "helpful_count": 3}]
        top_nh = []
        result = format_summary_result(totals, top_h, top_nh)
        assert result["total_feedback"] == 10
        assert result["helpful_count"] == 7
        assert result["not_helpful_count"] == 3
        assert result["unique_learnings_rated"] == 5
        assert result["helpfulness_rate"] == 70.0
        assert len(result["top_helpful"]) == 1
        assert result["top_not_helpful"] == []

    def test_formats_empty_summary(self):
        totals = {"total": 0, "helpful": 0, "not_helpful": 0, "unique_learnings": 0}
        result = format_summary_result(totals, [], [])
        assert result["total_feedback"] == 0
        assert result["helpfulness_rate"] == 0.0

    def test_formats_zero_helpful(self):
        totals = {"total": 5, "helpful": 0, "not_helpful": 5, "unique_learnings": 3}
        result = format_summary_result(totals, [], [])
        assert result["helpfulness_rate"] == 0.0


# ---------------------------------------------------------------------------
# Pure functions — empty_summary (no shared mutable state)
# ---------------------------------------------------------------------------

class TestEmptySummary:
    def test_returns_fresh_dict_each_call(self):
        a = empty_summary()
        b = empty_summary()
        assert a == b
        assert a is not b

    def test_mutation_does_not_leak(self):
        first = empty_summary()
        first["top_helpful"].append({"learning_id": "injected"})
        first["total_feedback"] = 999
        second = empty_summary()
        assert second["top_helpful"] == []
        assert second["total_feedback"] == 0


# ---------------------------------------------------------------------------
# store_feedback
# ---------------------------------------------------------------------------

class TestStoreFeedback:
    @pytest.mark.asyncio
    async def test_stores_helpful_feedback(self):
        learning_id = str(uuid4())
        conn = _mock_connection()
        conn.fetchval = AsyncMock(return_value=True)  # exists check
        conn.fetchrow = AsyncMock(return_value={
            "id": uuid4(),
            "created_at": MagicMock(isoformat=lambda: "2026-04-01T00:00:00+00:00"),
        })

        with patch(
            "scripts.core.memory_feedback.get_connection",
            return_value=_make_context_manager(conn),
        ):
            result = await store_feedback(
                learning_id=learning_id,
                helpful=True,
                session_id="test-session",
                context="was useful",
            )

        assert result["success"] is True
        assert result["helpful"] is True
        assert result["learning_id"] == learning_id

    @pytest.mark.asyncio
    async def test_stores_not_helpful_feedback(self):
        learning_id = str(uuid4())
        conn = _mock_connection()
        conn.fetchval = AsyncMock(return_value=True)
        conn.fetchrow = AsyncMock(return_value={
            "id": uuid4(),
            "created_at": MagicMock(isoformat=lambda: "2026-04-01T00:00:00+00:00"),
        })

        with patch(
            "scripts.core.memory_feedback.get_connection",
            return_value=_make_context_manager(conn),
        ):
            result = await store_feedback(
                learning_id=learning_id,
                helpful=False,
                session_id="test-session",
            )

        assert result["success"] is True
        assert result["helpful"] is False

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_learning(self):
        learning_id = str(uuid4())
        conn = _mock_connection()
        conn.fetchval = AsyncMock(return_value=False)  # does not exist

        with patch(
            "scripts.core.memory_feedback.get_connection",
            return_value=_make_context_manager(conn),
        ):
            result = await store_feedback(
                learning_id=learning_id,
                helpful=True,
                session_id="test-session",
            )

        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# get_feedback_for_learning
# ---------------------------------------------------------------------------

class TestGetFeedbackForLearning:
    @pytest.mark.asyncio
    async def test_returns_aggregated_feedback(self):
        learning_id = str(uuid4())
        conn = _mock_connection()
        conn.fetch = AsyncMock(return_value=[
            {
                "id": uuid4(),
                "session_id": "s1",
                "helpful": True,
                "context": "great",
                "source": "manual",
                "created_at": MagicMock(isoformat=lambda: "2026-04-01T00:00:00+00:00"),
            },
            {
                "id": uuid4(),
                "session_id": "s2",
                "helpful": False,
                "context": None,
                "source": "manual",
                "created_at": MagicMock(isoformat=lambda: "2026-04-01T01:00:00+00:00"),
            },
        ])

        with patch(
            "scripts.core.memory_feedback.get_connection",
            return_value=_make_context_manager(conn),
        ):
            result = await get_feedback_for_learning(learning_id)

        assert result["total_feedback"] == 2
        assert result["helpful_count"] == 1
        assert result["not_helpful_count"] == 1

    @pytest.mark.asyncio
    async def test_empty_feedback(self):
        learning_id = str(uuid4())
        conn = _mock_connection()
        conn.fetch = AsyncMock(return_value=[])

        with patch(
            "scripts.core.memory_feedback.get_connection",
            return_value=_make_context_manager(conn),
        ):
            result = await get_feedback_for_learning(learning_id)

        assert result["total_feedback"] == 0
        assert result["helpful_count"] == 0


# ---------------------------------------------------------------------------
# get_feedback_summary
# ---------------------------------------------------------------------------

class TestGetFeedbackSummary:
    @pytest.mark.asyncio
    async def test_returns_zeroes_when_table_missing(self):
        conn = _mock_connection()
        conn.fetchval = AsyncMock(return_value=False)  # table doesn't exist

        with patch(
            "scripts.core.memory_feedback.get_connection",
            return_value=_make_context_manager(conn),
        ):
            result = await get_feedback_summary()

        assert result["total_feedback"] == 0
        assert result["helpfulness_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_returns_stats_with_data(self):
        conn = _mock_connection()
        conn.fetchval = AsyncMock(return_value=True)  # table exists
        conn.fetchrow = AsyncMock(return_value={
            "total": 10,
            "helpful": 7,
            "not_helpful": 3,
            "unique_learnings": 5,
        })
        conn.fetch = AsyncMock(return_value=[])  # top helpful/not helpful

        with patch(
            "scripts.core.memory_feedback.get_connection",
            return_value=_make_context_manager(conn),
        ):
            result = await get_feedback_summary()

        assert result["total_feedback"] == 10
        assert result["helpful_count"] == 7
        assert result["helpfulness_rate"] == 70.0
        assert result["unique_learnings_rated"] == 5


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_store_helpful(self):
        parser = build_parser()
        args = parser.parse_args([
            "store", "--learning-id", "00000000-0000-0000-0000-000000000001", "--helpful",
        ])
        assert args.command == "store"
        assert args.learning_id == "00000000-0000-0000-0000-000000000001"
        assert args.helpful is True

    def test_store_not_helpful(self):
        parser = build_parser()
        args = parser.parse_args([
            "store", "--learning-id", "00000000-0000-0000-0000-000000000001", "--not-helpful",
        ])
        assert args.command == "store"
        assert args.not_helpful is True

    def test_store_requires_helpful_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["store", "--learning-id", "00000000-0000-0000-0000-000000000001"])

    def test_get_command(self):
        parser = build_parser()
        args = parser.parse_args(["get", "--learning-id", "00000000-0000-0000-0000-000000000001"])
        assert args.command == "get"
        assert args.learning_id == "00000000-0000-0000-0000-000000000001"

    def test_summary_command(self):
        parser = build_parser()
        args = parser.parse_args(["summary"])
        assert args.command == "summary"

    def test_rejects_invalid_uuid(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["store", "--learning-id", "not-a-uuid", "--helpful"])

    def test_rejects_invalid_source(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "store", "--learning-id", "00000000-0000-0000-0000-000000000001",
                "--helpful", "--source", "invalid",
            ])
