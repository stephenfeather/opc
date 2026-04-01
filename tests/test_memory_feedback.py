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
    build_parser,
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
