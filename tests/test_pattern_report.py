"""Tests for pattern detection reporting.

Validates:
1. Human-readable report format includes all sections
2. JSON output is valid and contains expected fields
3. Summary one-liner is concise
4. Empty run handled gracefully
5. Report truncation works
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.pattern_report import (  # noqa: E402  # noqa: E402
    _format_human,
    _format_json,
    _truncate,
    generate_report,
    generate_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pattern_row(
    *,
    pattern_type: str = "tool_cluster",
    label: str = "Vue testing patterns across 3 sessions",
    confidence: float = 0.85,
    member_count: int = 8,
    session_count: int = 3,
    tags: list[str] | None = None,
    representative_content: str = "Vue component testing needs...",
    temporal_span_days: int = 21,
) -> dict:
    return {
        "id": uuid.uuid4(),
        "pattern_type": pattern_type,
        "label": label,
        "confidence": confidence,
        "member_count": member_count,
        "session_count": session_count,
        "tags": tags or ["vue", "testing", "vitest"],
        "representative_id": uuid.uuid4(),
        "representative_content": representative_content,
        "metadata": {"temporal_span_days": temporal_span_days, "size": 8},
        "created_at": datetime.now(UTC),
        "run_id": uuid.uuid4(),
        "superseded_at": None,
        "synthesized_memory_id": None,
    }


def _make_meta(pattern_count: int = 3) -> dict:
    return {
        "run_id": uuid.uuid4(),
        "created_at": datetime.now(UTC) - timedelta(hours=2),
        "pattern_count": pattern_count,
    }


# ---------------------------------------------------------------------------
# Truncation tests
# ---------------------------------------------------------------------------

class TestTruncate:

    def test_short_text_unchanged(self):
        assert _truncate("hello", 80) == "hello"

    def test_long_text_truncated(self):
        text = "a" * 100
        result = _truncate(text, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_empty_string(self):
        assert _truncate("", 80) == ""

    def test_none_returns_empty(self):
        assert _truncate(None, 80) == ""

    def test_exact_length(self):
        text = "a" * 80
        assert _truncate(text, 80) == text


# ---------------------------------------------------------------------------
# Human format tests
# ---------------------------------------------------------------------------

class TestFormatHuman:

    def test_includes_header(self):
        meta = _make_meta()
        report = _format_human(meta, [], 2206, 756)
        assert "Pattern Detection Report" in report
        assert "2,206 learnings" in report
        assert "756 sessions" in report

    def test_no_patterns_message(self):
        meta = _make_meta(0)
        report = _format_human(meta, [], 100, 50)
        assert "No patterns detected" in report

    def test_includes_pattern_details(self):
        meta = _make_meta()
        patterns = [
            _make_pattern_row(pattern_type="tool_cluster"),
            _make_pattern_row(
                pattern_type="anti_pattern",
                label="Repeated failures",
                confidence=0.72,
            ),
        ]
        report = _format_human(meta, patterns, 2206, 756)
        assert "TOOL CLUSTERS" in report
        assert "ANTI-PATTERNS" in report
        assert "Vue testing" in report
        assert "Repeated failures" in report
        assert "0.85" in report
        assert "0.72" in report

    def test_includes_representative_content(self):
        meta = _make_meta()
        patterns = [_make_pattern_row(
            representative_content="Important pattern detail",
        )]
        report = _format_human(meta, patterns, 100, 10)
        assert "Important pattern detail" in report

    def test_includes_temporal_span(self):
        meta = _make_meta()
        patterns = [_make_pattern_row(temporal_span_days=21)]
        report = _format_human(meta, patterns, 100, 10)
        assert "21 days" in report

    def test_footer_summary(self):
        meta = _make_meta()
        patterns = [
            _make_pattern_row(member_count=8),
            _make_pattern_row(member_count=12),
        ]
        report = _format_human(meta, patterns, 100, 10)
        assert "Total: 2 patterns covering 20 learnings" in report
        assert "Average confidence" in report

    def test_type_ordering(self):
        meta = _make_meta()
        patterns = [
            _make_pattern_row(pattern_type="anti_pattern"),
            _make_pattern_row(pattern_type="cross_project"),
            _make_pattern_row(pattern_type="tool_cluster"),
        ]
        report = _format_human(meta, patterns, 100, 10)
        cross_pos = report.index("CROSS-PROJECT")
        tool_pos = report.index("TOOL CLUSTERS")
        anti_pos = report.index("ANTI-PATTERNS")
        assert cross_pos < tool_pos < anti_pos

    def test_many_tags_shows_overflow(self):
        meta = _make_meta()
        patterns = [_make_pattern_row(
            tags=["a", "b", "c", "d", "e", "f", "g"],
        )]
        report = _format_human(meta, patterns, 100, 10)
        assert "+2 more" in report


# ---------------------------------------------------------------------------
# JSON format tests
# ---------------------------------------------------------------------------

class TestFormatJson:

    def test_valid_json(self):
        meta = _make_meta()
        patterns = [_make_pattern_row()]
        result = _format_json(meta, patterns, 2206, 756)
        data = json.loads(result)
        assert data["total_learnings"] == 2206
        assert data["pattern_count"] == 1

    def test_pattern_fields_present(self):
        meta = _make_meta()
        patterns = [_make_pattern_row()]
        result = _format_json(meta, patterns, 100, 10)
        data = json.loads(result)
        pat = data["patterns"][0]
        assert "pattern_type" in pat
        assert "label" in pat
        assert "confidence" in pat
        assert "member_count" in pat
        assert "session_count" in pat
        assert "tags" in pat
        assert "temporal_span_days" in pat

    def test_empty_patterns(self):
        meta = _make_meta(0)
        result = _format_json(meta, [], 100, 10)
        data = json.loads(result)
        assert data["patterns"] == []
        assert data["pattern_count"] == 0


# ---------------------------------------------------------------------------
# Integration tests (with mocked DB)
# ---------------------------------------------------------------------------

class FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


def _make_pool(conn):
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))
    return pool


class TestGenerateReport:

    @pytest.mark.asyncio
    @patch("scripts.core.pattern_report._get_pool")
    async def test_no_runs_message(self, mock_pool):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.return_value = _make_pool(conn)

        result = await generate_report()
        assert "No pattern detection runs found" in result

    @pytest.mark.asyncio
    @patch("scripts.core.pattern_report._get_pool")
    async def test_returns_string(self, mock_pool):
        run_id = uuid.uuid4()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=[
            # _fetch_run_metadata
            {"run_id": run_id, "created_at": datetime.now(UTC), "pattern_count": 1},
            # _fetch_total_learnings
            {"cnt": 100},
            # _fetch_total_sessions
            {"cnt": 50},
        ])
        conn.fetch = AsyncMock(return_value=[])
        mock_pool.return_value = _make_pool(conn)

        result = await generate_report()
        assert isinstance(result, str)
        assert "Pattern Detection Report" in result


class TestGenerateSummary:

    @pytest.mark.asyncio
    @patch("scripts.core.pattern_report._get_pool")
    async def test_no_runs(self, mock_pool):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.return_value = _make_pool(conn)

        result = await generate_summary()
        assert "no runs yet" in result

    @pytest.mark.asyncio
    @patch("scripts.core.pattern_report._get_pool")
    async def test_summary_format(self, mock_pool):
        run_id = uuid.uuid4()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={
            "run_id": run_id,
            "created_at": datetime.now(UTC) - timedelta(hours=2),
            "pattern_count": 5,
        })
        conn.fetch = AsyncMock(return_value=[
            {"pattern_type": "tool_cluster", "cnt": 3},
            {"pattern_type": "cross_project", "cnt": 2},
        ])
        mock_pool.return_value = _make_pool(conn)

        result = await generate_summary()
        assert "2h ago" in result
        assert "5 patterns" in result
        assert "tool_cluster" in result
