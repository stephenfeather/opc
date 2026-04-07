"""Tests for pattern detection reporting.

Validates:
1. Human-readable report format includes all sections
2. JSON output is valid and contains expected fields
3. Summary one-liner is concise
4. Empty run handled gracefully
5. Report truncation works
6. Pure functions: age formatting, metadata parsing, type section formatting
7. I/O separation: generate_report_from_data / generate_summary_from_data
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

from scripts.core.pattern_report import (  # noqa: E402
    _format_human,
    _format_json,
    _truncate,
    format_age,
    format_type_breakdown,
    format_type_section,
    generate_report,
    generate_report_from_data,
    generate_summary,
    generate_summary_from_data,
    parse_pattern_metadata,
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
# format_age tests (pure function)
# ---------------------------------------------------------------------------


class TestFormatAge:

    def test_days_ago(self):
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        created = now - timedelta(days=3)
        result = format_age(created, now=now)
        assert result == "3d ago"

    def test_hours_ago(self):
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        created = now - timedelta(hours=5)
        result = format_age(created, now=now)
        assert result == "5h ago"

    def test_minutes_ago(self):
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        created = now - timedelta(minutes=15)
        result = format_age(created, now=now)
        assert result == "15m ago"

    def test_zero_minutes(self):
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        created = now - timedelta(seconds=30)
        result = format_age(created, now=now)
        assert result == "0m ago"

    def test_defaults_to_now(self):
        created = datetime.now(UTC) - timedelta(days=1)
        result = format_age(created)
        assert result == "1d ago"

    def test_future_timestamp_shows_skew(self):
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        created = now + timedelta(minutes=5)
        result = format_age(created, now=now)
        assert "ago" not in result
        assert "in 5m" == result

    def test_future_timestamp_hours(self):
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        created = now + timedelta(hours=2)
        result = format_age(created, now=now)
        assert "in 2h" == result

    def test_future_timestamp_days(self):
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        created = now + timedelta(days=1)
        result = format_age(created, now=now)
        assert "in 1d" == result


# ---------------------------------------------------------------------------
# parse_pattern_metadata tests (pure function)
# ---------------------------------------------------------------------------


class TestParsePatternMetadata:

    def test_dict_passthrough(self):
        meta = {"temporal_span_days": 21, "size": 8}
        assert parse_pattern_metadata(meta) == meta

    def test_json_string_parsed(self):
        meta_str = '{"temporal_span_days": 10}'
        result = parse_pattern_metadata(meta_str)
        assert result == {"temporal_span_days": 10}

    def test_none_returns_empty_dict(self):
        assert parse_pattern_metadata(None) == {}

    def test_empty_string_returns_empty_dict(self):
        assert parse_pattern_metadata("") == {}

    def test_malformed_json_returns_error_sentinel(self):
        result = parse_pattern_metadata("not-json")
        assert result.get("_parse_error") is True

    def test_numeric_value_returns_empty_dict(self):
        assert parse_pattern_metadata(42) == {}

    def test_json_array_returns_error_sentinel(self):
        result = parse_pattern_metadata("[]")
        assert result.get("_parse_error") is True

    def test_json_scalar_string_returns_error_sentinel(self):
        result = parse_pattern_metadata('"text"')
        assert result.get("_parse_error") is True

    def test_json_number_string_returns_error_sentinel(self):
        result = parse_pattern_metadata("1")
        assert result.get("_parse_error") is True


# ---------------------------------------------------------------------------
# format_type_section tests (pure, returns lines)
# ---------------------------------------------------------------------------


class TestFormatTypeSection:

    def test_returns_list_of_strings(self):
        patterns = [_make_pattern_row()]
        result = format_type_section("tool_cluster", patterns)
        assert isinstance(result, list)
        assert all(isinstance(line, str) for line in result)

    def test_includes_header(self):
        patterns = [_make_pattern_row()]
        result = format_type_section("tool_cluster", patterns)
        joined = "\n".join(result)
        assert "TOOL CLUSTERS" in joined

    def test_includes_pattern_label(self):
        patterns = [_make_pattern_row(label="My pattern")]
        result = format_type_section("tool_cluster", patterns)
        joined = "\n".join(result)
        assert "My pattern" in joined

    def test_unknown_type_uses_uppercase(self):
        patterns = [_make_pattern_row(pattern_type="custom_type")]
        result = format_type_section("custom_type", patterns)
        joined = "\n".join(result)
        assert "CUSTOM_TYPE" in joined

    def test_many_tags_overflow(self):
        patterns = [_make_pattern_row(
            tags=["a", "b", "c", "d", "e", "f", "g"],
        )]
        result = format_type_section("tool_cluster", patterns)
        joined = "\n".join(result)
        assert "+2 more" in joined


# ---------------------------------------------------------------------------
# format_type_breakdown tests (pure function)
# ---------------------------------------------------------------------------


class TestFormatTypeBreakdown:

    def test_formats_type_counts(self):
        rows = [
            {"pattern_type": "tool_cluster", "cnt": 3},
            {"pattern_type": "cross_project", "cnt": 2},
        ]
        result = format_type_breakdown(rows)
        assert "3 tool_cluster" in result
        assert "2 cross_project" in result

    def test_empty_rows(self):
        assert format_type_breakdown([]) == ""


# ---------------------------------------------------------------------------
# generate_report_from_data tests (pure orchestrator)
# ---------------------------------------------------------------------------


class TestGenerateReportFromData:

    def test_no_meta_returns_not_found(self):
        result = generate_report_from_data(
            meta=None, patterns=[], total_learnings=0,
            total_sessions=0, as_json=False,
        )
        assert "No pattern detection runs found" in result

    def test_human_format(self):
        meta = _make_meta()
        patterns = [_make_pattern_row()]
        result = generate_report_from_data(
            meta=meta, patterns=patterns, total_learnings=100,
            total_sessions=10, as_json=False,
        )
        assert "Pattern Detection Report" in result

    def test_json_format(self):
        meta = _make_meta()
        patterns = [_make_pattern_row()]
        result = generate_report_from_data(
            meta=meta, patterns=patterns, total_learnings=100,
            total_sessions=10, as_json=True,
        )
        data = json.loads(result)
        assert data["total_learnings"] == 100

    def test_malformed_metadata_degrades_gracefully(self):
        meta = _make_meta()
        bad_pattern = _make_pattern_row()
        bad_pattern["metadata"] = "not-json"
        result = generate_report_from_data(
            meta=meta, patterns=[bad_pattern], total_learnings=10,
            total_sessions=5, as_json=False,
        )
        assert "Pattern Detection Report" in result
        assert "span: N/A" in result

    def test_malformed_metadata_json_output(self):
        meta = _make_meta()
        bad_pattern = _make_pattern_row()
        bad_pattern["metadata"] = "not-json"
        result = generate_report_from_data(
            meta=meta, patterns=[bad_pattern], total_learnings=10,
            total_sessions=5, as_json=True,
        )
        data = json.loads(result)
        assert data["patterns"][0]["temporal_span_days"] is None
        assert data["patterns"][0]["metadata_error"] is True


class TestGenerateSummaryFromData:

    def test_no_meta_returns_no_runs(self):
        result = generate_summary_from_data(
            meta=None, type_rows=[], now=datetime.now(UTC),
        )
        assert "no runs yet" in result

    def test_includes_age_and_count(self):
        meta = _make_meta(5)
        rows = [
            {"pattern_type": "tool_cluster", "cnt": 3},
            {"pattern_type": "cross_project", "cnt": 2},
        ]
        now = datetime.now(UTC)
        result = generate_summary_from_data(
            meta=meta, type_rows=rows, now=now,
        )
        assert "2h ago" in result
        assert "5 patterns" in result
        assert "tool_cluster" in result


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
