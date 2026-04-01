"""Tests for memory_metrics.py.

Validates:
1. Period parsing (pure function)
2. Human-readable formatter (pure function)
3. CLI argument parsing
4. Metric collection against PostgreSQL (integration)
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.db.postgres_pool import reset_pool  # noqa: E402
from scripts.core.memory_metrics import (  # noqa: E402
    build_parser,
    collect_all_metrics,
    format_human,
    parse_period,
)

# ---------------------------------------------------------------------------
# Period parsing
# ---------------------------------------------------------------------------

class TestParsePeriod:
    def test_none_returns_none_tuple(self):
        start, end = parse_period(None)
        assert start is None
        assert end is None

    def test_empty_string_returns_none_tuple(self):
        start, end = parse_period("")
        assert start is None
        assert end is None

    def test_valid_period(self):
        start, end = parse_period("2026-03-01:2026-03-31")
        assert start == datetime(2026, 3, 1, tzinfo=UTC)
        assert end.year == 2026
        assert end.month == 3
        assert end.day == 31
        assert end.hour == 23
        assert end.minute == 59

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid period format"):
            parse_period("2026-03-01")

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            parse_period("not-a-date:also-not")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_default_is_json(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.human is False
        assert args.period is None

    def test_human_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--human"])
        assert args.human is True

    def test_period_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--period", "2026-03-01:2026-03-31"])
        assert args.period == "2026-03-01:2026-03-31"


# ---------------------------------------------------------------------------
# Human-readable formatter
# ---------------------------------------------------------------------------

def _sample_metrics() -> dict:
    """Return a minimal but complete metrics dict for formatter tests."""
    return {
        "generated_at": "2026-04-01T14:00:00+00:00",
        "period": None,
        "version": "0.7.3",
        "totals": {
            "active_learnings": 100,
            "superseded_learnings": 10,
            "total_learnings": 110,
        },
        "per_session": {
            "recent_10_sessions": {"average": 5.0, "min": 2, "max": 12},
            "overall": {"average": 4.0, "total_sessions_with_learnings": 25},
        },
        "confidence_distribution": {
            "high": {"count": 30, "pct": 30.0},
            "medium": {"count": 50, "pct": 50.0},
            "low": {"count": 20, "pct": 20.0},
        },
        "classification_distribution": {
            "WORKING_SOLUTION": {"count": 40, "pct": 40.0},
            "ERROR_FIX": {"count": 30, "pct": 30.0},
            "CODEBASE_PATTERN": {"count": 30, "pct": 30.0},
        },
        "dedup_stats": {
            "learnings_with_content_hash": 105,
            "learnings_without_content_hash": 5,
            "hash_coverage_pct": 95.5,
            "note": "Dedup rejections are not persisted; hash_coverage indicates dedup eligibility",
        },
        "extraction_stats": {
            "total_sessions": 50,
            "extracted": 40,
            "pending": 5,
            "failed": 3,
            "retried": 2,
            "extraction_rate_pct": 80.0,
        },
        "stale_learnings": {
            "never_recalled": 30,
            "total_active": 100,
            "never_recalled_pct": 30.0,
        },
        "top_tags": [
            {"tag": "hooks", "count": 20},
            {"tag": "python", "count": 15},
        ],
        "superseded": {
            "superseded_count": 10,
            "total_learnings": 110,
            "superseded_pct": 9.1,
        },
        "temporal": {
            "oldest_learning": "2026-01-15T09:00:00+00:00",
            "newest_learning": "2026-04-01T13:00:00+00:00",
            "last_7_days": 8,
            "last_30_days": 25,
        },
    }


class TestHumanFormatter:
    def test_contains_header(self):
        output = format_human(_sample_metrics())
        assert "Memory Metrics Report" in output
        assert "v0.7.3" in output

    def test_contains_totals(self):
        output = format_human(_sample_metrics())
        assert "100 active" in output
        assert "10 superseded" in output

    def test_contains_confidence(self):
        output = format_human(_sample_metrics())
        assert "Confidence Distribution" in output
        assert "high" in output
        assert "30.0%" in output

    def test_contains_classification(self):
        output = format_human(_sample_metrics())
        assert "WORKING_SOLUTION" in output

    def test_contains_tags(self):
        output = format_human(_sample_metrics())
        assert "hooks" in output
        assert "python" in output

    def test_contains_temporal(self):
        output = format_human(_sample_metrics())
        assert "last 7d: 8" in output
        assert "last 30d: 25" in output

    def test_period_shown_when_set(self):
        metrics = _sample_metrics()
        metrics["period"] = {
            "from": "2026-03-01T00:00:00+00:00",
            "to": "2026-03-31T23:59:59+00:00",
        }
        output = format_human(metrics)
        assert "Period:" in output
        assert "2026-03-01" in output

    def test_empty_metrics(self):
        metrics = _sample_metrics()
        metrics["totals"] = {
            "active_learnings": 0,
            "superseded_learnings": 0,
            "total_learnings": 0,
        }
        metrics["top_tags"] = []
        metrics["confidence_distribution"] = {}
        metrics["classification_distribution"] = {}
        output = format_human(metrics)
        assert "0 active" in output


# ---------------------------------------------------------------------------
# JSON output structure
# ---------------------------------------------------------------------------

class TestJSONStructure:
    def test_sample_metrics_is_valid_json(self):
        metrics = _sample_metrics()
        serialized = json.dumps(metrics)
        parsed = json.loads(serialized)
        assert parsed["version"] == "0.7.3"

    def test_all_top_level_keys_present(self):
        metrics = _sample_metrics()
        expected_keys = {
            "generated_at", "period", "totals", "per_session",
            "confidence_distribution", "classification_distribution",
            "dedup_stats", "extraction_stats", "stale_learnings",
            "top_tags", "superseded", "temporal", "version",
        }
        assert set(metrics.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Integration: collect_all_metrics against PostgreSQL
# ---------------------------------------------------------------------------

class TestCollectMetrics:
    """Integration tests that hit the real database.

    These verify the SQL queries execute without error and return
    the expected shape. They do NOT insert test data — they run
    against whatever is in the database.
    """

    def setup_method(self):
        reset_pool()

    @pytest.mark.asyncio
    async def test_collect_returns_all_keys(self):
        metrics = await collect_all_metrics()
        expected_keys = {
            "generated_at", "period", "totals", "per_session",
            "confidence_distribution", "classification_distribution",
            "dedup_stats", "extraction_stats", "stale_learnings",
            "top_tags", "superseded", "temporal", "version",
        }
        assert set(metrics.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_totals_are_integers(self):
        metrics = await collect_all_metrics()
        t = metrics["totals"]
        assert isinstance(t["active_learnings"], int)
        assert isinstance(t["superseded_learnings"], int)
        assert isinstance(t["total_learnings"], int)
        assert t["total_learnings"] == t["active_learnings"] + t["superseded_learnings"]

    @pytest.mark.asyncio
    async def test_period_is_none_by_default(self):
        metrics = await collect_all_metrics()
        assert metrics["period"] is None

    @pytest.mark.asyncio
    async def test_period_filter(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)
        metrics = await collect_all_metrics(start, end)
        assert metrics["period"] is not None
        assert "from" in metrics["period"]

    @pytest.mark.asyncio
    async def test_version_present(self):
        metrics = await collect_all_metrics()
        assert metrics["version"] == "0.7.3"

    @pytest.mark.asyncio
    async def test_top_tags_is_list(self):
        metrics = await collect_all_metrics()
        assert isinstance(metrics["top_tags"], list)
        if metrics["top_tags"]:
            assert "tag" in metrics["top_tags"][0]
            assert "count" in metrics["top_tags"][0]

    @pytest.mark.asyncio
    async def test_pct_fields_are_floats(self):
        metrics = await collect_all_metrics()
        for level_data in metrics["confidence_distribution"].values():
            assert isinstance(level_data["pct"], float)
        assert isinstance(metrics["extraction_stats"]["extraction_rate_pct"], float)
        assert isinstance(metrics["stale_learnings"]["never_recalled_pct"], float)
