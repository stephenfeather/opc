"""Tests for memory_metrics.py and memory_metrics_core.py.

Validates:
1. Pure core functions (calculate_pct, build_confidence_map, etc.)
2. Period parsing (pure function)
3. Human-readable formatter (pure function)
4. Report assembly (pure function)
5. CLI argument parsing
6. Metric collection against PostgreSQL (integration)
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.memory_metrics_core import (  # noqa: E402
    assemble_report,
    build_classification_map,
    build_confidence_map,
    calculate_pct,
    format_human,
    parse_period,
)

# Backwards-compat re-exports from the original module
from scripts.core.memory_metrics import (  # noqa: E402
    _get_version,
    build_parser,
    collect_all_metrics,
    format_human as format_human_reexport,
    parse_period as parse_period_reexport,
)


# ---------------------------------------------------------------------------
# calculate_pct
# ---------------------------------------------------------------------------


class TestCalculatePct:
    def test_basic_percentage(self):
        assert calculate_pct(25, 100) == 25.0

    def test_zero_total_returns_zero(self):
        assert calculate_pct(10, 0) == 0.0

    def test_rounding_to_one_decimal(self):
        assert calculate_pct(1, 3) == 33.3

    def test_full_percentage(self):
        assert calculate_pct(100, 100) == 100.0

    def test_zero_count(self):
        assert calculate_pct(0, 50) == 0.0


# ---------------------------------------------------------------------------
# build_confidence_map
# ---------------------------------------------------------------------------


class TestBuildConfidenceMap:
    def test_canonical_order(self):
        rows = [
            {"level": "low", "count": 5},
            {"level": "high", "count": 10},
            {"level": "medium", "count": 15},
        ]
        result = build_confidence_map(rows)
        keys = list(result.keys())
        assert keys == ["high", "medium", "low"]

    def test_percentages(self):
        rows = [
            {"level": "high", "count": 50},
            {"level": "medium", "count": 30},
            {"level": "low", "count": 20},
        ]
        result = build_confidence_map(rows)
        assert result["high"] == {"count": 50, "pct": 50.0}
        assert result["medium"] == {"count": 30, "pct": 30.0}
        assert result["low"] == {"count": 20, "pct": 20.0}

    def test_extra_levels_appended_sorted(self):
        rows = [
            {"level": "high", "count": 10},
            {"level": "unset", "count": 5},
            {"level": "custom", "count": 3},
        ]
        result = build_confidence_map(rows)
        keys = list(result.keys())
        # canonical first, then extras sorted
        assert keys[0] == "high"
        assert "custom" in keys
        assert "unset" in keys

    def test_empty_rows(self):
        result = build_confidence_map([])
        assert result == {}

    def test_does_not_mutate_input(self):
        rows = [
            {"level": "high", "count": 10},
            {"level": "low", "count": 5},
        ]
        original = [dict(r) for r in rows]
        build_confidence_map(rows)
        assert rows == original


# ---------------------------------------------------------------------------
# build_classification_map
# ---------------------------------------------------------------------------


class TestBuildClassificationMap:
    def test_basic(self):
        rows = [
            {"learning_type": "WORKING_SOLUTION", "count": 40},
            {"learning_type": "ERROR_FIX", "count": 30},
        ]
        result = build_classification_map(rows)
        assert result["WORKING_SOLUTION"] == {"count": 40, "pct": pytest.approx(57.1, abs=0.1)}

    def test_empty(self):
        result = build_classification_map([])
        assert result == {}

    def test_single_type(self):
        rows = [{"learning_type": "CODEBASE_PATTERN", "count": 10}]
        result = build_classification_map(rows)
        assert result["CODEBASE_PATTERN"]["pct"] == 100.0


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

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match="start date.*after end date"):
            parse_period("2026-04-01:2026-03-01")


# ---------------------------------------------------------------------------
# assemble_report
# ---------------------------------------------------------------------------


def _make_query_results() -> dict:
    """Build a complete set of query results for assemble_report."""
    return {
        "totals": {
            "active_learnings": 100,
            "superseded_learnings": 10,
            "total_learnings": 110,
        },
        "per_session": {
            "recent_10_sessions": {"average": 5.0, "min": 2, "max": 12},
            "overall": {"average": 4.0, "total_sessions_with_learnings": 25},
        },
        "confidence": {
            "high": {"count": 30, "pct": 30.0},
            "medium": {"count": 50, "pct": 50.0},
            "low": {"count": 20, "pct": 20.0},
        },
        "classification": {
            "WORKING_SOLUTION": {"count": 40, "pct": 40.0},
            "ERROR_FIX": {"count": 30, "pct": 30.0},
        },
        "dedup": {
            "learnings_with_content_hash": 105,
            "learnings_without_content_hash": 5,
            "hash_coverage_pct": 95.5,
            "note": "Dedup rejections are not persisted",
        },
        "embedding_coverage": {
            "with_embedding": 98,
            "without_embedding": 2,
            "coverage_pct": 98.0,
        },
        "extraction": {
            "total_sessions": 50,
            "extracted": 40,
            "pending": 5,
            "failed": 3,
            "retried": 2,
            "extraction_rate_pct": 80.0,
        },
        "stale": {
            "never_recalled": 30,
            "total_active": 100,
            "never_recalled_pct": 30.0,
        },
        "tags": [
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
        "feedback": {
            "total_feedback": 12,
            "helpful": 10,
            "not_helpful": 2,
            "unique_learnings_rated": 8,
            "helpfulness_rate_pct": 83.3,
        },
        "feedback_velocity": {
            "weeks": [],
            "avg_per_week": 0.0,
        },
        "supersession_candidates": {
            "total_candidates": 5,
            "by_confidence": {"low": 3, "medium": 1, "high": 1},
            "criteria": "active, never recalled, older than 30 days",
        },
        "recall_frequency": {
            "recalled_learnings": 70,
            "total_active": 100,
            "recall_rate_pct": 70.0,
            "total_recall_events": 200,
            "avg_recalls_per_recalled_learning": 2.9,
            "max_recalls_single_learning": 15,
        },
        "type_recall_correlation": {
            "WORKING_SOLUTION": {
                "stored": 40,
                "recalled": 30,
                "recall_rate_pct": 75.0,
                "total_recall_events": 120,
            },
        },
    }


_TEST_TIMESTAMP = "2026-04-01T12:00:00+00:00"


class TestAssembleReport:
    def test_no_period(self):
        qr = _make_query_results()
        report = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        assert report["period"] is None
        assert report["generated_at"] == _TEST_TIMESTAMP

    def test_with_period(self):
        qr = _make_query_results()
        start = datetime(2026, 3, 1, tzinfo=UTC)
        end = datetime(2026, 3, 31, 23, 59, 59, tzinfo=UTC)
        report = assemble_report(
            query_results=qr,
            start=start,
            end=end,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        assert report["period"]["from"] == start.isoformat()
        assert report["period"]["to"] == end.isoformat()

    def test_learnings_per_extraction_computed(self):
        qr = _make_query_results()
        report = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        extraction = report["extraction_stats_alltime"]
        assert extraction["learnings_per_extraction"] == round(110 / 40, 2)

    def test_learnings_per_extraction_zero_extracted(self):
        qr = _make_query_results()
        qr["extraction"]["extracted"] = 0
        report = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        assert report["extraction_stats_alltime"]["learnings_per_extraction"] == 0.0

    def test_does_not_mutate_input(self):
        qr = _make_query_results()
        original_extracted = qr["extraction"]["extracted"]
        assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        # Verify the input extraction dict was NOT mutated
        assert "learnings_per_extraction" not in qr["extraction"]
        assert qr["extraction"]["extracted"] == original_extracted

    def test_all_expected_keys_present(self):
        qr = _make_query_results()
        report = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        expected_keys = {
            "generated_at", "period", "totals", "per_session",
            "confidence_distribution", "classification_distribution",
            "dedup_stats_alltime", "embedding_coverage_alltime",
            "extraction_stats_alltime", "stale_learnings",
            "top_tags_alltime", "superseded_alltime", "temporal_alltime",
            "feedback_alltime", "feedback_velocity",
            "supersession_candidates", "recall_frequency",
            "type_recall_correlation", "version",
        }
        assert set(report.keys()) == expected_keys

    def test_version_passthrough(self):
        qr = _make_query_results()
        report = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="1.2.3",
            generated_at=_TEST_TIMESTAMP,
        )
        assert report["version"] == "1.2.3"

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        qr = _make_query_results()
        report1 = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        report2 = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        assert report1 == report2

    def test_no_output_to_input_aliasing(self):
        """Mutating the output must not change the input query_results."""
        qr = _make_query_results()
        report = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        # Mutate the output
        report["totals"]["active_learnings"] = 999
        report["feedback_alltime"]["helpful"] = 999
        # Verify input is untouched
        assert qr["totals"]["active_learnings"] == 100
        assert qr["feedback"]["helpful"] == 10

    def test_no_input_to_output_aliasing(self):
        """Mutating the input after assembly must not change the report."""
        qr = _make_query_results()
        report = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=_TEST_TIMESTAMP,
        )
        # Mutate the input
        qr["totals"]["active_learnings"] = 999
        qr["feedback"]["helpful"] = 999
        # Verify output is untouched
        assert report["totals"]["active_learnings"] == 100
        assert report["feedback_alltime"]["helpful"] == 10

    def test_generated_at_passthrough(self):
        """Explicit generated_at is used verbatim."""
        qr = _make_query_results()
        ts = "2026-01-01T00:00:00+00:00"
        report = assemble_report(
            query_results=qr,
            start=None,
            end=None,
            all_time_learnings=110,
            version="0.7.3",
            generated_at=ts,
        )
        assert report["generated_at"] == ts

    def test_generated_at_is_required(self):
        """assemble_report must not be callable without generated_at."""
        qr = _make_query_results()
        with pytest.raises(TypeError):
            assemble_report(
                query_results=qr,
                start=None,
                end=None,
                all_time_learnings=110,
                version="0.7.3",
            )


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
        assert args.json is False

    def test_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--json"])
        assert args.json is True
        assert args.human is False

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
        "version": _get_version(),
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
        "dedup_stats_alltime": {
            "learnings_with_content_hash": 105,
            "learnings_without_content_hash": 5,
            "hash_coverage_pct": 95.5,
            "note": "Dedup rejections are not persisted",
        },
        "embedding_coverage_alltime": {
            "with_embedding": 98,
            "without_embedding": 2,
            "coverage_pct": 98.0,
        },
        "extraction_stats_alltime": {
            "total_sessions": 50,
            "extracted": 40,
            "pending": 5,
            "failed": 3,
            "retried": 2,
            "extraction_rate_pct": 80.0,
            "learnings_per_extraction": 2.50,
        },
        "stale_learnings": {
            "never_recalled": 30,
            "total_active": 100,
            "never_recalled_pct": 30.0,
        },
        "top_tags_alltime": [
            {"tag": "hooks", "count": 20},
            {"tag": "python", "count": 15},
        ],
        "superseded_alltime": {
            "superseded_count": 10,
            "total_learnings": 110,
            "superseded_pct": 9.1,
        },
        "temporal_alltime": {
            "oldest_learning": "2026-01-15T09:00:00+00:00",
            "newest_learning": "2026-04-01T13:00:00+00:00",
            "last_7_days": 8,
            "last_30_days": 25,
        },
        "feedback_alltime": {
            "total_feedback": 12,
            "helpful": 10,
            "not_helpful": 2,
            "unique_learnings_rated": 8,
            "helpfulness_rate_pct": 83.3,
        },
        "feedback_velocity": {
            "weeks": [],
            "avg_per_week": 0.0,
        },
        "supersession_candidates": {
            "total_candidates": 5,
            "by_confidence": {"low": 3, "medium": 1, "high": 1},
            "criteria": "active, never recalled, older than 30 days",
        },
        "recall_frequency": {
            "recalled_learnings": 70,
            "total_active": 100,
            "recall_rate_pct": 70.0,
            "total_recall_events": 200,
            "avg_recalls_per_recalled_learning": 2.9,
            "max_recalls_single_learning": 15,
        },
        "type_recall_correlation": {
            "WORKING_SOLUTION": {
                "stored": 40,
                "recalled": 30,
                "recall_rate_pct": 75.0,
                "total_recall_events": 120,
            },
        },
    }


class TestHumanFormatter:
    def test_contains_header(self):
        output = format_human(_sample_metrics())
        assert "Memory Metrics Report" in output
        assert f"v{_get_version()}" in output

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
        metrics["top_tags_alltime"] = []
        metrics["confidence_distribution"] = {}
        metrics["classification_distribution"] = {}
        output = format_human(metrics)
        assert "0 active" in output

    def test_contains_embedding_coverage(self):
        output = format_human(_sample_metrics())
        assert "Embedding Coverage" in output
        assert "98.0%" in output

    def test_contains_feedback(self):
        output = format_human(_sample_metrics())
        assert "10 helpful" in output
        assert "2 not helpful" in output

    def test_contains_recall_quality(self):
        output = format_human(_sample_metrics())
        assert "Recall Rate" in output
        assert "70/100" in output

    def test_contains_supersession_candidates(self):
        output = format_human(_sample_metrics())
        assert "Supersession Candidates" in output

    def test_contains_type_recall_correlation(self):
        output = format_human(_sample_metrics())
        assert "Type vs Recall" in output

    def test_no_feedback_message(self):
        metrics = _sample_metrics()
        metrics["feedback_alltime"] = {
            "total_feedback": 0,
            "helpful": 0,
            "not_helpful": 0,
            "unique_learnings_rated": 0,
            "helpfulness_rate_pct": 0.0,
        }
        output = format_human(metrics)
        assert "No feedback recorded" in output


# ---------------------------------------------------------------------------
# Backwards compatibility: re-exports from memory_metrics
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    def test_parse_period_reexported(self):
        assert parse_period_reexport is parse_period

    def test_format_human_reexported(self):
        assert format_human_reexport is format_human


# ---------------------------------------------------------------------------
# JSON output structure
# ---------------------------------------------------------------------------


class TestJSONStructure:
    def test_sample_metrics_is_valid_json(self):
        metrics = _sample_metrics()
        serialized = json.dumps(metrics)
        parsed = json.loads(serialized)
        assert parsed["version"] == _get_version()

    def test_all_top_level_keys_present(self):
        metrics = _sample_metrics()
        expected_keys = {
            "generated_at", "period", "totals", "per_session",
            "confidence_distribution", "classification_distribution",
            "dedup_stats_alltime", "embedding_coverage_alltime",
            "extraction_stats_alltime", "stale_learnings",
            "top_tags_alltime", "superseded_alltime", "temporal_alltime",
            "feedback_alltime", "feedback_velocity",
            "supersession_candidates", "recall_frequency",
            "type_recall_correlation", "version",
        }
        assert set(metrics.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Integration: collect_all_metrics against PostgreSQL
# ---------------------------------------------------------------------------


def _db_available() -> bool:
    """Check if PostgreSQL is reachable by attempting a real connection."""
    import socket

    try:
        sock = socket.create_connection(("localhost", 5432), timeout=2)
        sock.close()
        return True
    except (OSError, TimeoutError):
        return False


@pytest.mark.skipif(not _db_available(), reason="PostgreSQL not available")
class TestCollectMetrics:
    """Integration tests that hit the real database.

    These verify the SQL queries execute without error and return
    the expected shape. They do NOT insert test data -- they run
    against whatever is in the database.
    """

    @pytest.fixture(autouse=True)
    def _reset_pool(self):
        from scripts.core.db.postgres_pool import reset_pool
        reset_pool()

    @pytest.mark.asyncio
    async def test_collect_returns_all_keys(self):
        metrics = await collect_all_metrics()
        expected_keys = {
            "generated_at", "period", "totals", "per_session",
            "confidence_distribution", "classification_distribution",
            "dedup_stats_alltime", "embedding_coverage_alltime",
            "extraction_stats_alltime", "stale_learnings",
            "top_tags_alltime", "superseded_alltime", "temporal_alltime",
            "feedback_alltime", "feedback_velocity",
            "supersession_candidates", "recall_frequency",
            "type_recall_correlation", "version",
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
        assert isinstance(metrics["version"], str)

    @pytest.mark.asyncio
    async def test_top_tags_is_list(self):
        metrics = await collect_all_metrics()
        assert isinstance(metrics["top_tags_alltime"], list)
        if metrics["top_tags_alltime"]:
            assert "tag" in metrics["top_tags_alltime"][0]
            assert "count" in metrics["top_tags_alltime"][0]

    @pytest.mark.asyncio
    async def test_pct_fields_are_floats(self):
        metrics = await collect_all_metrics()
        for level_data in metrics["confidence_distribution"].values():
            assert isinstance(level_data["pct"], float)
        assert isinstance(
            metrics["extraction_stats_alltime"]["extraction_rate_pct"], float
        )
        assert isinstance(metrics["stale_learnings"]["never_recalled_pct"], float)
