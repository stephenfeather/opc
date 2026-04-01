"""Tests for structured recall output (group_by_type and --structured flag).

Validates that:
1. group_by_type preserves relevance ordering within each group
2. group_by_type applies canonical type ordering (FAILED_APPROACH first, OPEN_THREAD last)
3. group_by_type handles unknown types (appended alphabetically at end)
4. group_by_type handles empty input
5. JSON structured output has correct shape: {"structured": true, "groups": {...}, "total": N}
6. JSON flat output is unchanged when --structured is not set
7. Human-readable structured output has type headers
8. --structured flag does not affect search behavior (purely post-processing)
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.recall_formatters import (
    LEARNING_TYPE_ORDER,
    format_human_output,
    format_json_output,
    group_by_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    similarity: float = 0.5,
    content: str = "test learning",
    session_id: str = "test-session",
    created_at: datetime | None = None,
    result_id: str | None = None,
    learning_type: str = "WORKING_SOLUTION",
) -> dict:
    """Create a fake recall result dict with learning_type in metadata."""
    return {
        "id": result_id or str(uuid.uuid4()),
        "similarity": similarity,
        "content": content,
        "session_id": session_id,
        "created_at": created_at or datetime(2026, 3, 29, 12, 0, tzinfo=UTC),
        "metadata": {"learning_type": learning_type},
    }


# ---------------------------------------------------------------------------
# group_by_type tests
# ---------------------------------------------------------------------------


class TestGroupByType:
    """Tests for group_by_type function."""

    def test_preserves_relevance_order_within_group(self):
        """Results within each group keep their original ranking."""
        results = [
            _make_result(similarity=0.9, content="best", learning_type="WORKING_SOLUTION"),
            _make_result(similarity=0.7, content="good", learning_type="WORKING_SOLUTION"),
            _make_result(similarity=0.5, content="ok", learning_type="WORKING_SOLUTION"),
        ]
        grouped = group_by_type(results)
        ws = grouped["WORKING_SOLUTION"]
        assert [r["content"] for r in ws] == ["best", "good", "ok"]

    def test_canonical_type_ordering(self):
        """Types appear in LEARNING_TYPE_ORDER: FAILED_APPROACH first, OPEN_THREAD last."""
        results = [
            _make_result(learning_type="OPEN_THREAD"),
            _make_result(learning_type="WORKING_SOLUTION"),
            _make_result(learning_type="FAILED_APPROACH"),
            _make_result(learning_type="ERROR_FIX"),
        ]
        grouped = group_by_type(results)
        keys = list(grouped.keys())
        assert keys == ["FAILED_APPROACH", "ERROR_FIX", "WORKING_SOLUTION", "OPEN_THREAD"]

    def test_unknown_types_appended_alphabetically(self):
        """Types not in LEARNING_TYPE_ORDER are appended after known types, sorted."""
        results = [
            _make_result(learning_type="WORKING_SOLUTION"),
            _make_result(learning_type="ZEBRA_TYPE"),
            _make_result(learning_type="ALPHA_TYPE"),
        ]
        grouped = group_by_type(results)
        keys = list(grouped.keys())
        assert keys == ["WORKING_SOLUTION", "ALPHA_TYPE", "ZEBRA_TYPE"]

    def test_empty_results(self):
        """Empty input returns empty dict."""
        grouped = group_by_type([])
        assert grouped == {}

    def test_missing_metadata_defaults_to_unknown(self):
        """Results without metadata.learning_type are grouped as UNKNOWN."""
        results = [
            {"id": "1", "similarity": 0.5, "content": "no metadata",
             "session_id": "s1", "created_at": datetime(2026, 1, 1, tzinfo=UTC),
             "metadata": {}},
        ]
        grouped = group_by_type(results)
        assert "UNKNOWN" in grouped
        assert len(grouped["UNKNOWN"]) == 1

    def test_all_canonical_types_in_order(self):
        """When all canonical types present, they appear in exact LEARNING_TYPE_ORDER."""
        results = [_make_result(learning_type=lt) for lt in reversed(LEARNING_TYPE_ORDER)]
        grouped = group_by_type(results)
        assert list(grouped.keys()) == LEARNING_TYPE_ORDER


# ---------------------------------------------------------------------------
# format_json_output tests
# ---------------------------------------------------------------------------


class TestFormatJsonOutput:
    """Tests for JSON output formatting."""

    def test_flat_output_shape(self):
        """Default (non-structured) output has {"results": [...], "total": N} shape."""
        results = [_make_result(learning_type="WORKING_SOLUTION")]
        output = json.loads(format_json_output(results, structured=False))
        assert "results" in output
        assert isinstance(output["results"], list)
        assert output["total"] == 1
        assert "groups" not in output
        assert "structured" not in output

    def test_structured_output_shape(self):
        """Structured output has consistent envelope: results + total + groups."""
        results = [
            _make_result(learning_type="WORKING_SOLUTION"),
            _make_result(learning_type="FAILED_APPROACH"),
        ]
        output = json.loads(format_json_output(results, structured=True))
        # Envelope is consistent — always has results and total
        assert "results" in output
        assert isinstance(output["results"], list)
        assert output["total"] == 2
        # Structured adds groups and structured flag
        assert output["structured"] is True
        assert "groups" in output

    def test_structured_groups_contain_correct_results(self):
        """Each group in structured output contains the right results."""
        results = [
            _make_result(content="sol1", learning_type="WORKING_SOLUTION"),
            _make_result(content="fail1", learning_type="FAILED_APPROACH"),
            _make_result(content="sol2", learning_type="WORKING_SOLUTION"),
        ]
        output = json.loads(format_json_output(results, structured=True))
        groups = output["groups"]
        assert len(groups["WORKING_SOLUTION"]) == 2
        assert len(groups["FAILED_APPROACH"]) == 1
        assert groups["WORKING_SOLUTION"][0]["content"] == "sol1"

    def test_structured_results_flat_list_matches_total(self):
        """Structured output still has flat results list with all items."""
        results = [
            _make_result(content="sol1", learning_type="WORKING_SOLUTION"),
            _make_result(content="fail1", learning_type="FAILED_APPROACH"),
        ]
        output = json.loads(format_json_output(results, structured=True))
        assert len(output["results"]) == output["total"] == 2

    def test_json_result_includes_learning_type(self):
        """Each JSON result includes learning_type field."""
        results = [_make_result(learning_type="ERROR_FIX")]
        output = json.loads(format_json_output(results))
        assert output["results"][0]["learning_type"] == "ERROR_FIX"

    def test_json_result_includes_id(self):
        """Each JSON result includes id field."""
        results = [_make_result(result_id="test-id-123")]
        output = json.loads(format_json_output(results))
        assert output["results"][0]["id"] == "test-id-123"


# ---------------------------------------------------------------------------
# format_human_output tests
# ---------------------------------------------------------------------------


class TestFormatHumanOutput:
    """Tests for human-readable output formatting."""

    def test_structured_has_type_headers(self):
        """Structured human output has ## TYPE_NAME headers."""
        results = [
            _make_result(content="a solution", learning_type="WORKING_SOLUTION"),
            _make_result(content="a failure", learning_type="FAILED_APPROACH"),
        ]
        output = format_human_output(results, structured=True)
        assert "## WORKING_SOLUTION" in output
        assert "## FAILED_APPROACH" in output

    def test_structured_headers_in_canonical_order(self):
        """Type headers in structured output follow canonical ordering."""
        results = [
            _make_result(learning_type="OPEN_THREAD"),
            _make_result(learning_type="FAILED_APPROACH"),
            _make_result(learning_type="WORKING_SOLUTION"),
        ]
        output = format_human_output(results, structured=True)
        fa_pos = output.index("## FAILED_APPROACH")
        ws_pos = output.index("## WORKING_SOLUTION")
        ot_pos = output.index("## OPEN_THREAD")
        assert fa_pos < ws_pos < ot_pos

    def test_flat_output_no_type_headers(self):
        """Non-structured output does NOT have type headers."""
        results = [_make_result(learning_type="WORKING_SOLUTION")]
        output = format_human_output(results, structured=False)
        assert "## WORKING_SOLUTION" not in output

    def test_structured_shows_total_and_type_count(self):
        """Structured output header shows total results and type count."""
        results = [
            _make_result(learning_type="WORKING_SOLUTION"),
            _make_result(learning_type="FAILED_APPROACH"),
            _make_result(learning_type="FAILED_APPROACH"),
        ]
        output = format_human_output(results, structured=True)
        assert "3 matching learnings in 2 types" in output

    def test_empty_results_message(self):
        """Empty results returns appropriate message."""
        output = format_human_output([], structured=True)
        assert "No matching learnings found" in output

    def test_structured_count_per_type(self):
        """Each type header shows the count of results in that type."""
        results = [
            _make_result(learning_type="WORKING_SOLUTION"),
            _make_result(learning_type="WORKING_SOLUTION"),
            _make_result(learning_type="FAILED_APPROACH"),
        ]
        output = format_human_output(results, structured=True)
        assert "## WORKING_SOLUTION (2)" in output
        assert "## FAILED_APPROACH (1)" in output
