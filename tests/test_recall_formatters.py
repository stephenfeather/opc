"""Tests for recall_formatters.py — pure output formatting functions.

Covers all public and private functions with AAA pattern.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from scripts.core.recall_formatters import (
    LEARNING_TYPE_ORDER,
    _build_json_result,
    _extract_learning_type,
    _extract_score,
    _format_created_at,
    _format_created_at_human,
    _format_result_line,
    format_human_output,
    format_json_full_output,
    format_json_output,
    format_result_preview,
    get_api_version,
    group_by_type,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    content: str = "Some learning content",
    similarity: float = 0.85,
    final_score: float | None = None,
    session_id: str = "test-session",
    created_at: datetime | str = "2026-01-15T10:30:00",
    learning_type: str = "WORKING_SOLUTION",
    result_id: str = "abc-123",
    rerank_details: dict | None = None,
    recall_count: int = 0,
    pattern_strength: float = 0.0,
    pattern_tags: list | None = None,
    kg_context: dict | None = None,
) -> dict:
    """Factory for building result dicts matching the recall pipeline shape."""
    result = {
        "id": result_id,
        "similarity": similarity,
        "session_id": session_id,
        "content": content,
        "created_at": created_at,
        "metadata": {"learning_type": learning_type},
        "recall_count": recall_count,
        "pattern_strength": pattern_strength,
        "pattern_tags": pattern_tags or [],
    }
    if final_score is not None:
        result["final_score"] = final_score
    if rerank_details is not None:
        result["rerank_details"] = rerank_details
    if kg_context is not None:
        result["kg_context"] = kg_context
    return result


# ---------------------------------------------------------------------------
# get_api_version
# ---------------------------------------------------------------------------

class TestGetApiVersion:
    def test_returns_string(self):
        version = get_api_version()
        assert isinstance(version, str)

    def test_returns_non_empty(self):
        version = get_api_version()
        assert len(version) > 0


# ---------------------------------------------------------------------------
# format_result_preview
# ---------------------------------------------------------------------------

class TestFormatResultPreview:
    def test_short_content_unchanged(self):
        content = "Short text"
        result = format_result_preview(content, max_length=200)
        assert result == "Short text"

    def test_exact_length_unchanged(self):
        content = "x" * 200
        result = format_result_preview(content, max_length=200)
        assert result == content
        assert "..." not in result

    def test_long_content_truncated(self):
        content = "x" * 250
        result = format_result_preview(content, max_length=200)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")

    def test_custom_max_length(self):
        content = "Hello world, this is a test"
        result = format_result_preview(content, max_length=5)
        assert result == "Hello..."

    def test_empty_string(self):
        result = format_result_preview("", max_length=200)
        assert result == ""

    def test_default_max_length_is_200(self):
        content = "x" * 201
        result = format_result_preview(content)
        assert result.endswith("...")
        assert len(result) == 203


# ---------------------------------------------------------------------------
# _format_created_at
# ---------------------------------------------------------------------------

class TestFormatCreatedAt:
    def test_datetime_to_isoformat(self):
        dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)
        result = _format_created_at(dt)
        assert result == "2026-01-15T10:30:00+00:00"

    def test_string_passthrough(self):
        result = _format_created_at("2026-01-15T10:30:00")
        assert result == "2026-01-15T10:30:00"

    def test_non_datetime_non_string_converted(self):
        result = _format_created_at(12345)
        assert result == "12345"

    def test_naive_datetime(self):
        dt = datetime(2026, 3, 1, 8, 0, 0)
        result = _format_created_at(dt)
        assert result == "2026-03-01T08:00:00"


# ---------------------------------------------------------------------------
# _format_created_at_human
# ---------------------------------------------------------------------------

class TestFormatCreatedAtHuman:
    def test_datetime_formatted(self):
        dt = datetime(2026, 3, 15, 14, 30, 0)
        result = _format_created_at_human(dt)
        assert result == "2026-03-15 14:30"

    def test_string_truncated_to_16_chars(self):
        result = _format_created_at_human("2026-01-15T10:30:00+00:00")
        assert result == "2026-01-15T10:30"

    def test_short_string_unchanged(self):
        result = _format_created_at_human("2026-01-15")
        assert result == "2026-01-15"


# ---------------------------------------------------------------------------
# _extract_learning_type
# ---------------------------------------------------------------------------

class TestExtractLearningType:
    def test_extracts_from_metadata(self):
        result = _make_result(learning_type="ERROR_FIX")
        assert _extract_learning_type(result) == "ERROR_FIX"

    def test_defaults_to_unknown_when_missing(self):
        result = _make_result()
        result["metadata"] = {}
        assert _extract_learning_type(result) == "UNKNOWN"

    def test_defaults_when_no_metadata_key(self):
        result = {"similarity": 0.5, "session_id": "s", "content": "c", "created_at": "d"}
        assert _extract_learning_type(result) == "UNKNOWN"

    def test_none_metadata_defaults_to_unknown(self):
        result = _make_result()
        result["metadata"] = None
        assert _extract_learning_type(result) == "UNKNOWN"

    def test_empty_string_learning_type_defaults_to_unknown(self):
        result = _make_result()
        result["metadata"] = {"learning_type": ""}
        assert _extract_learning_type(result) == "UNKNOWN"

    def test_none_learning_type_defaults_to_unknown(self):
        result = _make_result()
        result["metadata"] = {"learning_type": None}
        assert _extract_learning_type(result) == "UNKNOWN"


# ---------------------------------------------------------------------------
# _extract_score
# ---------------------------------------------------------------------------

class TestExtractScore:
    def test_prefers_final_score(self):
        result = _make_result(similarity=0.5, final_score=0.8)
        assert _extract_score(result) == 0.8

    def test_falls_back_to_similarity(self):
        result = _make_result(similarity=0.7)
        assert _extract_score(result) == 0.7

    def test_none_final_score_falls_back_to_similarity(self):
        result = _make_result(similarity=0.6)
        result["final_score"] = None
        assert _extract_score(result) == 0.6

    def test_returns_float(self):
        result = _make_result(similarity=1)
        assert isinstance(_extract_score(result), float)


# ---------------------------------------------------------------------------
# _format_result_line
# ---------------------------------------------------------------------------

class TestFormatResultLine:
    def test_basic_formatting(self):
        result = _make_result(
            content="A learning", similarity=0.85, final_score=0.9,
            session_id="sess-1", created_at="2026-01-15T10:30:00",
        )
        header, content = _format_result_line(1, result)
        assert header == "1. [0.900] Session: sess-1 (2026-01-15T10:30)"
        assert content == "   A learning"

    def test_default_content_indent_is_3_spaces(self):
        result = _make_result(content="test", similarity=0.5)
        _, content = _format_result_line(1, result)
        assert content.startswith("   ")  # 3 spaces

    def test_custom_content_indent(self):
        result = _make_result(content="test", similarity=0.5)
        _, content = _format_result_line(1, result, content_indent="     ")
        assert content == "     test"

    def test_with_indent(self):
        result = _make_result(content="test", similarity=0.5, session_id="s1")
        header, content = _format_result_line(
            3, result, indent="  ", content_indent="     "
        )
        assert header.startswith("  3.")
        assert content == "     test"

    def test_multiline_content_all_lines_indented(self):
        result = _make_result(content="line1\nline2\nline3", similarity=0.5)
        _, content = _format_result_line(1, result)
        lines = content.split("\n")
        assert all(line.startswith("   ") for line in lines)
        assert lines[0] == "   line1"
        assert lines[1] == "   line2"
        assert lines[2] == "   line3"

    def test_multiline_content_custom_indent(self):
        result = _make_result(content="a\nb", similarity=0.5)
        _, content = _format_result_line(1, result, content_indent="     ")
        lines = content.split("\n")
        assert lines[0] == "     a"
        assert lines[1] == "     b"

    def test_long_content_truncated(self):
        long_content = "x" * 400
        result = _make_result(content=long_content)
        _, content = _format_result_line(1, result)
        assert "..." in content


# ---------------------------------------------------------------------------
# _build_json_result
# ---------------------------------------------------------------------------

class TestBuildJsonResult:
    def test_basic_fields(self):
        raw = _make_result(content="test content", similarity=0.9, session_id="s1")
        result = _build_json_result(raw)

        assert result["id"] == "abc-123"
        assert result["raw_score"] == 0.9
        assert result["learning_type"] == "WORKING_SOLUTION"
        assert result["session_id"] == "s1"
        assert result["content"] == "test content"

    def test_score_uses_final_score_when_present(self):
        raw = _make_result(similarity=0.5, final_score=0.8)
        result = _build_json_result(raw)
        assert result["score"] == 0.8
        assert result["raw_score"] == 0.5

    def test_score_falls_back_to_similarity(self):
        raw = _make_result(similarity=0.7)
        result = _build_json_result(raw)
        assert result["score"] == 0.7

    def test_rerank_details_included_when_present(self):
        details = {"boost": 1.2, "reason": "recency"}
        raw = _make_result(rerank_details=details)
        result = _build_json_result(raw)
        assert result["rerank_details"] == details

    def test_rerank_details_absent_when_not_provided(self):
        raw = _make_result()
        result = _build_json_result(raw)
        assert "rerank_details" not in result

    def test_missing_metadata_defaults_to_unknown(self):
        raw = _make_result()
        raw["metadata"] = {}
        result = _build_json_result(raw)
        assert result["learning_type"] == "UNKNOWN"

    def test_missing_id_defaults_to_empty(self):
        raw = _make_result()
        del raw["id"]
        result = _build_json_result(raw)
        assert result["id"] == ""

    def test_datetime_created_at_serialized(self):
        dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)
        raw = _make_result(created_at=dt)
        result = _build_json_result(raw)
        assert result["created_at"] == "2026-01-15T10:30:00+00:00"


# ---------------------------------------------------------------------------
# format_json_output
# ---------------------------------------------------------------------------

class TestFormatJsonOutput:
    def test_empty_results(self):
        output = format_json_output([])
        parsed = json.loads(output)
        assert parsed["results"] == []
        assert parsed["total"] == 0
        assert "version" in parsed

    def test_single_result(self):
        results = [_make_result(content="learning 1")]
        output = format_json_output(results)
        parsed = json.loads(output)
        assert parsed["total"] == 1
        assert parsed["results"][0]["content"] == "learning 1"

    def test_multiple_results_preserve_order(self):
        results = [
            _make_result(content="first", similarity=0.9),
            _make_result(content="second", similarity=0.7),
        ]
        output = format_json_output(results)
        parsed = json.loads(output)
        assert parsed["results"][0]["content"] == "first"
        assert parsed["results"][1]["content"] == "second"

    def test_structured_adds_groups(self):
        results = [
            _make_result(learning_type="ERROR_FIX", content="fix 1"),
            _make_result(learning_type="WORKING_SOLUTION", content="sol 1"),
            _make_result(learning_type="ERROR_FIX", content="fix 2"),
        ]
        output = format_json_output(results, structured=True)
        parsed = json.loads(output)
        assert parsed["structured"] is True
        assert "groups" in parsed
        assert "ERROR_FIX" in parsed["groups"]
        assert len(parsed["groups"]["ERROR_FIX"]) == 2
        assert len(parsed["groups"]["WORKING_SOLUTION"]) == 1

    def test_non_structured_has_no_groups(self):
        results = [_make_result()]
        output = format_json_output(results, structured=False)
        parsed = json.loads(output)
        assert "groups" not in parsed
        assert "structured" not in parsed

    def test_output_is_valid_json(self):
        results = [_make_result(content='has "quotes" and \nnewlines')]
        output = format_json_output(results)
        parsed = json.loads(output)  # Should not raise
        assert parsed["results"][0]["content"] == 'has "quotes" and \nnewlines'

    def test_null_metadata_does_not_crash(self):
        result = _make_result()
        result["metadata"] = None
        output = format_json_output([result])
        parsed = json.loads(output)
        assert parsed["results"][0]["learning_type"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# format_json_full_output
# ---------------------------------------------------------------------------

class TestFormatJsonFullOutput:
    def test_includes_metadata(self):
        raw = _make_result(learning_type="ERROR_FIX")
        output = format_json_full_output([raw])
        parsed = json.loads(output)
        result = parsed["results"][0]
        assert "metadata" in result
        assert result["metadata"]["learning_type"] == "ERROR_FIX"

    def test_includes_recall_count(self):
        raw = _make_result(recall_count=5)
        output = format_json_full_output([raw])
        parsed = json.loads(output)
        assert parsed["results"][0]["recall_count"] == 5

    def test_includes_pattern_strength(self):
        raw = _make_result(pattern_strength=0.75)
        output = format_json_full_output([raw])
        parsed = json.loads(output)
        assert parsed["results"][0]["pattern_strength"] == 0.75

    def test_includes_pattern_tags(self):
        raw = _make_result(pattern_tags=["auth", "hooks"])
        output = format_json_full_output([raw])
        parsed = json.loads(output)
        assert parsed["results"][0]["pattern_tags"] == ["auth", "hooks"]

    def test_defaults_for_missing_fields(self):
        raw = _make_result()
        output = format_json_full_output([raw])
        parsed = json.loads(output)
        result = parsed["results"][0]
        assert result["recall_count"] == 0
        assert result["pattern_strength"] == 0.0
        assert result["pattern_tags"] == []

    def test_has_version(self):
        output = format_json_full_output([_make_result()])
        parsed = json.loads(output)
        assert "version" in parsed

    def test_null_metadata_normalized_to_empty_dict(self):
        raw = _make_result()
        raw["metadata"] = None
        output = format_json_full_output([raw])
        parsed = json.loads(output)
        assert parsed["results"][0]["metadata"] == {}


# ---------------------------------------------------------------------------
# format_human_output
# ---------------------------------------------------------------------------

class TestFormatHumanOutput:
    def test_empty_results(self):
        result = format_human_output([])
        assert result == "No matching learnings found."

    def test_single_result_flat(self):
        results = [_make_result(
            content="A learning",
            similarity=0.85,
            final_score=0.9,
            session_id="sess-1",
            created_at="2026-01-15T10:30:00",
        )]
        output = format_human_output(results)
        assert "1 matching learnings" in output
        assert "[0.900]" in output
        assert "sess-1" in output
        assert "A learning" in output

    def test_multiple_results_numbered(self):
        results = [
            _make_result(content="first", similarity=0.9),
            _make_result(content="second", similarity=0.7),
        ]
        output = format_human_output(results)
        assert "2 matching learnings" in output
        assert "1." in output
        assert "2." in output

    def test_structured_groups_by_type(self):
        results = [
            _make_result(learning_type="ERROR_FIX", content="fix"),
            _make_result(learning_type="WORKING_SOLUTION", content="sol"),
        ]
        output = format_human_output(results, structured=True)
        assert "## ERROR_FIX" in output
        assert "## WORKING_SOLUTION" in output
        assert "2 types" in output

    def test_datetime_created_at_formatted(self):
        dt = datetime(2026, 3, 15, 14, 30, 0)
        results = [_make_result(created_at=dt)]
        output = format_human_output(results)
        assert "2026-03-15 14:30" in output

    def test_string_created_at_truncated(self):
        results = [_make_result(created_at="2026-01-15T10:30:00+00:00")]
        output = format_human_output(results)
        assert "2026-01-15T10:30" in output

    def test_long_content_truncated_at_300(self):
        long_content = "x" * 400
        results = [_make_result(content=long_content)]
        output = format_human_output(results)
        # format_human_output uses max_length=300
        assert "..." in output

    def test_score_falls_back_to_similarity(self):
        results = [_make_result(similarity=0.456)]
        output = format_human_output(results)
        assert "[0.456]" in output

    def test_structured_continuous_numbering(self):
        results = [
            _make_result(learning_type="ERROR_FIX", content="fix1"),
            _make_result(learning_type="ERROR_FIX", content="fix2"),
            _make_result(learning_type="WORKING_SOLUTION", content="sol1"),
        ]
        output = format_human_output(results, structured=True)
        assert "  1." in output
        assert "  2." in output
        assert "  3." in output

    def test_null_metadata_does_not_crash(self):
        result = _make_result()
        result["metadata"] = None
        output = format_human_output([result])
        assert "UNKNOWN" not in output  # human output doesn't show type in flat mode
        assert "1 matching learnings" in output

    def test_multiline_content_indented_flat(self):
        results = [_make_result(content="line1\nline2")]
        output = format_human_output(results)
        lines = output.split("\n")
        content_lines = [line for line in lines if "line1" in line or "line2" in line]
        assert all(line.startswith("   ") for line in content_lines)

    def test_multiline_content_indented_structured(self):
        results = [_make_result(content="line1\nline2", learning_type="ERROR_FIX")]
        output = format_human_output(results, structured=True)
        lines = output.split("\n")
        content_lines = [line for line in lines if "line1" in line or "line2" in line]
        assert all(line.startswith("     ") for line in content_lines)

    def test_golden_flat_output(self):
        """Pin exact flat output format to prevent regressions."""
        results = [_make_result(
            content="Hook errors come from path issues",
            similarity=0.85,
            final_score=0.92,
            session_id="debug-hooks",
            created_at="2026-02-10T09:15:00",
        )]
        output = format_human_output(results)
        expected_lines = [
            "Found 1 matching learnings:",
            "",
            "1. [0.920] Session: debug-hooks (2026-02-10T09:15)",
            "   Hook errors come from path issues",
            "",
        ]
        assert output == "\n".join(expected_lines)

    def test_golden_structured_output(self):
        """Pin exact structured output format to prevent regressions."""
        results = [
            _make_result(
                learning_type="ERROR_FIX", content="Fix A",
                similarity=0.9, final_score=0.95,
                session_id="s1", created_at="2026-01-01T00:00:00",
            ),
            _make_result(
                learning_type="WORKING_SOLUTION", content="Sol B",
                similarity=0.8, final_score=0.85,
                session_id="s2", created_at="2026-01-02T00:00:00",
            ),
        ]
        output = format_human_output(results, structured=True)
        expected_lines = [
            "Found 2 matching learnings in 2 types:",
            "",
            "## ERROR_FIX (1)",
            "  1. [0.950] Session: s1 (2026-01-01T00:00)",
            "     Fix A",
            "",
            "## WORKING_SOLUTION (1)",
            "  2. [0.850] Session: s2 (2026-01-02T00:00)",
            "     Sol B",
            "",
        ]
        assert output == "\n".join(expected_lines)


# ---------------------------------------------------------------------------
# group_by_type
# ---------------------------------------------------------------------------

class TestGroupByType:
    def test_empty_results(self):
        result = group_by_type([])
        assert result == {}

    def test_single_type(self):
        results = [
            _make_result(learning_type="ERROR_FIX", content="a"),
            _make_result(learning_type="ERROR_FIX", content="b"),
        ]
        grouped = group_by_type(results)
        assert list(grouped.keys()) == ["ERROR_FIX"]
        assert len(grouped["ERROR_FIX"]) == 2

    def test_canonical_ordering(self):
        results = [
            _make_result(learning_type="WORKING_SOLUTION"),
            _make_result(learning_type="FAILED_APPROACH"),
            _make_result(learning_type="USER_PREFERENCE"),
        ]
        grouped = group_by_type(results)
        keys = list(grouped.keys())
        assert keys == ["FAILED_APPROACH", "WORKING_SOLUTION", "USER_PREFERENCE"]

    def test_unknown_types_appended_alphabetically(self):
        results = [
            _make_result(learning_type="WORKING_SOLUTION"),
            _make_result(learning_type="ZEBRA_TYPE"),
            _make_result(learning_type="ALPHA_TYPE"),
        ]
        grouped = group_by_type(results)
        keys = list(grouped.keys())
        assert keys == ["WORKING_SOLUTION", "ALPHA_TYPE", "ZEBRA_TYPE"]

    def test_preserves_relevance_order_within_group(self):
        results = [
            _make_result(learning_type="ERROR_FIX", content="first", similarity=0.9),
            _make_result(learning_type="ERROR_FIX", content="second", similarity=0.7),
        ]
        grouped = group_by_type(results)
        assert grouped["ERROR_FIX"][0]["content"] == "first"
        assert grouped["ERROR_FIX"][1]["content"] == "second"

    def test_missing_metadata_defaults_to_unknown(self):
        result = _make_result()
        result["metadata"] = {}
        grouped = group_by_type([result])
        assert "UNKNOWN" in grouped

    def test_all_canonical_types_ordered_correctly(self):
        """Verify LEARNING_TYPE_ORDER matches expected canonical order."""
        assert LEARNING_TYPE_ORDER == (
            "FAILED_APPROACH",
            "ERROR_FIX",
            "WORKING_SOLUTION",
            "ARCHITECTURAL_DECISION",
            "CODEBASE_PATTERN",
            "USER_PREFERENCE",
            "OPEN_THREAD",
        )

    def test_learning_type_order_is_immutable(self):
        assert isinstance(LEARNING_TYPE_ORDER, tuple)

    def test_does_not_mutate_input(self):
        results = [
            _make_result(learning_type="ERROR_FIX"),
            _make_result(learning_type="WORKING_SOLUTION"),
        ]
        original = [r.copy() for r in results]
        group_by_type(results)
        assert results[0] == original[0]
        assert results[1] == original[1]


# ---------------------------------------------------------------------------
# KG context serialization (Phase 3 fix for D2/E1 finding)
# ---------------------------------------------------------------------------


class TestKGContextSerialization:
    _KG_CTX = {
        "entities": [
            {"id": "e1", "name": "pytest", "type": "tool", "mention_count": 5}
        ],
        "edges": [
            {"source": "pytest", "target": "asyncpg", "relation": "used_with",
             "weight": 2.0}
        ],
    }

    def test_build_json_result_includes_kg_context(self):
        r = _make_result(kg_context=self._KG_CTX)
        out = _build_json_result(r)
        assert out["kg_context"] == self._KG_CTX

    def test_build_json_result_omits_kg_context_when_absent(self):
        r = _make_result()
        out = _build_json_result(r)
        assert "kg_context" not in out

    def test_format_json_output_round_trips_kg_context(self):
        r = _make_result(kg_context=self._KG_CTX)
        payload = json.loads(format_json_output([r]))
        assert payload["results"][0]["kg_context"] == self._KG_CTX

    def test_format_json_full_output_round_trips_kg_context(self):
        r = _make_result(kg_context=self._KG_CTX)
        payload = json.loads(format_json_full_output([r]))
        assert payload["results"][0]["kg_context"] == self._KG_CTX

    def test_format_json_output_no_kg_key_when_absent(self):
        r = _make_result()
        payload = json.loads(format_json_output([r]))
        assert "kg_context" not in payload["results"][0]
