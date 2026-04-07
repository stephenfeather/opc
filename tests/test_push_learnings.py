"""Tests for proactive memory push (push_learnings.py).

Validates that:
1. merge_candidates deduplicates and prioritizes pattern reps
2. truncate_content handles edge cases correctly
3. format_results builds expected JSON structure
4. get_push_candidates integrates queries and merge
5. _row_to_dict converts asyncpg Records correctly
6. build_stale_query / build_pattern_query return correct SQL params
7. parse_args builds CLI config from argv
8. build_cli_output produces text/JSON output
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from scripts.core.push_learnings import (
    _row_to_dict,
    build_cli_output,
    build_pattern_query_params,
    build_stale_query_params,
    format_results,
    get_push_candidates,
    merge_candidates,
    parse_args,
    parse_pattern_row,
    truncate_content,
    write_cache_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    *,
    id: str = "aaaa-bbbb-cccc-dddd",
    content: str = "Test learning content",
    learning_type: str = "WORKING_SOLUTION",
    confidence: str = "high",
    pattern_label: str | None = None,
    pattern_confidence: float | None = None,
    recall_count: int = 0,
    created_at: datetime | None = None,
) -> dict:
    """Build a minimal candidate dict for testing."""
    return {
        "id": id,
        "content": content,
        "metadata": {
            "type": "session_learning",
            "learning_type": learning_type,
            "confidence": confidence,
        },
        "created_at": created_at or datetime(2026, 3, 15, tzinfo=UTC),
        "recall_count": recall_count,
        "learning_type": learning_type,
        "confidence": confidence,
        "pattern_label": pattern_label,
        "pattern_confidence": pattern_confidence,
    }


_UNSET = object()


def _make_row(
    *,
    id: str = "aaaa-bbbb-cccc-dddd",
    content: str = "Test learning content",
    metadata: Any = _UNSET,
    created_at: datetime | None = None,
    recall_count: int = 0,
    pattern_label: str | None = None,
    pattern_confidence: float | None = None,
    pattern_type: str | None = None,
) -> Mapping[str, Any]:
    """Build a mock asyncpg-like row (Mapping)."""
    if metadata is _UNSET:
        metadata = {
            "type": "session_learning",
            "learning_type": "WORKING_SOLUTION",
            "confidence": "high",
        }
    row: dict[str, Any] = {
        "id": id,
        "content": content,
        "metadata": metadata,
        "created_at": created_at or datetime(2026, 3, 15, tzinfo=UTC),
        "recall_count": recall_count,
    }
    if pattern_label is not None:
        row["pattern_label"] = pattern_label
    if pattern_confidence is not None:
        row["pattern_confidence"] = pattern_confidence
    if pattern_type is not None:
        row["pattern_type"] = pattern_type
    return row


# ---------------------------------------------------------------------------
# _row_to_dict
# ---------------------------------------------------------------------------


class TestRowToDict:
    def test_basic_conversion(self):
        row = _make_row()
        result = _row_to_dict(row)
        assert result["id"] == "aaaa-bbbb-cccc-dddd"
        assert result["content"] == "Test learning content"
        assert result["learning_type"] == "WORKING_SOLUTION"
        assert result["confidence"] == "high"
        assert result["recall_count"] == 0
        assert result["pattern_label"] is None
        assert result["pattern_confidence"] is None

    def test_metadata_as_json_string(self):
        meta_str = json.dumps({"learning_type": "ERROR_FIX", "confidence": "low"})
        row = _make_row(metadata=meta_str)
        result = _row_to_dict(row)
        assert result["learning_type"] == "ERROR_FIX"
        assert result["confidence"] == "low"
        assert isinstance(result["metadata"], dict)

    def test_missing_metadata_keys_use_defaults(self):
        row = _make_row(metadata={"type": "session_learning"})
        result = _row_to_dict(row)
        assert result["learning_type"] == "UNKNOWN"
        assert result["confidence"] == "medium"

    def test_null_recall_count_defaults_to_zero(self):
        row = dict(_make_row())
        row["recall_count"] = None
        result = _row_to_dict(row)
        assert result["recall_count"] == 0

    def test_id_converted_to_string(self):
        import uuid

        uid = uuid.uuid4()
        row = _make_row(id=uid)
        result = _row_to_dict(row)
        assert result["id"] == str(uid)

    def test_malformed_json_metadata_defaults_to_empty(self):
        row = _make_row(metadata="{invalid json")
        result = _row_to_dict(row)
        assert result["metadata"] == {}
        assert result["learning_type"] == "UNKNOWN"
        assert result["confidence"] == "medium"

    def test_null_metadata_defaults_to_empty(self):
        row = _make_row(metadata=None)
        result = _row_to_dict(row)
        assert result["metadata"] == {}
        assert result["learning_type"] == "UNKNOWN"

    def test_list_metadata_defaults_to_empty(self):
        row = _make_row(metadata=["not", "a", "dict"])
        result = _row_to_dict(row)
        assert result["metadata"] == {}

    def test_numeric_metadata_defaults_to_empty(self):
        row = _make_row(metadata=42)
        result = _row_to_dict(row)
        assert result["metadata"] == {}


# ---------------------------------------------------------------------------
# parse_pattern_row
# ---------------------------------------------------------------------------


class TestParsePatternRow:
    def test_adds_pattern_fields(self):
        row = _make_row(
            pattern_label="hook errors",
            pattern_confidence=0.85,
        )
        result = parse_pattern_row(row)
        assert result["pattern_label"] == "hook errors"
        assert result["pattern_confidence"] == 0.85

    def test_none_pattern_confidence(self):
        row = _make_row(pattern_label="test")
        # No pattern_confidence key at all — .get returns None
        result = parse_pattern_row(row)
        assert result["pattern_confidence"] is None

    def test_preserves_base_fields(self):
        row = _make_row(
            id="pat-id",
            content="Pattern content",
            pattern_label="test",
            pattern_confidence=0.5,
        )
        result = parse_pattern_row(row)
        assert result["id"] == "pat-id"
        assert result["content"] == "Pattern content"


# ---------------------------------------------------------------------------
# build_stale_query_params / build_pattern_query_params
# ---------------------------------------------------------------------------


class TestBuildStaleQueryParams:
    def test_returns_project_and_k(self):
        sql, params = build_stale_query_params("opc", 5)
        assert "opc" in params
        assert 5 in params
        assert "archival_memory" in sql
        assert "recall_count = 0" in sql

    def test_different_project(self):
        sql, params = build_stale_query_params("myproject", 10)
        assert "myproject" in params
        assert 10 in params


class TestBuildPatternQueryParams:
    def test_returns_k(self):
        sql, params = build_pattern_query_params(3)
        assert 3 in params
        assert "detected_patterns" in sql
        assert "recall_count = 0" in sql


# ---------------------------------------------------------------------------
# merge_candidates
# ---------------------------------------------------------------------------


class TestMergeCandidates:
    def test_pattern_reps_come_first(self):
        pattern = _make_candidate(id="pat-1", pattern_label="hook errors")
        stale = _make_candidate(id="stale-1")
        result = merge_candidates([pattern], [stale], k=5)
        assert len(result) == 2
        assert result[0]["id"] == "pat-1"
        assert result[1]["id"] == "stale-1"

    def test_dedup_by_id(self):
        shared = _make_candidate(id="shared-id", pattern_label="from pattern")
        stale_dup = _make_candidate(id="shared-id")
        stale_unique = _make_candidate(id="unique-id")
        result = merge_candidates([shared], [stale_dup, stale_unique], k=5)
        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert ids == ["shared-id", "unique-id"]
        # Should keep the pattern version (first seen)
        assert result[0]["pattern_label"] == "from pattern"

    def test_cap_at_k(self):
        candidates = [_make_candidate(id=f"id-{i}") for i in range(10)]
        result = merge_candidates([], candidates, k=3)
        assert len(result) == 3

    def test_empty_inputs(self):
        assert merge_candidates([], [], k=5) == []

    def test_k_zero(self):
        stale = _make_candidate(id="stale-1")
        result = merge_candidates([], [stale], k=0)
        assert result == []

    def test_pattern_only(self):
        patterns = [_make_candidate(id=f"pat-{i}", pattern_label=f"p{i}") for i in range(3)]
        result = merge_candidates(patterns, [], k=5)
        assert len(result) == 3

    def test_stale_only(self):
        stales = [_make_candidate(id=f"st-{i}") for i in range(3)]
        result = merge_candidates([], stales, k=5)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# truncate_content
# ---------------------------------------------------------------------------


class TestTruncateContent:
    def test_short_content_unchanged(self):
        assert truncate_content("hello world", 150) == "hello world"

    def test_long_content_truncated(self):
        long = "x" * 200
        result = truncate_content(long, 150)
        assert len(result) == 153  # 150 + "..."
        assert result.endswith("...")

    def test_multiline_collapsed(self):
        multi = "line one\nline two\nline three"
        result = truncate_content(multi, 150)
        assert "\n" not in result
        assert result == "line one line two line three"

    def test_empty_lines_stripped(self):
        with_blanks = "first\n\n\nsecond\n\nthird"
        result = truncate_content(with_blanks, 150)
        assert result == "first second third"

    def test_exact_boundary(self):
        exact = "x" * 150
        result = truncate_content(exact, 150)
        assert result == exact  # no "..." appended

    def test_empty_content(self):
        assert truncate_content("", 150) == ""

    def test_whitespace_only(self):
        assert truncate_content("   \n  \n  ", 150) == ""


# ---------------------------------------------------------------------------
# format_results
# ---------------------------------------------------------------------------


class TestFormatResults:
    def test_basic_structure(self):
        candidates = [_make_candidate(id="test-id-1")]
        output = format_results(candidates, "opc", 150)
        assert output["push_source"] == "session_start"
        assert output["project"] == "opc"
        assert len(output["results"]) == 1
        assert "version" in output

    def test_result_fields(self):
        c = _make_candidate(
            id="abc-123",
            content="Some learning",
            learning_type="FAILED_APPROACH",
            confidence="high",
            pattern_label="hook errors",
        )
        output = format_results([c], "opc", 150)
        r = output["results"][0]
        assert r["id"] == "abc-123"
        assert r["learning_type"] == "FAILED_APPROACH"
        assert r["confidence"] == "high"
        assert r["pattern_label"] == "hook errors"
        assert "created_at" in r

    def test_content_truncated_in_output(self):
        c = _make_candidate(content="x" * 300)
        output = format_results([c], "opc", 100)
        assert len(output["results"][0]["content"]) == 103  # 100 + "..."

    def test_empty_candidates(self):
        output = format_results([], "opc", 150)
        assert output["results"] == []

    def test_pattern_label_null_when_absent(self):
        c = _make_candidate()
        output = format_results([c], "opc", 150)
        assert output["results"][0]["pattern_label"] is None


# ---------------------------------------------------------------------------
# write_cache_file
# ---------------------------------------------------------------------------


class TestWriteCacheFile:
    def test_writes_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        data = {"results": [{"id": "test"}]}
        write_cache_file(data)
        cache_file = tmp_path / ".claude" / "cache" / "memory-push.json"
        assert cache_file.exists()
        loaded = json.loads(cache_file.read_text())
        assert loaded["results"][0]["id"] == "test"

    def test_creates_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        write_cache_file({"results": []})
        assert (tmp_path / ".claude" / "cache").is_dir()


# ---------------------------------------------------------------------------
# parse_args (pure)
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        config = parse_args(["--project", "opc", "--json"])
        assert config["project"] == "opc"
        assert config["k"] == 5
        assert config["json_output"] is True
        assert config["no_record"] is False
        assert config["max_chars"] == 150

    def test_custom_values(self):
        config = parse_args([
            "--project", "myproj",
            "--k", "10",
            "--max-chars", "200",
            "--no-record",
        ])
        assert config["project"] == "myproj"
        assert config["k"] == 10
        assert config["max_chars"] == 200
        assert config["no_record"] is True

    def test_project_from_env(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/Users/test/myproject")
        config = parse_args([])
        assert config["project"] == "myproject"

    def test_no_project_returns_none(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        config = parse_args([])
        assert config["project"] is None

    def test_negative_k_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--project", "opc", "--k", "-1"])

    def test_negative_max_chars_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--project", "opc", "--max-chars", "-5"])


# ---------------------------------------------------------------------------
# build_cli_output (pure)
# ---------------------------------------------------------------------------


class TestBuildCliOutput:
    def test_json_output(self):
        candidates = [_make_candidate(id="test-1")]
        result = build_cli_output(candidates, "opc", max_chars=150, json_output=True)
        parsed = json.loads(result)
        assert parsed["project"] == "opc"
        assert len(parsed["results"]) == 1

    def test_text_output(self):
        candidates = [_make_candidate(id="test-1", content="Hello world")]
        result = build_cli_output(candidates, "opc", max_chars=150, json_output=False)
        assert "Hello world" in result
        assert "opc" in result

    def test_pattern_label_in_text(self):
        candidates = [
            _make_candidate(id="p-1", content="Fix hooks", pattern_label="hook errors")
        ]
        result = build_cli_output(candidates, "opc", max_chars=150, json_output=False)
        assert "hook errors" in result

    def test_empty_candidates_json(self):
        result = build_cli_output([], "opc", max_chars=150, json_output=True)
        parsed = json.loads(result)
        assert parsed["results"] == []

    def test_empty_candidates_text(self):
        result = build_cli_output([], "opc", max_chars=150, json_output=False)
        assert result == ""


# ---------------------------------------------------------------------------
# get_push_candidates (integration with mocked DB)
# ---------------------------------------------------------------------------


def _make_pool_mock() -> AsyncMock:
    """Create an AsyncMock pool that supports `async with pool.acquire() as conn`."""
    conn = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool = AsyncMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


class TestGetPushCandidates:
    @pytest.mark.asyncio
    async def test_returns_empty_for_sqlite(self):
        with patch(
            "scripts.core.push_learnings.get_backend", return_value="sqlite"
        ):
            result = await get_push_candidates("opc", k=5)
            assert result == []

    @pytest.mark.asyncio
    async def test_merges_pattern_and_stale(self):
        stale = [_make_candidate(id="stale-1", content="Stale learning")]
        patterns = [
            _make_candidate(
                id="pat-1", content="Pattern learning",
                confidence="medium", learning_type="FAILED_APPROACH",
                pattern_label="hook errors", pattern_confidence=0.85,
            ),
        ]

        with (
            patch("scripts.core.push_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.push_learnings.get_stale_learnings",
                new_callable=AsyncMock, return_value=stale,
            ),
            patch(
                "scripts.core.push_learnings.get_pattern_representatives",
                new_callable=AsyncMock, return_value=patterns,
            ),
            patch(
                "scripts.core.push_learnings.get_pool",
                new_callable=AsyncMock, return_value=_make_pool_mock(),
            ),
        ):
            result = await get_push_candidates("opc", k=5)

        assert len(result) == 2
        # Pattern rep should be first
        assert result[0]["id"] == "pat-1"
        assert result[1]["id"] == "stale-1"

    @pytest.mark.asyncio
    async def test_pattern_table_missing_graceful(self):
        stale = [_make_candidate(id="stale-1", content="Stale learning")]

        with (
            patch("scripts.core.push_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.push_learnings.get_stale_learnings",
                new_callable=AsyncMock, return_value=stale,
            ),
            patch(
                "scripts.core.push_learnings.get_pattern_representatives",
                new_callable=AsyncMock,
                side_effect=asyncpg.exceptions.UndefinedTableError("relation does not exist"),
            ),
            patch(
                "scripts.core.push_learnings.get_pool",
                new_callable=AsyncMock, return_value=_make_pool_mock(),
            ),
        ):
            result = await get_push_candidates("opc", k=5)

        assert len(result) == 1
        assert result[0]["id"] == "stale-1"

    @pytest.mark.asyncio
    async def test_no_results(self):
        with (
            patch("scripts.core.push_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.push_learnings.get_stale_learnings",
                new_callable=AsyncMock, return_value=[],
            ),
            patch(
                "scripts.core.push_learnings.get_pattern_representatives",
                new_callable=AsyncMock, return_value=[],
            ),
            patch(
                "scripts.core.push_learnings.get_pool",
                new_callable=AsyncMock, return_value=_make_pool_mock(),
            ),
        ):
            result = await get_push_candidates("opc", k=5)

        assert result == []
