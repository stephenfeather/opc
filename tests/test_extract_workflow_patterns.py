"""Tests for extract_workflow_patterns.py.

Validates that:
1. _extract_input extracts correct fields per tool type
2. _matches_step handles string and tuple patterns
3. detect_workflow_sequences finds patterns in tool use sequences
4. summarize_tool_usage computes correct frequency and file stats
5. format_pattern_as_learning formats patterns for human display
6. extract_tool_uses parses JSONL and correlates tool_use with tool_result
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.extract_workflow_patterns import (  # noqa: E402
    _build_pattern_result,
    _collect_files,
    _determine_success,
    _extract_input,
    _match_pattern_at,
    _matches_step,
    _parse_tool_result,
    _parse_tool_use_entry,
    detect_workflow_sequences,
    extract_tool_uses,
    format_pattern_as_learning,
    summarize_tool_usage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use(
    tool_name: str,
    input_data: dict | None = None,
    result_error: bool | None = None,
    timestamp: str | None = "2026-04-03T10:00:00Z",
) -> dict:
    """Build a minimal tool_use dict matching extract_tool_uses output shape."""
    return {
        "tool_name": tool_name,
        "input": input_data or {},
        "timestamp": timestamp,
        "result_error": result_error,
    }


def _write_jsonl(lines: list[dict], path: Path) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


def _make_assistant_tool_use(
    tool_name: str,
    tool_input: dict,
    tool_id: str = "tu_1",
    timestamp: str = "2026-04-03T10:00:00Z",
) -> dict:
    """Build a JSONL line for an assistant message containing a tool_use."""
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": tool_name,
                    "input": tool_input,
                    "id": tool_id,
                }
            ]
        },
    }


def _make_user_tool_result(
    tool_id: str = "tu_1",
    is_error: bool = False,
) -> dict:
    """Build a JSONL line for a user message containing a tool_result."""
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": is_error,
                }
            ]
        },
    }


# ===========================================================================
# _extract_input
# ===========================================================================


class TestExtractInput:
    """Tests for _extract_input: extracts relevant fields per tool type."""

    def test_read_tool_extracts_file_path(self) -> None:
        result = _extract_input("Read", {"file_path": "/tmp/foo.py", "limit": 100})
        assert result == {"file_path": "/tmp/foo.py"}

    def test_edit_tool_extracts_file_path(self) -> None:
        result = _extract_input("Edit", {"file_path": "/tmp/bar.py", "old_string": "x"})
        assert result == {"file_path": "/tmp/bar.py"}

    def test_write_tool_extracts_file_path(self) -> None:
        result = _extract_input("Write", {"file_path": "/tmp/baz.py", "content": "abc"})
        assert result == {"file_path": "/tmp/baz.py"}

    def test_multiedit_tool_extracts_file_path(self) -> None:
        result = _extract_input("MultiEdit", {"file_path": "/tmp/multi.py"})
        assert result == {"file_path": "/tmp/multi.py"}

    def test_bash_extracts_command_truncated(self) -> None:
        long_cmd = "x" * 1000
        result = _extract_input("Bash", {"command": long_cmd})
        assert result == {"command": long_cmd[:500]}

    def test_bash_empty_command(self) -> None:
        result = _extract_input("Bash", {})
        assert result == {"command": ""}

    def test_grep_extracts_pattern_and_path(self) -> None:
        result = _extract_input("Grep", {"pattern": "foo", "path": "/src"})
        assert result == {"pattern": "foo", "path": "/src"}

    def test_glob_extracts_pattern_and_path(self) -> None:
        result = _extract_input("Glob", {"pattern": "**/*.py", "path": "/src"})
        assert result == {"pattern": "**/*.py", "path": "/src"}

    def test_agent_extracts_subagent_and_description(self) -> None:
        result = _extract_input(
            "Agent", {"subagent_type": "scout", "description": "find stuff", "prompt": "long..."}
        )
        assert result == {"subagent_type": "scout", "description": "find stuff"}

    def test_unknown_tool_returns_empty(self) -> None:
        result = _extract_input("SomeNewTool", {"foo": "bar"})
        assert result == {}

    def test_missing_fields_use_defaults(self) -> None:
        result = _extract_input("Read", {})
        assert result == {"file_path": ""}

        result = _extract_input("Grep", {})
        assert result == {"pattern": "", "path": ""}

        result = _extract_input("Agent", {})
        assert result == {"subagent_type": "", "description": ""}


# ===========================================================================
# _matches_step
# ===========================================================================


class TestMatchesStep:
    """Tests for _matches_step: matches tool names against pattern steps."""

    def test_string_match(self) -> None:
        assert _matches_step("Bash", "Bash") is True

    def test_string_no_match(self) -> None:
        assert _matches_step("Read", "Bash") is False

    def test_tuple_match(self) -> None:
        assert _matches_step("Edit", ("Edit", "Write")) is True
        assert _matches_step("Write", ("Edit", "Write")) is True

    def test_tuple_no_match(self) -> None:
        assert _matches_step("Read", ("Edit", "Write")) is False

    def test_empty_string(self) -> None:
        assert _matches_step("", "Bash") is False
        assert _matches_step("", "") is True

    def test_empty_tuple(self) -> None:
        assert _matches_step("Bash", ()) is False


# ===========================================================================
# detect_workflow_sequences
# ===========================================================================


class TestDetectWorkflowSequences:
    """Tests for detect_workflow_sequences: finds patterns in tool sequences."""

    def test_empty_input(self) -> None:
        assert detect_workflow_sequences([]) == []

    def test_test_edit_test_pattern(self) -> None:
        """Bash -> Edit -> Bash should match test-edit-test."""
        tool_uses = [
            _make_tool_use("Bash", {"command": "pytest"}, result_error=True),
            _make_tool_use("Edit", {"file_path": "/tmp/foo.py"}, result_error=False),
            _make_tool_use("Bash", {"command": "pytest"}, result_error=False),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        assert len(patterns) >= 1
        tet = [p for p in patterns if p["pattern_type"] == "test-edit-test"]
        assert len(tet) == 1
        assert tet[0]["success"] is True

    def test_search_read_edit_pattern(self) -> None:
        """Grep -> Read -> Edit should match search-read-edit."""
        tool_uses = [
            _make_tool_use("Grep", {"pattern": "foo"}),
            _make_tool_use("Read", {"file_path": "/tmp/bar.py"}),
            _make_tool_use("Edit", {"file_path": "/tmp/bar.py"}, result_error=False),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        sre = [p for p in patterns if p["pattern_type"] == "search-read-edit"]
        assert len(sre) == 1
        assert "/tmp/bar.py" in sre[0]["files"]

    def test_read_edit_pattern(self) -> None:
        """Read -> Write should match read-edit."""
        tool_uses = [
            _make_tool_use("Read", {"file_path": "/tmp/x.py"}),
            _make_tool_use("Write", {"file_path": "/tmp/x.py"}, result_error=False),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        re_pats = [p for p in patterns if p["pattern_type"] == "read-edit"]
        assert len(re_pats) == 1

    def test_no_pattern_in_unrelated_tools(self) -> None:
        """Random tools shouldn't match any pattern."""
        tool_uses = [
            _make_tool_use("Agent", {"subagent_type": "scout"}),
            _make_tool_use("Agent", {"subagent_type": "oracle"}),
        ]
        assert detect_workflow_sequences(tool_uses) == []

    def test_success_determined_from_last_bash(self) -> None:
        """test-edit-test success comes from last Bash result."""
        tool_uses = [
            _make_tool_use("Bash", {"command": "pytest"}, result_error=True),
            _make_tool_use("Write", {"file_path": "/tmp/f.py"}, result_error=False),
            _make_tool_use("Bash", {"command": "pytest"}, result_error=True),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        tet = [p for p in patterns if p["pattern_type"] == "test-edit-test"]
        assert len(tet) == 1
        assert tet[0]["success"] is False

    def test_success_none_when_no_result_error_info(self) -> None:
        """Success is None when result_error is None on all tools."""
        tool_uses = [
            _make_tool_use("Bash", {"command": "pytest"}, result_error=None),
            _make_tool_use("Edit", {"file_path": "/tmp/f.py"}, result_error=None),
            _make_tool_use("Bash", {"command": "pytest"}, result_error=None),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        tet = [p for p in patterns if p["pattern_type"] == "test-edit-test"]
        assert len(tet) == 1
        assert tet[0]["success"] is None

    def test_multiple_patterns_detected(self) -> None:
        """A long sequence can contain multiple distinct patterns."""
        tool_uses = [
            _make_tool_use("Bash", {"command": "pytest"}, result_error=True),
            _make_tool_use("Edit", {"file_path": "/tmp/a.py"}, result_error=False),
            _make_tool_use("Bash", {"command": "pytest"}, result_error=False),
            _make_tool_use("Read", {"file_path": "/tmp/b.py"}),
            _make_tool_use("Edit", {"file_path": "/tmp/b.py"}, result_error=False),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        types = {p["pattern_type"] for p in patterns}
        assert "test-edit-test" in types
        assert "read-edit" in types

    def test_files_deduplicated(self) -> None:
        """Files list should contain unique entries."""
        tool_uses = [
            _make_tool_use("Read", {"file_path": "/tmp/same.py"}),
            _make_tool_use("Edit", {"file_path": "/tmp/same.py"}, result_error=False),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        re_pats = [p for p in patterns if p["pattern_type"] == "read-edit"]
        assert len(re_pats) == 1
        assert re_pats[0]["files"] == ["/tmp/same.py"]

    def test_leading_unrelated_tools_no_duplicate(self) -> None:
        """Pattern preceded by unrelated tools should be found exactly once."""
        tool_uses = [
            _make_tool_use("Agent", {"subagent_type": "scout"}),
            _make_tool_use("Agent", {"subagent_type": "oracle"}),
            _make_tool_use("Read", {"file_path": "/a.py"}),
            _make_tool_use("Edit", {"file_path": "/a.py"}, result_error=False),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        re_pats = [p for p in patterns if p["pattern_type"] == "read-edit"]
        assert len(re_pats) == 1

    def test_many_leading_unrelated_no_duplication(self) -> None:
        """Five leading Agents should still yield exactly one pattern."""
        tool_uses = [
            _make_tool_use("Agent", {}) for _ in range(5)
        ] + [
            _make_tool_use("Read", {"file_path": "/a.py"}),
            _make_tool_use("Edit", {"file_path": "/a.py"}, result_error=False),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        re_pats = [p for p in patterns if p["pattern_type"] == "read-edit"]
        assert len(re_pats) == 1

    def test_overlapping_patterns_prefer_longest(self) -> None:
        """Grep->Read->Edit matches search-read-edit, not also read-edit."""
        tool_uses = [
            _make_tool_use("Grep", {"pattern": "foo"}),
            _make_tool_use("Read", {"file_path": "/a.py"}),
            _make_tool_use("Edit", {"file_path": "/a.py"}, result_error=False),
        ]
        patterns = detect_workflow_sequences(tool_uses)
        types = [p["pattern_type"] for p in patterns]
        assert "search-read-edit" in types
        assert "read-edit" not in types
        assert len(patterns) == 1


# ===========================================================================
# summarize_tool_usage
# ===========================================================================


class TestSummarizeToolUsage:
    """Tests for summarize_tool_usage: computes frequency and file stats."""

    def test_empty_input(self) -> None:
        result = summarize_tool_usage([])
        assert result == {
            "tool_counts": {},
            "files_by_tool": {},
            "unique_commands": 0,
            "total_tool_calls": 0,
        }

    def test_counts_tools(self) -> None:
        tool_uses = [
            _make_tool_use("Read", {"file_path": "/a.py"}),
            _make_tool_use("Read", {"file_path": "/b.py"}),
            _make_tool_use("Bash", {"command": "pytest"}),
        ]
        result = summarize_tool_usage(tool_uses)
        assert result["tool_counts"] == {"Read": 2, "Bash": 1}
        assert result["total_tool_calls"] == 3

    def test_files_by_tool_sorted(self) -> None:
        tool_uses = [
            _make_tool_use("Edit", {"file_path": "/z.py"}),
            _make_tool_use("Edit", {"file_path": "/a.py"}),
        ]
        result = summarize_tool_usage(tool_uses)
        assert result["files_by_tool"]["Edit"] == ["/a.py", "/z.py"]

    def test_unique_commands(self) -> None:
        tool_uses = [
            _make_tool_use("Bash", {"command": "pytest"}),
            _make_tool_use("Bash", {"command": "pytest"}),
            _make_tool_use("Bash", {"command": "ruff check"}),
        ]
        result = summarize_tool_usage(tool_uses)
        assert result["unique_commands"] == 2

    def test_no_file_path_excluded(self) -> None:
        """Tools without file_path don't appear in files_by_tool."""
        tool_uses = [_make_tool_use("Bash", {"command": "ls"})]
        result = summarize_tool_usage(tool_uses)
        assert result["files_by_tool"] == {}


# ===========================================================================
# format_pattern_as_learning
# ===========================================================================


class TestFormatPatternAsLearning:
    """Tests for format_pattern_as_learning: human-readable pattern strings."""

    def test_success_pattern(self) -> None:
        pattern = {
            "pattern_type": "read-edit",
            "tools": [
                {"tool_name": "Read", "input": {}},
                {"tool_name": "Edit", "input": {}},
            ],
            "files": ["/tmp/foo.py"],
            "success": True,
        }
        result = format_pattern_as_learning(pattern)
        assert "read-edit" in result
        assert "Read -> Edit" in result
        assert "foo.py" in result
        assert "succeeded" in result

    def test_failed_pattern(self) -> None:
        pattern = {
            "pattern_type": "test-edit-test",
            "tools": [
                {"tool_name": "Bash", "input": {"command": "pytest tests/"}},
                {"tool_name": "Edit", "input": {}},
                {"tool_name": "Bash", "input": {"command": "pytest tests/"}},
            ],
            "files": [],
            "success": False,
        }
        result = format_pattern_as_learning(pattern)
        assert "failed" in result
        assert "Command: pytest tests/" in result

    def test_unknown_success(self) -> None:
        pattern = {
            "pattern_type": "read-edit",
            "tools": [
                {"tool_name": "Read", "input": {}},
                {"tool_name": "Edit", "input": {}},
            ],
            "files": [],
            "success": None,
        }
        result = format_pattern_as_learning(pattern)
        assert "unknown" in result

    def test_files_truncated_at_five(self) -> None:
        pattern = {
            "pattern_type": "read-edit",
            "tools": [
                {"tool_name": "Read", "input": {}},
                {"tool_name": "Edit", "input": {}},
            ],
            "files": [f"/tmp/f{i}.py" for i in range(10)],
            "success": True,
        }
        result = format_pattern_as_learning(pattern)
        # Only first 5 files should appear
        assert "f5.py" not in result
        assert "f4.py" in result


# ===========================================================================
# extract_tool_uses (JSONL I/O)
# ===========================================================================


class TestExtractToolUses:
    """Tests for extract_tool_uses: parses JSONL and correlates results."""

    def test_basic_extraction(self, tmp_path: Path) -> None:
        """Extract a single tool_use with its tool_result."""
        jsonl_path = tmp_path / "session.jsonl"
        _write_jsonl(
            [
                _make_assistant_tool_use("Read", {"file_path": "/tmp/x.py"}, "tu_1"),
                _make_user_tool_result("tu_1", is_error=False),
            ],
            jsonl_path,
        )
        result = extract_tool_uses(jsonl_path)
        assert len(result) == 1
        assert result[0]["tool_name"] == "Read"
        assert result[0]["input"] == {"file_path": "/tmp/x.py"}
        assert result[0]["result_error"] is False

    def test_error_result_correlation(self, tmp_path: Path) -> None:
        """Error results are correlated back to tool_use entries."""
        jsonl_path = tmp_path / "session.jsonl"
        _write_jsonl(
            [
                _make_assistant_tool_use("Bash", {"command": "fail"}, "tu_err"),
                _make_user_tool_result("tu_err", is_error=True),
            ],
            jsonl_path,
        )
        result = extract_tool_uses(jsonl_path)
        assert len(result) == 1
        assert result[0]["result_error"] is True

    def test_multiple_tool_uses(self, tmp_path: Path) -> None:
        """Multiple tool_use entries extracted in order."""
        jsonl_path = tmp_path / "session.jsonl"
        _write_jsonl(
            [
                _make_assistant_tool_use(
                    "Read", {"file_path": "/a.py"}, "t1", "2026-04-03T10:00:00Z"
                ),
                _make_user_tool_result("t1"),
                _make_assistant_tool_use(
                    "Edit", {"file_path": "/a.py"}, "t2", "2026-04-03T10:01:00Z"
                ),
                _make_user_tool_result("t2"),
            ],
            jsonl_path,
        )
        result = extract_tool_uses(jsonl_path)
        assert len(result) == 2
        assert result[0]["tool_name"] == "Read"
        assert result[1]["tool_name"] == "Edit"

    def test_malformed_json_lines_skipped(self, tmp_path: Path) -> None:
        """Invalid JSON lines are silently skipped."""
        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            f.write("not valid json\n")
            line = json.dumps(
                _make_assistant_tool_use("Read", {"file_path": "/x.py"}, "t1")
            )
            f.write(line + "\n")
            f.write(json.dumps(_make_user_tool_result("t1")) + "\n")
        result = extract_tool_uses(jsonl_path)
        assert len(result) == 1

    def test_no_tool_result_leaves_none(self, tmp_path: Path) -> None:
        """Tool_use without matching tool_result keeps result_error=None."""
        jsonl_path = tmp_path / "session.jsonl"
        _write_jsonl(
            [_make_assistant_tool_use("Read", {"file_path": "/x.py"}, "t1")],
            jsonl_path,
        )
        result = extract_tool_uses(jsonl_path)
        assert len(result) == 1
        assert result[0]["result_error"] is None

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty JSONL returns empty list."""
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")
        result = extract_tool_uses(jsonl_path)
        assert result == []

    def test_non_list_content_skipped(self, tmp_path: Path) -> None:
        """Messages with non-list content are skipped."""
        jsonl_path = tmp_path / "session.jsonl"
        _write_jsonl(
            [{"type": "assistant", "message": {"content": "just text"}}],
            jsonl_path,
        )
        result = extract_tool_uses(jsonl_path)
        assert result == []

    def test_top_level_list_skipped(self, tmp_path: Path) -> None:
        """JSON array at top level should be skipped."""
        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            f.write("[1, 2, 3]\n")
        result = extract_tool_uses(jsonl_path)
        assert result == []

    def test_non_dict_message_skipped(self, tmp_path: Path) -> None:
        """Non-dict message field should be skipped."""
        jsonl_path = tmp_path / "session.jsonl"
        _write_jsonl(
            [{"type": "assistant", "message": "oops"}],
            jsonl_path,
        )
        result = extract_tool_uses(jsonl_path)
        assert result == []

    def test_non_dict_input_skipped(self, tmp_path: Path) -> None:
        """tool_use with non-dict input should be skipped."""
        jsonl_path = tmp_path / "session.jsonl"
        _write_jsonl(
            [{
                "type": "assistant",
                "message": {
                    "content": [{
                        "type": "tool_use",
                        "name": "Read",
                        "input": "oops",
                        "id": "tu_1",
                    }]
                },
            }],
            jsonl_path,
        )
        result = extract_tool_uses(jsonl_path)
        assert result == []

    def test_empty_tool_use_id_not_tracked(self, tmp_path: Path) -> None:
        """tool_use entries without an ID should not be indexed."""
        jsonl_path = tmp_path / "session.jsonl"
        _write_jsonl(
            [{
                "type": "assistant",
                "message": {
                    "content": [{
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "/a.py"},
                        "id": "",
                    }]
                },
            }],
            jsonl_path,
        )
        result = extract_tool_uses(jsonl_path)
        assert result == []


# ===========================================================================
# _match_pattern_at (extracted helper)
# ===========================================================================


class TestMatchPatternAt:
    """Tests for _match_pattern_at: returns (matched, first_idx, end_idx)."""

    def test_full_match(self) -> None:
        tool_uses = [
            _make_tool_use("Read", {"file_path": "/a.py"}),
            _make_tool_use("Edit", {"file_path": "/a.py"}),
        ]
        result = _match_pattern_at(tool_uses, ["Read", ("Edit", "Write")], 0)
        assert result is not None
        matched, first_idx, end_idx = result
        assert len(matched) == 2
        assert first_idx == 0
        assert end_idx == 2

    def test_no_match(self) -> None:
        tool_uses = [_make_tool_use("Bash", {"command": "ls"})]
        result = _match_pattern_at(
            tool_uses, ["Read", ("Edit", "Write")], 0
        )
        assert result is None

    def test_partial_match_returns_none(self) -> None:
        tool_uses = [
            _make_tool_use("Read", {"file_path": "/a.py"}),
            _make_tool_use("Bash", {"command": "ls"}),
        ]
        result = _match_pattern_at(
            tool_uses, ["Read", ("Edit", "Write")], 0
        )
        assert result is None

    def test_start_offset(self) -> None:
        tool_uses = [
            _make_tool_use("Bash", {"command": "ls"}),
            _make_tool_use("Read", {"file_path": "/a.py"}),
            _make_tool_use("Edit", {"file_path": "/a.py"}),
        ]
        result = _match_pattern_at(
            tool_uses, ["Read", ("Edit", "Write")], 1
        )
        assert result is not None
        matched, first_idx, end_idx = result
        assert len(matched) == 2
        assert first_idx == 1
        assert end_idx == 3

    def test_skipped_entries_included_in_end_index(self) -> None:
        """end_index accounts for skipped non-matching entries."""
        tool_uses = [
            _make_tool_use("Agent", {}),
            _make_tool_use("Agent", {}),
            _make_tool_use("Read", {"file_path": "/a.py"}),
            _make_tool_use("Edit", {"file_path": "/a.py"}),
        ]
        result = _match_pattern_at(
            tool_uses, ["Read", ("Edit", "Write")], 0
        )
        assert result is not None
        matched, first_idx, end_idx = result
        assert len(matched) == 2
        assert first_idx == 2  # first actual match at index 2
        assert end_idx == 4  # past all 4 entries

    def test_empty_tool_uses(self) -> None:
        result = _match_pattern_at([], ["Read"], 0)
        assert result is None


# ===========================================================================
# _determine_success (extracted helper)
# ===========================================================================


class TestDetermineSuccess:
    """Tests for _determine_success: derives success from tool results."""

    def test_success_from_bash(self) -> None:
        matched = [
            _make_tool_use("Bash", {}, result_error=False),
            _make_tool_use("Edit", {}, result_error=None),
        ]
        assert _determine_success(matched) is True

    def test_failure_from_bash(self) -> None:
        matched = [
            _make_tool_use("Edit", {}, result_error=False),
            _make_tool_use("Bash", {}, result_error=True),
        ]
        assert _determine_success(matched) is False

    def test_fallback_to_edit(self) -> None:
        matched = [
            _make_tool_use("Read", {}),
            _make_tool_use("Edit", {}, result_error=False),
        ]
        assert _determine_success(matched) is True

    def test_none_when_no_info(self) -> None:
        matched = [
            _make_tool_use("Read", {}, result_error=None),
            _make_tool_use("Edit", {}, result_error=None),
        ]
        assert _determine_success(matched) is None

    def test_empty_matched(self) -> None:
        assert _determine_success([]) is None

    def test_last_bash_none_returns_none(self) -> None:
        """Last Bash with result_error=None should return None, not earlier result."""
        matched = [
            _make_tool_use("Bash", {}, result_error=True),
            _make_tool_use("Edit", {}, result_error=False),
            _make_tool_use("Bash", {}, result_error=None),
        ]
        assert _determine_success(matched) is None


# ===========================================================================
# _collect_files (extracted helper)
# ===========================================================================


class TestCollectFiles:
    """Tests for _collect_files: unique file paths from matched entries."""

    def test_unique_files(self) -> None:
        matched = [
            _make_tool_use("Read", {"file_path": "/a.py"}),
            _make_tool_use("Edit", {"file_path": "/a.py"}),
        ]
        assert _collect_files(matched) == ["/a.py"]

    def test_no_files(self) -> None:
        matched = [_make_tool_use("Bash", {"command": "ls"})]
        assert _collect_files(matched) == []

    def test_empty_matched(self) -> None:
        assert _collect_files([]) == []

    def test_preserves_encounter_order(self) -> None:
        """Files should be in first-seen order, not hash order."""
        matched = [
            _make_tool_use("Read", {"file_path": "/z.py"}),
            _make_tool_use("Edit", {"file_path": "/a.py"}),
            _make_tool_use("Read", {"file_path": "/z.py"}),
        ]
        assert _collect_files(matched) == ["/z.py", "/a.py"]


# ===========================================================================
# _build_pattern_result (extracted helper)
# ===========================================================================


class TestBuildPatternResult:
    """Tests for _build_pattern_result: assembles pattern dict."""

    def test_basic_build(self) -> None:
        matched = [
            _make_tool_use("Read", {"file_path": "/x.py"}),
            _make_tool_use(
                "Edit", {"file_path": "/x.py"}, result_error=False
            ),
        ]
        result = _build_pattern_result("read-edit", matched)
        assert result["pattern_type"] == "read-edit"
        assert len(result["tools"]) == 2
        assert result["files"] == ["/x.py"]
        assert result["success"] is True


# ===========================================================================
# _parse_tool_use_entry / _parse_tool_result (I/O helpers)
# ===========================================================================


class TestParseToolUseEntry:
    """Tests for _parse_tool_use_entry: parses content items."""

    def test_valid_tool_use(self) -> None:
        item = {
            "type": "tool_use",
            "name": "Read",
            "input": {"file_path": "/a.py"},
            "id": "tu_1",
        }
        data = {"timestamp": "2026-04-03T10:00:00Z"}
        result = _parse_tool_use_entry(item, data, 1)
        assert result is not None
        assert result["tool_name"] == "Read"
        assert result["tool_use_id"] == "tu_1"

    def test_non_tool_use_returns_none(self) -> None:
        assert _parse_tool_use_entry({"type": "text"}, {}, 1) is None

    def test_non_dict_returns_none(self) -> None:
        assert _parse_tool_use_entry("not a dict", {}, 1) is None

    def test_non_dict_input_returns_none(self) -> None:
        item = {
            "type": "tool_use",
            "name": "Read",
            "input": "not a dict",
            "id": "tu_1",
        }
        assert _parse_tool_use_entry(item, {}, 1) is None


class TestParseToolResult:
    """Tests for _parse_tool_result: parses tool_result items."""

    def test_valid_result(self) -> None:
        item = {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "is_error": True,
        }
        result = _parse_tool_result(item)
        assert result == ("tu_1", True)

    def test_non_tool_result_returns_none(self) -> None:
        assert _parse_tool_result({"type": "text"}) is None

    def test_defaults(self) -> None:
        item = {"type": "tool_result"}
        result = _parse_tool_result(item)
        assert result == ("", False)
