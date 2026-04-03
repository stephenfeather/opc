"""Tests for generate_mini_handoff.

Validates that:
1. JSONL parsing extracts valid entries and skips malformed lines
2. Pure extractors produce correct results from parsed entries
3. YAML formatting handles all value types correctly
4. Handoff assembly composes extractors into complete handoff dicts
5. Edge cases: empty inputs, missing fields, malformed data
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.generate_mini_handoff import (  # noqa: E402
    _format_yaml_value,
    extract_commands_run,
    extract_files_touched,
    extract_git_state,
    extract_timestamps,
    extract_tool_counts,
    format_as_yaml,
    parse_jsonl_entries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant_entry(
    tool_name: str,
    tool_input: dict,
    timestamp: str = "2025-01-15T10:00:00Z",
) -> dict:
    """Build a minimal assistant-type JSONL entry with a tool_use block."""
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": tool_name,
                    "input": tool_input,
                }
            ]
        },
    }


def _user_entry(text: str = "hello", timestamp: str = "2025-01-15T10:00:00Z") -> dict:
    """Build a minimal user-type JSONL entry."""
    return {
        "type": "user",
        "timestamp": timestamp,
        "message": {"content": text},
    }


def _state_event(
    tool: str,
    *,
    file: str = "",
    command: str = "",
    timestamp: str = "2025-01-15T10:00:00Z",
) -> dict:
    """Build a state-file event (Phase 3 format)."""
    event = {"tool": tool, "timestamp": timestamp}
    if file:
        event["file"] = file
    if command:
        event["command"] = command
    return event


def _entries_to_lines(entries: list[dict]) -> list[str]:
    """Convert a list of dicts to JSONL lines (strings)."""
    return [json.dumps(e) for e in entries]


# ===========================================================================
# parse_jsonl_entries
# ===========================================================================


class TestParseJsonlEntries:
    """Tests for parsing raw JSONL lines into entry dicts."""

    def test_parses_valid_json_lines(self):
        lines = _entries_to_lines([_user_entry(), _assistant_entry("Read", {"file_path": "a.py"})])
        result = parse_jsonl_entries(lines)
        assert len(result) == 2

    def test_skips_malformed_json(self):
        lines = [
            "not valid json",
            json.dumps(_user_entry()),
            "{broken",
        ]
        result = parse_jsonl_entries(lines)
        assert len(result) == 1

    def test_empty_input_returns_empty_list(self):
        assert parse_jsonl_entries([]) == []

    def test_all_malformed_returns_empty_list(self):
        assert parse_jsonl_entries(["bad", "{also bad", ""]) == []

    def test_preserves_entry_order(self):
        entries = [
            _user_entry(timestamp="2025-01-15T10:00:00Z"),
            _assistant_entry("Read", {"file_path": "a.py"}, timestamp="2025-01-15T10:01:00Z"),
            _user_entry(timestamp="2025-01-15T10:02:00Z"),
        ]
        result = parse_jsonl_entries(_entries_to_lines(entries))
        assert [r["timestamp"] for r in result] == [
            "2025-01-15T10:00:00Z",
            "2025-01-15T10:01:00Z",
            "2025-01-15T10:02:00Z",
        ]

    def test_skips_non_dict_json_values(self):
        lines = [
            json.dumps([1, 2, 3]),          # valid JSON array
            json.dumps("just a string"),     # valid JSON string
            json.dumps(42),                  # valid JSON number
            json.dumps(None),                # valid JSON null
            json.dumps(True),                # valid JSON bool
            json.dumps(_user_entry()),       # valid JSON dict — kept
        ]
        result = parse_jsonl_entries(lines)
        assert len(result) == 1
        assert result[0]["type"] == "user"

    def test_handles_empty_lines(self):
        lines = ["", json.dumps(_user_entry()), "", ""]
        result = parse_jsonl_entries(lines)
        assert len(result) == 1


# ===========================================================================
# extract_files_touched
# ===========================================================================


class TestExtractFilesTouched:
    """Tests for extracting file operations from parsed entries."""

    def test_read_files_tracked(self):
        entries = [_assistant_entry("Read", {"file_path": "src/main.py"})]
        result = extract_files_touched(entries)
        assert result["read"] == ["src/main.py"]

    def test_edit_files_tracked_as_modified(self):
        entries = [_assistant_entry("Edit", {"file_path": "src/main.py"})]
        result = extract_files_touched(entries)
        assert result["modified"] == ["src/main.py"]

    def test_multiedit_tracked_as_modified(self):
        entries = [_assistant_entry("MultiEdit", {"file_path": "src/main.py"})]
        result = extract_files_touched(entries)
        assert result["modified"] == ["src/main.py"]

    def test_write_without_prior_read_is_created(self):
        entries = [_assistant_entry("Write", {"file_path": "new_file.py"})]
        result = extract_files_touched(entries)
        assert result["created"] == ["new_file.py"]
        assert result["modified"] == []

    def test_write_with_prior_read_is_modified(self):
        entries = [
            _assistant_entry("Read", {"file_path": "existing.py"}),
            _assistant_entry("Write", {"file_path": "existing.py"}),
        ]
        result = extract_files_touched(entries)
        assert result["modified"] == ["existing.py"]
        assert result["created"] == []

    def test_deduplicates_read_files(self):
        entries = [
            _assistant_entry("Read", {"file_path": "a.py"}),
            _assistant_entry("Read", {"file_path": "a.py"}),
            _assistant_entry("Read", {"file_path": "b.py"}),
        ]
        result = extract_files_touched(entries)
        assert result["read"] == ["a.py", "b.py"]

    def test_deduplicates_edit_files(self):
        entries = [
            _assistant_entry("Edit", {"file_path": "a.py"}),
            _assistant_entry("Edit", {"file_path": "a.py"}),
        ]
        result = extract_files_touched(entries)
        assert result["modified"] == ["a.py"]

    def test_skips_non_assistant_entries(self):
        entries = [_user_entry()]
        result = extract_files_touched(entries)
        assert result == {"read": [], "modified": [], "created": []}

    def test_skips_entries_without_file_path(self):
        entries = [_assistant_entry("Read", {"other_param": "value"})]
        result = extract_files_touched(entries)
        assert result["read"] == []

    def test_skips_non_tool_use_content(self):
        entries = [
            {
                "type": "assistant",
                "timestamp": "2025-01-15T10:00:00Z",
                "message": {"content": [{"type": "text", "text": "hello"}]},
            }
        ]
        result = extract_files_touched(entries)
        assert result == {"read": [], "modified": [], "created": []}

    def test_empty_entries_returns_empty_categories(self):
        result = extract_files_touched([])
        assert result == {"read": [], "modified": [], "created": []}

    def test_content_not_a_list_skipped(self):
        entries = [
            {
                "type": "assistant",
                "timestamp": "2025-01-15T10:00:00Z",
                "message": {"content": "just a string"},
            }
        ]
        result = extract_files_touched(entries)
        assert result == {"read": [], "modified": [], "created": []}

    def test_combined_operations(self):
        entries = [
            _assistant_entry("Read", {"file_path": "existing.py"}),
            _assistant_entry("Read", {"file_path": "config.yaml"}),
            _assistant_entry("Edit", {"file_path": "existing.py"}),
            _assistant_entry("Write", {"file_path": "new.py"}),
            _assistant_entry("Write", {"file_path": "existing.py"}),
        ]
        result = extract_files_touched(entries)
        assert "existing.py" in result["read"]
        assert "config.yaml" in result["read"]
        assert "existing.py" in result["modified"]
        assert "new.py" in result["created"]
        assert "new.py" not in result["modified"]


# ===========================================================================
# extract_commands_run
# ===========================================================================


class TestExtractCommandsRun:
    """Tests for extracting bash commands from parsed entries."""

    def test_extracts_bash_commands(self):
        entries = [_assistant_entry("Bash", {"command": "git status"}, "2025-01-15T10:00:00Z")]
        result = extract_commands_run(entries)
        assert len(result) == 1
        assert result[0]["command"] == "git status"
        assert result[0]["timestamp"] == "2025-01-15T10:00:00Z"

    def test_truncates_long_commands(self):
        long_cmd = "x" * 600
        entries = [_assistant_entry("Bash", {"command": long_cmd})]
        result = extract_commands_run(entries)
        assert len(result[0]["command"]) == 500

    def test_skips_empty_commands(self):
        entries = [_assistant_entry("Bash", {"command": ""})]
        result = extract_commands_run(entries)
        assert result == []

    def test_skips_non_bash_tools(self):
        entries = [_assistant_entry("Read", {"file_path": "a.py"})]
        result = extract_commands_run(entries)
        assert result == []

    def test_preserves_order(self):
        entries = [
            _assistant_entry("Bash", {"command": "first"}, "2025-01-15T10:00:00Z"),
            _assistant_entry("Bash", {"command": "second"}, "2025-01-15T10:01:00Z"),
        ]
        result = extract_commands_run(entries)
        assert [r["command"] for r in result] == ["first", "second"]

    def test_empty_entries_returns_empty_list(self):
        assert extract_commands_run([]) == []


# ===========================================================================
# extract_git_state
# ===========================================================================


class TestExtractGitState:
    """Tests for extracting last git command from commands list."""

    def test_returns_last_git_command(self):
        commands = [
            {"command": "git status", "timestamp": "t1"},
            {"command": "npm test", "timestamp": "t2"},
            {"command": "git commit -m 'fix'", "timestamp": "t3"},
        ]
        result = extract_git_state(commands)
        assert result is not None
        assert result["last_command"] == "git commit -m 'fix'"
        assert result["timestamp"] == "t3"

    def test_returns_none_when_no_git_commands(self):
        commands = [
            {"command": "npm test", "timestamp": "t1"},
            {"command": "python script.py", "timestamp": "t2"},
        ]
        assert extract_git_state(commands) is None

    def test_empty_commands_returns_none(self):
        assert extract_git_state([]) is None

    def test_single_git_command(self):
        commands = [{"command": "git log", "timestamp": "t1"}]
        result = extract_git_state(commands)
        assert result["last_command"] == "git log"


# ===========================================================================
# extract_timestamps
# ===========================================================================


class TestExtractTimestamps:
    """Tests for extracting first/last timestamps from parsed entries."""

    def test_extracts_first_and_last(self):
        entries = [
            _user_entry(timestamp="2025-01-15T10:00:00Z"),
            _assistant_entry("Read", {"file_path": "a.py"}, "2025-01-15T10:05:00Z"),
            _user_entry(timestamp="2025-01-15T10:10:00Z"),
        ]
        result = extract_timestamps(entries)
        assert result["first_timestamp"] == "2025-01-15T10:00:00Z"
        assert result["last_timestamp"] == "2025-01-15T10:10:00Z"

    def test_empty_entries_returns_empty_strings(self):
        result = extract_timestamps([])
        assert result == {"first_timestamp": "", "last_timestamp": ""}

    def test_entries_without_timestamps_skipped(self):
        entries = [{"type": "user", "message": {"content": "hi"}}]
        result = extract_timestamps(entries)
        assert result == {"first_timestamp": "", "last_timestamp": ""}

    def test_single_entry(self):
        entries = [_user_entry(timestamp="2025-01-15T10:00:00Z")]
        result = extract_timestamps(entries)
        assert result["first_timestamp"] == "2025-01-15T10:00:00Z"
        assert result["last_timestamp"] == "2025-01-15T10:00:00Z"


# ===========================================================================
# extract_tool_counts
# ===========================================================================


class TestExtractToolCounts:
    """Tests for counting tool usage from parsed entries."""

    def test_counts_tool_usage(self):
        entries = [
            _assistant_entry("Read", {"file_path": "a.py"}),
            _assistant_entry("Read", {"file_path": "b.py"}),
            _assistant_entry("Edit", {"file_path": "a.py"}),
        ]
        result = extract_tool_counts(entries)
        assert result["Read"] == 2
        assert result["Edit"] == 1

    def test_sorted_by_count_descending(self):
        entries = [
            _assistant_entry("Edit", {"file_path": "a.py"}),
            _assistant_entry("Read", {"file_path": "a.py"}),
            _assistant_entry("Read", {"file_path": "b.py"}),
            _assistant_entry("Read", {"file_path": "c.py"}),
        ]
        result = extract_tool_counts(entries)
        keys = list(result.keys())
        assert keys[0] == "Read"
        assert keys[1] == "Edit"

    def test_empty_entries_returns_empty_dict(self):
        assert extract_tool_counts([]) == {}

    def test_skips_non_assistant_entries(self):
        entries = [_user_entry()]
        assert extract_tool_counts(entries) == {}

    def test_unknown_tool_name_uses_fallback(self):
        entries = [
            {
                "type": "assistant",
                "timestamp": "2025-01-15T10:00:00Z",
                "message": {
                    "content": [{"type": "tool_use", "input": {}}]
                },
            }
        ]
        result = extract_tool_counts(entries)
        assert result.get("unknown") == 1


# ===========================================================================
# _format_yaml_value
# ===========================================================================


class TestFormatYamlValue:
    """Tests for YAML value formatting (already pure)."""

    def test_none_value(self):
        assert _format_yaml_value(None) == "null"

    def test_bool_true(self):
        assert _format_yaml_value(True) == "true"

    def test_bool_false(self):
        assert _format_yaml_value(False) == "false"

    def test_int_value(self):
        assert _format_yaml_value(42) == "42"

    def test_float_value(self):
        assert _format_yaml_value(3.14) == "3.14"

    def test_simple_string(self):
        assert _format_yaml_value("hello") == "hello"

    def test_string_with_special_chars_quoted(self):
        assert _format_yaml_value("key: value") == '"key: value"'

    def test_string_with_newline_quoted(self):
        assert _format_yaml_value("line1\nline2") == '"line1\\nline2"'

    def test_string_starting_with_dash_quoted(self):
        assert _format_yaml_value("- item") == '"- item"'

    def test_empty_list(self):
        assert _format_yaml_value([]) == "[]"

    def test_empty_dict(self):
        assert _format_yaml_value({}) == "{}"

    def test_list_of_strings(self):
        result = _format_yaml_value(["a", "b"], indent=0)
        assert "- a" in result
        assert "- b" in result

    def test_dict_with_simple_values(self):
        result = _format_yaml_value({"key": "val"}, indent=0)
        assert "key: val" in result

    def test_nested_dict(self):
        result = _format_yaml_value({"outer": {"inner": "val"}}, indent=0)
        assert "outer:" in result
        assert "inner: val" in result

    def test_backslash_in_plain_string_unchanged(self):
        # Backslash alone doesn't trigger quoting in YAML
        result = _format_yaml_value("path\\to\\file")
        assert result == "path\\to\\file"

    def test_backslash_with_special_char_escaped(self):
        # When quoting is triggered by other chars, backslash is escaped
        result = _format_yaml_value("path\\to: file")
        assert '"' in result
        assert "path\\\\to" in result

    def test_quote_in_string_escaped(self):
        result = _format_yaml_value('say "hi"')
        assert result == '"say \\"hi\\""'


# ===========================================================================
# format_as_yaml
# ===========================================================================


class TestFormatAsYaml:
    """Tests for full handoff dict to YAML formatting."""

    def test_includes_frontmatter(self):
        handoff = {
            "session": "s-123",
            "date": "2025-01-15",
            "status": "complete",
            "outcome": "auto-extracted",
        }
        result = format_as_yaml(handoff)
        assert result.startswith("---\n")
        assert "session: s-123" in result
        assert "date: 2025-01-15" in result
        assert "status: complete" in result

    def test_body_fields_present(self):
        handoff = {
            "session": "s-123",
            "date": "2025-01-15",
            "status": "complete",
            "outcome": "auto-extracted",
            "goal": "Test goal",
            "tool_usage": {"Read": 5},
        }
        result = format_as_yaml(handoff)
        assert "goal: Test goal" in result
        assert "Read: 5" in result

    def test_none_body_fields_omitted(self):
        handoff = {
            "session": "s-123",
            "date": "2025-01-15",
            "status": "complete",
            "outcome": "auto-extracted",
            "git_state": None,
        }
        result = format_as_yaml(handoff)
        assert "git_state" not in result

    def test_ends_with_newline(self):
        handoff = {"session": "s-123", "date": "", "status": "", "outcome": ""}
        result = format_as_yaml(handoff)
        assert result.endswith("\n")

    def test_two_frontmatter_delimiters(self):
        handoff = {"session": "s-123", "date": "", "status": "", "outcome": ""}
        result = format_as_yaml(handoff)
        assert result.count("---") == 2


# ===========================================================================
# build_handoff_from_entries (assembles extractors)
# ===========================================================================


class TestBuildHandoffFromEntries:
    """Tests for assembling a handoff dict from parsed entries."""

    def test_complete_handoff_structure(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_entries

        entries = [
            _assistant_entry("Read", {"file_path": "a.py"}, "2025-01-15T10:00:00Z"),
            _assistant_entry("Bash", {"command": "git status"}, "2025-01-15T10:01:00Z"),
            _assistant_entry("Edit", {"file_path": "a.py"}, "2025-01-15T10:02:00Z"),
        ]
        result = build_handoff_from_entries(entries, "s-test", "/project")

        assert result["session"] == "s-test"
        assert result["status"] == "complete"
        assert result["date"] == "2025-01-15"
        assert "files" in result
        assert "commands_run" in result
        assert "tool_usage" in result
        assert "duration" in result

    def test_files_categorized_correctly(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_entries

        entries = [
            _assistant_entry("Read", {"file_path": "existing.py"}, "2025-01-15T10:00:00Z"),
            _assistant_entry("Write", {"file_path": "new.py"}, "2025-01-15T10:01:00Z"),
            _assistant_entry("Edit", {"file_path": "existing.py"}, "2025-01-15T10:02:00Z"),
        ]
        result = build_handoff_from_entries(entries, "s-test", "/project")
        assert "existing.py" in result["files"]["modified"]
        assert "new.py" in result["files"]["created"]

    def test_commands_limited_to_last_50(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_entries

        entries = [
            _assistant_entry("Bash", {"command": f"cmd-{i}"}, f"2025-01-15T10:{i:02d}:00Z")
            for i in range(60)
        ]
        result = build_handoff_from_entries(entries, "s-test", "/project")
        assert len(result["commands_run"]) == 50

    def test_empty_entries_produces_valid_handoff(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_entries

        result = build_handoff_from_entries([], "s-empty", "/project")
        assert result["session"] == "s-empty"
        assert result["files"] == {"read": [], "modified": [], "created": []}
        assert result["commands_run"] == []

    def test_git_state_included_when_present(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_entries

        entries = [
            _assistant_entry("Bash", {"command": "git push"}, "2025-01-15T10:00:00Z"),
        ]
        result = build_handoff_from_entries(entries, "s-test", "/project")
        assert result["git_state"] is not None
        assert "git push" in result["git_state"]["last_command"]

    def test_date_fallback_when_no_timestamps(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_entries

        entries = [{"type": "user", "message": {"content": "hi"}}]
        result = build_handoff_from_entries(entries, "s-test", "/project")
        # Should use current date as fallback
        assert result["date"] == datetime.now(UTC).strftime("%Y-%m-%d")

    def test_date_fallback_on_invalid_timestamp(self):
        from scripts.core.generate_mini_handoff import _extract_date

        result = _extract_date("not-a-date")
        assert result == datetime.now(UTC).strftime("%Y-%m-%d")


# ===========================================================================
# parse_state_events
# ===========================================================================


class TestParseStateEvents:
    """Tests for parsing Phase 3 state file events."""

    def test_parses_valid_events(self):
        from scripts.core.generate_mini_handoff import parse_state_events

        lines = _entries_to_lines([
            _state_event("Read", file="a.py"),
            _state_event("Bash", command="git status"),
        ])
        result = parse_state_events(lines)
        assert len(result) == 2

    def test_skips_malformed_lines(self):
        from scripts.core.generate_mini_handoff import parse_state_events

        lines = ["bad json", json.dumps(_state_event("Read", file="a.py"))]
        result = parse_state_events(lines)
        assert len(result) == 1

    def test_empty_input_returns_empty_list(self):
        from scripts.core.generate_mini_handoff import parse_state_events

        assert parse_state_events([]) == []


class TestBuildHandoffFromStateEvents:
    """Tests for building handoff from Phase 3 state events."""

    def test_read_files_tracked(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_state_events

        events = [_state_event("Read", file="a.py")]
        result = build_handoff_from_state_events(events, "s-test", "/project")
        assert "a.py" in result["files"]["read"]

    def test_edit_files_tracked_as_modified(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_state_events

        events = [_state_event("Edit", file="a.py")]
        result = build_handoff_from_state_events(events, "s-test", "/project")
        assert "a.py" in result["files"]["modified"]

    def test_write_files_tracked_as_created(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_state_events

        events = [_state_event("Write", file="new.py")]
        result = build_handoff_from_state_events(events, "s-test", "/project")
        assert "new.py" in result["files"]["created"]

    def test_bash_commands_extracted(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_state_events

        events = [_state_event("Bash", command="git status")]
        result = build_handoff_from_state_events(events, "s-test", "/project")
        assert "git status" in result["commands_run"]

    def test_tool_counts_sorted(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_state_events

        events = [
            _state_event("Read", file="a.py"),
            _state_event("Read", file="b.py"),
            _state_event("Edit", file="a.py"),
        ]
        result = build_handoff_from_state_events(events, "s-test", "/project")
        keys = list(result["tool_usage"].keys())
        assert keys[0] == "Read"

    def test_timestamps_extracted(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_state_events

        events = [
            _state_event("Read", file="a.py", timestamp="2025-01-15T10:00:00Z"),
            _state_event("Read", file="b.py", timestamp="2025-01-15T10:30:00Z"),
        ]
        result = build_handoff_from_state_events(events, "s-test", "/project")
        assert result["duration"]["first_timestamp"] == "2025-01-15T10:00:00Z"
        assert result["duration"]["last_timestamp"] == "2025-01-15T10:30:00Z"

    def test_deduplicates_files(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_state_events

        events = [
            _state_event("Read", file="a.py"),
            _state_event("Read", file="a.py"),
        ]
        result = build_handoff_from_state_events(events, "s-test", "/project")
        assert result["files"]["read"] == ["a.py"]

    def test_empty_events_produces_valid_handoff(self):
        from scripts.core.generate_mini_handoff import build_handoff_from_state_events

        result = build_handoff_from_state_events([], "s-empty", "/project")
        assert result["session"] == "s-empty"
        assert result["files"] == {"read": [], "modified": [], "created": []}


# ===========================================================================
# _sanitize_session_id
# ===========================================================================


class TestSanitizeSessionId:
    """Tests for session ID path traversal prevention."""

    def test_valid_simple_id(self):
        from scripts.core.generate_mini_handoff import _sanitize_session_id

        assert _sanitize_session_id("s-abc123") == "s-abc123"

    def test_valid_with_underscores(self):
        from scripts.core.generate_mini_handoff import _sanitize_session_id

        assert _sanitize_session_id("session_2025_01") == "session_2025_01"

    def test_rejects_path_traversal(self):
        import pytest

        from scripts.core.generate_mini_handoff import _sanitize_session_id

        with pytest.raises(ValueError, match="Invalid session_id"):
            _sanitize_session_id("../../../etc/passwd")

    def test_rejects_slashes(self):
        import pytest

        from scripts.core.generate_mini_handoff import _sanitize_session_id

        with pytest.raises(ValueError, match="Invalid session_id"):
            _sanitize_session_id("foo/bar")

    def test_rejects_empty_string(self):
        import pytest

        from scripts.core.generate_mini_handoff import _sanitize_session_id

        with pytest.raises(ValueError, match="Invalid session_id"):
            _sanitize_session_id("")

    def test_rejects_spaces(self):
        import pytest

        from scripts.core.generate_mini_handoff import _sanitize_session_id

        with pytest.raises(ValueError, match="Invalid session_id"):
            _sanitize_session_id("has spaces")
