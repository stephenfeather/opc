"""Tests for artifact_mark.py — TDD+FP compliance refactor (S18).

Tests pure functions extracted from artifact_mark.py:
- Argument parsing and validation
- Output formatting (confirmation, errors, recent list)
- Summary truncation
- Handoff ID resolution
"""

import argparse

import pytest

from scripts.core.artifact_mark import (
    build_arg_parser,
    format_confirmation,
    format_error_not_found,
    format_handoff_row,
    format_recent_list,
    resolve_handoff_id,
    truncate_summary,
)  # noqa: I001

# --- truncate_summary ---


class TestTruncateSummary:
    """Tests for truncate_summary pure function."""

    def test_short_text_unchanged(self):
        """Text shorter than max_len returns unchanged."""
        assert truncate_summary("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        """Text exactly at max_len returns unchanged."""
        assert truncate_summary("hello", 5) == "hello"

    def test_long_text_truncated_with_ellipsis(self):
        """Text longer than max_len is truncated with '...'."""
        result = truncate_summary("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8

    def test_none_returns_placeholder(self):
        """None input returns placeholder string."""
        assert truncate_summary(None, 50) == "(no summary)"

    def test_empty_string_returns_placeholder(self):
        """Empty string returns placeholder."""
        assert truncate_summary("", 50) == "(no summary)"


# --- format_handoff_row ---


class TestFormatHandoffRow:
    """Tests for format_handoff_row pure function."""

    def test_formats_row_with_summary(self):
        """Row with all fields formats correctly."""
        row = ("abc123def456", "session-1", "Fix the bug")
        result = format_handoff_row(row, max_summary_len=50)
        assert "abc123def456" in result
        assert "session-1" in result
        assert "Fix the bug" in result

    def test_formats_row_with_long_summary(self):
        """Long summary is truncated."""
        row = ("abc123def456", "session-1", "A" * 100)
        result = format_handoff_row(row, max_summary_len=50)
        assert "..." in result

    def test_formats_row_with_none_summary(self):
        """None summary shows placeholder."""
        row = ("abc123def456", "session-1", None)
        result = format_handoff_row(row, max_summary_len=50)
        assert "(no summary)" in result

    def test_id_truncated_to_12_chars(self):
        """ID is displayed truncated to 12 characters."""
        row = ("abcdef123456789", "session-1", "task")
        result = format_handoff_row(row, max_summary_len=50)
        assert "abcdef123456" in result


# --- format_recent_list ---


class TestFormatRecentList:
    """Tests for format_recent_list pure function."""

    def test_empty_list(self):
        """Empty list returns empty string."""
        assert format_recent_list([]) == ""

    def test_single_row(self):
        """Single row formats correctly."""
        rows = [("abc123def456", "session-1", "Fix bug")]
        result = format_recent_list(rows, max_summary_len=50)
        assert "abc123def456" in result
        assert "session-1" in result

    def test_multiple_rows(self):
        """Multiple rows each appear on separate lines."""
        rows = [
            ("id1", "session-1", "Fix bug"),
            ("id2", "session-2", "Add feature"),
        ]
        result = format_recent_list(rows, max_summary_len=50)
        lines = result.strip().split("\n")
        assert len(lines) == 2


# --- resolve_handoff_id ---


class TestResolveHandoffId:
    """Tests for resolve_handoff_id pure function."""

    def test_explicit_id_returned(self):
        """When handoff_id is provided, returns it directly."""
        result = resolve_handoff_id(handoff_id="abc123", use_latest=False, latest_id=None)
        assert result == ("abc123", None)

    def test_latest_flag_returns_latest_id(self):
        """When use_latest is True and latest_id exists, returns it."""
        result = resolve_handoff_id(handoff_id=None, use_latest=True, latest_id="xyz789")
        assert result == ("xyz789", None)

    def test_latest_flag_no_handoffs_returns_error(self):
        """When use_latest is True but no handoffs exist, returns error."""
        handoff_id, error = resolve_handoff_id(
            handoff_id=None, use_latest=True, latest_id=None
        )
        assert handoff_id is None
        assert error is not None
        assert "No handoffs found" in error

    def test_neither_flag_returns_error(self):
        """When neither handoff_id nor use_latest, returns error."""
        handoff_id, error = resolve_handoff_id(
            handoff_id=None, use_latest=False, latest_id=None
        )
        assert handoff_id is None
        assert error is not None


# --- format_confirmation ---


class TestFormatConfirmation:
    """Tests for format_confirmation pure function."""

    def test_basic_confirmation(self):
        """Confirmation includes outcome, db type, id, session."""
        handoff = ("abc123", "session-1", "Fix the bug")
        result = format_confirmation(handoff, "SUCCEEDED", "", "PostgreSQL")
        assert "SUCCEEDED" in result
        assert "PostgreSQL" in result
        assert "abc123" in result
        assert "session-1" in result

    def test_confirmation_with_notes(self):
        """Notes appear in confirmation when provided."""
        handoff = ("abc123", "session-1", "Fix the bug")
        result = format_confirmation(handoff, "PARTIAL_PLUS", "Almost done", "SQLite")
        assert "Almost done" in result

    def test_confirmation_without_notes(self):
        """Empty notes are not displayed."""
        handoff = ("abc123", "session-1", "Fix the bug")
        result = format_confirmation(handoff, "SUCCEEDED", "", "PostgreSQL")
        assert "Notes" not in result

    def test_confirmation_with_long_summary(self):
        """Long summary is truncated in confirmation."""
        handoff = ("abc123", "session-1", "A" * 100)
        result = format_confirmation(handoff, "SUCCEEDED", "", "PostgreSQL")
        assert "..." in result

    def test_confirmation_no_summary(self):
        """Missing summary doesn't cause error."""
        handoff = ("abc123", "session-1", None)
        result = format_confirmation(handoff, "SUCCEEDED", "", "PostgreSQL")
        assert "SUCCEEDED" in result
        # Summary line should not appear when None
        assert "Summary" not in result


# --- format_error_not_found ---


class TestFormatErrorNotFound:
    """Tests for format_error_not_found pure function."""

    def test_error_includes_handoff_id(self):
        """Error message includes the handoff ID."""
        result = format_error_not_found("abc123", "PostgreSQL", [])
        assert "abc123" in result

    def test_error_includes_db_type(self):
        """Error message includes the database type."""
        result = format_error_not_found("abc123", "SQLite", [])
        assert "SQLite" in result

    def test_error_includes_recent_handoffs(self):
        """Error message includes recent handoff list."""
        recent = [("id1", "session-1", "Task 1")]
        result = format_error_not_found("abc123", "PostgreSQL", recent)
        assert "session-1" in result

    def test_error_empty_recent(self):
        """Error with no recent handoffs still works."""
        result = format_error_not_found("abc123", "PostgreSQL", [])
        assert "abc123" in result


# --- build_arg_parser ---


class TestBuildArgParser:
    """Tests for build_arg_parser pure function."""

    def test_parser_returns_argparse(self):
        """Returns an ArgumentParser instance."""
        parser = build_arg_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_accepts_handoff_and_outcome(self):
        """Parser accepts --handoff and --outcome."""
        parser = build_arg_parser()
        args = parser.parse_args(["--handoff", "abc123", "--outcome", "SUCCEEDED"])
        assert args.handoff == "abc123"
        assert args.outcome == "SUCCEEDED"

    def test_parser_accepts_latest_flag(self):
        """Parser accepts --latest flag."""
        parser = build_arg_parser()
        args = parser.parse_args(["--latest", "--outcome", "FAILED"])
        assert args.latest is True
        assert args.outcome == "FAILED"

    def test_parser_accepts_get_latest_id(self):
        """Parser accepts --get-latest-id flag."""
        parser = build_arg_parser()
        args = parser.parse_args(["--get-latest-id"])
        assert args.get_latest_id is True

    def test_parser_accepts_notes(self):
        """Parser accepts --notes."""
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--handoff", "abc", "--outcome", "SUCCEEDED", "--notes", "All good"]
        )
        assert args.notes == "All good"

    def test_parser_notes_default_empty(self):
        """Notes default to empty string."""
        parser = build_arg_parser()
        args = parser.parse_args(["--handoff", "abc", "--outcome", "SUCCEEDED"])
        assert args.notes == ""

    def test_parser_rejects_invalid_outcome(self):
        """Invalid outcome values are rejected."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--handoff", "abc", "--outcome", "INVALID"])

    def test_parser_rejects_handoff_and_latest_together(self):
        """--handoff and --latest are mutually exclusive."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--handoff", "abc", "--latest", "--outcome", "SUCCEEDED"]
            )

    def test_parser_rejects_get_latest_id_with_handoff(self):
        """--get-latest-id and --handoff are mutually exclusive."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--get-latest-id", "--handoff", "abc", "--outcome", "SUCCEEDED"]
            )

    def test_parser_rejects_get_latest_id_with_latest(self):
        """--get-latest-id and --latest are mutually exclusive."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--get-latest-id", "--latest", "--outcome", "SUCCEEDED"])
