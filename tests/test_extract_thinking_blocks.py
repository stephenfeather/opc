"""Tests for extract_thinking_blocks.py — TDD + FP compliance refactor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.core.extract_thinking_blocks import (
    compute_stats,
    extract_thinking_blocks,
    format_blocks_json,
    format_blocks_text,
    has_perception_signal,
    main,
    parse_jsonl_entry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_jsonl_line(
    *,
    msg_type: str = "assistant",
    content: list | str | None = None,
    timestamp: str = "2026-04-02T12:00:00Z",
) -> str:
    """Build a single JSONL line for testing."""
    return json.dumps({"type": msg_type, "timestamp": timestamp, "message": {"content": content}})


def _thinking_content(text: str) -> list[dict]:
    return [{"type": "thinking", "thinking": text}]


def _text_content(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


@pytest.fixture()
def sample_jsonl(tmp_path: Path) -> Path:
    """Create a sample JSONL file with mixed content."""
    lines = [
        _make_jsonl_line(content=_thinking_content("I realized the issue is in the parser")),
        _make_jsonl_line(content=_text_content("Here is the answer")),
        _make_jsonl_line(content=_thinking_content("Let me check the docs")),
        _make_jsonl_line(
            content=[
                {"type": "thinking", "thinking": "Actually, turns out the bug is here"},
                {"type": "text", "text": "Found it"},
            ],
        ),
        _make_jsonl_line(msg_type="system", content=_thinking_content("system thinking")),
        "not valid json\n",
        _make_jsonl_line(content="just a string, not a list"),
    ]
    p = tmp_path / "session.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.fixture()
def empty_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    return p


# ---------------------------------------------------------------------------
# Tests: has_perception_signal (pure)
# ---------------------------------------------------------------------------

class TestHasPerceptionSignal:
    """Pure function: text -> bool."""

    def test_detects_realized(self):
        assert has_perception_signal("I realized the issue is in the parser") is True

    def test_detects_actually(self):
        assert has_perception_signal("Actually, this is different") is True

    def test_detects_turns_out(self):
        assert has_perception_signal("It turns out the bug was here") is True

    def test_detects_wait(self):
        assert has_perception_signal("Wait, that doesn't make sense") is True

    def test_detects_this_works(self):
        assert has_perception_signal("this works perfectly") is True

    def test_no_signal_in_plain_text(self):
        assert has_perception_signal("Let me check the documentation") is False

    def test_empty_string(self):
        assert has_perception_signal("") is False

    def test_case_insensitive(self):
        assert has_perception_signal("ACTUALLY this is wrong") is True


# ---------------------------------------------------------------------------
# Tests: parse_jsonl_entry (pure)
# ---------------------------------------------------------------------------

class TestParseJsonlEntry:
    """Pure function: str -> list[dict]."""

    def test_extracts_thinking_block(self):
        line = _make_jsonl_line(content=_thinking_content("deep thought"))
        result = parse_jsonl_entry(line)
        assert len(result) == 1
        assert result[0]["thinking"] == "deep thought"
        assert result[0]["line_num"] == 0  # default

    def test_extracts_multiple_thinking_blocks(self):
        content = [
            {"type": "thinking", "thinking": "first"},
            {"type": "text", "text": "hello"},
            {"type": "thinking", "thinking": "second"},
        ]
        line = _make_jsonl_line(content=content)
        result = parse_jsonl_entry(line)
        assert len(result) == 2
        assert result[0]["thinking"] == "first"
        assert result[1]["thinking"] == "second"

    def test_skips_non_assistant_user_types(self):
        line = _make_jsonl_line(msg_type="system", content=_thinking_content("nope"))
        assert parse_jsonl_entry(line) == []

    def test_skips_string_content(self):
        line = _make_jsonl_line(content="just a string")
        assert parse_jsonl_entry(line) == []

    def test_skips_empty_thinking(self):
        line = _make_jsonl_line(content=[{"type": "thinking", "thinking": ""}])
        assert parse_jsonl_entry(line) == []

    def test_invalid_json_returns_empty(self):
        assert parse_jsonl_entry("not json at all") == []

    def test_skips_non_dict_message(self):
        """Schema drift: message is a string instead of a dict."""
        line = json.dumps({"type": "assistant", "message": "oops"})
        assert parse_jsonl_entry(line) == []

    def test_skips_non_dict_content_items(self):
        """Schema drift: content array contains non-dict items."""
        line = _make_jsonl_line(content=["just a string", 42, None])
        assert parse_jsonl_entry(line) == []

    def test_skips_non_string_thinking_value(self):
        """Schema drift: thinking value is truthy but not a string."""
        line = _make_jsonl_line(content=[{"type": "thinking", "thinking": 12345}])
        assert parse_jsonl_entry(line) == []

    def test_skips_missing_message_key(self):
        """Entry has type but no message key at all."""
        line = json.dumps({"type": "assistant", "timestamp": "2026-01-01T00:00:00Z"})
        assert parse_jsonl_entry(line) == []

    def test_line_num_passed_through(self):
        line = _make_jsonl_line(content=_thinking_content("thought"))
        result = parse_jsonl_entry(line, line_num=42)
        assert result[0]["line_num"] == 42

    def test_timestamp_preserved(self):
        line = _make_jsonl_line(
            content=_thinking_content("thought"),
            timestamp="2026-01-01T00:00:00Z",
        )
        result = parse_jsonl_entry(line)
        assert result[0]["timestamp"] == "2026-01-01T00:00:00Z"

    def test_has_perception_signal_set(self):
        line = _make_jsonl_line(content=_thinking_content("I realized something"))
        result = parse_jsonl_entry(line)
        assert result[0]["has_perception_signal"] is True

    def test_no_perception_signal_set(self):
        line = _make_jsonl_line(content=_thinking_content("checking docs"))
        result = parse_jsonl_entry(line)
        assert result[0]["has_perception_signal"] is False


# ---------------------------------------------------------------------------
# Tests: compute_stats (pure)
# ---------------------------------------------------------------------------

class TestComputeStats:
    """Pure function: list[dict] -> dict."""

    def test_basic_stats(self):
        blocks = [
            {"has_perception_signal": True},
            {"has_perception_signal": False},
            {"has_perception_signal": True},
        ]
        stats = compute_stats(blocks)
        assert stats["total"] == 3
        assert stats["with_signal"] == 2
        assert stats["ratio"] == pytest.approx(66.7, abs=0.1)

    def test_empty_blocks(self):
        stats = compute_stats([])
        assert stats["total"] == 0
        assert stats["with_signal"] == 0
        assert stats["ratio"] is None

    def test_all_signals(self):
        blocks = [{"has_perception_signal": True}] * 5
        stats = compute_stats(blocks)
        assert stats["ratio"] == pytest.approx(100.0)

    def test_no_signals(self):
        blocks = [{"has_perception_signal": False}] * 3
        stats = compute_stats(blocks)
        assert stats["ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: format_blocks_text (pure)
# ---------------------------------------------------------------------------

class TestFormatBlocksText:
    """Pure function: list[dict] -> str."""

    def test_single_block(self):
        blocks = [{"line_num": 1, "has_perception_signal": False, "thinking": "hello"}]
        result = format_blocks_text(blocks)
        assert "[Line 1]" in result
        assert "hello" in result
        assert "*" not in result  # no perception marker

    def test_perception_marker(self):
        blocks = [{"line_num": 5, "has_perception_signal": True, "thinking": "aha"}]
        result = format_blocks_text(blocks)
        assert "*" in result

    def test_multiple_blocks_separated(self):
        blocks = [
            {"line_num": 1, "has_perception_signal": False, "thinking": "first"},
            {"line_num": 2, "has_perception_signal": False, "thinking": "second"},
        ]
        result = format_blocks_text(blocks)
        assert "---" in result
        assert "first" in result
        assert "second" in result

    def test_empty_blocks(self):
        assert format_blocks_text([]) == ""


# ---------------------------------------------------------------------------
# Tests: format_blocks_json (pure)
# ---------------------------------------------------------------------------

class TestFormatBlocksJson:
    """Pure function: list[dict] -> str."""

    def test_valid_json_output(self):
        blocks = [{"line_num": 1, "thinking": "x", "has_perception_signal": False}]
        result = format_blocks_json(blocks)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["thinking"] == "x"

    def test_empty_blocks(self):
        result = format_blocks_json([])
        assert json.loads(result) == []


# ---------------------------------------------------------------------------
# Tests: extract_thinking_blocks (I/O boundary)
# ---------------------------------------------------------------------------

class TestExtractThinkingBlocks:
    """I/O function: reads file, delegates to pure parse_jsonl_entry."""

    def test_extracts_all_blocks(self, sample_jsonl: Path):
        blocks = extract_thinking_blocks(sample_jsonl)
        # 3 thinking blocks: line 1, line 3, line 4 (system is skipped)
        assert len(blocks) == 3

    def test_filter_perception(self, sample_jsonl: Path):
        blocks = extract_thinking_blocks(sample_jsonl, filter_perception=True)
        assert all(b["has_perception_signal"] for b in blocks)
        # "realized" (line 1) and "turns out" (line 4)
        assert len(blocks) == 2

    def test_empty_file(self, empty_jsonl: Path):
        blocks = extract_thinking_blocks(empty_jsonl)
        assert blocks == []

    def test_malformed_line_does_not_stop_extraction(self, tmp_path: Path):
        """One schema-drifted line must not prevent later valid blocks from being extracted."""
        lines = [
            json.dumps({"type": "assistant", "message": "not a dict"}),  # malformed
            _make_jsonl_line(content=_thinking_content("valid block after bad line")),
        ]
        p = tmp_path / "mixed.jsonl"
        p.write_text("\n".join(lines) + "\n")
        blocks = extract_thinking_blocks(p)
        assert len(blocks) == 1
        assert blocks[0]["thinking"] == "valid block after bad line"

    def test_line_numbers_are_1_indexed(self, sample_jsonl: Path):
        blocks = extract_thinking_blocks(sample_jsonl)
        assert blocks[0]["line_num"] == 1


# ---------------------------------------------------------------------------
# Tests: main() CLI handler (I/O boundary)
# ---------------------------------------------------------------------------

class TestMainCLI:
    """CLI entry point tests — verify arg parsing and output routing."""

    def test_text_output_to_stdout(self, sample_jsonl: Path, capsys):
        sys.argv = ["prog", "--jsonl", str(sample_jsonl)]
        main()
        out = capsys.readouterr().out
        assert "[Line 1]" in out

    def test_json_output_to_stdout(self, sample_jsonl: Path, capsys):
        sys.argv = ["prog", "--jsonl", str(sample_jsonl), "--format", "json"]
        main()
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert len(parsed) == 3

    def test_stats_output(self, sample_jsonl: Path, capsys):
        sys.argv = ["prog", "--jsonl", str(sample_jsonl), "--stats"]
        main()
        out = capsys.readouterr().out
        assert "Total thinking blocks: 3" in out
        assert "With perception signals: 2" in out

    def test_stats_empty_file(self, empty_jsonl: Path, capsys):
        sys.argv = ["prog", "--jsonl", str(empty_jsonl), "--stats"]
        main()
        out = capsys.readouterr().out
        assert "Ratio: N/A" in out

    def test_filter_flag(self, sample_jsonl: Path, capsys):
        sys.argv = ["prog", "--jsonl", str(sample_jsonl), "--filter", "--format", "json"]
        main()
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert len(parsed) == 2
        assert all(b["has_perception_signal"] for b in parsed)

    def test_output_to_file(self, sample_jsonl: Path, tmp_path: Path, capsys):
        out_file = tmp_path / "out.txt"
        sys.argv = ["prog", "--jsonl", str(sample_jsonl), "--output", str(out_file)]
        main()
        assert out_file.exists()
        assert "[Line 1]" in out_file.read_text()
        err = capsys.readouterr().err
        assert "Wrote 3 blocks" in err

    def test_file_not_found(self, tmp_path: Path):
        sys.argv = ["prog", "--jsonl", str(tmp_path / "nope.jsonl")]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
