"""Tests for memory_daemon_extractors — extraction subprocess and post-extraction pipeline.

Phase 4 of S30 TDD+FP refactor.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Step 4.1 — _is_extraction_blocked
# ---------------------------------------------------------------------------


class TestIsExtractionBlocked:
    """_is_extraction_blocked checks for .claude/no-extract sentinel."""

    def test_returns_false_for_empty_project(self):
        from scripts.core.memory_daemon_extractors import is_extraction_blocked

        assert is_extraction_blocked("") is False

    def test_returns_false_when_no_sentinel(self, tmp_path):
        from scripts.core.memory_daemon_extractors import is_extraction_blocked

        assert is_extraction_blocked(str(tmp_path)) is False

    def test_returns_true_when_sentinel_exists(self, tmp_path):
        sentinel = tmp_path / ".claude" / "no-extract"
        sentinel.parent.mkdir(parents=True)
        sentinel.touch()

        from scripts.core.memory_daemon_extractors import is_extraction_blocked

        assert is_extraction_blocked(str(tmp_path)) is True


# ---------------------------------------------------------------------------
# Step 4.1 — extract_memories_impl
# ---------------------------------------------------------------------------


class TestExtractMemoriesImpl:
    """extract_memories_impl uses only injected dependencies."""

    def test_blocked_project_marks_extracted_and_returns_false(self, tmp_path):
        from scripts.core.memory_daemon_extractors import extract_memories_impl

        mock_log = MagicMock()
        mock_mark = MagicMock()

        result = extract_memories_impl(
            session_id="s1",
            project_dir=str(tmp_path),
            transcript_path=None,
            active_extractions={},
            subprocess_popen=MagicMock(),
            is_blocked_fn=lambda _: True,
            mark_extracted_fn=mock_mark,
            mark_failed_fn=MagicMock(),
            log_fn=mock_log,
            daemon_cfg=MagicMock(),
            allowed_models=frozenset({"sonnet"}),
            strip_frontmatter_fn=lambda c: c,
        )

        assert result is False
        mock_mark.assert_called_once_with("s1")

    def test_no_transcript_marks_extracted_and_returns_false(self, tmp_path):
        from scripts.core.memory_daemon_extractors import extract_memories_impl

        mock_mark = MagicMock()

        result = extract_memories_impl(
            session_id="s1",
            project_dir=str(tmp_path),
            transcript_path=None,
            active_extractions={},
            subprocess_popen=MagicMock(),
            is_blocked_fn=lambda _: False,
            mark_extracted_fn=mock_mark,
            mark_failed_fn=MagicMock(),
            log_fn=MagicMock(),
            daemon_cfg=MagicMock(),
            allowed_models=frozenset({"sonnet"}),
            strip_frontmatter_fn=lambda c: c,
        )

        assert result is False
        mock_mark.assert_called_once_with("s1")

    def test_invalid_model_marks_failed_and_returns_false(self, tmp_path):
        from scripts.core.memory_daemon_extractors import extract_memories_impl

        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text('{"type":"msg"}\n')
        mock_mark_extracted = MagicMock()
        mock_mark_failed = MagicMock()
        cfg = MagicMock()
        cfg.extraction_model = "gpt-evil"

        result = extract_memories_impl(
            session_id="s1",
            project_dir=str(tmp_path),
            transcript_path=str(jsonl),
            active_extractions={},
            subprocess_popen=MagicMock(),
            is_blocked_fn=lambda _: False,
            mark_extracted_fn=mock_mark_extracted,
            mark_failed_fn=mock_mark_failed,
            log_fn=MagicMock(),
            daemon_cfg=cfg,
            allowed_models=frozenset({"sonnet", "haiku", "opus"}),
            strip_frontmatter_fn=lambda c: c,
        )

        assert result is False
        mock_mark_extracted.assert_not_called()
        mock_mark_failed.assert_called_once_with("s1")

    def test_successful_extraction_starts_subprocess(self, tmp_path):
        from scripts.core.memory_daemon_extractors import extract_memories_impl

        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text('{"type":"msg"}\n')
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_popen = MagicMock(return_value=mock_proc)
        ae = {}
        cfg = MagicMock()
        cfg.extraction_model = "sonnet"
        cfg.extraction_max_turns = 10

        result = extract_memories_impl(
            session_id="s1",
            project_dir=str(tmp_path),
            transcript_path=str(jsonl),
            active_extractions=ae,
            subprocess_popen=mock_popen,
            is_blocked_fn=lambda _: False,
            mark_extracted_fn=MagicMock(),
            mark_failed_fn=MagicMock(),
            log_fn=MagicMock(),
            daemon_cfg=cfg,
            allowed_models=frozenset({"sonnet", "haiku", "opus"}),
            strip_frontmatter_fn=lambda c: c,
        )

        assert result is True
        assert 999 in ae
        mock_popen.assert_called_once()


# ---------------------------------------------------------------------------
# Step 4.2 — archive_session_jsonl
# ---------------------------------------------------------------------------


class TestArchiveSessionJsonl:
    """archive_session_jsonl compresses and uploads to S3."""

    def test_noop_without_bucket(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_SESSION_ARCHIVE_BUCKET", raising=False)
        from scripts.core.memory_daemon_extractors import archive_session_jsonl

        # Should return without error
        archive_session_jsonl("s1", Path("/nonexistent"), log_fn=MagicMock(),
                              mark_archived_fn=MagicMock())

    def test_noop_when_no_jsonl(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_SESSION_ARCHIVE_BUCKET", "test-bucket")
        from scripts.core.memory_daemon_extractors import archive_session_jsonl

        mock_log = MagicMock()
        archive_session_jsonl("s1", None, log_fn=mock_log,
                              mark_archived_fn=MagicMock())


# ---------------------------------------------------------------------------
# Step 4.3d — _count_session_rejections
# ---------------------------------------------------------------------------


class TestCountSessionRejections:
    """count_session_rejections wraps store_learning.get_rejection_count."""

    @patch("scripts.core.memory_daemon_extractors.get_rejection_count", return_value=3)
    def test_returns_count(self, mock_get):
        from scripts.core.memory_daemon_extractors import count_session_rejections

        assert count_session_rejections("s1") == 3

    @patch("scripts.core.memory_daemon_extractors.get_rejection_count",
           side_effect=Exception("db error"))
    def test_returns_none_on_error(self, mock_get):
        from scripts.core.memory_daemon_extractors import count_session_rejections

        assert count_session_rejections("s1") is None
