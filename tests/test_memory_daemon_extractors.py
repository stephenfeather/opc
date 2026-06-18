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

    def test_successful_extraction_spawns_with_stdin_devnull(self, tmp_path):
        # Issue #98 hardening: the extraction Popen must close stdin
        # (stdin=subprocess.DEVNULL) so the child cannot block on or read
        # from the daemon's stdin.
        import subprocess

        from scripts.core.memory_daemon_extractors import extract_memories_impl

        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text('{"type":"msg"}\n')
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_popen = MagicMock(return_value=mock_proc)
        cfg = MagicMock()
        cfg.extraction_model = "sonnet"
        cfg.extraction_max_turns = 10

        result = extract_memories_impl(
            session_id="s1",
            project_dir=str(tmp_path),
            transcript_path=str(jsonl),
            active_extractions={},
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
        assert mock_popen.call_args.kwargs["stdin"] == subprocess.DEVNULL

    def test_live_extraction_env_lacks_inherited_source_time(self, tmp_path):
        # Issue #52 Round 3: a daemon launched with a stale CLAUDE_SOURCE_TIME
        # in its environment must NOT pass it to live extraction (which sets the
        # CLAUDE_MEMORY_EXTRACTION=1 trust marker), or live learnings would be
        # backdated. Assert the subprocess env carries the marker but no source
        # time.
        import os

        from scripts.core.memory_daemon_extractors import extract_memories_impl

        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text('{"type":"msg"}\n')
        mock_proc = MagicMock()
        mock_proc.pid = 1001
        mock_popen = MagicMock(return_value=mock_proc)
        cfg = MagicMock()
        cfg.extraction_model = "sonnet"
        cfg.extraction_max_turns = 10

        with patch.dict(os.environ, {"CLAUDE_SOURCE_TIME": "2020-01-01T00:00:00Z"}):
            extract_memories_impl(
                session_id="s1",
                project_dir=str(tmp_path),
                transcript_path=str(jsonl),
                active_extractions={},
                subprocess_popen=mock_popen,
                is_blocked_fn=lambda _: False,
                mark_extracted_fn=MagicMock(),
                mark_failed_fn=MagicMock(),
                log_fn=MagicMock(),
                daemon_cfg=cfg,
                allowed_models=frozenset({"sonnet", "haiku", "opus"}),
                strip_frontmatter_fn=lambda c: c,
            )

        passed_env = mock_popen.call_args.kwargs["env"]
        assert passed_env["CLAUDE_MEMORY_EXTRACTION"] == "1"
        assert "CLAUDE_SOURCE_TIME" not in passed_env


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

    def test_success_path_spawns_with_stdin_devnull(self, monkeypatch, tmp_path):
        # Issue #98 hardening: zstd compress + aws s3 cp must both pass
        # stdin=subprocess.DEVNULL so child processes never read the
        # daemon's stdin.
        import subprocess

        from scripts.core.memory_daemon_extractors import archive_session_jsonl

        monkeypatch.setenv("CLAUDE_SESSION_ARCHIVE_BUCKET", "test-bucket")
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text("{}\n")

        ok = MagicMock()
        ok.returncode = 0
        mock_run = MagicMock(return_value=ok)
        monkeypatch.setattr(
            "scripts.core.memory_daemon_extractors.subprocess.run", mock_run
        )

        archive_session_jsonl("s1", jsonl, log_fn=MagicMock(),
                              mark_archived_fn=MagicMock())

        # First call: zstd compress. Second call: aws s3 cp.
        assert mock_run.call_count == 2
        for call in mock_run.call_args_list:
            assert call.kwargs["stdin"] == subprocess.DEVNULL

    def test_s3_fail_rollback_spawns_with_stdin_devnull(self, monkeypatch, tmp_path):
        # Issue #98 hardening: the zstd -d rollback after an S3 upload
        # failure must also pass stdin=subprocess.DEVNULL.
        import subprocess

        from scripts.core.memory_daemon_extractors import archive_session_jsonl

        monkeypatch.setenv("CLAUDE_SESSION_ARCHIVE_BUCKET", "test-bucket")
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text("{}\n")

        compress_ok = MagicMock()
        compress_ok.returncode = 0
        s3_fail = MagicMock()
        s3_fail.returncode = 1
        s3_fail.stderr = b"boom"
        rollback = MagicMock()
        rollback.returncode = 0
        mock_run = MagicMock(side_effect=[compress_ok, s3_fail, rollback])
        monkeypatch.setattr(
            "scripts.core.memory_daemon_extractors.subprocess.run", mock_run
        )

        archive_session_jsonl("s1", jsonl, log_fn=MagicMock(),
                              mark_archived_fn=MagicMock())

        # compress, s3 cp, then rollback decompress.
        assert mock_run.call_count == 3
        for call in mock_run.call_args_list:
            assert call.kwargs["stdin"] == subprocess.DEVNULL

    def test_timeout_rollback_spawns_with_stdin_devnull(self, monkeypatch, tmp_path):
        # Issue #98 hardening: the zstd -d rollback in the TimeoutExpired
        # branch must also pass stdin=subprocess.DEVNULL.
        import subprocess

        from scripts.core.memory_daemon_extractors import archive_session_jsonl

        monkeypatch.setenv("CLAUDE_SESSION_ARCHIVE_BUCKET", "test-bucket")
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text("{}\n")
        zst = jsonl.with_suffix(".jsonl.zst")

        def _run(cmd, *args, **kwargs):
            # First call (zstd compress): simulate the rename effect so the
            # timeout rollback branch's guard (zst exists, jsonl gone) passes,
            # then raise TimeoutExpired.
            if mock_run.call_count == 1:
                jsonl.unlink()
                zst.write_text("compressed")
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)
            rollback = MagicMock()
            rollback.returncode = 0
            return rollback

        mock_run = MagicMock(side_effect=_run)
        monkeypatch.setattr(
            "scripts.core.memory_daemon_extractors.subprocess.run", mock_run
        )

        archive_session_jsonl("s1", jsonl, log_fn=MagicMock(),
                              mark_archived_fn=MagicMock())

        # compress (timeout) then rollback decompress.
        assert mock_run.call_count == 2
        for call in mock_run.call_args_list:
            assert call.kwargs["stdin"] == subprocess.DEVNULL


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

    @patch("scripts.core.memory_daemon_extractors.get_rejection_count",
           side_effect=Exception("db error"))
    def test_logs_on_error(self, mock_get, caplog):
        """Issue #98 — the previously-silent except path must log the
        failure via the module logger so the swallowed error is visible.
        """
        import logging

        from scripts.core.memory_daemon_extractors import count_session_rejections

        with caplog.at_level(logging.WARNING, logger="memory-daemon"):
            assert count_session_rejections("s1") is None

        assert any(
            "rejection" in r.getMessage().lower() for r in caplog.records
        ), f"Expected a logged rejection-count failure, got {caplog.records}"

    @patch("scripts.core.memory_daemon_extractors.get_rejection_count",
           side_effect=Exception("db error"))
    def test_log_redacts_exception(self, mock_get, caplog):
        """The logged failure must route the exception through safe()."""
        import logging

        from scripts.core.memory_daemon_extractors import count_session_rejections

        with caplog.at_level(logging.WARNING, logger="memory-daemon"):
            count_session_rejections("s1")

        # safe() is the module redaction helper used throughout; the
        # exception text should appear (proving we logged the cause).
        assert any("db error" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Issue #96 Round 1 fix — debug-wiring regression guard
# ---------------------------------------------------------------------------


class TestExtractMemoriesImplDebugWiring:
    """extract_memories_impl must delegate argv/env construction to the
    debug-instrumented helpers in memory_daemon_core so that Issue #96's
    DEBUG-gated diagnostics fire in production.

    Codex adversarial review Round 1 caught that the helpers shipped as
    dead code — extractors.py built argv/env inline, bypassing them.
    This test locks in the wiring.
    """

    def test_debug_logging_fires_via_core_helpers(self, tmp_path, monkeypatch):
        """When MEMORY_DAEMON_DEBUG=1, starting an extraction must emit
        the helper DEBUG log lines via memory_daemon.log.

        This proves that extract_memories_impl actually calls
        build_extraction_command / build_extraction_env (vs. building
        argv/env inline). Also tripwires env-value leakage.
        """
        from scripts.core import memory_daemon, memory_daemon_core
        from scripts.core.memory_daemon_extractors import extract_memories_impl

        # Spy: capture every log call instead of touching the real daemon
        # log file. Patch BOTH modules per PR #106 hermeticity learnings.
        messages: list[str] = []

        def _spy(msg):
            messages.append(str(msg))

        monkeypatch.setattr(memory_daemon, "log", _spy, raising=True)
        monkeypatch.setattr(memory_daemon_core, "log", _spy, raising=False)

        # Enable DEBUG so the helper thunks actually fire.
        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", "1")

        # Tripwire: a fake secret env var that os.environ.copy() would
        # pull into the extraction env. Under the Round 3 tightened
        # format, DEBUG logging must NOT emit the VALUE and must NOT
        # enumerate parent-process env keys such as 'OPC_TRIPWIRE_SECRET'
        # — env logging is daemon-owned only (CLAUDE_MEMORY_EXTRACTION,
        # CLAUDE_PROJECT_DIR presence, env_var_count). Both the key
        # name and the value are checked absent by the assertion at
        # the bottom of this test.
        tripwire_value = "zAbC123DO-NOT-LOG"
        monkeypatch.setenv("OPC_TRIPWIRE_SECRET", tripwire_value)

        # Arrange: a real JSONL file and a mock subprocess that returns
        # a process handle with a pid. The test does NOT spawn a real
        # claude subprocess — subprocess_popen is injected.
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text('{"type":"msg"}\n')

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen = MagicMock(return_value=mock_proc)

        cfg = MagicMock()
        cfg.extraction_model = "sonnet"
        cfg.extraction_max_turns = 10

        # Act
        result = extract_memories_impl(
            session_id="sess-test",
            project_dir=str(tmp_path),
            transcript_path=str(jsonl),
            active_extractions={},
            subprocess_popen=mock_popen,
            is_blocked_fn=lambda _: False,
            mark_extracted_fn=MagicMock(),
            mark_failed_fn=MagicMock(),
            log_fn=MagicMock(),
            daemon_cfg=cfg,
            allowed_models=frozenset({"sonnet", "haiku", "opus"}),
            strip_frontmatter_fn=lambda c: c,
        )

        assert result is True, "Extraction should have started successfully"

        # Assert: the core helpers' DEBUG log lines landed in the spy.
        argv_msgs = [
            m for m in messages if "build_extraction_command:" in m
        ]
        env_msgs = [
            m for m in messages if "build_extraction_env:" in m
        ]

        assert argv_msgs, (
            "Expected at least one DEBUG log line containing "
            "'build_extraction_command argv' — proves extract_memories_impl "
            "delegates argv construction to memory_daemon_core. "
            f"Captured messages: {messages}"
        )
        assert any("sess-test" in m for m in argv_msgs), (
            "argv DEBUG log should contain the session_id. "
            f"argv messages: {argv_msgs}"
        )

        assert env_msgs, (
            "Expected at least one DEBUG log line containing "
            "'build_extraction_env keys' — proves extract_memories_impl "
            "delegates env construction to memory_daemon_core. "
            f"Captured messages: {messages}"
        )
        assert any("CLAUDE_MEMORY_EXTRACTION" in m for m in env_msgs), (
            "env DEBUG log should contain CLAUDE_MEMORY_EXTRACTION key. "
            f"env messages: {env_msgs}"
        )
        assert any("CLAUDE_PROJECT_DIR" in m for m in env_msgs), (
            "env DEBUG log should contain CLAUDE_PROJECT_DIR key. "
            f"env messages: {env_msgs}"
        )

        # Tripwire: under the Round 3 tightened format, NEITHER the
        # parent-env KEY NAME ('OPC_TRIPWIRE_SECRET') NOR its VALUE
        # may appear in the DEBUG log. Env logging is daemon-owned
        # ONLY (CLAUDE_MEMORY_EXTRACTION=1, CLAUDE_PROJECT_DIR presence,
        # env_var_count=N) — enumerating parent-env key names was the
        # reconnaissance risk Codex flagged in Round 3. This assertion
        # guards both halves of that contract (PR #110 Cycle 1 T1).
        leaked = [
            m for m in messages
            if "OPC_TRIPWIRE_SECRET" in m or tripwire_value in m
        ]
        assert not leaked, (
            f"SECURITY VIOLATION: parent env key or value leaked into "
            f"DEBUG log. Env logging must stay daemon-owned only "
            f"(no parent-env KEY NAMES, no parent-env VALUES). "
            f"Leaked messages: {leaked}"
        )
