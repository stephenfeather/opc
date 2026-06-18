"""Tests for DaemonState and state management in memory_daemon.py.

Phase 3 of S30 TDD+FP refactor.
"""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


class TestDaemonState:
    """DaemonState dataclass holds all mutable daemon state."""

    def test_has_expected_fields(self):
        from scripts.core.memory_daemon import DaemonState

        names = {f.name for f in fields(DaemonState)}
        assert names == {
            "active_extractions",
            "pending_queue",
            "pattern_proc",
            "last_pattern_run",
        }

    def test_factory_creates_empty_state(self):
        from scripts.core.memory_daemon import create_daemon_state

        state = create_daemon_state()
        assert state.active_extractions == {}
        assert state.pending_queue == []
        assert state.pattern_proc is None
        assert state.last_pattern_run == 0.0

    def test_active_extractions_is_mutable_dict(self):
        from scripts.core.memory_daemon import create_daemon_state

        state = create_daemon_state()
        state.active_extractions[123] = ("sess", "proc", "path", "proj", 0)
        assert 123 in state.active_extractions

    def test_pending_queue_is_mutable_list(self):
        from scripts.core.memory_daemon import create_daemon_state

        state = create_daemon_state()
        state.pending_queue.append(("sess", "proj", None))
        assert len(state.pending_queue) == 1


class TestGetActiveExtractions:
    """get_active_extractions returns the live dict from _daemon_state."""

    def test_lazy_inits_without_state(self):
        import scripts.core.memory_daemon as mod

        original = mod._daemon_state
        try:
            mod._daemon_state = None
            result = mod.get_active_extractions()
            assert isinstance(result, dict)
            assert mod._daemon_state is not None  # lazy-initialized
        finally:
            mod._daemon_state = original

    def test_returns_dict_when_state_set(self):
        import scripts.core.memory_daemon as mod

        state = mod.create_daemon_state()
        state.active_extractions[42] = ("s", "p", "j", "d", 0)
        original = mod._daemon_state
        try:
            mod._daemon_state = state
            result = mod.get_active_extractions()
            assert result is state.active_extractions
            assert 42 in result
        finally:
            mod._daemon_state = original


class TestGetPendingQueue:
    """get_pending_queue returns the live list from _daemon_state."""

    def test_lazy_inits_without_state(self):
        import scripts.core.memory_daemon as mod

        original = mod._daemon_state
        try:
            mod._daemon_state = None
            result = mod.get_pending_queue()
            assert isinstance(result, list)
            assert mod._daemon_state is not None
        finally:
            mod._daemon_state = original

    def test_returns_list_when_state_set(self):
        import scripts.core.memory_daemon as mod

        state = mod.create_daemon_state()
        state.pending_queue.append(("s", "p", None))
        original = mod._daemon_state
        try:
            mod._daemon_state = state
            result = mod.get_pending_queue()
            assert result is state.pending_queue
        finally:
            mod._daemon_state = original


# ---------------------------------------------------------------------------
# Step 3.2 — daemon_tick
# ---------------------------------------------------------------------------


class TestDaemonTick:
    """daemon_tick executes one iteration of the daemon loop."""

    @pytest.fixture(autouse=True)
    def _setup_state(self):
        """Set up DaemonState for daemon_tick tests."""
        import scripts.core.memory_daemon as mod

        self.mod = mod
        self.state = mod.create_daemon_state()
        self.original = mod._daemon_state
        mod._daemon_state = self.state
        yield
        mod._daemon_state = self.original

    @patch("scripts.core.memory_daemon._run_pattern_detection_batch")
    @patch("scripts.core.memory_daemon._check_pattern_detection")
    @patch("scripts.core.memory_daemon.get_stale_sessions", return_value=[])
    @patch("scripts.core.memory_daemon.process_pending_queue")
    @patch("scripts.core.memory_daemon.watchdog_stuck_extractions")
    @patch("scripts.core.memory_daemon.reap_completed_extractions")
    def test_calls_reap_watchdog_queue_in_order(
        self, mock_reap, mock_watchdog, mock_queue, mock_stale,
        mock_check, mock_run
    ):
        from scripts.core.memory_daemon import daemon_tick

        daemon_tick()

        mock_reap.assert_called_once()
        mock_watchdog.assert_called_once()
        mock_queue.assert_called_once()
        mock_stale.assert_called_once()

    @patch("scripts.core.memory_daemon._run_pattern_detection_batch")
    @patch("scripts.core.memory_daemon._check_pattern_detection")
    @patch("scripts.core.memory_daemon.get_stale_sessions", return_value=[])
    @patch("scripts.core.memory_daemon.process_pending_queue")
    @patch("scripts.core.memory_daemon.watchdog_stuck_extractions")
    @patch("scripts.core.memory_daemon.reap_completed_extractions")
    def test_checks_pattern_detection(
        self, mock_reap, mock_watchdog, mock_queue, mock_stale,
        mock_check, mock_run
    ):
        from scripts.core.memory_daemon import daemon_tick

        daemon_tick()
        mock_check.assert_called_once()

    @patch("scripts.core.memory_daemon._run_pattern_detection_batch")
    @patch("scripts.core.memory_daemon._check_pattern_detection")
    @patch("scripts.core.memory_daemon.get_stale_sessions", return_value=[])
    @patch("scripts.core.memory_daemon.process_pending_queue")
    @patch("scripts.core.memory_daemon.watchdog_stuck_extractions")
    @patch("scripts.core.memory_daemon.reap_completed_extractions")
    @patch("scripts.core.memory_daemon.use_postgres", return_value=True)
    def test_triggers_pattern_detection_when_due(
        self, mock_use_pg, mock_reap, mock_watchdog, mock_queue,
        mock_stale, mock_check, mock_run
    ):
        from scripts.core.memory_daemon import daemon_tick

        # Set last_pattern_run far in the past
        self.state.last_pattern_run = 0
        daemon_tick()
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Step 3.3 — daemon_tick uses filter_truly_stale_sessions
# ---------------------------------------------------------------------------


class TestDaemonTickStaleFiltering:
    """daemon_tick uses filter_truly_stale_sessions and marks newly dead."""

    @pytest.fixture(autouse=True)
    def _setup_state(self):
        import scripts.core.memory_daemon as mod

        self.mod = mod
        self.state = mod.create_daemon_state()
        self.original = mod._daemon_state
        mod._daemon_state = self.state
        yield
        mod._daemon_state = self.original

    @patch("scripts.core.memory_daemon._run_pattern_detection_batch")
    @patch("scripts.core.memory_daemon._check_pattern_detection")
    @patch("scripts.core.memory_daemon.process_pending_queue")
    @patch("scripts.core.memory_daemon.watchdog_stuck_extractions")
    @patch("scripts.core.memory_daemon.reap_completed_extractions")
    @patch("scripts.core.memory_daemon.queue_or_extract")
    @patch("scripts.core.memory_daemon.mark_extracting")
    @patch("scripts.core.memory_daemon.mark_session_exited")
    @patch("scripts.core.memory_daemon.log")
    @patch("scripts.core.memory_daemon._is_process_alive", return_value=False)
    @patch("scripts.core.memory_daemon.get_stale_sessions")
    def test_marks_newly_dead_and_skips(
        self, mock_stale, mock_alive, mock_log, mock_mark_exited,
        mock_mark_extracting, mock_queue, mock_reap, mock_watchdog,
        mock_ppq, mock_check, mock_run
    ):
        """Sessions with no exited_at get marked exited, not extracted."""
        from scripts.core.memory_daemon import daemon_tick

        # Return a session with exited_at=None (newly dead)
        mock_stale.return_value = [
            ("sess-1", "proj", "/t.jsonl", 1234, None),
        ]
        daemon_tick()

        mock_mark_exited.assert_called_once_with("sess-1")
        mock_mark_extracting.assert_not_called()

    @patch("scripts.core.memory_daemon._run_pattern_detection_batch")
    @patch("scripts.core.memory_daemon._check_pattern_detection")
    @patch("scripts.core.memory_daemon.process_pending_queue")
    @patch("scripts.core.memory_daemon.watchdog_stuck_extractions")
    @patch("scripts.core.memory_daemon.reap_completed_extractions")
    @patch("scripts.core.memory_daemon.queue_or_extract")
    @patch("scripts.core.memory_daemon.mark_extracting")
    @patch("scripts.core.memory_daemon.mark_session_exited")
    @patch("scripts.core.memory_daemon.log")
    @patch("scripts.core.memory_daemon._is_process_alive", return_value=False)
    @patch("scripts.core.memory_daemon.get_stale_sessions")
    def test_extracts_truly_stale(
        self, mock_stale, mock_alive, mock_log, mock_mark_exited,
        mock_mark_extracting, mock_queue, mock_reap, mock_watchdog,
        mock_ppq, mock_check, mock_run
    ):
        """Sessions with exited_at set (past grace) get queued for extraction.

        Note: mark_extracting is NOT called by daemon_tick -- it is called
        inside queue_or_extract/process_pending_queue when extraction actually
        starts (fix for issue #82).
        """
        from datetime import datetime

        from scripts.core.memory_daemon import daemon_tick

        # Return a session with exited_at set (truly stale)
        mock_stale.return_value = [
            ("sess-2", "proj", "/t.jsonl", 1234, datetime(2020, 1, 1)),
        ]
        daemon_tick()

        mock_mark_extracting.assert_not_called()
        mock_queue.assert_called_once_with("sess-2", "proj", "/t.jsonl")


# ---------------------------------------------------------------------------
# Step 4.4 — os.waitpid zombie reaping in reap_completed_extractions
# ---------------------------------------------------------------------------


class TestReapCompletedExtractions:
    """reap_completed_extractions removes completed PIDs from active set."""

    @pytest.fixture(autouse=True)
    def _setup_state(self):
        import scripts.core.memory_daemon as mod

        self.mod = mod
        self.state = mod.create_daemon_state()
        self.original = mod._daemon_state
        mod._daemon_state = self.state
        yield
        mod._daemon_state = self.original

    @patch("scripts.core.memory_daemon.archive_session_jsonl")
    @patch("scripts.core.memory_daemon._generate_mini_handoff")
    @patch("scripts.core.memory_daemon._extract_and_store_workflows")
    @patch("scripts.core.memory_daemon._calibrate_session_confidence")
    @patch("scripts.core.memory_daemon.mark_extracted")
    @patch("scripts.core.memory_daemon._count_session_rejections", return_value=None)
    @patch("scripts.core.memory_daemon._count_session_learnings", return_value=None)
    @patch("scripts.core.memory_daemon.log")
    def test_removes_completed_pid_from_active(
        self, mock_log, mock_count, mock_rej,
        mock_mark, mock_cal, mock_wf, mock_hoff, mock_arch
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.pid = 42
        self.state.active_extractions[42] = (
            "sess-1", mock_proc, Path("/t.jsonl"), "proj", 0
        )

        from scripts.core.memory_daemon import reap_completed_extractions

        count = reap_completed_extractions()

        assert count == 1
        assert 42 not in self.state.active_extractions
        mock_mark.assert_called_once_with("sess-1")

    @patch("scripts.core.memory_daemon.mark_extraction_failed")
    @patch("scripts.core.memory_daemon._count_session_rejections", return_value=None)
    @patch("scripts.core.memory_daemon._count_session_learnings", return_value=None)
    @patch("scripts.core.memory_daemon.log")
    def test_marks_failed_on_nonzero_exit(
        self, mock_log, mock_count, mock_rej, mock_fail
    ):
        import io

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.pid = 42
        mock_proc.stderr = io.BytesIO(b"some error")
        self.state.active_extractions[42] = (
            "sess-1", mock_proc, Path("/t.jsonl"), "proj", 0
        )

        from scripts.core.memory_daemon import reap_completed_extractions

        count = reap_completed_extractions()

        assert count == 1
        mock_fail.assert_called_once_with("sess-1", last_error="some error")

    @patch("scripts.core.memory_daemon.archive_session_jsonl")
    @patch("scripts.core.memory_daemon._generate_mini_handoff")
    @patch("scripts.core.memory_daemon._extract_and_store_workflows")
    @patch("scripts.core.memory_daemon._calibrate_session_confidence")
    @patch("scripts.core.memory_daemon.mark_extracted")
    @patch("scripts.core.memory_daemon._count_session_rejections", return_value=None)
    @patch("scripts.core.memory_daemon._count_session_learnings", return_value=None)
    @patch("scripts.core.memory_daemon.log")
    def test_success_reap_does_not_block_on_open_stderr_pipes(
        self, mock_log, mock_count, mock_rej,
        mock_mark, mock_cal, mock_wf, mock_hoff, mock_arch
    ):
        """The success-path pipe close must be opportunistic (timeout=0). Reap
        of N completed extractions whose stderr write ends stay open must
        finish near-instantly — not N * the failure-diagnostic budget."""
        import os
        import time as _time

        open_write_fds = []
        try:
            for i in range(4):
                read_fd, write_fd = os.pipe()
                open_write_fds.append(write_fd)
                os.write(write_fd, b"noise")  # buffered; write end never closed
                proc = MagicMock()
                proc.poll.return_value = 0
                proc.pid = 100 + i
                proc.stderr = os.fdopen(read_fd, "rb", buffering=0)
                self.state.active_extractions[100 + i] = (
                    f"sess-{i}", proc, Path("/t.jsonl"), "proj", 0
                )

            from scripts.core.memory_daemon import reap_completed_extractions

            start = _time.monotonic()
            count = reap_completed_extractions()
            elapsed = _time.monotonic() - start

            assert count == 4
            assert elapsed < 0.5, f"reap blocked for {elapsed:.2f}s on open pipes"
        finally:
            for write_fd in open_write_fds:
                os.close(write_fd)


# ---------------------------------------------------------------------------
# Issue #98 — per-stage pipeline timing helper (_timed_stage)
# ---------------------------------------------------------------------------


class TestTimedStage:
    """_timed_stage wraps a pipeline stage, returns its result, and emits a
    DEBUG-gated timing line (stage name + elapsed ms). Lines only surface
    under --debug so normal operation stays quiet.
    """

    def test_returns_wrapped_result(self, monkeypatch):
        from scripts.core import memory_daemon

        monkeypatch.setattr(memory_daemon, "DEBUG", False, raising=False)
        result = memory_daemon._timed_stage("calibrate", lambda: "ok")
        assert result == "ok"

    def test_emits_debug_line_when_debug_on(self, monkeypatch):
        from scripts.core import memory_daemon

        messages: list[str] = []
        monkeypatch.setattr(memory_daemon, "log", lambda m: messages.append(str(m)))
        monkeypatch.setattr(memory_daemon, "DEBUG", True, raising=False)

        memory_daemon._timed_stage("calibrate", lambda: None)

        assert any("calibrate" in m for m in messages), (
            f"Expected a debug line naming the stage, got {messages}"
        )
        assert any("ms" in m for m in messages), (
            f"Expected an elapsed-ms duration in the debug line, got {messages}"
        )

    def test_no_line_when_debug_off(self, monkeypatch):
        from scripts.core import memory_daemon

        messages: list[str] = []
        monkeypatch.setattr(memory_daemon, "log", lambda m: messages.append(str(m)))
        monkeypatch.setattr(memory_daemon, "DEBUG", False, raising=False)

        memory_daemon._timed_stage("calibrate", lambda: None)

        assert messages == [], (
            f"Expected zero log output with DEBUG off, got {messages}"
        )


# ---------------------------------------------------------------------------
# Issue #98 — watchdog stderr parity + non-blocking _drain_proc_stderr
# ---------------------------------------------------------------------------


class TestWatchdogStuckExtractions:
    """The watchdog-killed path captures the child stderr for parity with the
    reap path, using the shared non-blocking _drain_proc_stderr helper."""

    @pytest.fixture(autouse=True)
    def _setup_state(self):
        import scripts.core.memory_daemon as mod

        self.mod = mod
        self.state = mod.create_daemon_state()
        self.original = mod._daemon_state
        mod._daemon_state = self.state
        yield
        mod._daemon_state = self.original

    @patch("scripts.core.memory_daemon.mark_extraction_failed")
    @patch("scripts.core.memory_daemon.log")
    def test_captures_stderr_on_kill(self, mock_log, mock_fail):
        import io

        mock_proc = MagicMock()
        mock_proc.stderr = io.BytesIO(b"hung child traceback")
        # start_time 0 → elapsed is effectively "now", far past any timeout.
        self.state.active_extractions[42] = (
            "sess-1", mock_proc, Path("/t.jsonl"), "proj", 0
        )

        from scripts.core.memory_daemon import watchdog_stuck_extractions

        killed = watchdog_stuck_extractions()

        assert killed == 1
        logged = " ".join(str(c.args[0]) for c in mock_log.call_args_list)
        assert "hung child traceback" in logged
        assert mock_fail.call_count == 1
        assert "hung child traceback" in (
            mock_fail.call_args.kwargs.get("last_error") or ""
        )

    @patch("scripts.core.memory_daemon.mark_extraction_failed")
    @patch("scripts.core.memory_daemon.log")
    def test_skips_blocking_drain_when_kill_times_out(self, mock_log, mock_fail):
        """If kill()/wait() fails and the child is still alive, the watchdog
        must NOT call the blocking stderr read — doing so would hang the daemon
        waiting for EOF on a pipe whose write-end is still open."""
        import subprocess

        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=5)
        mock_proc.poll.return_value = None  # still alive after failed kill
        mock_proc.stderr = MagicMock()
        self.state.active_extractions[42] = (
            "sess-1", mock_proc, Path("/t.jsonl"), "proj", 0
        )

        from scripts.core.memory_daemon import watchdog_stuck_extractions

        killed = watchdog_stuck_extractions()

        assert killed == 1
        mock_proc.stderr.read.assert_not_called()
        assert mock_fail.call_count == 1
        assert "not drained" in (mock_fail.call_args.kwargs.get("last_error") or "")

    @patch("scripts.core.memory_daemon.mark_extraction_failed")
    @patch("scripts.core.memory_daemon.log")
    def test_persisted_last_error_is_sanitized(self, mock_log, mock_fail):
        """Captured stderr is persisted to last_error, which is interpolated
        raw into a downstream log line. It must be control-char sanitized
        (no raw newlines/ESC) before storage."""
        import io

        mock_proc = MagicMock()
        mock_proc.stderr = io.BytesIO(b"forged\nlog line\x1b[31mred")
        self.state.active_extractions[42] = (
            "sess-1", mock_proc, Path("/t.jsonl"), "proj", 0
        )

        from scripts.core.memory_daemon import watchdog_stuck_extractions

        watchdog_stuck_extractions()

        last_error = mock_fail.call_args.kwargs.get("last_error") or ""
        assert "\n" not in last_error
        assert "\x1b" not in last_error
        assert "forged" in last_error

    def test_drain_is_time_bounded_when_write_end_stays_open(self):
        """Even when the parent has 'exited', a surviving descendant can keep
        the stderr write-end open so EOF never arrives. The drain must return
        within its time budget and still surface partial bytes."""
        import os
        import time as _time

        from scripts.core.memory_daemon import _drain_proc_stderr

        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"partial diagnostic before hang")
            reader = os.fdopen(read_fd, "rb", buffering=0)
            read_fd = -1
            proc = MagicMock()
            proc.stderr = reader

            start = _time.monotonic()
            text = _drain_proc_stderr(proc, timeout=0.3)
            elapsed = _time.monotonic() - start

            assert elapsed < 3.0, f"drain blocked for {elapsed:.1f}s (should be bounded)"
            assert "partial diagnostic before hang" in text
        finally:
            os.close(write_fd)
            if read_fd != -1:
                os.close(read_fd)

    def test_drain_timeout_zero_still_reads_buffered_data(self):
        """timeout=0.0 is an opportunistic non-blocking poll, not a no-op: it
        must still drain bytes already buffered in the pipe."""
        import os

        from scripts.core.memory_daemon import _drain_proc_stderr

        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"buffered before the loop")
            reader = os.fdopen(read_fd, "rb", buffering=0)
            read_fd = -1
            proc = MagicMock()
            proc.stderr = reader

            text = _drain_proc_stderr(proc, timeout=0.0)

            assert "buffered before the loop" in text
        finally:
            os.close(write_fd)
            if read_fd != -1:
                os.close(read_fd)


# ---------------------------------------------------------------------------
# Step 5.1 — Daemon lifecycle tests
# ---------------------------------------------------------------------------


class TestIsRunning:
    """is_running checks PID file and process liveness."""

    def test_returns_false_when_no_pid_file(self, tmp_path):
        import scripts.core.memory_daemon as mod

        original = mod.PID_FILE
        try:
            mod.PID_FILE = tmp_path / "nonexistent.pid"
            running, pid = mod.is_running()
            assert running is False
            assert pid is None
        finally:
            mod.PID_FILE = original

    def test_returns_true_for_own_pid(self, tmp_path):
        import scripts.core.memory_daemon as mod

        pid_file = tmp_path / "test.pid"
        pid_file.write_text(str(os.getpid()))
        original = mod.PID_FILE
        try:
            mod.PID_FILE = pid_file
            running, pid = mod.is_running()
            assert running is True
            assert pid == os.getpid()
        finally:
            mod.PID_FILE = original

    def test_cleans_stale_pid_file(self, tmp_path):
        import scripts.core.memory_daemon as mod

        pid_file = tmp_path / "test.pid"
        pid_file.write_text("999999999")  # very unlikely to be running
        original = mod.PID_FILE
        try:
            mod.PID_FILE = pid_file
            running, pid = mod.is_running()
            assert running is False
            assert not pid_file.exists()  # cleaned up stale file
        finally:
            mod.PID_FILE = original


class TestStopDaemon:
    """stop_daemon sends SIGTERM and cleans PID file."""

    def test_reports_not_running(self, tmp_path):
        import scripts.core.memory_daemon as mod

        original = mod.PID_FILE
        try:
            mod.PID_FILE = tmp_path / "nonexistent.pid"
            rc = mod.stop_daemon()
            assert rc == 0
        finally:
            mod.PID_FILE = original


# ---------------------------------------------------------------------------
# Step 5.2 — CLI dispatch tests
# ---------------------------------------------------------------------------


class TestMainCli:
    """main() dispatches to start/stop/status."""

    @patch("scripts.core.memory_daemon.start_daemon", return_value=0)
    def test_start_command(self, mock_start):
        import scripts.core.memory_daemon as mod

        with patch("sys.argv", ["memory_daemon.py", "start"]):
            rc = mod.main()
        assert rc == 0
        mock_start.assert_called_once()

    @patch("scripts.core.memory_daemon.stop_daemon", return_value=0)
    def test_stop_command(self, mock_stop):
        import scripts.core.memory_daemon as mod

        with patch("sys.argv", ["memory_daemon.py", "stop"]):
            rc = mod.main()
        assert rc == 0
        mock_stop.assert_called_once()

    @patch("scripts.core.memory_daemon.status_daemon")
    def test_status_command(self, mock_status):
        import scripts.core.memory_daemon as mod

        with patch("sys.argv", ["memory_daemon.py", "status"]):
            mod.main()
        mock_status.assert_called_once()

    def test_no_command_prints_help(self):
        import scripts.core.memory_daemon as mod

        with patch("sys.argv", ["memory_daemon.py"]):
            rc = mod.main()
        assert rc == 1


# ---------------------------------------------------------------------------
# Step 5.3 — Import cycle smoke test
# ---------------------------------------------------------------------------


class TestImportSmoke:
    """All four modules import without cycles."""

    def test_all_modules_import_cleanly(self):
        import importlib

        for name in [
            "scripts.core.memory_daemon_core",
            "scripts.core.memory_daemon_db",
            "scripts.core.memory_daemon_extractors",
            "scripts.core.memory_daemon",
        ]:
            mod = importlib.import_module(name)
            assert mod is not None
