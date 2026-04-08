"""Tests for DaemonState and state management in memory_daemon.py.

Phase 3 of S30 TDD+FP refactor.
"""

from __future__ import annotations

from dataclasses import fields
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

    def test_raises_without_state(self):
        import scripts.core.memory_daemon as mod

        original = mod._daemon_state
        try:
            mod._daemon_state = None
            with pytest.raises(RuntimeError, match="daemon context"):
                mod.get_active_extractions()
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

    def test_raises_without_state(self):
        import scripts.core.memory_daemon as mod

        original = mod._daemon_state
        try:
            mod._daemon_state = None
            with pytest.raises(RuntimeError, match="daemon context"):
                mod.get_pending_queue()
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

        daemon_tick(self.state)

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

        daemon_tick(self.state)
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
        daemon_tick(self.state)
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
        daemon_tick(self.state)

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
        """Sessions with exited_at set (past grace) get extracted."""
        from datetime import datetime

        from scripts.core.memory_daemon import daemon_tick

        # Return a session with exited_at set (truly stale)
        mock_stale.return_value = [
            ("sess-2", "proj", "/t.jsonl", 1234, datetime(2020, 1, 1)),
        ]
        daemon_tick(self.state)

        mock_mark_extracting.assert_called_once_with("sess-2")
        mock_queue.assert_called_once_with("sess-2", "proj", "/t.jsonl")
