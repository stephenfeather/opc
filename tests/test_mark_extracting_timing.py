"""Tests for issue #82: mark_extracting must only fire on actual extraction.

The bug: daemon_tick() calls mark_extracting() before queue_or_extract(),
so sessions that get queued (not immediately extracted) are incorrectly
marked as 'extracting' with extraction_attempts incremented.

Fix: move mark_extracting() into the actual spawn paths so it only fires
when extraction actually starts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestQueueOrExtractMarkTiming:
    """mark_extracting should only be called when extraction actually starts."""

    @pytest.fixture(autouse=True)
    def _setup_state(self):
        import scripts.core.memory_daemon as mod

        self.mod = mod
        self.state = mod.create_daemon_state()
        self.original = mod._daemon_state
        mod._daemon_state = self.state
        yield
        mod._daemon_state = self.original

    @patch("scripts.core.memory_daemon.extract_memories")
    @patch("scripts.core.memory_daemon.mark_extracting")
    @patch("scripts.core.memory_daemon.log")
    def test_mark_extracting_called_on_immediate_extract(
        self, mock_log, mock_mark, mock_extract
    ):
        """When under concurrency limit, mark_extracting fires before extract."""
        # No active extractions -> should extract immediately
        assert len(self.state.active_extractions) == 0

        self.mod.queue_or_extract("sess-1", "proj", "/t.jsonl")

        mock_mark.assert_called_once_with("sess-1")
        mock_extract.assert_called_once_with("sess-1", "proj", "/t.jsonl")

    @patch("scripts.core.memory_daemon.extract_memories")
    @patch("scripts.core.memory_daemon.mark_extracting")
    @patch("scripts.core.memory_daemon.log")
    def test_mark_extracting_not_called_when_queued(
        self, mock_log, mock_mark, mock_extract
    ):
        """When at concurrency limit, mark_extracting must NOT fire."""
        # Fill active_extractions to max_concurrent
        for i in range(self.mod._max_concurrent()):
            self.state.active_extractions[i] = (
                f"busy-{i}", MagicMock(), Path("/t.jsonl"), "proj", 0
            )

        self.mod.queue_or_extract("sess-queued", "proj", "/t.jsonl")

        # Session was queued, not extracted -- mark_extracting must not fire
        mock_mark.assert_not_called()
        mock_extract.assert_not_called()
        assert len(self.state.pending_queue) == 1

    @patch("scripts.core.memory_daemon.extract_memories")
    @patch("scripts.core.memory_daemon.mark_extracting")
    @patch("scripts.core.memory_daemon.log")
    def test_mark_extracting_called_on_dequeue(
        self, mock_log, mock_mark, mock_extract
    ):
        """When dequeuing from pending_queue, mark_extracting fires."""
        # Pre-populate queue with a pending session
        self.state.pending_queue.append(("sess-pending", "proj", "/t.jsonl"))
        # No active extractions -> dequeue should proceed
        assert len(self.state.active_extractions) == 0

        self.mod.process_pending_queue()

        mock_mark.assert_called_once_with("sess-pending")
        mock_extract.assert_called_once_with("sess-pending", "proj", "/t.jsonl")


class TestMarkExtractingConditional:
    """mark_extracting must NOT be called when extract_memories returns False."""

    @pytest.fixture(autouse=True)
    def _setup_state(self):
        import scripts.core.memory_daemon as mod

        self.mod = mod
        self.state = mod.create_daemon_state()
        self.original = mod._daemon_state
        mod._daemon_state = self.state
        yield
        mod._daemon_state = self.original

    @patch("scripts.core.memory_daemon.extract_memories", return_value=False)
    @patch("scripts.core.memory_daemon.mark_extracting")
    @patch("scripts.core.memory_daemon.log")
    def test_mark_extracting_not_called_when_extract_returns_false_immediate(
        self, mock_log, mock_mark, mock_extract
    ):
        """queue_or_extract: mark_extracting must NOT fire when extract_memories returns False."""
        assert len(self.state.active_extractions) == 0

        self.mod.queue_or_extract("sess-fail", "proj", "/t.jsonl")

        mock_extract.assert_called_once_with("sess-fail", "proj", "/t.jsonl")
        mock_mark.assert_not_called()

    @patch("scripts.core.memory_daemon.extract_memories", return_value=False)
    @patch("scripts.core.memory_daemon.mark_extracting")
    @patch("scripts.core.memory_daemon.log")
    def test_mark_extracting_not_called_when_dequeue_extract_returns_false(
        self, mock_log, mock_mark, mock_extract
    ):
        """process_pending_queue: mark_extracting must NOT fire when extract_memories returns False."""
        self.state.pending_queue.append(("sess-fail-dequeue", "proj", "/t.jsonl"))
        assert len(self.state.active_extractions) == 0

        self.mod.process_pending_queue()

        mock_extract.assert_called_once_with("sess-fail-dequeue", "proj", "/t.jsonl")
        mock_mark.assert_not_called()


class TestDedupGuard:
    """queue_or_extract must not add duplicate entries to pending_queue."""

    @pytest.fixture(autouse=True)
    def _setup_state(self):
        import scripts.core.memory_daemon as mod

        self.mod = mod
        self.state = mod.create_daemon_state()
        self.original = mod._daemon_state
        mod._daemon_state = self.state
        yield
        mod._daemon_state = self.original

    @patch("scripts.core.memory_daemon.extract_memories")
    @patch("scripts.core.memory_daemon.mark_extracting")
    @patch("scripts.core.memory_daemon.log")
    def test_no_duplicate_in_pending_queue(self, mock_log, mock_mark, mock_extract):
        """Multiple daemon_tick calls at concurrency limit must not duplicate queue entries."""
        # Fill active_extractions to max_concurrent
        for i in range(self.mod._max_concurrent()):
            self.state.active_extractions[i] = (
                f"busy-{i}", MagicMock(), Path("/t.jsonl"), "proj", 0
            )

        # Call queue_or_extract twice for the same session (simulates two ticks)
        self.mod.queue_or_extract("sess-dup", "proj", "/t.jsonl")
        self.mod.queue_or_extract("sess-dup", "proj", "/t.jsonl")

        # Should appear only once in queue
        queued = [item[0] for item in self.state.pending_queue]
        assert queued.count("sess-dup") == 1

    @patch("scripts.core.memory_daemon.extract_memories")
    @patch("scripts.core.memory_daemon.mark_extracting")
    @patch("scripts.core.memory_daemon.log")
    def test_no_duplicate_when_already_active(self, mock_log, mock_mark, mock_extract):
        """queue_or_extract skips session already in active_extractions."""
        # Put sess-active in active_extractions (under concurrency limit)
        self.state.active_extractions[999] = (
            "sess-active", MagicMock(), Path("/t.jsonl"), "proj", 0
        )
        # active count is 1, below max_concurrent -- but should still skip

        self.mod.queue_or_extract("sess-active", "proj", "/t.jsonl")

        # Should not be queued and extract_memories should not be called again
        mock_extract.assert_not_called()
        assert len(self.state.pending_queue) == 0


class TestDaemonTickNoEarlyMark:
    """daemon_tick must not call mark_extracting directly -- it delegates."""

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
    def test_daemon_tick_does_not_call_mark_extracting(
        self, mock_stale, mock_alive, mock_log, mock_mark_exited,
        mock_mark_extracting, mock_queue, mock_reap, mock_watchdog,
        mock_ppq, mock_check, mock_run
    ):
        """daemon_tick should NOT call mark_extracting -- queue_or_extract handles it."""
        mock_stale.return_value = [
            ("sess-2", "proj", "/t.jsonl", 1234, datetime(2020, 1, 1)),
        ]

        self.mod.daemon_tick()

        # daemon_tick should delegate to queue_or_extract without calling mark_extracting
        mock_mark_extracting.assert_not_called()
        mock_queue.assert_called_once_with("sess-2", "proj", "/t.jsonl")
