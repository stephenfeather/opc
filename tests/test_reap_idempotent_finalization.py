"""Tests for idempotent completed-extraction finalization (GitHub issue #207).

`reap_completed_extractions()` must remove a completed (exited) child from the
active set even when post-completion DB finalization raises. Otherwise a single
completed extraction replays on every daemon poll under a persistent DB outage:
re-running the diagnostic count queries (WARNING flood, #97/#98) and the
``pg_connect()`` retry storm indefinitely.

Verifies:
- A successful-exit child whose ``mark_extracted()`` raises is still removed
  from ``active_extractions`` (no replay).
- A failed-exit child whose ``mark_extraction_failed()`` raises is still removed.
- Across two ticks, the diagnostic count helpers run at most once — the second
  tick is a no-op because the PID was removed.
- The finalization failure is logged (once), and reap still reports the PID as
  reaped (return count includes it).
"""

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_active_extractions():
    """Set up DaemonState so get_active_extractions() works in tests."""
    import scripts.core.memory_daemon as mod

    state = mod.create_daemon_state()
    original = mod._daemon_state
    mod._daemon_state = state
    yield
    mod._daemon_state = original


@pytest.fixture(autouse=True)
def _silence_daemon_log():
    """Prevent test calls from writing to the production daemon log."""
    with patch("scripts.core.memory_daemon.log"):
        yield


@pytest.fixture()
def tmp_jsonl(tmp_path: Path) -> Path:
    jsonl = tmp_path / "session.jsonl"
    jsonl.write_text('{"type":"message"}\n')
    return jsonl


def _make_mock_proc(pid: int, exit_code: int, stderr_text: str = "") -> MagicMock:
    mock = MagicMock()
    mock.pid = pid
    mock.poll.return_value = exit_code
    mock.stderr = io.BytesIO(stderr_text.encode())
    return mock


def test_reap_removes_pid_when_mark_extracted_raises(tmp_jsonl):
    """A successful child must be reaped even if mark_extracted() raises (#207)."""
    from scripts.core.memory_daemon import (
        get_active_extractions,
        reap_completed_extractions,
    )

    proc = _make_mock_proc(pid=701, exit_code=0)
    ae = get_active_extractions()
    ae[701] = ("sess-db-outage", proc, tmp_jsonl, "/tmp/proj", time.time() - 30)

    with (
        patch(
            "scripts.core.memory_daemon.mark_extracted",
            side_effect=RuntimeError("pg_connect failed after max_retries"),
        ),
        patch("scripts.core.memory_daemon._count_session_learnings", return_value=3),
        patch("scripts.core.memory_daemon._count_session_rejections", return_value=0),
    ):
        # Must not propagate the finalization error.
        reaped = reap_completed_extractions()

    assert 701 not in ae, "Completed PID must be removed even when mark_extracted raises"
    assert reaped == 1, "reap must still report the dead child as reaped"


def test_reap_removes_pid_when_mark_extraction_failed_raises(tmp_jsonl):
    """A failed child must be reaped even if mark_extraction_failed() raises (#207)."""
    from scripts.core.memory_daemon import (
        get_active_extractions,
        reap_completed_extractions,
    )

    proc = _make_mock_proc(pid=702, exit_code=1, stderr_text="boom")
    ae = get_active_extractions()
    ae[702] = ("sess-fail-outage", proc, tmp_jsonl, "/tmp/proj", time.time() - 30)

    with patch(
        "scripts.core.memory_daemon.mark_extraction_failed",
        side_effect=RuntimeError("pg_connect failed after max_retries"),
    ):
        reaped = reap_completed_extractions()

    assert 702 not in ae, "Failed PID must be removed even when mark_extraction_failed raises"
    assert reaped == 1


def test_reap_no_replay_of_count_queries_across_ticks(tmp_jsonl):
    """The diagnostic count helpers must not re-run on a second tick (#207).

    Under a persistent DB outage, re-running the count queries every poll is the
    log-flood / retry-storm symptom. Removing the PID on the first tick prevents
    the second tick from re-polling the same dead child.
    """
    from scripts.core.memory_daemon import (
        get_active_extractions,
        reap_completed_extractions,
    )

    proc = _make_mock_proc(pid=703, exit_code=0)
    ae = get_active_extractions()
    ae[703] = ("sess-replay", proc, tmp_jsonl, "/tmp/proj", time.time() - 30)

    with (
        patch(
            "scripts.core.memory_daemon.mark_extracted",
            side_effect=RuntimeError("pg_connect failed after max_retries"),
        ),
        patch(
            "scripts.core.memory_daemon._count_session_learnings", return_value=1
        ) as count_learnings,
        patch("scripts.core.memory_daemon._count_session_rejections", return_value=0),
    ):
        reap_completed_extractions()  # tick 1
        reap_completed_extractions()  # tick 2 — must be a no-op

    assert count_learnings.call_count == 1, (
        "Count query must run at most once; a second call means the dead PID "
        "replayed (the #207 flood)."
    )


def test_reap_logs_finalization_failure(tmp_jsonl):
    """A swallowed finalization failure must still be logged for diagnosability."""
    from scripts.core.memory_daemon import (
        get_active_extractions,
        reap_completed_extractions,
    )

    proc = _make_mock_proc(pid=704, exit_code=0)
    ae = get_active_extractions()
    ae[704] = ("sess-logged", proc, tmp_jsonl, "/tmp/proj", time.time() - 30)

    with (
        patch(
            "scripts.core.memory_daemon.mark_extracted",
            side_effect=RuntimeError("pg_connect failed after max_retries"),
        ),
        patch("scripts.core.memory_daemon._count_session_learnings", return_value=0),
        patch("scripts.core.memory_daemon._count_session_rejections", return_value=0),
        patch("scripts.core.memory_daemon.log") as mock_log,
    ):
        reap_completed_extractions()

    log_messages = " ".join(str(c) for c in mock_log.call_args_list)
    assert (
        "pg_connect failed" in log_messages or "finaliz" in log_messages.lower()
    ), "Finalization failure must be logged"
