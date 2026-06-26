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


# ---------------------------------------------------------------------------
# Bounded finalization retry (recover from a *transient* DB outage without a
# daemon restart). reap removes the dead PID immediately, but the completed
# outcome is queued and retried with backoff so the session does not strand in
# 'extracting' forever (adversarial-review R1 finding).
# ---------------------------------------------------------------------------


def test_finalization_retried_after_transient_db_recovery(tmp_jsonl):
    """A finalization that fails once is retried and succeeds after DB recovery."""
    from scripts.core.memory_daemon import (
        get_active_extractions,
        get_pending_finalizations,
        reap_completed_extractions,
        retry_pending_finalizations,
    )

    proc = _make_mock_proc(pid=801, exit_code=0)
    ae = get_active_extractions()
    ae[801] = ("sess-transient", proc, tmp_jsonl, "/tmp/proj", time.time() - 30)

    calls = {"n": 0}

    def flaky_mark(_sid):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("pg_connect failed after max_retries")

    with (
        patch("scripts.core.memory_daemon.mark_extracted", side_effect=flaky_mark),
        patch(
            "scripts.core.memory_daemon._count_session_learnings", return_value=2
        ) as count_learnings,
        patch("scripts.core.memory_daemon._count_session_rejections", return_value=0),
        patch("scripts.core.memory_daemon._calibrate_session_confidence"),
        patch("scripts.core.memory_daemon._extract_and_store_workflows"),
        patch("scripts.core.memory_daemon._generate_mini_handoff"),
        patch("scripts.core.memory_daemon.archive_session_jsonl"),
    ):
        reap_completed_extractions()  # tick 1: mark_extracted raises -> enqueued
        assert 801 not in ae, "dead PID removed immediately"

        pq = get_pending_finalizations()
        assert len(pq) == 1, "failed finalization must be queued for retry"

        # Force the backoff deadline due so the retry fires this tick.
        pq[0].next_attempt_at = 0.0
        finalized = retry_pending_finalizations()  # tick 2: DB recovered

    assert finalized == 1
    assert calls["n"] == 2, "mark_extracted retried exactly once after the failure"
    assert count_learnings.call_count == 1, "diagnostics must not re-run on retry"
    assert len(get_pending_finalizations()) == 0, "queue drains after success"


def test_finalization_gives_up_after_max_attempts(tmp_jsonl):
    """A persistent DB outage drains the queue after the bounded attempt cap."""
    from scripts.core.memory_daemon import (
        _FINALIZATION_MAX_ATTEMPTS,
        get_active_extractions,
        get_pending_finalizations,
        reap_completed_extractions,
        retry_pending_finalizations,
    )

    proc = _make_mock_proc(pid=802, exit_code=0)
    ae = get_active_extractions()
    ae[802] = ("sess-persist", proc, tmp_jsonl, "/tmp/proj", time.time() - 30)

    with (
        patch(
            "scripts.core.memory_daemon.mark_extracted",
            side_effect=RuntimeError("db down"),
        ),
        patch("scripts.core.memory_daemon._count_session_learnings", return_value=0),
        patch("scripts.core.memory_daemon._count_session_rejections", return_value=0),
    ):
        reap_completed_extractions()  # attempt 1, enqueued
        pq = get_pending_finalizations()
        # Drive every remaining attempt to exhaustion.
        for _ in range(_FINALIZATION_MAX_ATTEMPTS + 1):
            for pf in pq:
                pf.next_attempt_at = 0.0
            retry_pending_finalizations()

    assert (
        len(get_pending_finalizations()) == 0
    ), "queue must drain (bounded) after giving up on a persistent outage"


def test_best_effort_stage_failure_does_not_enqueue_retry(tmp_jsonl):
    """A post-extraction stage failure must not strand or re-queue finalization."""
    from scripts.core.memory_daemon import (
        get_active_extractions,
        get_pending_finalizations,
        reap_completed_extractions,
    )

    proc = _make_mock_proc(pid=803, exit_code=0)
    ae = get_active_extractions()
    ae[803] = ("sess-stage-fail", proc, tmp_jsonl, "/tmp/proj", time.time() - 30)

    with (
        patch("scripts.core.memory_daemon.mark_extracted"),  # critical op succeeds
        patch(
            "scripts.core.memory_daemon._calibrate_session_confidence",
            side_effect=RuntimeError("stage boom"),
        ),
        patch("scripts.core.memory_daemon._extract_and_store_workflows"),
        patch("scripts.core.memory_daemon._generate_mini_handoff"),
        patch("scripts.core.memory_daemon.archive_session_jsonl"),
        patch("scripts.core.memory_daemon._count_session_learnings", return_value=0),
        patch("scripts.core.memory_daemon._count_session_rejections", return_value=0),
    ):
        reaped = reap_completed_extractions()

    assert reaped == 1
    assert 803 not in ae
    assert (
        len(get_pending_finalizations()) == 0
    ), "a best-effort stage failure must not enqueue a finalization retry"
