"""Tests for extraction stderr logging (GitHub issue #84).

Verifies:
- stderr is captured (subprocess.PIPE) when extraction subprocess launched
- stderr is read and logged on failure in reap_completed_extractions
- last_error is passed through mark_extraction_failed to DB
- stderr is truncated to 500 chars
- stderr fd is closed on success (no fd leak)
- last_error column added to pg_ensure_column and sqlite_ensure_table
"""

import io
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _make_mock_proc(
    pid: int = 12345,
    exit_code: int | None = None,
    stderr_text: str = "",
) -> MagicMock:
    """Create a mock subprocess with configurable stderr."""
    mock = MagicMock()
    mock.pid = pid
    mock.poll.return_value = exit_code
    mock.stderr = io.BytesIO(stderr_text.encode())
    return mock


# ---------------------------------------------------------------------------
# Step 1: stderr=subprocess.PIPE in extract_memories
# ---------------------------------------------------------------------------


@patch("scripts.core.memory_daemon.subprocess.Popen")
@patch("scripts.core.memory_daemon._is_extraction_blocked", return_value=False)
@patch("scripts.core.memory_daemon.mark_extracted")
def test_extract_memories_uses_stderr_pipe(
    _mark, _blocked, mock_popen, tmp_jsonl
):
    """extract_memories must pass stderr=subprocess.PIPE to Popen."""
    from scripts.core.memory_daemon import extract_memories

    mock_popen.return_value = _make_mock_proc()
    extract_memories("sess-stderr-1", "/tmp/proj", str(tmp_jsonl))

    kwargs = mock_popen.call_args[1]
    assert kwargs["stderr"] == subprocess.PIPE, (
        "stderr must be subprocess.PIPE, not DEVNULL"
    )


# ---------------------------------------------------------------------------
# Step 2: reap reads stderr on failure
# ---------------------------------------------------------------------------


def test_reap_logs_stderr_on_failure(tmp_jsonl):
    """When extraction fails, reap_completed_extractions should read stderr and log it."""
    from scripts.core.memory_daemon import (
        get_active_extractions,
        reap_completed_extractions,
    )

    stderr_msg = "Error: model not found"
    proc = _make_mock_proc(pid=111, exit_code=1, stderr_text=stderr_msg)

    ae = get_active_extractions()
    ae[111] = ("sess-fail", proc, tmp_jsonl, "/tmp/proj", time.time() - 60)

    with (
        patch("scripts.core.memory_daemon.mark_extraction_failed"),
        patch("scripts.core.memory_daemon.log") as mock_log,
    ):
        reap_completed_extractions()

    # Verify stderr was read and included in the log message
    log_messages = " ".join(str(c) for c in mock_log.call_args_list)
    assert "stderr" in log_messages.lower() or stderr_msg in log_messages, (
        "Log output must include stderr text on failure"
    )


def test_reap_passes_last_error_to_mark_failed(tmp_jsonl):
    """mark_extraction_failed must receive last_error kwarg with stderr text."""
    from scripts.core.memory_daemon import (
        get_active_extractions,
        reap_completed_extractions,
    )

    stderr_msg = "Segmentation fault"
    proc = _make_mock_proc(pid=222, exit_code=139, stderr_text=stderr_msg)

    ae = get_active_extractions()
    ae[222] = ("sess-segfault", proc, tmp_jsonl, "/tmp/proj", time.time() - 60)

    with (
        patch("scripts.core.memory_daemon.mark_extraction_failed") as mock_fail,
        patch("scripts.core.memory_daemon.log"),
    ):
        reap_completed_extractions()

    mock_fail.assert_called_once()
    _, kwargs = mock_fail.call_args
    assert "last_error" in kwargs or (
        len(mock_fail.call_args[0]) > 1
    ), "mark_extraction_failed must receive last_error"
    # Check actual value
    if "last_error" in kwargs:
        assert stderr_msg in kwargs["last_error"]


def test_reap_truncates_stderr_to_500_chars(tmp_jsonl):
    """stderr text longer than 500 chars must be truncated."""
    from scripts.core.memory_daemon import (
        get_active_extractions,
        reap_completed_extractions,
    )

    long_stderr = "x" * 1000
    proc = _make_mock_proc(pid=333, exit_code=1, stderr_text=long_stderr)

    ae = get_active_extractions()
    ae[333] = ("sess-long", proc, tmp_jsonl, "/tmp/proj", time.time() - 60)

    with (
        patch("scripts.core.memory_daemon.mark_extraction_failed") as mock_fail,
        patch("scripts.core.memory_daemon.log"),
    ):
        reap_completed_extractions()

    mock_fail.assert_called_once()
    _, kwargs = mock_fail.call_args
    last_error = kwargs.get("last_error", "")
    assert len(last_error) <= 500, (
        f"last_error must be truncated to 500 chars, got {len(last_error)}"
    )


def test_reap_closes_stderr_on_success(tmp_jsonl):
    """On successful extraction (exit 0), stderr fd must be closed."""
    from scripts.core.memory_daemon import (
        get_active_extractions,
        reap_completed_extractions,
    )

    proc = MagicMock()
    proc.pid = 444
    proc.poll.return_value = 0
    proc.stderr = MagicMock()

    ae = get_active_extractions()
    ae[444] = ("sess-ok", proc, tmp_jsonl, "/tmp/proj", time.time() - 60)

    with (
        patch("scripts.core.memory_daemon.mark_extracted"),
        patch("scripts.core.memory_daemon._calibrate_session_confidence"),
        patch("scripts.core.memory_daemon._extract_and_store_workflows"),
        patch("scripts.core.memory_daemon._generate_mini_handoff"),
        patch("scripts.core.memory_daemon.archive_session_jsonl"),
        patch("scripts.core.memory_daemon.log"),
    ):
        reap_completed_extractions()

    proc.stderr.close.assert_called_once()


# ---------------------------------------------------------------------------
# Step 3: mark_extraction_failed accepts last_error
# ---------------------------------------------------------------------------


def test_pg_mark_extraction_failed_accepts_last_error():
    """pg_mark_extraction_failed must accept and use last_error parameter."""
    from scripts.core.memory_daemon_db import pg_mark_extraction_failed

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    # Simulate attempts >= max_retries (permanent failure)
    mock_cur.fetchone.return_value = (3,)

    with patch("scripts.core.memory_daemon_db.pg_connect", return_value=mock_conn):
        pg_mark_extraction_failed("sess-pg", max_retries=3, last_error="OOM killed")

    # The UPDATE for permanent failure should include last_error
    update_calls = [
        c for c in mock_cur.execute.call_args_list
        if "UPDATE" in str(c) and "failed" in str(c)
    ]
    assert len(update_calls) > 0, "Should execute UPDATE with 'failed' status"
    sql = str(update_calls[0])
    assert "last_error" in sql, "UPDATE SQL must set last_error column"


def test_sqlite_mark_extraction_failed_accepts_last_error():
    """sqlite_mark_extraction_failed must accept and use last_error parameter."""
    from scripts.core.memory_daemon_db import sqlite_mark_extraction_failed

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (3,)
    mock_conn.execute.return_value = mock_cursor

    with (
        patch("scripts.core.memory_daemon_db.get_sqlite_path") as mock_path,
        patch("sqlite3.connect", return_value=mock_conn),
    ):
        mock_path.return_value = Path("/tmp/test.db")
        sqlite_mark_extraction_failed("sess-sq", max_retries=3, last_error="OOM killed")

    update_calls = [
        c for c in mock_conn.execute.call_args_list
        if "UPDATE" in str(c) and "failed" in str(c)
    ]
    assert len(update_calls) > 0, "Should execute UPDATE with 'failed' status"
    sql = str(update_calls[0])
    assert "last_error" in sql, "UPDATE SQL must set last_error column"


def test_mark_extraction_failed_dispatcher_passes_last_error():
    """The dispatcher mark_extraction_failed must forward last_error."""
    from scripts.core.memory_daemon_db import mark_extraction_failed

    with patch("scripts.core.memory_daemon_db.use_postgres", return_value=True):
        with patch(
            "scripts.core.memory_daemon_db.pg_mark_extraction_failed"
        ) as mock_pg:
            mark_extraction_failed("sess-disp", max_retries=3, last_error="timeout")
            mock_pg.assert_called_once_with(
                "sess-disp", max_retries=3, last_error="timeout"
            )


# ---------------------------------------------------------------------------
# Step 4: Schema includes last_error column
# ---------------------------------------------------------------------------


def test_pg_ensure_column_includes_last_error():
    """pg_ensure_column must add last_error TEXT column."""
    from scripts.core.memory_daemon_db import pg_ensure_column

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    with patch("scripts.core.memory_daemon_db.pg_connect", return_value=mock_conn):
        pg_ensure_column()

    all_sql = " ".join(str(c) for c in mock_cur.execute.call_args_list)
    assert "last_error" in all_sql, (
        "pg_ensure_column must include last_error column migration"
    )


def test_sqlite_ensure_table_includes_last_error():
    """sqlite_ensure_table must include last_error column."""
    from scripts.core.memory_daemon_db import sqlite_ensure_table

    mock_conn = MagicMock()

    with (
        patch("scripts.core.memory_daemon_db.get_sqlite_path") as mock_path,
        patch("sqlite3.connect", return_value=mock_conn),
    ):
        mock_path.return_value = Path("/tmp/test.db")
        sqlite_ensure_table()

    all_sql = " ".join(str(c) for c in mock_conn.execute.call_args_list)
    assert "last_error" in all_sql, (
        "sqlite_ensure_table must include last_error column"
    )
