"""Tests for index_single_file row-id surfacing (issue #191).

`index_single_file` must return the *actual stored* primary-key id so the
MCP `index_artifacts` tool can echo it back to callers (e.g. for an
immediate `mark_handoff`) without a separate DB round-trip.

Covers:
- index_single_file returns {success, id, type, file} matching the row written
- the returned id equals the value persisted in the DB
- unknown file types / write failures return {success: False, error}
- a write that yields no id is reported as a failure, not a hollow success
- the --file --json CLI emits exactly one JSON object on every path
- db_execute(return_id=True) appends RETURNING id and returns the stored id (PG)
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.core.artifact_index import (
    db_execute,
    index_single_file,
    init_sqlite,
)
from scripts.core.artifact_index_core import generate_file_id

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _PROJECT_ROOT / "scripts" / "core" / "artifact_index.py"

_HANDOFF_MD = """---
status: SUCCEEDED
date: 2026-06-17
---

## What Was Done

Implemented the row-id surfacing fix.

## Key Decisions

- Used RETURNING id.
"""


@pytest.fixture
def conn(tmp_path):
    """Real SQLite connection initialized with the production schema."""
    connection = init_sqlite(tmp_path / "ctx.db")
    yield connection
    connection.close()


def _write_handoff(tmp_path: Path) -> Path:
    handoff_dir = tmp_path / "thoughts" / "shared" / "handoffs" / "main"
    handoff_dir.mkdir(parents=True)
    f = handoff_dir / "2026-06-17_task-01.md"
    f.write_text(_HANDOFF_MD)
    return f


class TestIndexSingleFileReturnsId:
    def test_returns_structured_result(self, conn, tmp_path):
        f = _write_handoff(tmp_path)
        result = index_single_file(conn, f)
        assert result["success"] is True
        assert result["type"] == "handoff"
        assert result["file"] == f.name
        assert result["id"]

    def test_returned_id_matches_stored_row(self, conn, tmp_path):
        f = _write_handoff(tmp_path)
        result = index_single_file(conn, f)
        row = conn.execute(
            "SELECT id FROM handoffs WHERE file_path = ?", (str(f.resolve()),)
        ).fetchone()
        assert row is not None
        assert result["id"] == row[0]

    def test_sqlite_id_is_deterministic_file_id(self, conn, tmp_path):
        # On the SQLite backend the PK is the deterministic file id.
        f = _write_handoff(tmp_path)
        result = index_single_file(conn, f)
        assert result["id"] == generate_file_id(str(f.resolve()))

    def test_reindex_is_idempotent(self, conn, tmp_path):
        f = _write_handoff(tmp_path)
        first = index_single_file(conn, f)
        second = index_single_file(conn, f)
        assert first["id"] == second["id"]
        count = conn.execute(
            "SELECT COUNT(*) FROM handoffs WHERE file_path = ?", (str(f.resolve()),)
        ).fetchone()[0]
        assert count == 1

    def test_unknown_type_returns_failure_with_error(self, conn, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("not an artifact")
        result = index_single_file(conn, f)
        assert result["success"] is False
        assert "error" in result
        assert "id" not in result

    def test_db_failure_returns_error(self, tmp_path):
        # A connection whose write fails must surface a machine-readable error,
        # not just success: False.
        f = _write_handoff(tmp_path)
        result = index_single_file(_RaisingPgConn(), f)
        assert result["success"] is False
        assert "boom db" in result["error"]

    def test_write_without_id_is_failure_not_hollow_success(self, tmp_path):
        # If the write returns no row id, the surfacing contract is violated;
        # report failure rather than success with a missing id.
        f = _write_handoff(tmp_path)
        result = index_single_file(_NoRowPgConn(), f)
        assert result["success"] is False
        assert "no row id" in result["error"]
        assert "id" not in result


class TestCliJson:
    """End-to-end: the --file --json CLI path emits the stored id on stdout."""

    def _run(self, args):
        return subprocess.run(
            [sys.executable, str(_SCRIPT), *args],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

    def test_json_flag_emits_id(self, tmp_path):
        handoff_dir = tmp_path / "handoffs"
        handoff_dir.mkdir()
        f = handoff_dir / "task-09.md"
        f.write_text(_HANDOFF_MD)
        db = tmp_path / "ctx.db"

        proc = self._run(["--file", str(f), "--db", str(db), "--json"])
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["success"] is True
        assert payload["type"] == "handoff"
        assert payload["file"] == "task-09.md"
        assert payload["id"] == generate_file_id(str(f.resolve()))

    def test_without_json_flag_emits_human_line(self, tmp_path):
        handoff_dir = tmp_path / "handoffs"
        handoff_dir.mkdir()
        f = handoff_dir / "task-10.md"
        f.write_text(_HANDOFF_MD)
        db = tmp_path / "ctx.db"

        proc = self._run(["--file", str(f), "--db", str(db)])
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "Indexed handoff: task-10.md"

    def test_json_unknown_type_stdout_is_pure_json(self, tmp_path):
        # Not under a handoffs/plans path and not a continuity ledger.
        f = tmp_path / "notes.txt"
        f.write_text("not an artifact")
        db = tmp_path / "ctx.db"

        proc = self._run(["--file", str(f), "--db", str(db), "--json"])
        assert proc.returncode == 1
        # stdout must be parseable JSON only — diagnostics belong on stderr.
        payload = json.loads(proc.stdout)
        assert payload["success"] is False
        assert "error" in payload
        assert "Unknown file type" in proc.stderr

    def test_json_file_not_found_emits_json_failure(self, tmp_path):
        missing = tmp_path / "handoffs" / "nope.md"
        db = tmp_path / "ctx.db"

        proc = self._run(["--file", str(missing), "--db", str(db), "--json"])
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert payload["success"] is False
        assert "error" in payload
        assert "File not found" in proc.stderr


class _FakePgCursor:
    """Minimal psycopg2-like cursor recording the executed SQL."""

    def __init__(self, returned_id):
        self._returned_id = returned_id
        self.executed_sql = None

    def execute(self, sql, params=()):
        self.executed_sql = sql

    def fetchone(self):
        return (self._returned_id,)

    def close(self):
        pass


class _FakePgConn:
    """Fake non-sqlite connection so db_execute takes the PostgreSQL path."""

    def __init__(self, returned_id):
        self.cursor_obj = _FakePgCursor(returned_id)

    def cursor(self):
        return self.cursor_obj


class _RaisingPgCursor:
    def execute(self, sql, params=()):
        raise RuntimeError("boom db")

    def close(self):
        pass


class _RaisingPgConn:
    """Fake non-sqlite connection whose writes fail, to exercise error paths."""

    def cursor(self):
        return _RaisingPgCursor()

    def commit(self):
        pass

    def rollback(self):
        pass


class _NoRowPgCursor:
    """Cursor whose RETURNING fetch yields no row (id cannot be surfaced)."""

    def execute(self, sql, params=()):
        pass

    def fetchone(self):
        return None

    def close(self):
        pass


class _NoRowPgConn:
    def cursor(self):
        return _NoRowPgCursor()

    def commit(self):
        pass

    def rollback(self):
        pass


class TestDbExecutePostgresReturningId:
    """db_execute must append RETURNING id and return the DB-stored id on PG."""

    def test_returns_db_generated_id_and_appends_returning(self):
        uuid_value = "48d985fb-e318-4f07-ab16-2727b6a66dec"
        conn = _FakePgConn(uuid_value)
        result = db_execute(
            conn,
            "INSERT OR REPLACE INTO plans (id, title) VALUES (?, ?)",
            ("hex123", "A plan"),
            return_id=True,
        )
        assert result == uuid_value
        assert "RETURNING id" in conn.cursor_obj.executed_sql

    def test_no_returning_clause_when_return_id_false(self):
        conn = _FakePgConn("ignored")
        result = db_execute(
            conn,
            "INSERT OR REPLACE INTO plans (id, title) VALUES (?, ?)",
            ("hex123", "A plan"),
            return_id=False,
        )
        assert result is None
        assert "RETURNING id" not in conn.cursor_obj.executed_sql
