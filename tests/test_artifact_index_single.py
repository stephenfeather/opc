"""Tests for index_single_file row-id surfacing (issue #191).

`index_single_file` must return the *actual stored* primary-key id so the
MCP `index_artifacts` tool can echo it back to callers (e.g. for an
immediate `mark_handoff`) without a separate DB round-trip.

Covers:
- index_single_file returns {id, type, file} matching the row written
- the returned id equals the value persisted in the DB
- unknown file types return None
- build_index_payload shapes the --json CLI response
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.core.artifact_index import (
    build_index_payload,
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
        assert result is not None
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

    def test_unknown_type_returns_none(self, conn, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("not an artifact")
        assert index_single_file(conn, f) is None


class TestBuildIndexPayload:
    def test_success_payload_includes_fields(self):
        result = {"id": "abc123", "type": "handoff", "file": "x.md"}
        payload = build_index_payload(result)
        assert payload == {
            "success": True,
            "id": "abc123",
            "type": "handoff",
            "file": "x.md",
        }

    def test_failure_payload(self):
        payload = build_index_payload(None)
        assert payload == {"success": False}


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
