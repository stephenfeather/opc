"""Issue #283: stable artifact ids/paths, plan created_at, and the PostgreSQL
adapter carrying every handoff field.

Findings this covers:
- Bulk indexers (index_handoffs/index_plans/index_continuity) hashed the path
  *string* they were given, so a relative glob and the hook's absolute --file
  produced two rows for one artifact. Paths are now resolved before id/file_path.
- index_plans never stored created_at on either backend.
- adapt_for_postgres dropped task_number, files_modified, turn_span_id,
  braintrust_session_id and created_at (so PG created_at was now()).
- init_postgres must add the missing columns idempotently.
"""

import sqlite3
from pathlib import Path

import pytest

from scripts.core import artifact_index as ai
from scripts.core.artifact_index_core import adapt_for_postgres, generate_file_id
from scripts.core.artifact_query_sql import sql_for

_SCHEMA = Path(__file__).resolve().parent.parent / "scripts" / "core" / "artifact_schema.sql"

HANDOFF_MD = """---
date: 2026-03-04
status: SUCCEEDED
root_span_id: span-1
---
## What was done
Built the thing.
"""

PLAN_MD = """---
date: 2026-02-01
---
# Plan: Something

## Overview
Do it.
"""

LEDGER_MD = """# Continuity Ledger

## Goal
Finish.
"""


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(_SCHEMA.read_text())
    yield c
    c.close()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repo with one handoff, one plan, one ledger; cwd inside it."""
    (tmp_path / "thoughts/shared/handoffs/sess").mkdir(parents=True)
    (tmp_path / "thoughts/shared/plans").mkdir(parents=True)
    (tmp_path / "thoughts/shared/handoffs/sess/task-01.md").write_text(HANDOFF_MD)
    (tmp_path / "thoughts/shared/plans/p.md").write_text(PLAN_MD)
    (tmp_path / "CONTINUITY_CLAUDE-sess.md").write_text(LEDGER_MD)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# B. Stable ids / absolute paths
# ---------------------------------------------------------------------------


class TestStablePaths:
    def test_index_handoffs_stores_absolute_path_and_matching_id(self, conn, repo):
        ai.index_handoffs(conn, Path("thoughts/shared/handoffs"))
        row = conn.execute("SELECT id, file_path FROM handoffs").fetchone()
        expected = (repo / "thoughts/shared/handoffs/sess/task-01.md").resolve()
        assert row[1] == str(expected)
        assert row[0] == generate_file_id(str(expected))

    def test_index_plans_stores_absolute_path_and_matching_id(self, conn, repo):
        ai.index_plans(conn, Path("thoughts/shared/plans"))
        row = conn.execute("SELECT id, file_path FROM plans").fetchone()
        expected = (repo / "thoughts/shared/plans/p.md").resolve()
        assert row[1] == str(expected)
        assert row[0] == generate_file_id(str(expected))

    def test_index_continuity_id_matches_absolute_path(self, conn, repo):
        ai.index_continuity(conn, Path("."))
        row = conn.execute("SELECT id FROM continuity").fetchone()
        expected = (repo / "CONTINUITY_CLAUDE-sess.md").resolve()
        assert row[0] == generate_file_id(str(expected))

    def test_bulk_and_single_file_agree(self, conn, repo):
        """The hook's --file path and the bulk glob must upsert the SAME row."""
        ai.index_plans(conn, Path("thoughts/shared/plans"))
        ai.index_single_file(conn, Path("thoughts/shared/plans/p.md"))
        assert conn.execute("SELECT count(*) FROM plans").fetchone()[0] == 1

    def test_relative_and_absolute_input_same_id(self, conn, repo):
        rel = Path("thoughts/shared/handoffs/sess/task-01.md")
        assert ai.parse_handoff(rel)["id"] == ai.parse_handoff(rel.resolve())["id"]
        rel_plan = Path("thoughts/shared/plans/p.md")
        assert ai.parse_plan(rel_plan)["id"] == ai.parse_plan(rel_plan.resolve())["id"]


# ---------------------------------------------------------------------------
# Plans created_at
# ---------------------------------------------------------------------------


class TestPlanCreatedAt:
    def test_parse_plan_exposes_created_at(self, repo):
        data = ai.parse_plan(Path("thoughts/shared/plans/p.md"))
        assert data["created_at"] == "2026-02-01"

    def test_index_plans_stores_created_at(self, conn, repo):
        ai.index_plans(conn, Path("thoughts/shared/plans"))
        assert conn.execute("SELECT created_at FROM plans").fetchone()[0] == "2026-02-01"

    def test_filename_date_prefix_used_when_frontmatter_has_none(self, repo):
        p = repo / "thoughts/shared/plans/2026-05-09_big-refactor.md"
        p.write_text("# Big refactor\n")
        assert ai.parse_plan(p)["created_at"] == "2026-05-09"

    def test_frontmatter_date_wins_over_filename(self, repo):
        p = repo / "thoughts/shared/plans/2026-05-09_x.md"
        p.write_text("---\ndate: 2026-06-01\n---\n# X\n")
        assert ai.parse_plan(p)["created_at"] == "2026-06-01"

    def test_missing_date_stores_null_not_empty_string(self, conn, repo):
        (repo / "thoughts/shared/plans/p.md").write_text("# No date\n")
        ai.index_plans(conn, Path("thoughts/shared/plans"))
        assert conn.execute("SELECT created_at FROM plans").fetchone()[0] is None


# ---------------------------------------------------------------------------
# C. Adapter carries every field
# ---------------------------------------------------------------------------

_SQLITE_ORDER = (
    "id",
    "session_name",
    "session_uuid",
    "task_number",
    "file_path",
    "task_summary",
    "what_worked",
    "what_failed",
    "key_decisions",
    "files_modified",
    "outcome",
    "root_span_id",
    "turn_span_id",
    "session_id",
    "braintrust_session_id",
    "created_at",
)


class TestHandoffAdapter:
    def _adapt(self, params=None):
        params = params or tuple(_SQLITE_ORDER)  # each value names its column
        return adapt_for_postgres("INSERT INTO handoffs (col) VALUES (?)", params, "handoffs")

    def test_all_fields_bound_except_id(self):
        _, new_params = self._adapt()
        assert set(new_params) == set(_SQLITE_ORDER) - {"id"}
        assert len(new_params) == 15

    @pytest.mark.parametrize(
        "col",
        ("task_number", "files_modified", "turn_span_id", "braintrust_session_id", "created_at"),
    )
    def test_sql_lists_previously_dropped_column(self, col):
        sql, _ = self._adapt()
        assert col in sql.split("VALUES")[0]

    def test_param_order_matches_sql_column_order(self):
        sql, new_params = self._adapt()
        cols_clause = sql.split("(", 1)[1].split(")", 1)[0]
        cols = [c.strip() for c in cols_clause.split(",")]
        assert cols[0] == "id"
        assert cols[1:] == [("goal" if c == "task_summary" else c) for c in list(new_params)]

    def test_created_at_never_regresses_on_conflict(self):
        sql, _ = self._adapt()
        assert "created_at = COALESCE(EXCLUDED.created_at, handoffs.created_at)" in sql

    def test_empty_created_at_becomes_null(self):
        params: list = list(_SQLITE_ORDER[:15]) + [""]
        _, new_params = self._adapt(tuple(params))
        assert new_params[-1] is None

    def test_still_upserts_on_file_path(self):
        sql, _ = self._adapt()
        assert "ON CONFLICT (file_path)" in sql
        assert "goal = EXCLUDED.goal" in sql

    def test_wrong_param_count_raises(self):
        with pytest.raises(ValueError, match="Expected 16 handoff params"):
            self._adapt(tuple("x" for _ in range(14)))


class TestPlanAdapter:
    def test_plan_insert_includes_created_at_and_upserts_it(self):
        sql = ai._PLAN_INSERT_SQL
        assert "created_at" in sql
        new_sql, _ = adapt_for_postgres(sql, tuple(range(8)), "plans")
        assert "created_at = EXCLUDED.created_at" in new_sql
        assert "ON CONFLICT (id)" in new_sql


# ---------------------------------------------------------------------------
# Schema migration in init_postgres
# ---------------------------------------------------------------------------


class _Cur:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params=None):
        self.sink.append(" ".join(sql.split()))

    def close(self):
        pass


class _Conn:
    def __init__(self):
        self.executed = []

    def cursor(self):
        return _Cur(self.executed)

    def commit(self):
        pass

    def close(self):
        pass


class TestInitPostgresMigration:
    @pytest.mark.parametrize(
        "ddl",
        (
            "ALTER TABLE handoffs ADD COLUMN IF NOT EXISTS task_number INTEGER",
            "ALTER TABLE handoffs ADD COLUMN IF NOT EXISTS files_modified TEXT",
            "ALTER TABLE handoffs ADD COLUMN IF NOT EXISTS turn_span_id TEXT",
            "ALTER TABLE handoffs ADD COLUMN IF NOT EXISTS braintrust_session_id TEXT",
            "ALTER TABLE plans ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ",
        ),
    )
    def test_adds_missing_columns_idempotently(self, monkeypatch, ddl):
        c = _Conn()
        monkeypatch.setattr(ai, "pg_connect", lambda: c)
        ai.init_postgres()
        assert ddl in c.executed

    def test_docker_schema_has_the_columns(self):
        schema = (Path(__file__).resolve().parent.parent / "docker" / "init-schema.sql").read_text()
        handoffs = schema.split("CREATE TABLE IF NOT EXISTS handoffs", 1)[1].split(");", 1)[0]
        for col in ("task_number", "files_modified", "turn_span_id", "braintrust_session_id"):
            assert col in handoffs, col
        plans = schema.split("CREATE TABLE IF NOT EXISTS plans", 1)[1].split(");", 1)[0]
        assert "created_at" in plans


# ---------------------------------------------------------------------------
# E. Query prefers the stored task_number
# ---------------------------------------------------------------------------


class TestQueryUsesStoredTaskNumber:
    @pytest.mark.parametrize("stmt", ("search_handoffs", "get_handoff_by_span_id"))
    def test_coalesce_column_then_filename(self, stmt):
        sql = sql_for("postgres", stmt)
        expected = "COALESCE(h.task_number, substring(h.file_path from 'task-([0-9]{1,6})')::int)"
        assert expected in sql
