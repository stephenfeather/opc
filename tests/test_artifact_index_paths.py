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
from scripts.core.artifact_index_core import (
    adapt_for_postgres,
    generate_file_id,
    normalize_artifact_date,
)
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

    def test_invalid_filename_date_is_ignored(self, repo):
        """2026-13-45 would abort the TIMESTAMPTZ insert (aegis MEDIUM)."""
        p = repo / "thoughts/shared/plans/2026-13-45-foo.md"
        p.write_text("# Bad date\n")
        assert ai.parse_plan(p)["created_at"] == ""

    def test_frontmatter_date_wins_over_filename(self, repo):
        p = repo / "thoughts/shared/plans/2026-05-09_x.md"
        p.write_text("---\ndate: 2026-06-01\n---\n# X\n")
        assert ai.parse_plan(p)["created_at"] == "2026-06-01"

    def test_missing_date_stores_null_not_empty_string(self, conn, repo):
        (repo / "thoughts/shared/plans/p.md").write_text("# No date\n")
        ai.index_plans(conn, Path("thoughts/shared/plans"))
        assert conn.execute("SELECT created_at FROM plans").fetchone()[0] is None


# ---------------------------------------------------------------------------
# Date normalisation (review R1: quoted YAML dates fail the TIMESTAMPTZ cast)
# ---------------------------------------------------------------------------


class TestNormalizeArtifactDate:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        (
            ("2026-02-01", "2026-02-01"),
            ('"2026-02-01"', "2026-02-01"),
            ("'2026-02-01'", "2026-02-01"),
            ("2026-02-01T10:20:30", "2026-02-01T10:20:30"),
            ("2026-02-01T10:20:30+00:00", "2026-02-01T10:20:30+00:00"),
            ("2026-02-01 10:20:30Z", "2026-02-01 10:20:30Z"),
            ("2026-08-26_22-57", "2026-08-26T22:57:00"),  # handoff filename style
            ("  2026-02-01  ", "2026-02-01"),
        ),
    )
    def test_accepted_forms(self, raw, expected):
        assert normalize_artifact_date(raw) == expected

    @pytest.mark.parametrize("raw", ("", None, "yesterday", "2026-13-40", "02/01/2026", '""'))
    def test_rejected_forms_become_none(self, raw):
        assert normalize_artifact_date(raw) is None

    def test_quoted_frontmatter_date_reaches_plan_and_handoff(self, repo):
        (repo / "thoughts/shared/plans/p.md").write_text('---\ndate: "2026-02-01"\n---\n# X\n')
        assert ai.parse_plan(Path("thoughts/shared/plans/p.md"))["created_at"] == "2026-02-01"
        (repo / "thoughts/shared/handoffs/sess/task-01.md").write_text(
            "---\ndate: '2026-03-04'\n---\n## What was done\nx\n"
        )
        assert (
            ai.parse_handoff(Path("thoughts/shared/handoffs/sess/task-01.md"))["created_at"]
            == "2026-03-04"
        )

    def test_garbage_handoff_date_falls_back_to_now(self, repo):
        (repo / "thoughts/shared/handoffs/sess/task-01.md").write_text(
            "---\ndate: someday\n---\n## What was done\nx\n"
        )
        created = ai.parse_handoff(Path("thoughts/shared/handoffs/sess/task-01.md"))["created_at"]
        assert normalize_artifact_date(created) == created  # valid ISO, not "someday"

    def test_adapter_rejects_unnormalised_created_at(self):
        params = list(_SQLITE_ORDER[:15]) + ['"2026-02-01"']
        _, new_params = adapt_for_postgres(
            "INSERT INTO handoffs (col) VALUES (?)", tuple(params), "handoffs"
        )
        assert new_params[-1] == "2026-02-01"


# ---------------------------------------------------------------------------
# Legacy relative-path rows (review R1: canonical paths must not leave dupes)
# ---------------------------------------------------------------------------


class TestLegacyRowPruning:
    def test_bulk_index_replaces_legacy_relative_plan_row(self, conn, repo):
        conn.execute(
            "INSERT INTO plans (id, title, file_path) VALUES (?, ?, ?)",
            ("legacyid", "old", "thoughts/shared/plans/p.md"),
        )
        conn.commit()
        ai.index_plans(conn, Path("thoughts/shared/plans"))
        rows = conn.execute("SELECT file_path FROM plans").fetchall()
        assert rows == [(str((repo / "thoughts/shared/plans/p.md").resolve()),)]

    def test_continuity_never_deletes_other_rows_with_same_session_name(self, conn, repo):
        """A session name is not a project discriminator; another project's
        ledger with the same name must survive a bulk run (review R2)."""
        conn.execute(
            "INSERT INTO continuity (id, session_name) VALUES (?, ?)",
            ("other-project", "sess"),
        )
        conn.commit()
        ai.index_continuity(conn, Path("."))
        ids = {r[0] for r in conn.execute("SELECT id FROM continuity").fetchall()}
        assert ids == {
            "other-project",
            generate_file_id(str((repo / "CONTINUITY_CLAUDE-sess.md").resolve())),
        }

    def test_prune_only_removes_the_exact_legacy_twin_of_the_written_row(self, conn, repo):
        absolute = str((repo / "thoughts/shared/plans/p.md").resolve())
        rows = (
            ("a", absolute),  # canonical row — kept
            ("twin", "thoughts/shared/plans/p.md"),  # legacy twin of `absolute` — removed
            ("partial", "shared/plans/p.md"),  # not the legacy form — kept
            ("wild", "%.md"),  # a LIKE pattern is just a string here — kept
            ("foreign", "thoughts/shared/plans/other.md"),  # another file — kept
            ("elsewhere", "some/other/project/thoughts/shared/plans/p.md"),  # kept
        )
        for rid, path in rows:
            conn.execute(
                "INSERT INTO plans (id, title, file_path) VALUES (?, ?, ?)", (rid, "t", path)
            )
        conn.commit()
        assert ai.prune_legacy_rows(conn, "plans", absolute) == 1
        assert ai.prune_legacy_rows(conn, "plans", absolute) == 0  # idempotent
        kept = {r[0] for r in conn.execute("SELECT id FROM plans").fetchall()}
        assert kept == {"a", "partial", "wild", "foreign", "elsewhere"}

    def test_prune_is_noop_for_paths_outside_a_thoughts_tree(self, conn):
        assert ai.prune_legacy_rows(conn, "plans", "/somewhere/plan.md") == 0

    def test_prune_is_noop_on_postgres(self):
        """A relative path is not unique across projects sharing one PostgreSQL
        database; re-homing there is #287's project-aware backfill (PR review)."""
        pg = _Conn()
        assert ai.prune_legacy_rows(pg, "plans", "/x/thoughts/shared/plans/p.md") == 0
        assert not any("DELETE" in s for s in pg.executed)

    def test_single_file_index_also_retires_the_legacy_twin(self, conn, repo):
        """The hook path (--file) must not leave the pre-#283 row behind (R3)."""
        conn.execute(
            "INSERT INTO plans (id, title, file_path) VALUES (?, ?, ?)",
            ("legacyid", "old", "thoughts/shared/plans/p.md"),
        )
        conn.commit()
        ai.index_single_file(conn, Path("thoughts/shared/plans/p.md"))
        rows = conn.execute("SELECT file_path FROM plans").fetchall()
        assert rows == [(str((repo / "thoughts/shared/plans/p.md").resolve()),)]

    def test_prune_requires_absolute_canonical_path(self, conn):
        with pytest.raises(ValueError):
            ai.prune_legacy_rows(conn, "plans", "thoughts/shared/plans/p.md")

    def test_legacy_row_survives_when_its_file_fails_to_index(self, conn, repo, monkeypatch):
        """Deletion happens only after the replacement is written (review R2)."""
        conn.execute(
            "INSERT INTO plans (id, title, file_path) VALUES (?, ?, ?)",
            ("legacyid", "old", "thoughts/shared/plans/p.md"),
        )
        conn.commit()

        def boom(_path):
            raise OSError("unreadable")

        monkeypatch.setattr(ai, "parse_plan", boom)
        ai.index_plans(conn, Path("thoughts/shared/plans"))
        assert conn.execute("SELECT id FROM plans").fetchall() == [("legacyid",)]

    def test_handoffs_are_not_pruned_here(self, conn, repo):
        """Legacy handoff rows span ~600 project roots; #287 backfill owns them."""
        conn.execute(
            "INSERT INTO handoffs (id, session_name, file_path) VALUES (?, ?, ?)",
            ("legacy", "s", "thoughts/shared/handoffs/s/x.md"),
        )
        conn.commit()
        ai.index_handoffs(conn, Path("thoughts/shared/handoffs"))
        assert conn.execute("SELECT count(*) FROM handoffs").fetchone()[0] == 2

    def test_prune_rejects_unknown_table(self, conn):
        with pytest.raises(ValueError):
            ai.prune_legacy_rows(conn, "handoffs; DROP TABLE plans", "/abs/p.md")


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


_DATE = "2026-01-01"
# Each value names its column, except created_at which must be a real date
# (the adapter normalises it and binds NULL for anything unparseable).
_NAMED_PARAMS = tuple(_SQLITE_ORDER[:15]) + (_DATE,)


class TestHandoffAdapter:
    def _adapt(self, params=None):
        params = params or _NAMED_PARAMS
        return adapt_for_postgres("INSERT INTO handoffs (col) VALUES (?)", params, "handoffs")

    def test_all_fields_bound_except_id(self):
        _, new_params = self._adapt()
        assert set(new_params) == (set(_SQLITE_ORDER) - {"id", "created_at"}) | {_DATE}
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
        bound = [("goal" if c == "task_summary" else c) for c in list(new_params)]
        bound[-1] = "created_at"  # the date value stands for its column
        assert cols[1:] == bound

    def test_created_at_never_regresses_on_conflict(self):
        sql, _ = self._adapt()
        assert "created_at = COALESCE(EXCLUDED.created_at, handoffs.created_at)" in sql

    @pytest.mark.parametrize(
        "col",
        ("session_name", "root_span_id", "turn_span_id", "session_id", "braintrust_session_id"),
    )
    def test_identity_fields_survive_blank_reindex(self, col):
        """Parsers emit '' for absent frontmatter; a re-index must not erase
        previously captured correlation ids with blanks (review R2)."""
        sql, _ = self._adapt()
        flat = "".join(sql.split())  # whitespace-insensitive: clauses may wrap
        assert f"{col}=COALESCE(NULLIF(EXCLUDED.{col},''),handoffs.{col})" in flat

    def test_task_number_survives_null_reindex(self):
        sql, _ = self._adapt()
        assert "task_number = COALESCE(EXCLUDED.task_number, handoffs.task_number)" in sql

    @pytest.mark.parametrize(
        "col", ("goal", "what_worked", "what_failed", "key_decisions", "files_modified", "outcome")
    )
    def test_content_fields_mirror_the_file(self, col):
        sql, _ = self._adapt()
        assert f"{col} = EXCLUDED.{col}" in sql

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
    def __init__(self, sink, schema_current):
        self.sink = sink
        self.schema_current = schema_current

    def execute(self, sql, params=None):
        self.sink.append(" ".join(sql.split()))

    def fetchone(self):
        return (self.schema_current,)

    def close(self):
        pass


class _Conn:
    def __init__(self, schema_current=False):
        self.executed = []
        self.schema_current = schema_current

    def cursor(self):
        return _Cur(self.executed, self.schema_current)

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

    def test_skips_all_ddl_when_schema_is_current(self, monkeypatch):
        """The hook's --file fast path calls init_postgres on every write; an
        up-to-date database must pay one catalog SELECT, no ALTER/CREATE (R3)."""
        c = _Conn(schema_current=True)
        monkeypatch.setattr(ai, "pg_connect", lambda: c)
        ai.init_postgres()
        ddl = [
            s
            for s in c.executed
            if s.startswith(("ALTER TABLE", "CREATE INDEX IF NOT EXISTS idx_"))
        ]
        assert ddl == []
        probes = [s for s in c.executed if "information_schema.columns" in s]
        assert len(probes) == 1

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


class TestPlanQueryTimestamp:
    def test_pg_plans_return_authored_date_with_indexed_fallback(self):
        sql = sql_for("postgres", "search_plans")
        assert "COALESCE(p.created_at, p.indexed_at) AS created_at" in sql
        assert "ORDER BY score DESC, COALESCE(p.created_at, p.indexed_at) DESC" in sql
        assert "p.indexed_at AS created_at" not in sql


class TestQueryUsesStoredTaskNumber:
    @pytest.mark.parametrize("stmt", ("search_handoffs", "get_handoff_by_span_id"))
    def test_coalesce_column_then_filename(self, stmt):
        sql = sql_for("postgres", stmt)
        expected = "COALESCE(h.task_number, substring(h.file_path from 'task-([0-9]{1,6})')::int)"
        assert expected in sql
