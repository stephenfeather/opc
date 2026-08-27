"""Tests for the PostgreSQL read path in artifact_query.py (issue #282).

The indexer writes to PostgreSQL when it is the active backend, but the query
script only ever opened SQLite. These tests cover:

- search_past_queries degrading to [] when the FTS table is absent (the crash)
- the per-backend SQL table in artifact_query_sql.py
- the _execute helper on both backends (stub PG connection, no live DB)
- backend selection (--db always forces SQLite)
- search/lookup functions and search_dispatch routed through a stub PG conn
- _run_search / _run_span_lookup on the PG path (connection lifetime, --save)

Existing SQLite behaviour is covered by tests/test_artifact_query.py and must
stay green unchanged.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.core import artifact_query as aq
from scripts.core.artifact_query import (
    _execute,
    _select_backend,
    get_handoff_by_span_id,
    get_ledger_for_session,
    handle_span_id_lookup,
    pg_search_query,
    search_continuity,
    search_dispatch,
    search_handoffs,
    search_past_queries,
    search_plans,
)
from scripts.core.artifact_query_sql import (
    BACKENDS,
    CONTINUITY_DOC_COLUMNS,
    HANDOFF_DOC_COLUMNS,
    PG_FTS_INDEX_DDL,
    PLAN_DOC_COLUMNS,
    pg_document_expression,
    sql_for,
)

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "scripts" / "core" / "artifact_schema.sql"


# ---------------------------------------------------------------------------
# Stub PostgreSQL connection (psycopg2-shaped: cursor().execute, no conn.execute)
# ---------------------------------------------------------------------------


class _StubCursor:
    def __init__(self, conn, rows, description):
        self._conn = conn
        self._rows = rows
        self.description = description
        self.closed = False

    def execute(self, sql, params=None):
        self._conn.calls.append((sql, list(params or [])))

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        self.closed = True


class _StubPgConn:
    """Records executed SQL; returns canned rows. Has no .execute on purpose."""

    def __init__(self, rows=(), columns=()):
        self.rows = list(rows)
        self.description = [(c,) for c in columns]
        self.calls = []
        self.close_count = 0

    def cursor(self):
        return _StubCursor(self, self.rows, self.description)

    def commit(self):
        pass

    def close(self):
        self.close_count += 1


@pytest.fixture
def sqlite_conn():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA_PATH.read_text())
    yield conn
    conn.close()


@pytest.fixture
def sqlite_without_queries(sqlite_conn):
    """Production SQLite schema with the queries tables dropped (legacy DB shape)."""
    sqlite_conn.execute("DROP TABLE queries_fts")
    sqlite_conn.execute("DROP TABLE queries")
    sqlite_conn.commit()
    return sqlite_conn


# ===========================================================================
# The crash: queries_fts missing
# ===========================================================================


class TestSearchPastQueriesMissingTable:
    def test_returns_empty_when_fts_table_absent(self, sqlite_without_queries):
        assert search_past_queries(sqlite_without_queries, "anything") == []

    def test_still_works_when_table_present(self, sqlite_conn):
        sqlite_conn.execute(
            "INSERT INTO queries (id, question, answer, was_helpful, handoffs_matched,"
            " plans_matched, continuity_matched) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("q1", "How does auth work?", "JWT", 1, "[]", "[]", "[]"),
        )
        sqlite_conn.commit()
        rows = search_past_queries(sqlite_conn, "auth")
        assert [r["id"] for r in rows] == ["q1"]

    def test_returns_empty_on_postgres(self):
        conn = _StubPgConn()
        assert search_past_queries(conn, "anything", backend="postgres") == []
        assert conn.calls == []  # no queries table exists on PG; never queried

    def test_other_operational_errors_still_raise(self, sqlite_conn):
        sqlite_conn.execute("DROP TABLE queries")  # keep queries_fts, break the JOIN
        with pytest.raises(sqlite3.OperationalError):
            search_past_queries(sqlite_conn, "anything")


# ===========================================================================
# Per-backend SQL table
# ===========================================================================


class TestSqlTable:
    STATEMENTS = (
        "search_handoffs",
        "search_plans",
        "search_continuity",
        "get_handoff_by_span_id",
        "get_ledger_for_session",
    )

    def test_backends(self):
        assert set(BACKENDS) == {"sqlite", "postgres"}

    @pytest.mark.parametrize("name", STATEMENTS)
    def test_every_statement_exists_for_both_backends(self, name):
        assert sql_for("sqlite", name)
        assert sql_for("postgres", name)

    def test_unknown_backend_raises(self):
        with pytest.raises(KeyError):
            sql_for("mysql", "search_handoffs")

    def test_unknown_statement_raises(self):
        with pytest.raises(KeyError):
            sql_for("postgres", "nope")

    @pytest.mark.parametrize("name", STATEMENTS)
    def test_postgres_uses_no_sqlite_only_constructs(self, name):
        sql = sql_for("postgres", name)
        assert "rowid" not in sql
        assert "datetime(" not in sql
        assert "MATCH" not in sql
        assert "_fts" not in sql
        assert "?" not in sql  # psycopg2 placeholders

    @pytest.mark.parametrize("name", ("search_handoffs", "search_plans", "search_continuity"))
    def test_postgres_search_uses_websearch_tsquery(self, name):
        sql = sql_for("postgres", name)
        # websearch_to_tsquery understands OR and never raises on odd input;
        # plainto_tsquery would AND the terms (recall regression vs SQLite).
        assert "websearch_to_tsquery" in sql
        assert "plainto_tsquery" not in sql
        assert "to_tsquery('english', %s)" not in sql.replace("websearch_to_tsquery", "")
        assert "@@" in sql
        assert "ts_rank" in sql

    def test_postgres_handoffs_maps_schema_shape(self):
        sql = sql_for("postgres", "search_handoffs")
        assert "goal AS task_summary" in sql
        # task_number derived from file_path; bounded so an absurd number in a
        # poisoned path cannot overflow ::int and abort the whole search.
        assert "task-([0-9]{1,6})" in sql
        assert "AS task_number" in sql

    @pytest.mark.parametrize(
        "name", ("search_plans", "search_continuity", "get_ledger_for_session")
    )
    def test_postgres_aliases_indexed_at_as_created_at(self, name):
        assert "indexed_at AS created_at" in sql_for("postgres", name)

    @pytest.mark.parametrize("name", ("search_handoffs", "search_plans", "search_continuity"))
    def test_sqlite_search_still_fts5(self, name):
        assert "MATCH ?" in sql_for("sqlite", name)


class TestPgFtsIndexParity:
    """The GIN expression indexes only help if the query expression matches exactly."""

    def test_document_expression_is_immutable_safe(self):
        expr = pg_document_expression(("a", "b"), "x.")
        assert expr == "to_tsvector('english', COALESCE(x.a, '') || ' ' || COALESCE(x.b, ''))"
        assert "concat_ws" not in expr  # STABLE, cannot back an expression index

    @pytest.mark.parametrize(
        ("statement", "alias", "columns", "table"),
        (
            ("search_handoffs", "h.", HANDOFF_DOC_COLUMNS, "handoffs"),
            ("search_plans", "p.", PLAN_DOC_COLUMNS, "plans"),
            ("search_continuity", "c.", CONTINUITY_DOC_COLUMNS, "continuity"),
        ),
    )
    def test_query_expression_matches_index_ddl(self, statement, alias, columns, table):
        query_expr = pg_document_expression(columns, alias)
        assert query_expr in sql_for("postgres", statement)
        index_expr = pg_document_expression(columns)
        ddl = next(d for d in PG_FTS_INDEX_DDL if f" ON {table} " in d)
        assert ddl.startswith("CREATE INDEX IF NOT EXISTS ")
        assert f"USING gin({index_expr})" in ddl
        # Same expression modulo the table alias.
        assert query_expr.replace(alias, "") == index_expr

    def test_docker_schema_carries_the_same_indexes(self):
        schema = (Path(__file__).resolve().parent.parent / "docker" / "init-schema.sql").read_text()
        for ddl in PG_FTS_INDEX_DDL:
            head, _, expr = ddl.partition(" USING gin(")
            assert head in schema, head
            assert expr.rstrip(")") in schema.replace("\n    ", " ")

    def test_init_postgres_applies_index_ddl(self, monkeypatch):
        from scripts.core import artifact_index as ai

        conn = _StubPgConn()
        monkeypatch.setattr(ai, "pg_connect", lambda: conn)
        ai.init_postgres()
        executed = [sql for sql, _ in conn.calls]
        for ddl in PG_FTS_INDEX_DDL:
            assert ddl in executed


# ===========================================================================
# _execute helper
# ===========================================================================


class TestExecute:
    def test_sqlite_returns_columns_and_rows(self, sqlite_conn):
        sqlite_conn.execute(
            "INSERT INTO plans (id, title, file_path) VALUES (?, ?, ?)", ("p1", "T", "f")
        )
        columns, rows = _execute(sqlite_conn, "SELECT id, title FROM plans", [], "sqlite")
        assert columns == ["id", "title"]
        assert rows == [("p1", "T")]

    def test_postgres_uses_cursor(self):
        conn = _StubPgConn(rows=[("x", 1)], columns=("id", "n"))
        columns, rows = _execute(conn, "SELECT 1 WHERE %s", ["p"], "postgres")
        assert columns == ["id", "n"]
        assert rows == [("x", 1)]
        assert conn.calls == [("SELECT 1 WHERE %s", ["p"])]

    def test_postgres_closes_cursor(self, monkeypatch):
        conn = _StubPgConn()
        made = []
        real_cursor = conn.cursor

        def tracking_cursor():
            cur = real_cursor()
            made.append(cur)
            return cur

        monkeypatch.setattr(conn, "cursor", tracking_cursor)
        _execute(conn, "SELECT 1", [], "postgres")
        assert made and made[0].closed

    def test_unknown_backend_raises(self, sqlite_conn):
        with pytest.raises(ValueError):
            _execute(sqlite_conn, "SELECT 1", [], "oracle")


# ===========================================================================
# Search / lookup functions on the PG path (stub conn)
# ===========================================================================


_HANDOFF_COLS = (
    "id",
    "session_name",
    "task_number",
    "task_summary",
    "what_worked",
    "what_failed",
    "key_decisions",
    "outcome",
    "file_path",
    "created_at",
    "score",
)


class TestSearchHandoffsPostgres:
    def test_params_order_without_outcome(self):
        conn = _StubPgConn()
        search_handoffs(conn, "oauth login", limit=7, backend="postgres")
        sql, params = conn.calls[0]
        assert params == ['"oauth" OR "login"', '"oauth" OR "login"', 7]
        assert "outcome = %s" not in sql

    def test_params_order_with_outcome(self):
        conn = _StubPgConn()
        search_handoffs(conn, "oauth", outcome="SUCCEEDED", limit=3, backend="postgres")
        sql, params = conn.calls[0]
        assert params == ['"oauth"', '"oauth"', "SUCCEEDED", 3]
        assert "h.outcome = %s" in sql

    def test_query_is_or_joined_like_sqlite(self):
        """SQLite FTS5 ORs the terms; PostgreSQL must keep that recall contract."""
        conn = _StubPgConn()
        search_handoffs(conn, "oauth login", backend="postgres")
        _, params = conn.calls[0]
        assert params[0] == '"oauth" OR "login"'
        assert params[0] == params[1]

    def test_rows_become_dicts(self):
        row = ("uuid-1", "sess", 3, "goal text", None, None, None, "SUCCEEDED", "f", None, 0.5)
        conn = _StubPgConn(rows=[row], columns=_HANDOFF_COLS)
        result = search_handoffs(conn, "goal", backend="postgres")
        assert result == [dict(zip(_HANDOFF_COLS, row))]


class TestSearchPlansContinuityPostgres:
    def test_plans_params(self):
        conn = _StubPgConn()
        search_plans(conn, "api design", limit=2, backend="postgres")
        _, params = conn.calls[0]
        assert params == ['"api" OR "design"', '"api" OR "design"', 2]

    def test_continuity_params(self):
        conn = _StubPgConn()
        search_continuity(conn, "deploy", limit=4, backend="postgres")
        _, params = conn.calls[0]
        assert params == ['"deploy"', '"deploy"', 4]


class TestPgSearchQuery:
    """websearch_to_tsquery input: quoted tokens OR-joined, never a syntax error."""

    def test_or_joined_and_quoted(self):
        assert pg_search_query("oauth login") == '"oauth" OR "login"'

    def test_empty_and_whitespace_give_empty_query(self):
        assert pg_search_query("") == ""
        assert pg_search_query("   \t ") == ""

    def test_embedded_quotes_stripped(self):
        assert pg_search_query('say "hi" there') == '"say" OR "hi" OR "there"'

    def test_quote_only_tokens_dropped(self):
        assert pg_search_query('"" foo') == '"foo"'

    def test_operators_are_neutralised_by_quoting(self):
        # websearch syntax: leading '-' negates, bare OR/AND are operators.
        assert pg_search_query("-secret OR") == '"-secret" OR "OR"'

    def test_same_tokenisation_as_sqlite(self):
        q = "a  b\tc"
        assert pg_search_query(q).count(" OR ") == aq.escape_fts5_query(q).count(" OR ")


class TestLookupsPostgres:
    def test_span_lookup(self):
        cols = ("id", "session_name", "root_span_id")
        conn = _StubPgConn(rows=[("u1", "s", "span-x")], columns=cols)
        result = get_handoff_by_span_id(conn, "span-x", backend="postgres")
        assert result == {"id": "u1", "session_name": "s", "root_span_id": "span-x"}
        sql, params = conn.calls[0]
        assert params == ["span-x"]
        assert "root_span_id = %s" in sql

    def test_span_lookup_none(self):
        conn = _StubPgConn(columns=("id",))
        assert get_handoff_by_span_id(conn, "missing", backend="postgres") is None

    def test_ledger_lookup(self):
        cols = ("id", "session_name", "created_at")
        conn = _StubPgConn(rows=[("c1", "s", "t")], columns=cols)
        assert get_ledger_for_session(conn, "s", backend="postgres") == {
            "id": "c1",
            "session_name": "s",
            "created_at": "t",
        }
        _, params = conn.calls[0]
        assert params == ["s"]

    def test_handle_span_id_lookup_threads_backend(self):
        cols = ("id", "session_name", "file_path")
        conn = _StubPgConn(rows=[("u1", "s", None)], columns=cols)
        result = handle_span_id_lookup(conn, "span-x", backend="postgres")
        assert result["id"] == "u1"


class TestSearchDispatchPostgres:
    def test_past_queries_empty_and_all_types_queried(self):
        conn = _StubPgConn()
        results = search_dispatch(conn, "auth", search_type="all", backend="postgres")
        assert results["past_queries"] == []
        assert set(results) == {"past_queries", "handoffs", "plans", "continuity"}
        assert len(conn.calls) == 3  # one per artifact type, none for queries

    def test_single_type(self):
        conn = _StubPgConn()
        results = search_dispatch(conn, "auth", search_type="plans", backend="postgres")
        assert set(results) == {"past_queries", "plans"}
        assert len(conn.calls) == 1

    def test_sqlite_default_unchanged(self, sqlite_conn):
        results = search_dispatch(sqlite_conn, "auth", search_type="all")
        assert set(results) == {"past_queries", "handoffs", "plans", "continuity"}


# ===========================================================================
# Backend selection
# ===========================================================================


class TestSelectBackend:
    def _args(self, *argv):
        return aq._build_parser().parse_args(list(argv))

    def test_db_flag_forces_sqlite_even_when_postgres_active(self, monkeypatch):
        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        assert _select_backend(self._args("q", "--db", "x.db")) == "sqlite"

    def test_postgres_when_active(self, monkeypatch):
        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        assert _select_backend(self._args("q")) == "postgres"

    def test_sqlite_when_postgres_not_active(self, monkeypatch):
        monkeypatch.setattr(aq, "use_postgres", lambda: False)
        assert _select_backend(self._args("q")) == "sqlite"

    def test_resolver_error_propagates(self, monkeypatch):
        def boom():
            raise ValueError("bad backend")

        monkeypatch.setattr(aq, "use_postgres", boom)
        with pytest.raises(ValueError):
            _select_backend(self._args("q"))


# ===========================================================================
# CLI runners on the PG path
# ===========================================================================


class TestRunnersPostgres:
    def _args(self, *argv):
        return aq._build_parser().parse_args(list(argv))

    def test_run_search_json_uses_pg_and_closes(self, monkeypatch, capsys):
        conn = _StubPgConn()
        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        monkeypatch.setattr(aq, "_open_pg", lambda: conn)
        aq._run_search(self._args("hello", "--json"), "hello")
        data = json.loads(capsys.readouterr().out)
        assert data["past_queries"] == []
        assert conn.close_count == 1

    def test_run_search_closes_on_error(self, monkeypatch):
        conn = _StubPgConn()
        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        monkeypatch.setattr(aq, "_open_pg", lambda: conn)

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(aq, "search_dispatch", boom)
        with pytest.raises(RuntimeError):
            aq._run_search(self._args("hello"), "hello")
        assert conn.close_count == 1

    def test_run_search_save_rejected_on_pg_before_searching(self, monkeypatch, capsys):
        """--save must not report success when nothing is persisted (no queries table on PG)."""
        conn = _StubPgConn()
        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        monkeypatch.setattr(aq, "_open_pg", lambda: conn)
        called = []
        monkeypatch.setattr(aq, "save_query", lambda *a, **k: called.append(a))
        rc = aq._run_search(self._args("hello", "--save"), "hello")
        assert rc == 2
        assert called == []
        assert conn.calls == []  # rejected before any search ran
        assert "--save" in capsys.readouterr().err

    def test_run_search_save_still_works_on_sqlite(self, tmp_path, capsys):
        db_file = tmp_path / "t.db"
        c = sqlite3.connect(str(db_file))
        c.executescript(_SCHEMA_PATH.read_text())
        c.close()
        rc = aq._run_search(self._args("q", "--save", "--db", str(db_file)), "q")
        assert rc == 0
        assert "Query saved" in capsys.readouterr().out
        c = sqlite3.connect(str(db_file))
        assert c.execute("SELECT count(*) FROM queries").fetchone()[0] == 1
        c.close()

    def test_run_search_config_error_is_clean(self, monkeypatch, capsys):
        def boom():
            raise ValueError("AGENTICA_MEMORY_BACKEND=bogus")

        monkeypatch.setattr(aq, "use_postgres", boom)
        rc = aq._run_search(self._args("hello"), "hello")
        assert rc == 1
        assert "Backend configuration error" in capsys.readouterr().err

    def test_run_search_success_returns_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        monkeypatch.setattr(aq, "_open_pg", lambda: _StubPgConn())
        assert aq._run_search(self._args("hello", "--json"), "hello") == 0


class _FakePgError(Exception):
    """Stands in for psycopg2.Error (module name is what the boundary checks)."""

    __module__ = "psycopg2.errors"


class TestPostgresFailureBoundary:
    """Connection/schema failures on the PG path must be diagnostics, not tracebacks."""

    def _args(self, *argv):
        return aq._build_parser().parse_args(list(argv))

    def test_connection_failure(self, monkeypatch, capsys):
        monkeypatch.setattr(aq, "use_postgres", lambda: True)

        def refuse():
            raise _FakePgError("connection refused")

        monkeypatch.setattr(aq, "_open_pg", refuse)
        rc = aq._run_search(self._args("hello"), "hello")
        assert rc == 1
        err = capsys.readouterr().err
        assert "PostgreSQL" in err and "connection refused" in err

    def test_missing_schema_points_to_indexer(self, monkeypatch, capsys):
        conn = _StubPgConn()

        def undefined(*a, **k):
            raise _FakePgError('relation "handoffs" does not exist')

        monkeypatch.setattr(conn, "cursor", undefined)
        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        monkeypatch.setattr(aq, "_open_pg", lambda: conn)
        rc = aq._run_search(self._args("hello"), "hello")
        assert rc == 1
        err = capsys.readouterr().err
        assert "artifact_index.py --all" in err
        assert conn.close_count == 1  # still closed on failure

    def test_span_lookup_failure(self, monkeypatch, capsys):
        monkeypatch.setattr(aq, "use_postgres", lambda: True)

        def refuse():
            raise _FakePgError("timeout expired")

        monkeypatch.setattr(aq, "_open_pg", refuse)
        rc = aq._run_span_lookup(self._args("--by-span-id", "x"))
        assert rc == 1
        assert "timeout expired" in capsys.readouterr().err

    def test_malformed_dsn_error_does_not_echo_password(self, monkeypatch, capsys):
        """libpq echoes a malformed DSN verbatim; credentials must be redacted (aegis LOW)."""
        monkeypatch.setattr(aq, "use_postgres", lambda: True)

        def bad_dsn():
            raise _FakePgError(
                'invalid dsn: missing "=" after "postgres//claude:s3cr3t@localhost/db" '
                "in connection info string"
            )

        monkeypatch.setattr(aq, "_open_pg", bad_dsn)
        rc = aq._run_search(self._args("hello"), "hello")
        err = capsys.readouterr().err
        assert rc == 1
        assert "s3cr3t" not in err
        assert "invalid dsn" in err

    @pytest.mark.parametrize(
        ("raw", "expected"),
        (
            ("postgresql://u:pw@h:5432/db", "postgresql://***:***@h:5432/db"),
            ("host=h password=pw user=u", "host=h password=*** user=u"),
            ("connection refused at h:5432", "connection refused at h:5432"),
        ),
    )
    def test_redact_credentials(self, raw, expected):
        assert aq.redact_credentials(raw) == expected

    def test_non_pg_errors_still_propagate(self, monkeypatch):
        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        monkeypatch.setattr(aq, "_open_pg", lambda: _StubPgConn())

        def boom(*a, **k):
            raise RuntimeError("bug")

        monkeypatch.setattr(aq, "search_dispatch", boom)
        with pytest.raises(RuntimeError):
            aq._run_search(self._args("hello"), "hello")

    def test_main_exits_nonzero_on_pg_failure(self, monkeypatch):
        from unittest.mock import patch

        monkeypatch.setattr(aq, "use_postgres", lambda: True)

        def refuse():
            raise _FakePgError("connection refused")

        monkeypatch.setattr(aq, "_open_pg", refuse)
        with (
            patch("scripts.core.artifact_query._enable_faulthandler"),
            patch("sys.argv", ["artifact_query.py", "hello"]),
            pytest.raises(SystemExit) as exc,
        ):
            aq.main()
        assert exc.value.code == 1

    def test_main_returns_normally_on_success(self, monkeypatch, capsys):
        from unittest.mock import patch

        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        monkeypatch.setattr(aq, "_open_pg", lambda: _StubPgConn())
        with (
            patch("scripts.core.artifact_query._enable_faulthandler"),
            patch("sys.argv", ["artifact_query.py", "hello", "--json"]),
        ):
            aq.main()  # no SystemExit
        assert json.loads(capsys.readouterr().out)["past_queries"] == []

    def test_run_span_lookup_uses_pg(self, monkeypatch, capsys):
        cols = ("id", "session_name", "task_number", "outcome", "file_path")
        conn = _StubPgConn(rows=[("u1", "sess", None, "SUCCEEDED", None)], columns=cols)
        monkeypatch.setattr(aq, "use_postgres", lambda: True)
        monkeypatch.setattr(aq, "_open_pg", lambda: conn)
        aq._run_span_lookup(self._args("--by-span-id", "span-x", "--json"))
        data = json.loads(capsys.readouterr().out)
        assert data["session_name"] == "sess"
        assert conn.close_count == 1

    def test_sqlite_path_ignores_open_pg(self, monkeypatch, tmp_path, capsys):
        """--db must never touch PG even when it is the active backend."""
        db_file = tmp_path / "t.db"
        c = sqlite3.connect(str(db_file))
        c.executescript(_SCHEMA_PATH.read_text())
        c.close()
        monkeypatch.setattr(aq, "use_postgres", lambda: True)

        def never():
            raise AssertionError("_open_pg called on --db path")

        monkeypatch.setattr(aq, "_open_pg", never)
        aq._run_search(self._args("q", "--json", "--db", str(db_file)), "q")
        assert isinstance(json.loads(capsys.readouterr().out), dict)
