"""Tests for memory_daemon_db — database connection and query helpers.

Phase 2 of S30 TDD+FP refactor. Each step adds tests before moving functions.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Step 2.1 — Connection helpers
# ---------------------------------------------------------------------------


class TestGetPostgresUrl:
    """get_postgres_url returns the first available env var."""

    def test_returns_canonical_url(self, monkeypatch):
        monkeypatch.setenv("CONTINUOUS_CLAUDE_DB_URL", "postgresql://canon")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from scripts.core.memory_daemon_db import get_postgres_url

        assert get_postgres_url() == "postgresql://canon"

    def test_falls_back_to_database_url(self, monkeypatch):
        monkeypatch.delenv("CONTINUOUS_CLAUDE_DB_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgresql://fallback")
        from scripts.core.memory_daemon_db import get_postgres_url

        assert get_postgres_url() == "postgresql://fallback"

    def test_returns_none_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("CONTINUOUS_CLAUDE_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("OPC_POSTGRES_URL", raising=False)
        from scripts.core.memory_daemon_db import get_postgres_url

        assert get_postgres_url() is None

    def test_falls_back_to_opc_postgres_url(self, monkeypatch):
        # Issue #71: the daemon now honors the legacy OPC_POSTGRES_URL too.
        monkeypatch.delenv("CONTINUOUS_CLAUDE_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("OPC_POSTGRES_URL", "postgresql://legacy")
        from scripts.core.memory_daemon_db import get_postgres_url

        assert get_postgres_url() == "postgresql://legacy"

    def test_canonical_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("CONTINUOUS_CLAUDE_DB_URL", "postgresql://canon")
        monkeypatch.setenv("DATABASE_URL", "postgresql://fallback")
        from scripts.core.memory_daemon_db import get_postgres_url

        assert get_postgres_url() == "postgresql://canon"


class TestUsePostgres:
    """use_postgres checks URL availability AND psycopg2 importability."""

    def test_false_when_no_url(self, monkeypatch):
        monkeypatch.delenv("CONTINUOUS_CLAUDE_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("OPC_POSTGRES_URL", raising=False)
        from scripts.core.memory_daemon_db import use_postgres

        assert use_postgres() is False

    def test_false_when_sqlite_override_with_url(self, monkeypatch):
        # Issue #71 split-brain fix: an explicit AGENTICA_MEMORY_BACKEND=sqlite
        # override must win for the daemon too, even when a PostgreSQL URL is
        # present, so the daemon stays on the same backend as store/recall.
        monkeypatch.setenv("AGENTICA_MEMORY_BACKEND", "sqlite")
        monkeypatch.setenv("CONTINUOUS_CLAUDE_DB_URL", "postgresql://canon")
        monkeypatch.setenv("OPC_POSTGRES_URL", "postgresql://legacy")
        from scripts.core.memory_daemon_db import use_postgres

        assert use_postgres() is False

    def test_true_when_url_and_psycopg2(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://test")
        from scripts.core.memory_daemon_db import use_postgres

        # psycopg2 is installed in test env
        assert use_postgres() is True

    def test_false_when_url_but_no_psycopg2(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://test")
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psycopg2":
                raise ImportError("no psycopg2")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from scripts.core.memory_daemon_db import use_postgres

        assert use_postgres() is False


class TestPgConnect:
    """pg_connect wraps psycopg2.connect with exponential backoff retry."""

    @patch("scripts.core.memory_daemon_db.get_postgres_url", return_value="postgresql://test")
    def test_returns_connection_on_success(self, mock_url):
        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value = MagicMock()
            from scripts.core.memory_daemon_db import pg_connect

            conn = pg_connect()
            assert conn is not None
            mock_connect.assert_called_once_with("postgresql://test")

    @patch("scripts.core.memory_daemon_db.get_postgres_url", return_value="postgresql://test")
    def test_retries_on_operational_error(self, mock_url):
        import psycopg2

        with patch("psycopg2.connect") as mock_connect, \
             patch("time.sleep"):
            mock_connect.side_effect = [
                psycopg2.OperationalError("conn refused"),
                MagicMock(),
            ]
            from scripts.core.memory_daemon_db import pg_connect

            conn = pg_connect(max_retries=2, base_delay=0.01)
            assert conn is not None
            assert mock_connect.call_count == 2

    @patch("scripts.core.memory_daemon_db.get_postgres_url", return_value="postgresql://test")
    def test_raises_after_max_retries(self, mock_url):
        import psycopg2

        with patch("psycopg2.connect") as mock_connect, \
             patch("time.sleep"):
            mock_connect.side_effect = psycopg2.OperationalError("down")
            from scripts.core.memory_daemon_db import pg_connect

            with pytest.raises(psycopg2.OperationalError):
                pg_connect(max_retries=2, base_delay=0.01)
            assert mock_connect.call_count == 2


class TestGetSqlitePath:
    """get_sqlite_path returns ~/.claude/sessions.db."""

    def test_returns_expected_path(self):
        from scripts.core.memory_daemon_db import get_sqlite_path

        result = get_sqlite_path()
        assert isinstance(result, Path)
        assert result == Path.home() / ".claude" / "sessions.db"


# ---------------------------------------------------------------------------
# Step 2.2 — Schema setup
# ---------------------------------------------------------------------------


class TestPgEnsureColumn:
    """pg_ensure_column adds extraction columns to sessions and push-tracking columns to archival_memory."""  # noqa: E501

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_adds_all_extraction_and_push_columns(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_ensure_column

        pg_ensure_column()

        # 7 ALTER TABLE on sessions (extraction columns) +
        # 2 ALTER TABLE on archival_memory (push tracking columns) = 9
        assert mock_cur.execute.call_count == 9
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_adds_expected_column_names(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_ensure_column

        pg_ensure_column()

        sql_calls = [c.args[0] for c in mock_cur.execute.call_args_list]
        sessions_sql = " ".join(s for s in sql_calls if "ALTER TABLE sessions" in s)
        archival_sql = " ".join(s for s in sql_calls if "ALTER TABLE archival_memory" in s)
        for col in [
            "memory_extracted_at",
            "extraction_status",
            "extraction_attempts",
            "transcript_path",
            "archived_at",
            "archive_path",
            "last_error",
        ]:
            assert col in sessions_sql, f"{col} must be ALTERed on sessions"
        for col in ["push_count", "last_pushed_at"]:
            assert col in archival_sql, f"{col} must be ALTERed on archival_memory"


class TestSqliteEnsureTable:
    """sqlite_ensure_table creates the sessions table and adds columns."""

    def test_creates_table_in_temp_db(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_ensure_table

            sqlite_ensure_table()

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "id", "project", "working_on", "started_at", "last_heartbeat",
            "memory_extracted_at", "extraction_status", "extraction_attempts",
            "transcript_path", "archived_at", "archive_path",
        }
        assert expected.issubset(columns)

    def test_idempotent_on_existing_table(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_ensure_table

            sqlite_ensure_table()
            sqlite_ensure_table()  # should not raise


class TestEnsureSchema:
    """ensure_schema dispatches to pg or sqlite based on backend."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_ensure_column")
    def test_calls_pg_when_postgres(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import ensure_schema

        ensure_schema()
        mock_pg.assert_called_once()

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=False)
    @patch("scripts.core.memory_daemon_db.sqlite_ensure_table")
    def test_calls_sqlite_when_no_postgres(self, mock_sqlite, mock_use):
        from scripts.core.memory_daemon_db import ensure_schema

        ensure_schema()
        mock_sqlite.assert_called_once()


# ---------------------------------------------------------------------------
# Step 2.3 — Stale session queries (explicit config params per D3)
# ---------------------------------------------------------------------------


class TestPgGetStaleSessions:
    """pg_get_stale_sessions queries with explicit config params."""

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_passes_config_params_to_query(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_get_stale_sessions

        pg_get_stale_sessions(
            stale_threshold=900, max_retries=3, harvest_grace_period=300
        )

        # Verify config values passed as query params
        args = mock_cur.execute.call_args
        assert args[0][1] == (900, 3, 300)
        mock_conn.close.assert_called_once()

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_returns_stale_session_tuples(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("sess-1", "proj-a", "/path/t.jsonl", 1234, None),
        ]
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_get_stale_sessions

        rows = pg_get_stale_sessions(
            stale_threshold=900, max_retries=3, harvest_grace_period=300
        )
        assert len(rows) == 1
        assert rows[0][0] == "sess-1"


class TestSqliteGetStaleSessions:
    """sqlite_get_stale_sessions queries with explicit config params."""

    def test_returns_empty_when_no_db(self, tmp_path):
        db_path = tmp_path / "nonexistent.db"
        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_get_stale_sessions

            result = sqlite_get_stale_sessions(
                stale_threshold=900, max_retries=3, harvest_grace_period=300
            )
            assert result == []

    def test_returns_stale_rows(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, project TEXT, transcript_path TEXT,
                pid INTEGER, exited_at TEXT, last_heartbeat TEXT,
                extraction_status TEXT DEFAULT 'pending',
                extraction_attempts INTEGER DEFAULT 0
            )
        """)
        # Insert a session with very old heartbeat and exited_at
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", "proj", "/t.jsonl", 100, "2020-01-01T00:00:00",
             "2020-01-01T00:00:00", "pending", 0),
        )
        conn.commit()
        conn.close()

        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_get_stale_sessions

            rows = sqlite_get_stale_sessions(
                stale_threshold=900, max_retries=3, harvest_grace_period=300
            )
            assert len(rows) == 1
            assert rows[0][0] == "s1"


class TestGetStaleSessions:
    """get_stale_sessions dispatches to pg or sqlite."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_get_stale_sessions", return_value=[])
    def test_delegates_to_pg(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import get_stale_sessions

        get_stale_sessions(stale_threshold=900, max_retries=3, harvest_grace_period=300)
        mock_pg.assert_called_once_with(
            stale_threshold=900, max_retries=3, harvest_grace_period=300
        )

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=False)
    @patch("scripts.core.memory_daemon_db.sqlite_get_stale_sessions", return_value=[])
    def test_delegates_to_sqlite(self, mock_sqlite, mock_use):
        from scripts.core.memory_daemon_db import get_stale_sessions

        get_stale_sessions(stale_threshold=900, max_retries=3, harvest_grace_period=300)
        mock_sqlite.assert_called_once_with(
            stale_threshold=900, max_retries=3, harvest_grace_period=300
        )


# ---------------------------------------------------------------------------
# Step 2.4 — mark_* functions (D3: max_retries as explicit param where needed)
# ---------------------------------------------------------------------------


class TestPgMarkExtractingDb:
    """pg_mark_extracting updates extraction_status in PostgreSQL."""

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_executes_update(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_mark_extracting

        pg_mark_extracting("sess-1")
        mock_cur.execute.assert_called_once()
        assert "extracting" in mock_cur.execute.call_args[0][0]
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()


class TestPgMarkExtractedDb:
    """pg_mark_extracted sets extracted status."""

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_executes_update(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_mark_extracted

        pg_mark_extracted("sess-1")
        mock_cur.execute.assert_called_once()
        assert "extracted" in mock_cur.execute.call_args[0][0]
        mock_conn.commit.assert_called_once()


class TestPgMarkExtractionFailedDb:
    """pg_mark_extraction_failed retries or gives up based on max_retries."""

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_retries_when_under_max(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (1,)  # 1 attempt so far
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_mark_extraction_failed

        pg_mark_extraction_failed("sess-1", max_retries=3)
        # Should set back to pending
        calls = [c[0][0] for c in mock_cur.execute.call_args_list]
        assert any("pending" in c for c in calls)

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_fails_permanently_at_max(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (3,)  # at max
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_mark_extraction_failed

        pg_mark_extraction_failed("sess-1", max_retries=3)
        calls = [c[0][0] for c in mock_cur.execute.call_args_list]
        assert any("failed" in c for c in calls)


class TestPgMarkArchivedDb:
    """pg_mark_archived stamps sessions and archival_memory."""

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_executes_two_updates(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_mark_archived

        pg_mark_archived("sess-1", "/archive/path.zst")
        assert mock_cur.execute.call_count == 2
        mock_conn.commit.assert_called_once()


class TestPgMarkSessionExitedDb:
    """pg_mark_session_exited stamps exited_at."""

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_executes_update(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_mark_session_exited

        pg_mark_session_exited("sess-1")
        mock_cur.execute.assert_called_once()
        assert "exited_at" in mock_cur.execute.call_args[0][0]


class TestSqliteMarkExtractingDb:
    """sqlite_mark_extracting updates extraction_status."""

    def test_updates_status(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, extraction_status TEXT DEFAULT 'pending',
                extraction_attempts INTEGER DEFAULT 0
            )
        """)
        conn.execute("INSERT INTO sessions VALUES ('s1', 'pending', 0)")
        conn.commit()
        conn.close()

        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_mark_extracting

            sqlite_mark_extracting("s1")

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT extraction_status, extraction_attempts FROM sessions").fetchone()
        conn.close()
        assert row == ("extracting", 1)


class TestSqliteMarkExtractedDb:
    """sqlite_mark_extracted sets extracted status with timestamp."""

    def test_updates_status(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, extraction_status TEXT,
                memory_extracted_at TEXT
            )
        """)
        conn.execute("INSERT INTO sessions VALUES ('s1', 'extracting', NULL)")
        conn.commit()
        conn.close()

        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_mark_extracted

            sqlite_mark_extracted("s1")

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT extraction_status FROM sessions").fetchone()
        conn.close()
        assert row[0] == "extracted"


class TestSqliteMarkExtractionFailedDb:
    """sqlite_mark_extraction_failed retries or gives up."""

    def test_retries_when_under_max(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, extraction_status TEXT,
                extraction_attempts INTEGER DEFAULT 0
            )
        """)
        conn.execute("INSERT INTO sessions VALUES ('s1', 'extracting', 1)")
        conn.commit()
        conn.close()

        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_mark_extraction_failed

            sqlite_mark_extraction_failed("s1", max_retries=3)

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT extraction_status FROM sessions").fetchone()
        conn.close()
        assert row[0] == "pending"

    def test_fails_permanently_at_max(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, extraction_status TEXT,
                extraction_attempts INTEGER DEFAULT 0,
                last_error TEXT
            )
        """)
        conn.execute("INSERT INTO sessions VALUES ('s1', 'extracting', 3, NULL)")
        conn.commit()
        conn.close()

        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_mark_extraction_failed

            sqlite_mark_extraction_failed("s1", max_retries=3)

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT extraction_status FROM sessions").fetchone()
        conn.close()
        assert row[0] == "failed"


class TestSqliteMarkSessionExitedDb:
    """sqlite_mark_session_exited stamps exited_at."""

    def test_sets_exited_at(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (id TEXT PRIMARY KEY, exited_at TEXT)
        """)
        conn.execute("INSERT INTO sessions VALUES ('s1', NULL)")
        conn.commit()
        conn.close()

        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_mark_session_exited

            sqlite_mark_session_exited("s1")

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT exited_at FROM sessions").fetchone()
        conn.close()
        assert row[0] is not None


# ---------------------------------------------------------------------------
# Step 2.5 — Recovery functions
# ---------------------------------------------------------------------------


class TestPgRecoverStalledDb:
    """pg_recover_stalled_extractions resets stuck sessions."""

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_resets_extracting_to_pending(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [("sess-1",), ("sess-2",)]
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_recover_stalled_extractions

        pg_recover_stalled_extractions()
        assert "pending" in mock_cur.execute.call_args[0][0]
        mock_conn.commit.assert_called_once()


class TestSqliteRecoverStalledDb:
    """sqlite_recover_stalled_extractions resets stuck sessions."""

    def test_resets_extracting_to_pending(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (id TEXT PRIMARY KEY, extraction_status TEXT)
        """)
        conn.execute("INSERT INTO sessions VALUES ('s1', 'extracting')")
        conn.commit()
        conn.close()

        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_recover_stalled_extractions

            sqlite_recover_stalled_extractions()

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT extraction_status FROM sessions").fetchone()
        conn.close()
        assert row[0] == "pending"

    def test_noop_when_no_db(self, tmp_path):
        db_path = tmp_path / "nonexistent.db"
        with patch("scripts.core.memory_daemon_db.get_sqlite_path", return_value=db_path):
            from scripts.core.memory_daemon_db import sqlite_recover_stalled_extractions

            sqlite_recover_stalled_extractions()  # should not raise


class TestRecoverStalledDb:
    """recover_stalled_extractions dispatches to pg or sqlite."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_recover_stalled_extractions")
    def test_delegates_to_pg(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import recover_stalled_extractions

        recover_stalled_extractions()
        mock_pg.assert_called_once()

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=False)
    @patch("scripts.core.memory_daemon_db.sqlite_recover_stalled_extractions")
    def test_delegates_to_sqlite(self, mock_sqlite, mock_use):
        from scripts.core.memory_daemon_db import recover_stalled_extractions

        recover_stalled_extractions()
        mock_sqlite.assert_called_once()


# ---------------------------------------------------------------------------
# Step 2.6 — Dispatcher functions and remaining queries
# ---------------------------------------------------------------------------


class TestMarkExtractingDb:
    """mark_extracting dispatches to pg or sqlite."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_mark_extracting")
    def test_delegates_to_pg(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import mark_extracting

        mark_extracting("sess-1")
        mock_pg.assert_called_once_with("sess-1")

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=False)
    @patch("scripts.core.memory_daemon_db.sqlite_mark_extracting")
    def test_delegates_to_sqlite(self, mock_sqlite, mock_use):
        from scripts.core.memory_daemon_db import mark_extracting

        mark_extracting("sess-1")
        mock_sqlite.assert_called_once_with("sess-1")


class TestMarkExtractedDb:
    """mark_extracted dispatches to pg or sqlite."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_mark_extracted")
    def test_delegates_to_pg(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import mark_extracted

        mark_extracted("sess-1")
        mock_pg.assert_called_once_with("sess-1")


class TestMarkExtractionFailedDb:
    """mark_extraction_failed dispatches with max_retries."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_mark_extraction_failed")
    def test_delegates_to_pg(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import mark_extraction_failed

        mark_extraction_failed("sess-1", max_retries=3)
        mock_pg.assert_called_once_with("sess-1", max_retries=3, last_error=None)

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=False)
    @patch("scripts.core.memory_daemon_db.sqlite_mark_extraction_failed")
    def test_delegates_to_sqlite(self, mock_sqlite, mock_use):
        from scripts.core.memory_daemon_db import mark_extraction_failed

        mark_extraction_failed("sess-1", max_retries=3)
        mock_sqlite.assert_called_once_with("sess-1", max_retries=3, last_error=None)


class TestMarkSessionExitedDb:
    """mark_session_exited dispatches to pg or sqlite."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_mark_session_exited")
    def test_delegates_to_pg(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import mark_session_exited

        mark_session_exited("sess-1")
        mock_pg.assert_called_once_with("sess-1")


class TestCountSessionLearningsDb:
    """_count_session_learnings queries archival_memory."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_returns_count(self, mock_pg_connect, mock_use):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (5,)
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import count_session_learnings

        assert count_session_learnings("sess-1") == 5

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=False)
    def test_returns_none_when_no_postgres(self, mock_use):
        from scripts.core.memory_daemon_db import count_session_learnings

        assert count_session_learnings("sess-1") is None

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect", side_effect=Exception("db down"))
    def test_returns_none_on_error(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import count_session_learnings

        assert count_session_learnings("sess-1") is None

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect", side_effect=Exception("db down"))
    def test_logs_warning_on_error(self, mock_pg, mock_use, caplog):
        from scripts.core.memory_daemon_db import count_session_learnings

        with caplog.at_level(logging.WARNING, logger="memory-daemon"):
            assert count_session_learnings("sess-1") is None

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING record for swallowed DB error"
        msg = warnings[0].getMessage()
        assert "count_session_learnings failed" in msg
        # Exception rendered via safe_exception(): "ClassName: message".
        assert "Exception: db down" in msg

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch(
        "scripts.core.memory_daemon_db.pg_connect",
        side_effect=Exception("INSERT failed for id = 'sk-secret-value'"),
    )
    def test_warning_redacts_db_values_in_exception(self, mock_pg, mock_use, caplog):
        # Issue #117: a single-quoted VALUE in the exception text must not leak.
        from scripts.core.memory_daemon_db import count_session_learnings

        with caplog.at_level(logging.WARNING, logger="memory-daemon"):
            assert count_session_learnings("sess-1") is None

        msg = caplog.records[0].getMessage()
        assert "sk-secret-value" not in msg
        assert "'<redacted>'" in msg

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect", side_effect=Exception("db down"))
    def test_warning_escapes_db_sourced_session_id(self, mock_pg, mock_use, caplog):
        # Issue #104: session_id is a DB-sourced string; logging it RAW is a
        # log-injection/forgery vector. It must be wrapped with safe() so
        # control chars are escaped (newline -> \x0a, ESC -> \x1b).
        from scripts.core.memory_daemon_db import count_session_learnings

        hostile = "s\n1\x1b[31m"
        with caplog.at_level(logging.WARNING, logger="memory-daemon"):
            assert count_session_learnings(hostile) is None

        msg = caplog.records[0].getMessage()
        # Raw control bytes must NOT survive into the log message.
        assert "\n1" not in msg
        assert "\x1b[31m" not in msg
        # safe() renders them as escaped markers instead.
        assert "\\x0a" in msg
        assert "\\x1b" in msg


class TestSeedLastPatternRunDb:
    """seed_last_pattern_run reads latest pattern detection timestamp."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=False)
    def test_returns_zero_without_postgres(self, mock_use):
        from scripts.core.memory_daemon_db import seed_last_pattern_run

        assert seed_last_pattern_run() == 0

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_returns_timestamp_from_db(self, mock_pg_connect, mock_use):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_dt = MagicMock()
        mock_dt.timestamp.return_value = 1700000000.0
        mock_cur.fetchone.return_value = (mock_dt,)
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import seed_last_pattern_run

        assert seed_last_pattern_run() == 1700000000.0

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect", side_effect=Exception("db down"))
    def test_returns_zero_on_error(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import seed_last_pattern_run

        assert seed_last_pattern_run() == 0

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect", side_effect=Exception("db down"))
    def test_logs_warning_on_error(self, mock_pg, mock_use, caplog):
        from scripts.core.memory_daemon_db import seed_last_pattern_run

        with caplog.at_level(logging.WARNING, logger="memory-daemon"):
            assert seed_last_pattern_run() == 0

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING record for swallowed DB error"
        msg = warnings[0].getMessage()
        assert "seed_last_pattern_run failed" in msg
        # Exception rendered via safe_exception(): "ClassName: message".
        assert "Exception: db down" in msg


# ---------------------------------------------------------------------------
# Step 2.x — recall_log retention pruning (issue #146)
# ---------------------------------------------------------------------------


class _RowcountCursor:
    """Fake DB cursor whose ``rowcount`` follows a scripted per-execute sequence.

    Each ``execute()`` pops the next value from ``rowcounts`` and exposes it as
    ``rowcount`` (psycopg2 semantics: rows affected by the last statement). When
    ``error_after`` is set, the (error_after+1)-th execute raises, simulating a
    mid-loop DB failure.
    """

    def __init__(self, rowcounts, *, error_after=None):
        self._rowcounts = list(rowcounts)
        self._error_after = error_after
        self.execute_count = 0
        self.rowcount = 0
        self.execute_calls: list[tuple] = []

    def execute(self, sql, params=None):
        self.execute_count += 1
        if self._error_after is not None and self.execute_count > self._error_after:
            raise RuntimeError("boom mid-loop")
        self.execute_calls.append((sql, params))
        self.rowcount = self._rowcounts.pop(0) if self._rowcounts else 0


class TestPruneRecallLog:
    """prune_recall_log: batched retention delete for the recall_log table."""

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=False)
    def test_returns_zero_complete_without_postgres(self, mock_use):
        from scripts.core.memory_daemon_db import prune_recall_log

        assert prune_recall_log(90) == (0, True)

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_disabled_when_retention_non_positive(self, mock_pg, mock_use):
        from scripts.core.memory_daemon_db import prune_recall_log

        assert prune_recall_log(0) == (0, True)
        assert prune_recall_log(-5) == (0, True)
        mock_pg.assert_not_called()

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_single_partial_batch_is_complete(self, mock_pg, mock_use):
        conn = MagicMock()
        cur = _RowcountCursor([3])
        conn.cursor.return_value = cur
        mock_pg.return_value = conn

        from scripts.core.memory_daemon_db import prune_recall_log

        # 3 < batch_size -> drained, so complete is True.
        assert prune_recall_log(90, batch_size=10) == (3, True)
        assert cur.execute_count == 1
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_loops_until_batch_not_full(self, mock_pg, mock_use):
        conn = MagicMock()
        cur = _RowcountCursor([10, 10, 4])  # full, full, partial -> stop
        conn.cursor.return_value = cur
        mock_pg.return_value = conn

        from scripts.core.memory_daemon_db import prune_recall_log

        assert prune_recall_log(90, batch_size=10) == (24, True)
        assert cur.execute_count == 3
        # Commit after every batch so locks release between deletes.
        assert conn.commit.call_count == 3
        conn.close.assert_called_once()

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_interval_is_parameterized_not_interpolated(self, mock_pg, mock_use):
        conn = MagicMock()
        cur = _RowcountCursor([0])
        conn.cursor.return_value = cur
        mock_pg.return_value = conn

        from scripts.core.memory_daemon_db import prune_recall_log

        prune_recall_log(90, batch_size=500)
        sql, params = cur.execute_calls[0]
        assert "recall_log" in sql
        assert "make_interval" in sql
        assert params == (90, 500)
        # Retention/limit must never be string-formatted into the SQL text.
        assert "90" not in sql
        assert "500" not in sql

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_cap_with_full_final_batch_is_incomplete(self, mock_pg, mock_use):
        conn = MagicMock()
        cur = _RowcountCursor([5] * 10)  # always full -> never drains
        conn.cursor.return_value = cur
        mock_pg.return_value = conn

        from scripts.core.memory_daemon_db import prune_recall_log

        # Cap hit with a still-full final batch -> backlog remains -> complete=False
        # so the scheduler continues promptly instead of waiting a full interval.
        assert prune_recall_log(90, batch_size=5, max_batches=3) == (15, False)
        assert cur.execute_count == 3
        conn.close.assert_called_once()

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect", side_effect=Exception("db down"))
    def test_raises_on_connect_error(self, mock_pg, mock_use):
        # Failures propagate (not swallowed) so the scheduler can tell a real
        # failure from an empty prune and retry promptly (issue #146 review).
        from scripts.core.memory_daemon_db import prune_recall_log

        with pytest.raises(Exception, match="db down"):
            prune_recall_log(90)

    @patch("scripts.core.memory_daemon_db.use_postgres", return_value=True)
    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_raises_on_midloop_error_and_closes(self, mock_pg, mock_use):
        conn = MagicMock()
        cur = _RowcountCursor([10], error_after=1)  # batch 1 ok, batch 2 raises
        conn.cursor.return_value = cur
        mock_pg.return_value = conn

        from scripts.core.memory_daemon_db import prune_recall_log

        with pytest.raises(RuntimeError, match="boom mid-loop"):
            prune_recall_log(90, batch_size=10)
        # Batch 1 committed before the failure (committed batches persist)...
        assert conn.commit.call_count == 1
        # ...and the connection is always closed, even when a batch raises.
        conn.close.assert_called_once()
