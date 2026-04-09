"""Tests for memory_daemon_db — database connection and query helpers.

Phase 2 of S30 TDD+FP refactor. Each step adds tests before moving functions.
"""

from __future__ import annotations

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
        from scripts.core.memory_daemon_db import get_postgres_url

        assert get_postgres_url() is None

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
    """pg_ensure_column adds extraction columns to sessions table."""

    @patch("scripts.core.memory_daemon_db.pg_connect")
    def test_adds_six_columns(self, mock_pg_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_pg_connect.return_value = mock_conn

        from scripts.core.memory_daemon_db import pg_ensure_column

        pg_ensure_column()

        assert mock_cur.execute.call_count == 7
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
        combined = " ".join(sql_calls)
        for col in [
            "memory_extracted_at",
            "extraction_status",
            "extraction_attempts",
            "transcript_path",
            "archived_at",
            "archive_path",
            "last_error",
        ]:
            assert col in combined


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
