"""Tests for pg_connect() bug fixes (issues #80 and #81).

Issue #80: pg_connect(None URL) raises TypeError instead of OperationalError.
Issue #81: pg_connect(max_retries=0) raises TypeError from `raise None`.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _import_pg_connect():
    """Import pg_connect lazily to avoid top-level psycopg2 dependency."""
    from scripts.core.memory_daemon_db import pg_connect

    return pg_connect


class TestPgConnectNoneUrl:
    """Issue #80: pg_connect should raise OperationalError when URL is None."""

    def test_none_url_raises_operational_error(self):
        """When get_postgres_url() returns None, pg_connect must raise
        psycopg2.OperationalError (not TypeError)."""
        import psycopg2

        pg_connect = _import_pg_connect()

        with patch(
            "scripts.core.memory_daemon_db.get_postgres_url", return_value=None
        ):
            with pytest.raises(psycopg2.OperationalError, match="DATABASE_URL not configured"):
                pg_connect(max_retries=1)

    def test_none_url_does_not_raise_type_error(self):
        """Ensure TypeError is never raised for a None URL."""
        import psycopg2

        pg_connect = _import_pg_connect()

        with patch(
            "scripts.core.memory_daemon_db.get_postgres_url", return_value=None
        ):
            with pytest.raises(psycopg2.OperationalError):
                pg_connect(max_retries=1)


class TestPgConnectZeroRetries:
    """Issue #81: pg_connect(max_retries=0) should raise OperationalError, not TypeError."""

    def test_zero_retries_raises_operational_error(self):
        """When max_retries=0, pg_connect must raise OperationalError
        (not TypeError from `raise None`)."""
        import psycopg2

        pg_connect = _import_pg_connect()

        with patch(
            "scripts.core.memory_daemon_db.get_postgres_url",
            return_value="postgresql://fake:5432/db",
        ):
            with pytest.raises(psycopg2.OperationalError, match="no connection attempts made"):
                pg_connect(max_retries=0)

    def test_zero_retries_does_not_raise_type_error(self):
        """Ensure TypeError is never raised when max_retries=0."""
        import psycopg2

        pg_connect = _import_pg_connect()

        with patch(
            "scripts.core.memory_daemon_db.get_postgres_url",
            return_value="postgresql://fake:5432/db",
        ):
            with pytest.raises(psycopg2.OperationalError):
                pg_connect(max_retries=0)


class TestPgConnectRetryStillWorks:
    """Ensure the fix doesn't break normal retry behavior."""

    def test_successful_connection_returns_conn(self):
        """pg_connect returns the connection object on success."""
        pg_connect = _import_pg_connect()
        fake_conn = object()

        with patch("scripts.core.memory_daemon_db.get_postgres_url", return_value="postgresql://x"):
            with patch("psycopg2.connect", return_value=fake_conn):
                result = pg_connect(max_retries=1)
                assert result is fake_conn

    def test_retry_on_operational_error(self):
        """pg_connect retries on OperationalError and succeeds on second attempt."""
        import psycopg2

        pg_connect = _import_pg_connect()
        fake_conn = object()

        with patch("scripts.core.memory_daemon_db.get_postgres_url", return_value="postgresql://x"):
            with patch(
                "psycopg2.connect",
                side_effect=[psycopg2.OperationalError("transient"), fake_conn],
            ):
                with patch("time.sleep"):  # skip actual delay
                    result = pg_connect(max_retries=2, base_delay=0.01)
                    assert result is fake_conn

    def test_exhausted_retries_raises_last_error(self):
        """After exhausting retries, pg_connect raises the last OperationalError."""
        import psycopg2

        pg_connect = _import_pg_connect()

        with patch("scripts.core.memory_daemon_db.get_postgres_url", return_value="postgresql://x"):
            with patch(
                "psycopg2.connect",
                side_effect=psycopg2.OperationalError("persistent failure"),
            ):
                with patch("time.sleep"):
                    with pytest.raises(
                        psycopg2.OperationalError, match="persistent failure"
                    ):
                        pg_connect(max_retries=2, base_delay=0.01)
