"""Database layer for the memory extraction daemon.

Connection helpers, schema setup, queries, and mark_* functions.
All config is passed as explicit parameters (D3) — no import-time capture.
Logging via logging.getLogger (D4).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("memory-daemon")


# ---------------------------------------------------------------------------
# Step 2.1 — Connection helpers
# ---------------------------------------------------------------------------


def get_postgres_url() -> str | None:
    """Get PostgreSQL URL from environment (canonical first)."""
    return os.environ.get("CONTINUOUS_CLAUDE_DB_URL") or os.environ.get("DATABASE_URL")


def use_postgres() -> bool:
    """Check if PostgreSQL is available."""
    url = get_postgres_url()
    if not url:
        return False
    try:
        import psycopg2  # noqa: F401

        return True
    except ImportError:
        return False


def pg_connect(max_retries: int = 3, base_delay: float = 2.0):
    """Connect to PostgreSQL with retry logic for transient failures."""
    import psycopg2

    last_error = None
    for attempt in range(max_retries):
        try:
            return psycopg2.connect(get_postgres_url())
        except psycopg2.OperationalError as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.info(
                    "DB connection failed (attempt %d/%d), retrying in %ss: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    e,
                )
                time.sleep(delay)
    raise last_error


def get_sqlite_path() -> Path:
    """Get SQLite database path."""
    return Path.home() / ".claude" / "sessions.db"


# ---------------------------------------------------------------------------
# Step 2.2 — Schema setup
# ---------------------------------------------------------------------------


def pg_ensure_column():
    """Ensure extraction columns exist in PostgreSQL."""
    conn = pg_connect()
    cur = conn.cursor()
    for col, typedef in [
        ("memory_extracted_at", "TIMESTAMP"),
        ("extraction_status", "TEXT DEFAULT 'pending'"),
        ("extraction_attempts", "INTEGER DEFAULT 0"),
        ("transcript_path", "TEXT"),
        ("archived_at", "TIMESTAMP"),
        ("archive_path", "TEXT"),
        ("last_error", "TEXT"),
    ]:
        cur.execute(f"""
            ALTER TABLE sessions
            ADD COLUMN IF NOT EXISTS {col} {typedef}
        """)
    # Ensure push tracking columns on archival_memory
    for col, typedef in [
        ("push_count", "INTEGER NOT NULL DEFAULT 0"),
        ("last_pushed_at", "TIMESTAMPTZ"),
    ]:
        cur.execute(f"""
            ALTER TABLE archival_memory
            ADD COLUMN IF NOT EXISTS {col} {typedef}
        """)
    conn.commit()
    conn.close()


def sqlite_ensure_table():
    """Ensure sessions table exists in SQLite with required columns."""
    import sqlite3

    db_path = get_sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            project TEXT,
            working_on TEXT,
            started_at TIMESTAMP,
            last_heartbeat TIMESTAMP,
            pid INTEGER,
            exited_at TIMESTAMP,
            memory_extracted_at TIMESTAMP,
            extraction_status TEXT DEFAULT 'pending',
            extraction_attempts INTEGER DEFAULT 0,
            transcript_path TEXT,
            archived_at TIMESTAMP,
            archive_path TEXT,
            last_error TEXT
        )
    """)
    # Add columns if table already exists without them
    for col, typedef in [
        ("pid", "INTEGER"),
        ("exited_at", "TIMESTAMP"),
        ("memory_extracted_at", "TIMESTAMP"),
        ("extraction_status", "TEXT DEFAULT 'pending'"),
        ("extraction_attempts", "INTEGER DEFAULT 0"),
        ("transcript_path", "TEXT"),
        ("archived_at", "TIMESTAMP"),
        ("archive_path", "TEXT"),
        ("last_error", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    conn.close()


def ensure_schema():
    """Ensure database schema is ready."""
    if use_postgres():
        pg_ensure_column()
    else:
        sqlite_ensure_table()


# ---------------------------------------------------------------------------
# Step 2.3 — Stale session queries (explicit config params per D3)
# ---------------------------------------------------------------------------


def pg_get_stale_sessions(
    stale_threshold: int, max_retries: int, harvest_grace_period: int
) -> list:
    """Get sessions with stale heartbeat that need extraction.

    Returns rows where either:
      - exited_at IS NULL (daemon must mark and wait), or
      - exited_at is older than harvest_grace_period (ready to harvest).
    Sessions within the grace period are excluded by the DB clock.
    """
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, project, transcript_path, pid, exited_at FROM sessions
        WHERE last_heartbeat < NOW() - INTERVAL '%s seconds'
        AND extraction_status = 'pending'
        AND extraction_attempts < %s
        AND (exited_at IS NULL
             OR exited_at < NOW() - INTERVAL '%s seconds')
    """,
        (stale_threshold, max_retries, harvest_grace_period),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def sqlite_get_stale_sessions(
    stale_threshold: int, max_retries: int, harvest_grace_period: int
) -> list:
    """Get sessions with stale heartbeat that need extraction.

    Returns rows where either:
      - exited_at IS NULL (daemon must mark and wait), or
      - exited_at is older than harvest_grace_period (ready to harvest).
    Sessions within the grace period are excluded.
    """
    import sqlite3
    from datetime import datetime, timedelta

    db_path = get_sqlite_path()
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    stale_dt = (datetime.now() - timedelta(seconds=stale_threshold)).isoformat()
    grace_dt = (datetime.now() - timedelta(seconds=harvest_grace_period)).isoformat()
    cursor = conn.execute(
        """
        SELECT id, project, transcript_path, pid, exited_at FROM sessions
        WHERE last_heartbeat < ?
        AND extraction_status = 'pending'
        AND COALESCE(extraction_attempts, 0) < ?
        AND (exited_at IS NULL
             OR exited_at < ?)
    """,
        (stale_dt, max_retries, grace_dt),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_stale_sessions(
    stale_threshold: int, max_retries: int, harvest_grace_period: int
) -> list:
    """Get stale sessions from database."""
    if use_postgres():
        return pg_get_stale_sessions(
            stale_threshold=stale_threshold,
            max_retries=max_retries,
            harvest_grace_period=harvest_grace_period,
        )
    return sqlite_get_stale_sessions(
        stale_threshold=stale_threshold,
        max_retries=max_retries,
        harvest_grace_period=harvest_grace_period,
    )


# ---------------------------------------------------------------------------
# Step 2.4 — mark_* functions (D3: max_retries as explicit param where needed)
# ---------------------------------------------------------------------------


def pg_mark_extracting(session_id: str):
    """Mark session as actively being extracted in PostgreSQL."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE sessions
        SET extraction_status = 'extracting',
            extraction_attempts = COALESCE(extraction_attempts, 0) + 1
        WHERE id = %s
    """,
        (session_id,),
    )
    conn.commit()
    conn.close()


def pg_mark_extracted(session_id: str):
    """Mark session as successfully extracted in PostgreSQL."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE sessions
        SET memory_extracted_at = NOW(),
            extraction_status = 'extracted'
        WHERE id = %s
    """,
        (session_id,),
    )
    conn.commit()
    conn.close()


def pg_mark_extraction_failed(
    session_id: str, max_retries: int, last_error: str | None = None
):
    """Mark extraction as failed; retry if under max_retries, else give up."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT extraction_attempts FROM sessions WHERE id = %s",
        (session_id,),
    )
    row = cur.fetchone()
    attempts = row[0] if row else 0

    if attempts < max_retries:
        cur.execute(
            "UPDATE sessions SET extraction_status = 'pending' WHERE id = %s",
            (session_id,),
        )
        logger.info(
            "Extraction failed for %s (attempt %d/%d), will retry",
            session_id,
            attempts,
            max_retries,
        )
    else:
        cur.execute(
            "UPDATE sessions SET extraction_status = 'failed', "
            "last_error = %s WHERE id = %s",
            (last_error, session_id),
        )
        suffix = f" (last error: {last_error})" if last_error else ""
        logger.info(
            "Extraction permanently failed for %s after %d attempts%s",
            session_id,
            attempts,
            suffix,
        )

    conn.commit()
    conn.close()


def pg_mark_archived(session_id: str, archive_path: str):
    """Mark session as archived in PostgreSQL and stamp learnings with archive_path."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE sessions
        SET archived_at = NOW(), archive_path = %s
        WHERE id = %s
    """,
        (archive_path, session_id),
    )
    cur.execute(
        """
        UPDATE archival_memory
        SET metadata = COALESCE(metadata, '{}'::jsonb) ||
            jsonb_build_object('source_session_id', %s, 'archive_path', %s)
        WHERE session_id = %s
        AND (metadata->>'archive_path') IS NULL
    """,
        (session_id, archive_path, session_id),
    )
    conn.commit()
    conn.close()


def sqlite_mark_archived(session_id: str, archive_path: str):
    """Mark session as archived in SQLite."""
    import sqlite3
    from datetime import datetime

    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE sessions
        SET archived_at = ?, archive_path = ?
        WHERE id = ?
    """,
        (datetime.now().isoformat(), archive_path, session_id),
    )
    conn.commit()
    conn.close()


def mark_archived(session_id: str, archive_path: str):
    """Mark session as archived."""
    if use_postgres():
        pg_mark_archived(session_id, archive_path)
    else:
        sqlite_mark_archived(session_id, archive_path)


def pg_mark_session_exited(session_id: str):
    """Set exited_at for a session the daemon observed as dead (no clean exit)."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE sessions SET exited_at = NOW()
        WHERE id = %s AND exited_at IS NULL
    """,
        (session_id,),
    )
    conn.commit()
    conn.close()


def sqlite_mark_extracting(session_id: str):
    """Mark session as actively being extracted in SQLite."""
    import sqlite3

    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE sessions
        SET extraction_status = 'extracting',
            extraction_attempts = COALESCE(extraction_attempts, 0) + 1
        WHERE id = ?
    """,
        (session_id,),
    )
    conn.commit()
    conn.close()


def sqlite_mark_extracted(session_id: str):
    """Mark session as extracted in SQLite."""
    import sqlite3
    from datetime import datetime

    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE sessions
        SET memory_extracted_at = ?,
            extraction_status = 'extracted'
        WHERE id = ?
    """,
        (datetime.now().isoformat(), session_id),
    )
    conn.commit()
    conn.close()


def sqlite_mark_extraction_failed(
    session_id: str, max_retries: int, last_error: str | None = None
):
    """Mark extraction as failed in SQLite; retry if under max_retries."""
    import sqlite3

    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT extraction_attempts FROM sessions WHERE id = ?", (session_id,)
    )
    row = cursor.fetchone()
    attempts = row[0] if row else 0

    if attempts < max_retries:
        conn.execute(
            "UPDATE sessions SET extraction_status = 'pending' WHERE id = ?",
            (session_id,),
        )
    else:
        conn.execute(
            "UPDATE sessions SET extraction_status = 'failed', "
            "last_error = ? WHERE id = ?",
            (last_error, session_id),
        )
        suffix = f" (last error: {last_error})" if last_error else ""
        logger.info(
            "Extraction permanently failed for %s after %d attempts%s",
            session_id,
            attempts,
            suffix,
        )
    conn.commit()
    conn.close()


def sqlite_mark_session_exited(session_id: str):
    """Set exited_at for a session the daemon observed as dead (no clean exit)."""
    import sqlite3
    from datetime import datetime

    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE sessions SET exited_at = ? WHERE id = ? AND exited_at IS NULL",
        (datetime.now().isoformat(), session_id),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Step 2.5 — Recovery functions
# ---------------------------------------------------------------------------


def pg_recover_stalled_extractions():
    """Reset sessions stuck in 'extracting' back to 'pending' on daemon startup."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sessions
        SET extraction_status = 'pending'
        WHERE extraction_status = 'extracting'
        RETURNING id
    """)
    recovered = [row[0] for row in cur.fetchall()]
    conn.commit()
    conn.close()
    if recovered:
        logger.info(
            "Startup recovery: reset %d stalled sessions: %s",
            len(recovered),
            ", ".join(recovered),
        )


def sqlite_recover_stalled_extractions():
    """Reset sessions stuck in 'extracting' back to 'pending' on daemon startup."""
    import sqlite3

    db_path = get_sqlite_path()
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        UPDATE sessions
        SET extraction_status = 'pending'
        WHERE extraction_status = 'extracting'
        RETURNING id
    """)
    recovered = [row[0] for row in cursor.fetchall()]
    conn.commit()
    conn.close()
    if recovered:
        logger.info(
            "Startup recovery: reset %d stalled sessions: %s",
            len(recovered),
            ", ".join(recovered),
        )


def recover_stalled_extractions():
    """Reset stalled extractions on daemon startup (handles sleep/crash recovery)."""
    if use_postgres():
        pg_recover_stalled_extractions()
    else:
        sqlite_recover_stalled_extractions()


# ---------------------------------------------------------------------------
# Step 2.6 — Dispatcher functions and remaining queries
# ---------------------------------------------------------------------------


def mark_extracting(session_id: str):
    """Mark session as actively being extracted."""
    if use_postgres():
        pg_mark_extracting(session_id)
    else:
        sqlite_mark_extracting(session_id)


def mark_extracted(session_id: str):
    """Mark session as extracted."""
    if use_postgres():
        pg_mark_extracted(session_id)
    else:
        sqlite_mark_extracted(session_id)


def mark_extraction_failed(
    session_id: str, max_retries: int, last_error: str | None = None
):
    """Mark extraction as failed (will retry if under max_retries)."""
    if use_postgres():
        pg_mark_extraction_failed(
            session_id, max_retries=max_retries, last_error=last_error
        )
    else:
        sqlite_mark_extraction_failed(
            session_id, max_retries=max_retries, last_error=last_error
        )


def mark_session_exited(session_id: str):
    """Record exited_at for a session whose PID the daemon observed as dead."""
    if use_postgres():
        pg_mark_session_exited(session_id)
    else:
        sqlite_mark_session_exited(session_id)


def count_session_learnings(session_id: str) -> int | None:
    """Count learnings stored for a session.

    Returns None on error or when using SQLite (not implemented for SQLite).
    """
    if not use_postgres():
        return None
    try:
        conn = pg_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM archival_memory WHERE session_id = %s",
            (session_id,),
        )
        count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.debug("count_session_learnings failed for %s: %s", session_id, e)
        return None


def seed_last_pattern_run() -> float:
    """Read the most recent pattern detection timestamp from PostgreSQL.

    Returns a Unix timestamp so the daemon skips an immediate re-run
    after restart if a recent detection already happened.
    """
    if not use_postgres():
        return 0
    try:
        conn = pg_connect()
        cur = conn.cursor()
        cur.execute("SELECT MAX(created_at) FROM detected_patterns")
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return row[0].timestamp()
    except Exception as e:
        logger.debug("seed_last_pattern_run failed: %s", e)
    return 0
