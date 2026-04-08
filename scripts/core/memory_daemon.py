#!/usr/bin/env python3
"""
Global Memory Extraction Daemon

A singleton daemon that monitors for stale Claude sessions and automatically
extracts learnings when sessions end (heartbeat goes stale).

USAGE:
    # Start daemon (if not already running)
    uv run python scripts/core/memory_daemon.py start

    # Check status
    uv run python scripts/core/memory_daemon.py status

    # Stop daemon
    uv run python scripts/core/memory_daemon.py stop

ARCHITECTURE:
    - Single global instance (PID file at ~/.claude/memory-daemon.pid)
    - Works with PostgreSQL or SQLite
    - Polls every 60 seconds for stale sessions (heartbeat > 15 min)
    - Runs headless `claude -p` for memory extraction
    - 4-state extraction lifecycle: pending -> extracting -> extracted | failed
    - S3 archival after successful extraction (zstd + upload)
    - DB retry logic for transient PostgreSQL failures

The session_start hook ensures this daemon is running.
"""

import argparse
import logging
import logging.handlers
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path so `scripts.*` imports work
# when launched via `uv run` (which doesn't add cwd to sys.path)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import faulthandler

faulthandler.enable(file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"), all_threads=True)

# Load .env files for DATABASE_URL (cross-platform)
# 1. Local opc/.env first (relative to script location)
# Script is at opc/scripts/core/memory_daemon.py, .env is at opc/.env
opc_env = Path(__file__).parent.parent.parent / ".env"
if opc_env.exists():
    load_dotenv(opc_env, override=True)

# 2. Global ~/.claude/.env (API keys, S3 bucket, etc.)
# Loaded with override=True so global secrets always take effect,
# even if the hook environment had empty/missing values.
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env, override=True)

try:
    from importlib.metadata import version as _pkg_version
    DAEMON_VERSION = _pkg_version("mcp-execution")
except Exception:
    DAEMON_VERSION = "0.7.3"  # fallback


def _is_extraction_blocked(project_dir: str) -> bool:
    """Return True if this project has opted out of memory extraction."""
    if not project_dir:
        return False
    sentinel = Path(project_dir) / ".claude" / "no-extract"
    return sentinel.exists()


# Re-exports from memory_daemon_core (D12: shims per step)
from scripts.core.memory_daemon_core import (  # noqa: E402
    StaleSession,
    _ALLOWED_EXTRACTION_MODELS,
    _normalize_project,
)

# Config from opc.toml [daemon]
from scripts.core.config import get_config as _get_config
_daemon_cfg = _get_config().daemon

POLL_INTERVAL = _daemon_cfg.poll_interval
STALE_THRESHOLD = _daemon_cfg.stale_threshold
MAX_CONCURRENT_EXTRACTIONS = _daemon_cfg.max_concurrent_extractions
MAX_RETRIES = _daemon_cfg.max_retries
EXTRACTION_TIMEOUT = _daemon_cfg.extraction_timeout
HARVEST_GRACE_PERIOD = _daemon_cfg.harvest_grace_period
PID_FILE = Path.home() / ".claude" / "memory-daemon.pid"
LOG_FILE = Path.home() / ".claude" / "memory-daemon.log"

# Worker queue state (module-level for daemon process)
active_extractions: dict = {}  # pid -> (session_id, proc, jsonl_path, project, start_time)
pending_queue: list[tuple[str, str, str | None]] = []  # [(session_id, project, transcript_path), ...]

# Pattern detection state (module-level for daemon process)
# Interval configurable via PATTERN_DETECTION_INTERVAL_HOURS env var
_PATTERN_DETECTION_INTERVAL = _daemon_cfg.pattern_detection_interval_hours * 3600
_pattern_proc: subprocess.Popen | None = None
_last_pattern_run: float = 0


def _setup_logging() -> logging.Logger:
    """Configure rotating logger for the daemon.

    Uses TimedRotatingFileHandler to rotate the log every N days,
    keeping a configurable number of backups. Rotated files are
    named memory-daemon.log.YYYY-MM-DD.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("memory-daemon")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.handlers.TimedRotatingFileHandler(
            LOG_FILE,
            when="D",
            interval=_daemon_cfg.log_rotation_days,
            backupCount=_daemon_cfg.log_backup_count,
        )
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    return logger


_logger = _setup_logging()


def _seed_last_pattern_run() -> float:
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
    except Exception:
        pass
    return 0


def log(msg: str):
    """Write timestamped log message via rotating file handler."""
    try:
        _logger.info(msg)
    except Exception:
        pass  # Don't crash on log failures


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


# Database operations - PostgreSQL

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
                delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s
                log(f"DB connection failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay}s: {e}")
                time.sleep(delay)
    raise last_error


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
    ]:
        cur.execute(f"""
            ALTER TABLE sessions
            ADD COLUMN IF NOT EXISTS {col} {typedef}
        """)
    conn.commit()
    conn.close()


def pg_get_stale_sessions() -> list:
    """Get sessions with stale heartbeat that need extraction.

    Returns rows where either:
      - exited_at IS NULL (daemon must mark and wait), or
      - exited_at is older than harvest_grace_period (ready to harvest).
    Sessions within the grace period are excluded by the DB clock.
    """
    conn = pg_connect()
    cur = conn.cursor()
    # Use DB clock for all time comparisons to avoid local-vs-UTC mismatch
    cur.execute("""
        SELECT id, project, transcript_path, pid, exited_at FROM sessions
        WHERE last_heartbeat < NOW() - INTERVAL '%s seconds'
        AND extraction_status = 'pending'
        AND extraction_attempts < %s
        AND (exited_at IS NULL
             OR exited_at < NOW() - INTERVAL '%s seconds')
    """, (STALE_THRESHOLD, MAX_RETRIES, HARVEST_GRACE_PERIOD))
    rows = cur.fetchall()
    conn.close()
    return rows


def pg_mark_extracting(session_id: str):
    """Mark session as actively being extracted in PostgreSQL."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sessions
        SET extraction_status = 'extracting',
            extraction_attempts = COALESCE(extraction_attempts, 0) + 1
        WHERE id = %s
    """, (session_id,))
    conn.commit()
    conn.close()


def pg_mark_extracted(session_id: str):
    """Mark session as successfully extracted in PostgreSQL."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sessions
        SET memory_extracted_at = NOW(),
            extraction_status = 'extracted'
        WHERE id = %s
    """, (session_id,))
    conn.commit()
    conn.close()


def pg_mark_extraction_failed(session_id: str):
    """Mark extraction as failed; retry if under MAX_RETRIES, else give up."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT extraction_attempts FROM sessions WHERE id = %s
    """, (session_id,))
    row = cur.fetchone()
    attempts = row[0] if row else 0

    if attempts < MAX_RETRIES:
        cur.execute("""
            UPDATE sessions SET extraction_status = 'pending' WHERE id = %s
        """, (session_id,))
        log(f"Extraction failed for {session_id} (attempt {attempts}/{MAX_RETRIES}), will retry")
    else:
        cur.execute("""
            UPDATE sessions SET extraction_status = 'failed' WHERE id = %s
        """, (session_id,))
        log(f"Extraction permanently failed for {session_id} after {attempts} attempts")

    conn.commit()
    conn.close()


def pg_mark_archived(session_id: str, archive_path: str):
    """Mark session as archived in PostgreSQL and stamp learnings with archive_path."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sessions
        SET archived_at = NOW(), archive_path = %s
        WHERE id = %s
    """, (archive_path, session_id))
    # Stamp archival_memory with source traceability
    cur.execute("""
        UPDATE archival_memory
        SET metadata = COALESCE(metadata, '{}'::jsonb) ||
            jsonb_build_object('source_session_id', %s, 'archive_path', %s)
        WHERE session_id = %s
        AND (metadata->>'archive_path') IS NULL
    """, (session_id, archive_path, session_id))
    conn.commit()
    conn.close()


def pg_mark_session_exited(session_id: str):
    """Set exited_at for a session the daemon observed as dead (no clean exit)."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sessions SET exited_at = NOW()
        WHERE id = %s AND exited_at IS NULL
    """, (session_id,))
    conn.commit()
    conn.close()


# Database operations - SQLite
def get_sqlite_path() -> Path:
    """Get SQLite database path."""
    return Path.home() / ".claude" / "sessions.db"


def sqlite_ensure_table():
    """Ensure sessions table exists in SQLite with required columns."""
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
            memory_extracted_at TIMESTAMP,
            extraction_status TEXT DEFAULT 'pending',
            extraction_attempts INTEGER DEFAULT 0,
            transcript_path TEXT,
            archived_at TIMESTAMP,
            archive_path TEXT
        )
    """)
    # Add columns if table already exists without them
    for col, typedef in [
        ("memory_extracted_at", "TIMESTAMP"),
        ("extraction_status", "TEXT DEFAULT 'pending'"),
        ("extraction_attempts", "INTEGER DEFAULT 0"),
        ("transcript_path", "TEXT"),
        ("archived_at", "TIMESTAMP"),
        ("archive_path", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    conn.close()


def sqlite_get_stale_sessions() -> list:
    """Get sessions with stale heartbeat that need extraction.

    Returns rows where either:
      - exited_at IS NULL (daemon must mark and wait), or
      - exited_at is older than harvest_grace_period (ready to harvest).
    Sessions within the grace period are excluded.
    """
    db_path = get_sqlite_path()
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    stale_threshold = (datetime.now() - timedelta(seconds=STALE_THRESHOLD)).isoformat()
    grace_threshold = (datetime.now() - timedelta(seconds=HARVEST_GRACE_PERIOD)).isoformat()
    cursor = conn.execute("""
        SELECT id, project, transcript_path, pid, exited_at FROM sessions
        WHERE last_heartbeat < ?
        AND extraction_status = 'pending'
        AND COALESCE(extraction_attempts, 0) < ?
        AND (exited_at IS NULL
             OR exited_at < ?)
    """, (stale_threshold, MAX_RETRIES, grace_threshold))
    rows = cursor.fetchall()
    conn.close()
    return rows


def sqlite_mark_extracting(session_id: str):
    """Mark session as actively being extracted in SQLite."""
    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    conn.execute("""
        UPDATE sessions
        SET extraction_status = 'extracting',
            extraction_attempts = COALESCE(extraction_attempts, 0) + 1
        WHERE id = ?
    """, (session_id,))
    conn.commit()
    conn.close()


def sqlite_mark_extracted(session_id: str):
    """Mark session as extracted in SQLite."""
    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    conn.execute("""
        UPDATE sessions
        SET memory_extracted_at = ?,
            extraction_status = 'extracted'
        WHERE id = ?
    """, (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()


def sqlite_mark_extraction_failed(session_id: str):
    """Mark extraction as failed in SQLite; retry if under MAX_RETRIES."""
    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT extraction_attempts FROM sessions WHERE id = ?", (session_id,)
    )
    row = cursor.fetchone()
    attempts = row[0] if row else 0

    if attempts < MAX_RETRIES:
        conn.execute(
            "UPDATE sessions SET extraction_status = 'pending' WHERE id = ?",
            (session_id,),
        )
    else:
        conn.execute(
            "UPDATE sessions SET extraction_status = 'failed' WHERE id = ?",
            (session_id,),
        )
    conn.commit()
    conn.close()


def sqlite_mark_session_exited(session_id: str):
    """Set exited_at for a session the daemon observed as dead (no clean exit)."""
    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE sessions SET exited_at = ? WHERE id = ? AND exited_at IS NULL",
        (datetime.now().isoformat(), session_id),
    )
    conn.commit()
    conn.close()


# Unified interface
def _is_process_alive(pid: int | None) -> bool:
    """Check if a process is still running via kill(0) signal."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


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
        log(f"Startup recovery: reset {len(recovered)} stalled sessions: {', '.join(recovered)}")


def sqlite_recover_stalled_extractions():
    """Reset sessions stuck in 'extracting' back to 'pending' on daemon startup."""
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
        log(f"Startup recovery: reset {len(recovered)} stalled sessions: {', '.join(recovered)}")


def recover_stalled_extractions():
    """Reset stalled extractions on daemon startup (handles sleep/crash recovery)."""
    if use_postgres():
        pg_recover_stalled_extractions()
    else:
        sqlite_recover_stalled_extractions()


def ensure_schema():
    """Ensure database schema is ready."""
    if use_postgres():
        pg_ensure_column()
    else:
        sqlite_ensure_table()


def get_stale_sessions() -> list:
    """Get stale sessions from database."""
    if use_postgres():
        return pg_get_stale_sessions()
    return sqlite_get_stale_sessions()


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


def mark_extraction_failed(session_id: str):
    """Mark extraction as failed (will retry if under MAX_RETRIES)."""
    if use_postgres():
        pg_mark_extraction_failed(session_id)
    else:
        sqlite_mark_extraction_failed(session_id)


def mark_session_exited(session_id: str):
    """Record exited_at for a session whose PID the daemon observed as dead."""
    if use_postgres():
        pg_mark_session_exited(session_id)
    else:
        sqlite_mark_session_exited(session_id)


def extract_memories(
    session_id: str,
    project_dir: str,
    transcript_path: str | None = None,
) -> bool:
    """Run memory extraction for a session. Returns True if subprocess started."""
    log(f"Extracting memories for session {session_id} "
        f"(project={project_dir or 'unknown'})")

    if _is_extraction_blocked(project_dir):
        log(f"Extraction blocked by .claude/no-extract sentinel "
            f"(project={project_dir}), marking as extracted (skip)")
        mark_extracted(session_id)
        return False

    # Use transcript_path from DB — no glob fallback (wrong-file guessing caused orphaned extractions)
    jsonl_path = None
    if transcript_path:
        candidate = Path(transcript_path)
        if candidate.exists():
            jsonl_path = candidate

    if not jsonl_path:
        reason = "no transcript_path in DB" if not transcript_path else "file missing from disk"
        log(f"No JSONL for session {session_id} "
            f"(project={project_dir or 'unknown'}, {reason}), marking as extracted (skip)")
        mark_extracted(session_id)
        return False

    # Run headless memory extraction
    try:
        # Read agent prompt from memory-extractor.md (strip YAML frontmatter)
        config_dir = Path(os.environ.get('CLAUDE_CONFIG_DIR', str(Path.home() / '.claude')))
        agent_file = config_dir / "agents" / "memory-extractor.md"

        agent_prompt = ""
        if agent_file.exists():
            content = agent_file.read_text()
            # Strip YAML frontmatter if present
            if content.startswith("---"):
                parts = content.split("---", 2)
                agent_prompt = parts[2].strip() if len(parts) >= 3 else content
            else:
                agent_prompt = content
        else:
            # Fallback minimal prompt
            agent_prompt = """Extract learnings from this Claude Code session.
Look for decisions, what worked, what failed, and patterns discovered.
Store each learning using store_learning.py with appropriate type and tags."""

        if _daemon_cfg.extraction_model not in _ALLOWED_EXTRACTION_MODELS:
            log(
                f"Invalid extraction_model '{_daemon_cfg.extraction_model}', "
                f"must be one of {sorted(_ALLOWED_EXTRACTION_MODELS)}"
            )
            mark_extracted(session_id)
            return False

        env = os.environ.copy()
        env["CLAUDE_MEMORY_EXTRACTION"] = "1"  # Prevent session-register from registering extraction sessions
        if project_dir:
            env["CLAUDE_PROJECT_DIR"] = project_dir

        proc = subprocess.Popen(
            [
                "claude", "-p",
                "--model", _daemon_cfg.extraction_model,
                "--dangerously-skip-permissions",
                "--allowedTools", "Bash,Read",
                "--max-turns", str(_daemon_cfg.extraction_max_turns),
                "--append-system-prompt", agent_prompt,
                f"Extract learnings from session {session_id}. JSONL path: {jsonl_path}"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        active_extractions[proc.pid] = (
            session_id, proc, jsonl_path, project_dir, time.time()
        )
        log(f"Started extraction for {session_id} "
            f"(pid={proc.pid}, file={jsonl_path.name}, "
            f"active={len(active_extractions)})")
        return True
    except Exception as e:
        log(f"Failed to start extraction: {e}")
        return False


def archive_session_jsonl(session_id: str, jsonl_path: Path | None = None):
    """Compress and upload session JSONL to S3, then delete local copy."""
    bucket = os.environ.get("CLAUDE_SESSION_ARCHIVE_BUCKET")
    if not bucket:
        return

    if not jsonl_path or not jsonl_path.exists():
        log(f"Archive skipped for {session_id}: JSONL not found")
        return

    project_name = jsonl_path.parent.name
    s3_key = f"s3://{bucket}/sessions/{project_name}/{jsonl_path.stem}.jsonl.zst"
    zst_path = jsonl_path.with_suffix(".jsonl.zst")

    try:
        # Compress with zstd
        result = subprocess.run(
            ["zstd", "-q", "--rm", str(jsonl_path)],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0:
            log(f"zstd failed for {session_id}: {result.stderr.decode()}")
            return

        # Upload to S3
        result = subprocess.run(
            ["aws", "s3", "cp", str(zst_path), s3_key, "--quiet"],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            log(f"S3 upload failed for {session_id}: {result.stderr.decode()}")
            # Restore original file on upload failure
            subprocess.run(
                ["zstd", "-d", "-q", "--rm", str(zst_path)],
                capture_output=True, timeout=300,
            )
            return

        # Clean up local compressed file
        zst_path.unlink(missing_ok=True)

        # Mark archived in DB
        try:
            pg_mark_archived(session_id, s3_key)
        except Exception as e:
            log(f"Archive DB update failed for {session_id} (file already in S3): {e}")

        log(f"Archived {session_id} -> {s3_key}")

    except subprocess.TimeoutExpired:
        log(f"Archive timeout for {session_id}")
        # Restore if compressed but not uploaded
        if zst_path.exists() and not jsonl_path.exists():
            subprocess.run(
                ["zstd", "-d", "-q", "--rm", str(zst_path)],
                capture_output=True, timeout=300,
            )
    except Exception as e:
        log(f"Archive error for {session_id}: {e}")


def _calibrate_session_confidence(session_id: str):
    """Run confidence calibration on learnings from a completed extraction."""
    try:
        import asyncio

        from scripts.core.confidence_calibrator import calibrate_session

        result = asyncio.run(calibrate_session(session_id))
        stats = result["stats"]
        if stats["total"] > 0:
            log(
                f"Confidence calibration for {session_id}: "
                f"{stats['updated']} updated, "
                f"{stats['unchanged']} unchanged"
            )
    except Exception as e:
        log(f"Confidence calibration failed for {session_id}: {e}")


def _extract_and_store_workflows(
    session_id: str,
    jsonl_path: Path,
    project: str | None,
):
    """Extract workflow patterns and store as learnings. Non-fatal."""
    try:
        from scripts.core.extract_workflow_patterns import (
            detect_workflow_sequences,
            extract_tool_uses,
            format_pattern_as_learning,
        )
    except ImportError as e:
        log(f"Workflow extraction unavailable: {e}")
        return

    try:
        tool_uses = extract_tool_uses(jsonl_path, max_entries=50_000)
        patterns = detect_workflow_sequences(tool_uses)
        successful = [p for p in patterns if p.get("success") is True]

        if not successful:
            log(f"No successful workflow patterns for {session_id}")
            return

        from scripts.core.store_learning import store_learning_v2

        stored = 0
        for pattern in successful:
            content = format_pattern_as_learning(pattern)
            try:
                import asyncio
                project_name = _normalize_project(project) if project else None
                result = asyncio.run(store_learning_v2(
                    session_id=session_id,
                    content=content,
                    learning_type="WORKING_SOLUTION",
                    context=project or "unknown",
                    tags=["workflow", pattern["pattern_type"]],
                    confidence="high",
                    project=project_name,
                ))
                if result.get("success") and not result.get("skipped"):
                    stored += 1
            except Exception as e:
                log(f"Failed to store workflow learning: {e}")

        log(f"Stored {stored} workflow patterns for {session_id}")
    except Exception as e:
        log(f"Workflow extraction failed for {session_id}: {e}")


def _generate_mini_handoff(
    session_id: str,
    jsonl_path: Path,
    project: str | None,
):
    """Generate a mini-handoff YAML from session data. Non-fatal.

    Prefers state file (real-time hook data) over JSONL (post-session transcript).
    Cleans up state file after successful generation.
    """
    try:
        from scripts.core.generate_mini_handoff import (
            generate_handoff,
            write_handoff,
        )
    except ImportError as e:
        log(f"Mini-handoff generation unavailable: {e}")
        return

    if not project:
        log(f"Mini-handoff skipped for {session_id}: no project dir")
        return

    # Check for state file from session-state-collector hook
    state_file = Path(project) / ".claude" / "cache" / "session-state" / f"{session_id}.jsonl"
    use_state_file = state_file.exists() and state_file.stat().st_size > 0

    try:
        handoff = generate_handoff(
            session_id=session_id,
            project_dir=project,
            jsonl_path=jsonl_path,
            state_file=state_file if use_state_file else None,
        )
        output_path = write_handoff(handoff, Path(project), session_id)
        source = "state_file" if use_state_file else "jsonl"
        log(f"Mini-handoff written for {session_id} (source={source}): {output_path}")

        # Clean up state file after successful generation
        if use_state_file:
            try:
                state_file.unlink()
                log(f"State file cleaned up for {session_id}")
            except OSError as cleanup_err:
                log(f"State file cleanup failed for {session_id}: {cleanup_err}")
    except Exception as e:
        log(f"Mini-handoff generation failed for {session_id}: {e}")


def _count_session_learnings(session_id: str) -> int | None:
    """Count learnings stored for a session. Returns None on error."""
    try:
        if use_postgres():
            conn = pg_connect()
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM archival_memory WHERE session_id = %s",
                (session_id,),
            )
            count = cur.fetchone()[0]
            conn.close()
            return count
    except Exception:
        return None
    return None


def _count_session_rejections(session_id: str) -> int | None:
    """Count rejected learnings for a session. Returns None on error."""
    try:
        from scripts.core.store_learning import get_rejection_count

        return get_rejection_count(session_id)
    except Exception:
        return None


def reap_completed_extractions():
    """Check for completed extraction processes and remove from active set."""
    completed = []
    for pid, (session_id, proc, jsonl_path, project, _start) in list(active_extractions.items()):
        exit_code = proc.poll()
        if exit_code is not None:
            completed.append(pid)
            elapsed = int(time.time() - _start)
            learnings_count = _count_session_learnings(session_id) if exit_code == 0 else None
            learnings_info = f", learnings={learnings_count}" if learnings_count is not None else ""
            rejections_count = _count_session_rejections(session_id) if exit_code == 0 else None
            rejections_info = f", rejections={rejections_count}" if rejections_count is not None else ""
            log(f"Extraction completed for {session_id} "
                f"(pid={pid}, project={project}, "
                f"exit={exit_code}, elapsed={elapsed}s{learnings_info}{rejections_info})")
            if exit_code == 0:
                mark_extracted(session_id)
                _calibrate_session_confidence(session_id)
                _extract_and_store_workflows(session_id, jsonl_path, project)
                _generate_mini_handoff(session_id, jsonl_path, project)
                archive_session_jsonl(session_id, jsonl_path)
            else:
                mark_extraction_failed(session_id)

    for pid in completed:
        del active_extractions[pid]

    return len(completed)


def watchdog_stuck_extractions():
    """Kill extraction subprocesses that exceed EXTRACTION_TIMEOUT."""
    now = time.time()
    killed = []
    for pid, (session_id, proc, jsonl_path, project, start_time) in list(
        active_extractions.items()
    ):
        elapsed = now - start_time
        if elapsed > EXTRACTION_TIMEOUT:
            elapsed_min = int(elapsed / 60)
            log(f"Watchdog: killing stuck extraction "
                f"{session_id} (pid={pid}, "
                f"running {elapsed_min}m)")
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception as e:
                log(f"Watchdog: failed to kill pid {pid}: {e}")
            killed.append(pid)
            mark_extraction_failed(session_id)

    for pid in killed:
        del active_extractions[pid]

    if killed:
        log(f"Watchdog: killed {len(killed)} stuck extractions")
    return len(killed)


def process_pending_queue():
    """Spawn extractions from queue if under concurrency limit."""
    spawned = 0
    while pending_queue and len(active_extractions) < MAX_CONCURRENT_EXTRACTIONS:
        session_id, project, transcript_path = pending_queue.pop(0)
        log(f"Dequeuing {session_id} (project={project or 'unknown'}, "
            f"queue remaining: {len(pending_queue)})")
        extract_memories(session_id, project, transcript_path)
        spawned += 1
    return spawned


def queue_or_extract(
    session_id: str,
    project: str,
    transcript_path: str | None = None,
):
    """Queue extraction if at limit, otherwise extract immediately."""
    if len(active_extractions) >= MAX_CONCURRENT_EXTRACTIONS:
        pending_queue.append((session_id, project, transcript_path))
        log(f"Queued {session_id} (active={len(active_extractions)}, "
            f"queue={len(pending_queue)})")
    else:
        extract_memories(session_id, project, transcript_path)


# ---------------------------------------------------------------------------
# Pattern detection (non-blocking subprocess)
# ---------------------------------------------------------------------------

def _run_pattern_detection_batch():
    """Launch pattern detection as a non-blocking subprocess.

    Uses Popen to avoid blocking the daemon loop (detection can take
    minutes on large datasets). Only one detection run at a time.
    """
    global _pattern_proc, _last_pattern_run
    # Don't start if already running
    if _pattern_proc is not None and _pattern_proc.poll() is None:
        log("Pattern detection already running, skipping")
        return
    try:
        project_root = Path(__file__).parent.parent.parent
        log("Starting pattern detection batch...")
        _pattern_proc = subprocess.Popen(
            [sys.executable, "-m", "scripts.core.pattern_batch"],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _last_pattern_run = time.time()
    except Exception as e:
        log(f"Pattern detection launch error: {e}")


def _check_pattern_detection():
    """Check if background pattern detection finished.

    Called from daemon loop each iteration.
    """
    global _pattern_proc
    if _pattern_proc is None:
        return
    rc = _pattern_proc.poll()
    if rc is not None:
        if rc == 0:
            stdout = _pattern_proc.stdout.read().decode()
            try:
                import json as _json
                data = _json.loads(stdout)
                total = data.get("patterns_detected", "?")
                analyzed = data.get("learnings_analyzed", "?")
                by_type = data.get("patterns_by_type", {})
                type_summary = ", ".join(
                    f"{k}={v}" for k, v in sorted(by_type.items())
                )
                log(f"Pattern detection completed: {total} patterns "
                    f"from {analyzed} learnings ({type_summary})")
            except (_json.JSONDecodeError, KeyError):
                log(f"Pattern detection completed: {stdout[:200]}")
        else:
            stderr = _pattern_proc.stderr.read().decode()[:200]
            log(f"Pattern detection failed (rc={rc}): {stderr}")
        _pattern_proc = None


def daemon_loop():
    """Main daemon loop."""
    db_type = "PostgreSQL" if use_postgres() else "SQLite"
    log(f"Memory daemon v{DAEMON_VERSION} started "
        f"(using {db_type}, "
        f"max_concurrent={MAX_CONCURRENT_EXTRACTIONS})")
    ensure_schema()
    recover_stalled_extractions()

    global _last_pattern_run
    _last_pattern_run = _seed_last_pattern_run()
    if _last_pattern_run:
        log(f"Seeded last pattern run from DB: "
            f"{datetime.fromtimestamp(_last_pattern_run).isoformat()}")

    while True:
        try:
            # Reap completed, kill stuck, then process pending queue
            reap_completed_extractions()
            watchdog_stuck_extractions()
            process_pending_queue()

            # Find stale sessions (SQL excludes those within grace period)
            stale = get_stale_sessions()
            if stale:
                # Filter: skip alive processes, mark newly-dead sessions
                truly_stale = []
                for row in stale:
                    session_id = row[0]
                    pid = row[3] if len(row) > 3 else None
                    exited_at = row[4] if len(row) > 4 else None
                    if _is_process_alive(pid):
                        log(f"Skipping {session_id}: process {pid} "
                            f"still alive")
                        continue
                    if exited_at is None:
                        # First time daemon sees this session as dead;
                        # record exited_at so grace period starts
                        mark_session_exited(session_id)
                        log(f"Skipping {session_id}: marked exited, "
                            f"grace period {HARVEST_GRACE_PERIOD}s")
                        continue
                    # exited_at exists and past grace period (SQL filtered)
                    truly_stale.append(row)

                if truly_stale:
                    summary = ", ".join(
                        f"{sid}({proj or '?'})"
                        for sid, proj, *_ in truly_stale
                    )
                    log(f"Found {len(truly_stale)} stale sessions: "
                        f"{summary}")
                    for row in truly_stale:
                        session_id, project = row[0], row[1]
                        tp = row[2] if len(row) > 2 else None
                        mark_extracting(session_id)
                        queue_or_extract(
                            session_id, project or "", tp
                        )
            # Pattern detection: check completion, trigger if due
            # Only runs on PostgreSQL — pattern_batch.py requires asyncpg
            _check_pattern_detection()
            if use_postgres():
                elapsed = time.time() - _last_pattern_run
                if elapsed > _PATTERN_DETECTION_INTERVAL:
                    _run_pattern_detection_batch()

        except Exception as e:
            log(f"Error in daemon loop: {e}")

        time.sleep(POLL_INTERVAL)


def is_running() -> tuple[bool, int | None]:
    """Check if daemon is already running."""
    if not PID_FILE.exists():
        return False, None

    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        return True, pid
    except (ValueError, ProcessLookupError, PermissionError):
        # Stale PID file
        PID_FILE.unlink(missing_ok=True)
        return False, None


def _run_as_daemon():
    """Run the daemon loop (called by subprocess on Windows, directly after fork on Unix)."""
    # Write PID file
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    # Close standard file descriptors
    sys.stdin.close()
    sys.stdout.close()
    sys.stderr.close()

    # Run daemon
    try:
        daemon_loop()
    finally:
        PID_FILE.unlink(missing_ok=True)


def start_daemon():
    """Start the daemon if not already running.

    Cross-platform: Uses subprocess.DETACHED_PROCESS on Windows,
    double-fork on Unix (macOS/Linux).
    """
    running, pid = is_running()
    if running:
        print(f"Memory daemon already running (PID {pid})")
        return 0

    if sys.platform == "win32":
        # Windows: spawn as detached subprocess
        DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        try:
            with open(os.devnull, "r+b") as devnull:
                subprocess.Popen(
                    [sys.executable, __file__, "--daemon-subprocess"],
                    creationflags=DETACHED_PROCESS,
                    stdin=devnull,
                    stdout=devnull,
                    stderr=devnull,
                )
            print("Memory daemon started")
            return 0
        except Exception as e:
            print(f"Failed to start daemon: {e}")
            return 1
    else:
        # Unix (macOS/Linux): classic double-fork
        if os.fork() > 0:
            print("Memory daemon started")
            return 0

        # Detach from terminal
        os.setsid()

        # Fork again to prevent zombie
        if os.fork() > 0:
            sys.exit(0)

        _run_as_daemon()


def stop_daemon():
    """Stop the daemon."""
    running, pid = is_running()
    if not running:
        print("Memory daemon not running")
        return 0

    try:
        log(f"Memory daemon stopping (PID {pid})")
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped memory daemon (PID {pid})")
        PID_FILE.unlink(missing_ok=True)
        return 0
    except ProcessLookupError:
        log(f"Memory daemon already exited (PID {pid}), cleaning up PID file")
        PID_FILE.unlink(missing_ok=True)
        return 0


def status_daemon():
    """Show daemon status."""
    running, pid = is_running()
    db_type = "PostgreSQL" if use_postgres() else "SQLite"

    print(f"Memory Daemon Status (v{DAEMON_VERSION})")
    print(f"  Running: {'Yes' if running else 'No'}")
    if running:
        print(f"  PID: {pid}")
    print(f"  Database: {db_type}")
    print(f"  PID file: {PID_FILE}")
    print(f"  Log file: {LOG_FILE}")

    # Show extraction status counts
    if use_postgres():
        try:
            conn = pg_connect()
            cur = conn.cursor()
            cur.execute("""
                SELECT extraction_status, COUNT(*)
                FROM sessions
                GROUP BY extraction_status
            """)
            counts = dict(cur.fetchall())
            conn.close()
            print("\nExtraction Status:")
            for status in ['pending', 'extracting', 'extracted', 'failed']:
                print(f"  {status}: {counts.get(status, 0)}")
        except Exception as e:
            print(f"\n  (DB query failed: {e})")

    # Show pattern detection status
    if use_postgres():
        try:
            conn = pg_connect()
            cur = conn.cursor()
            cur.execute("""
                SELECT run_id, MIN(created_at), COUNT(*)
                FROM detected_patterns
                WHERE superseded_at IS NULL
                GROUP BY run_id
                ORDER BY MIN(created_at) DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            conn.close()
            if row:
                _, last_run, count = row
                print(f"\nPattern Detection:")
                print(f"  Last run: {last_run}")
                print(f"  Active patterns: {count}")
                interval_h = _PATTERN_DETECTION_INTERVAL // 3600
                print(f"  Interval: every {interval_h}h")
            else:
                print("\nPattern Detection: no runs yet")
        except Exception:
            # Table may not exist yet — that's fine
            pass

    # Show recent log
    if LOG_FILE.exists():
        print("\nRecent log:")
        lines = LOG_FILE.read_text().strip().split("\n")[-5:]
        for line in lines:
            print(f"  {line}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Global Memory Extraction Daemon")
    parser.add_argument("command", nargs="?", choices=["start", "stop", "status"], help="Command")
    parser.add_argument("--daemon-subprocess", action="store_true",
                        help="Internal: run as daemon subprocess (Windows)")
    args = parser.parse_args()

    # Windows subprocess entry point
    if args.daemon_subprocess:
        _run_as_daemon()
        return 0

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "start":
        return start_daemon()
    elif args.command == "stop":
        return stop_daemon()
    elif args.command == "status":
        return status_daemon()


if __name__ == "__main__":
    sys.exit(main() or 0)
