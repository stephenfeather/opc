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
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

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

# Global config
POLL_INTERVAL = 60  # seconds
STALE_THRESHOLD = 900  # 15 minutes in seconds
MAX_CONCURRENT_EXTRACTIONS = 4
MAX_RETRIES = 5
PID_FILE = Path.home() / ".claude" / "memory-daemon.pid"
LOG_FILE = Path.home() / ".claude" / "memory-daemon.log"

# Worker queue state (module-level for daemon process)
active_extractions: dict = {}  # pid -> (session_id, proc, jsonl_path, project)
pending_queue: list[tuple[str, str, str | None]] = []  # [(session_id, project, transcript_path), ...]


def log(msg: str):
    """Write timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line)
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
    """Get sessions with stale heartbeat that need extraction."""
    conn = pg_connect()
    cur = conn.cursor()
    # Use DB clock for comparison to avoid local-vs-UTC timezone mismatch
    cur.execute("""
        SELECT id, project, transcript_path FROM sessions
        WHERE last_heartbeat < NOW() - INTERVAL '%s seconds'
        AND extraction_status = 'pending'
        AND extraction_attempts < %s
    """, (STALE_THRESHOLD, MAX_RETRIES))
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
    """Get sessions with stale heartbeat that need extraction."""
    db_path = get_sqlite_path()
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    threshold = (datetime.now() - timedelta(seconds=STALE_THRESHOLD)).isoformat()
    cursor = conn.execute("""
        SELECT id, project, transcript_path FROM sessions
        WHERE last_heartbeat < ?
        AND extraction_status = 'pending'
        AND COALESCE(extraction_attempts, 0) < ?
    """, (threshold, MAX_RETRIES))
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


# Unified interface
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


def extract_memories(
    session_id: str,
    project_dir: str,
    transcript_path: str | None = None,
) -> bool:
    """Run memory extraction for a session. Returns True if subprocess started."""
    log(f"Extracting memories for session {session_id} "
        f"(project={project_dir or 'unknown'})")

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

        env = os.environ.copy()
        env["CLAUDE_MEMORY_EXTRACTION"] = "1"  # Prevent session-register from registering extraction sessions

        proc = subprocess.Popen(
            [
                "claude", "-p",
                "--model", "sonnet",
                "--dangerously-skip-permissions",
                "--max-turns", "15",
                "--append-system-prompt", agent_prompt,
                f"Extract learnings from session {session_id}. JSONL path: {jsonl_path}"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        active_extractions[proc.pid] = (session_id, proc, jsonl_path, project_dir)
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


def _extract_and_store_workflows(
    session_id: str,
    jsonl_path: Path,
    project: str | None,
):
    """Extract workflow patterns and store as learnings. Non-fatal."""
    try:
        from scripts.core.extract_workflow_patterns import (
            extract_tool_uses,
            detect_workflow_sequences,
            format_pattern_as_learning,
        )
    except ImportError as e:
        log(f"Workflow extraction unavailable: {e}")
        return

    try:
        tool_uses = extract_tool_uses(jsonl_path)
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
                result = asyncio.run(store_learning_v2(
                    session_id=session_id,
                    content=content,
                    learning_type="WORKING_SOLUTION",
                    context=project or "unknown",
                    tags=["workflow", pattern["pattern_type"]],
                    confidence="high",
                ))
                if result.get("success") and not result.get("skipped"):
                    stored += 1
            except Exception as e:
                log(f"Failed to store workflow learning: {e}")

        log(f"Stored {stored} workflow patterns for {session_id}")
    except Exception as e:
        log(f"Workflow extraction failed for {session_id}: {e}")


def reap_completed_extractions():
    """Check for completed extraction processes and remove from active set."""
    completed = []
    for pid, (session_id, proc, jsonl_path, project) in list(active_extractions.items()):
        exit_code = proc.poll()
        if exit_code is not None:
            completed.append(pid)
            log(f"Extraction completed for {session_id} "
                f"(pid={pid}, project={project}, exit={exit_code})")
            if exit_code == 0:
                mark_extracted(session_id)
                _extract_and_store_workflows(session_id, jsonl_path, project)
                archive_session_jsonl(session_id, jsonl_path)
            else:
                mark_extraction_failed(session_id)

    for pid in completed:
        del active_extractions[pid]

    return len(completed)


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


def daemon_loop():
    """Main daemon loop."""
    db_type = "PostgreSQL" if use_postgres() else "SQLite"
    log(f"Memory daemon started (using {db_type}, max_concurrent={MAX_CONCURRENT_EXTRACTIONS})")
    ensure_schema()

    while True:
        try:
            # Reap completed processes and process pending queue
            reap_completed_extractions()
            process_pending_queue()

            # Find new stale sessions
            stale = get_stale_sessions()
            if stale:
                summary = ", ".join(
                    f"{sid}({proj or '?'})"
                    for sid, proj, *_ in stale
                )
                log(f"Found {len(stale)} stale sessions: {summary}")
                for row in stale:
                    session_id, project = row[0], row[1]
                    tp = row[2] if len(row) > 2 else None
                    mark_extracting(session_id)
                    queue_or_extract(session_id, project or "", tp)
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

    print("Memory Daemon Status")
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
