#!/usr/bin/env python3
"""
Backfill unregistered sessions into PostgreSQL for daemon extraction.

Finds JSONL session files that have no corresponding database record,
inserts them with the JSONL UUID as session ID (so the daemon can match
the file), and lets the daemon handle extraction in batches.

Usage:
    # Dry run - show what would be inserted
    uv run python scripts/core/backfill_sessions.py --dry-run

    # Insert first batch (default 10)
    uv run python scripts/core/backfill_sessions.py

    # Insert specific batch size
    uv run python scripts/core/backfill_sessions.py --batch-size 20

    # Insert all at once (daemon queues in-memory, lost on restart)
    uv run python scripts/core/backfill_sessions.py --all

    # Only backfill sessions after a specific date
    uv run python scripts/core/backfill_sessions.py --after 2026-02-15
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import psycopg2

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

BASE_PID = 900000


def get_pg_url() -> str:
    """Resolve PostgreSQL URL from environment with fallback chain."""
    return (
        os.environ.get("CONTINUOUS_CLAUDE_DB_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql://claude:claude_dev@localhost:5432/continuous_claude"
    )


def naive_decode_path(dir_name: str) -> str:
    """Decode an encoded directory name by replacing all dashes with slashes."""
    encoded = dir_name.lstrip("-")
    return "/" + encoded.replace("-", "/") if encoded else "/"


def decode_project_path_pure(
    dir_name: str,
    dir_exists: Callable[[str], bool],
) -> str:
    """Decode a project path, using dir_exists to resolve ambiguous hyphens.

    Pure function: filesystem check is injected via dir_exists callback.
    """
    encoded = dir_name.lstrip("-")
    parts = encoded.split("-")

    def resolve(idx: int, current: str) -> str | None:
        if idx == len(parts):
            return current
        for end in range(len(parts), idx, -1):
            segment = "-".join(parts[idx:end])
            candidate = f"{current}/{segment}"
            if end == len(parts):
                if dir_exists(candidate):
                    return candidate
            elif dir_exists(candidate):
                result = resolve(end, candidate)
                if result:
                    return result
        return None

    return resolve(0, "") or naive_decode_path(dir_name)


def decode_project_path(dir_name: str) -> str:
    """Decode project path using real filesystem checks."""
    return decode_project_path_pure(dir_name, os.path.isdir)


def is_subagent_file(path_str: str) -> bool:
    """Check if a JSONL path belongs to a subagent transcript."""
    return "subagents" in path_str or Path(path_str).stem.startswith("agent-")


def is_daemon_extraction_content(first_line: str) -> bool:
    """Check if the first line indicates a daemon extraction session."""
    return bool(first_line) and "Extract learnings from session" in first_line


def filter_sessions_by_date(
    sessions: list[dict],
    after_date: str | None,
) -> list[dict]:
    """Return sessions with mtime on or after the cutoff date.

    Returns a new list; does not mutate the input.
    """
    if after_date is None:
        return list(sessions)
    cutoff = datetime.strptime(after_date, "%Y-%m-%d")
    return [s for s in sessions if s["mtime"] >= cutoff]


def sort_sessions_by_mtime(sessions: list[dict]) -> list[dict]:
    """Return sessions sorted by mtime ascending. Does not mutate input."""
    return sorted(sessions, key=lambda s: s["mtime"])


def build_session_record(session: dict, fake_pid: int) -> dict:
    """Build an insertion record from a session dict and fake PID.

    Sets exited_at to mtime so crash-recovery hooks skip these historical rows.
    """
    return {
        "id": session["uuid"],
        "project": session["project"],
        "working_on": "backfill",
        "started_at": session["mtime"],
        "last_heartbeat": session["mtime"],
        "exited_at": session["mtime"],
        "pid": fake_pid,
        "transcript_path": session.get("jsonl_path", ""),
    }


def compute_fake_pid(index: int, base: int = BASE_PID) -> int:
    """Compute a fake PID for a session at the given index."""
    return base + index


def format_dry_run_line(session: dict) -> str:
    """Format a single session for dry-run output."""
    size_kb = session["size"] / 1024
    project_name = session["project"].split("/")[-1]
    return (
        f"  {session['uuid'][:8]}...  "
        f"{session['mtime'].strftime('%Y-%m-%d %H:%M')}  "
        f"{size_kb:6.0f}KB  {project_name}"
    )


def select_batch(
    sessions: list[dict],
    batch_size: int,
    select_all: bool,
) -> list[dict]:
    """Select a batch of sessions to insert. Does not mutate input."""
    return list(sessions) if select_all else list(sessions[:batch_size])


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------


def _read_first_line(path: Path) -> str:
    """Read the first line of a file, returning empty string on error."""
    try:
        with open(path) as f:
            return f.readline(500)
    except Exception:
        return ""


def _build_session_info(
    jsonl_path: Path,
    project_path: str,
) -> dict:
    """Build a session info dict from a JSONL path and its decoded project."""
    mtime = datetime.fromtimestamp(jsonl_path.stat().st_mtime)
    return {
        "uuid": jsonl_path.stem,
        "project": project_path,
        "mtime": mtime,
        "size": jsonl_path.stat().st_size,
        "jsonl_path": str(jsonl_path),
    }


def find_unregistered_sessions(after_date: str | None = None) -> list[dict]:
    """Find JSONL files with no matching database session."""

    config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    projects_dir = config_dir / "projects"

    if not projects_dir.exists():
        print("No projects directory found")
        return []

    # Get all registered session IDs from DB
    conn = psycopg2.connect(get_pg_url())
    cur = conn.cursor()
    cur.execute("SELECT id FROM sessions")
    registered_ids = {row[0] for row in cur.fetchall()}
    conn.close()

    # Collect candidate sessions from JSONL files
    sessions = [
        _build_session_info(jsonl_path, decode_project_path(jsonl_path.parent.name))
        for jsonl_path in projects_dir.glob("*/*.jsonl")
        if not is_subagent_file(str(jsonl_path))
        and not is_daemon_extraction_content(_read_first_line(jsonl_path))
        and jsonl_path.stem not in registered_ids
    ]

    filtered = filter_sessions_by_date(sessions, after_date)
    return sort_sessions_by_mtime(filtered)


def insert_sessions(sessions: list[dict], dry_run: bool = False) -> None:
    """Insert session records into PostgreSQL."""
    if dry_run:
        print(f"\nDry run: would insert {len(sessions)} sessions:\n")
        for s in sessions:
            print(format_dry_run_line(s))
        return

    conn = psycopg2.connect(get_pg_url())
    cur = conn.cursor()

    inserted = 0
    for i, s in enumerate(sessions):
        record = build_session_record(s, compute_fake_pid(i))
        try:
            cur.execute("SAVEPOINT sp_insert")
            cur.execute(
                """
                INSERT INTO sessions
                    (id, project, working_on, started_at, last_heartbeat,
                     pid, transcript_path, exited_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    record["id"],
                    record["project"],
                    record["working_on"],
                    record["started_at"],
                    record["last_heartbeat"],
                    record["pid"],
                    record["transcript_path"],
                    record["exited_at"],
                ),
            )
            cur.execute("RELEASE SAVEPOINT sp_insert")
            inserted += 1
        except Exception as e:
            cur.execute("ROLLBACK TO SAVEPOINT sp_insert")
            print(f"  Error inserting {s['uuid']}: {e}")

    conn.commit()
    conn.close()
    print(
        f"Inserted {inserted} sessions. Daemon will process them over the next "
        f"~{inserted // 4 + 1} minutes (4 concurrent extractions)."
    )


def _bootstrap() -> None:
    """Initialize faulthandler and load .env files. Called only from main()."""
    from dotenv import load_dotenv

    log_path = os.path.expanduser("~/.claude/logs/opc_crash.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    faulthandler.enable(
        file=open(log_path, "a"),  # noqa: SIM115
        all_threads=True,
    )
    global_env = Path.home() / ".claude" / ".env"
    if global_env.exists():
        load_dotenv(global_env)
    opc_env = Path(__file__).parent.parent.parent / ".env"
    if opc_env.exists():
        load_dotenv(opc_env, override=True)


def main() -> int:
    """CLI entry point for backfill_sessions."""
    _bootstrap()
    parser = argparse.ArgumentParser(description="Backfill unregistered sessions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be inserted")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Sessions to insert (default: 10)"
    )
    parser.add_argument("--all", action="store_true", help="Insert all unregistered sessions")
    parser.add_argument(
        "--after", type=str, default=None, help="Only after this date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    sessions = find_unregistered_sessions(after_date=args.after)

    if not sessions:
        print("No unregistered sessions found.")
        return 0

    print(f"Found {len(sessions)} unregistered sessions")

    if args.dry_run:
        insert_sessions(sessions, dry_run=True)
        return 0

    batch = select_batch(sessions, batch_size=args.batch_size, select_all=args.all)
    print(f"Inserting batch of {len(batch)} (of {len(sessions)} total)")
    insert_sessions(batch)

    remaining = len(sessions) - len(batch)
    if remaining > 0:
        print(f"\n{remaining} sessions remaining. Run again to insert the next batch.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
