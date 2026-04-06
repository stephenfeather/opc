#!/usr/bin/env python3
"""
USAGE: artifact_mark.py --handoff ID --outcome OUTCOME [--notes NOTES]
       artifact_mark.py --latest --outcome OUTCOME [--notes NOTES]
       artifact_mark.py --get-latest-id

Mark a handoff with user outcome in the database (PostgreSQL or SQLite).

Supports PostgreSQL (via DATABASE_URL or CONTINUOUS_CLAUDE_DB_URL) with
automatic fallback to SQLite for installations without PostgreSQL.

Examples:
    # Mark a specific handoff as succeeded
    uv run python scripts/core/artifact_mark.py --handoff abc123 --outcome SUCCEEDED

    # Mark the most recent handoff
    uv run python scripts/core/artifact_mark.py --latest --outcome SUCCEEDED

    # Just get the latest handoff ID (for scripts)
    uv run python scripts/core/artifact_mark.py --get-latest-id

    # Mark with additional notes
    uv run python scripts/core/artifact_mark.py --latest --outcome PARTIAL_PLUS \\
        --notes "Almost done"
"""

import argparse
import faulthandler
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

_crash_log_dir = Path.home() / ".claude" / "logs"
_crash_log_dir.mkdir(parents=True, exist_ok=True)
_crash_log_file = open(_crash_log_dir / "opc_crash.log", "a")  # noqa: SIM115
faulthandler.enable(file=_crash_log_file, all_threads=True)

# Load .env files for DATABASE_URL (cross-platform)
# 1. Global ~/.claude/.env
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)

# 2. Local opc/.env (relative to script location)
opc_env = Path(__file__).parent.parent.parent / ".env"
if opc_env.exists():
    load_dotenv(opc_env, override=True)

VALID_OUTCOMES = ("SUCCEEDED", "PARTIAL_PLUS", "PARTIAL_MINUS", "FAILED")


# --- Pure functions ---


def truncate_summary(text: str | None, max_len: int) -> str:
    """Truncate text with ellipsis, or return placeholder for empty/None."""
    if not text:
        return "(no summary)"
    if len(text) <= max_len:
        return text
    if max_len < 4:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def format_handoff_row(row: tuple, max_summary_len: int = 50) -> str:
    """Format a single handoff row for display: '  ID: session - summary'."""
    return f"  {row[0][:12]}: {row[1]} - {truncate_summary(row[2], max_summary_len)}"


def format_recent_list(rows: list, max_summary_len: int = 50) -> str:
    """Format a list of handoff rows for display."""
    if not rows:
        return ""
    return "\n".join(format_handoff_row(row, max_summary_len) for row in rows)


def resolve_handoff_id(
    handoff_id: str | None, use_latest: bool, latest_id: str | None
) -> tuple[str | None, str | None]:
    """Determine which handoff ID to use. Returns (id, error_message)."""
    if handoff_id:
        return (handoff_id, None)
    if use_latest:
        if latest_id:
            return (latest_id, None)
        return (None, "Error: No handoffs found in database")
    return (None, "Either --handoff ID or --latest is required")


def format_confirmation(
    handoff: tuple, outcome: str, notes: str, db_type: str
) -> str:
    """Format the success confirmation message."""
    lines = [
        f"\u2713 Marked handoff as {outcome}",
        f"  Database: {db_type}",
        f"  ID: {handoff[0]}",
        f"  Session: {handoff[1]}",
    ]
    if handoff[2]:
        lines.append(f"  Summary: {truncate_summary(handoff[2], 80)}")
    if notes:
        lines.append(f"  Notes: {notes}")
    return "\n".join(lines)


def format_error_not_found(handoff_id: str, db_type: str, recent_rows: list) -> str:
    """Format the 'handoff not found' error message."""
    lines = [
        f"Error: Handoff not found: {handoff_id}",
        f"Database: {db_type}",
        "Available handoffs:",
    ]
    if recent_rows:
        lines.append(format_recent_list(recent_rows))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser (pure -- no side effects)."""
    parser = argparse.ArgumentParser(
        description="Mark handoff outcome (PostgreSQL or SQLite)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mark specific handoff
  %(prog)s --handoff abc123 --outcome SUCCEEDED

  # Mark latest handoff
  %(prog)s --latest --outcome SUCCEEDED

  # Get latest handoff ID (for scripts)
  %(prog)s --get-latest-id
""",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--handoff", help="Handoff ID to mark")
    mode_group.add_argument(
        "--latest", action="store_true", help="Mark the most recent handoff"
    )
    mode_group.add_argument(
        "--get-latest-id",
        action="store_true",
        help="Print latest handoff ID and exit",
    )
    parser.add_argument(
        "--outcome",
        choices=list(VALID_OUTCOMES),
        help="Outcome of the handoff",
    )
    parser.add_argument(
        "--notes", default="", help="Optional notes about the outcome"
    )
    return parser


# --- I/O: Configuration ---


def get_postgres_url() -> str | None:
    """Get PostgreSQL URL from environment if available (canonical first)."""
    return os.environ.get("CONTINUOUS_CLAUDE_DB_URL") or os.environ.get("DATABASE_URL")


def get_sqlite_path() -> Path:
    """Get SQLite database path."""
    return Path(".claude/cache/artifact-index/context.db")


def use_postgres() -> bool:
    """Check if PostgreSQL should be used."""
    url = get_postgres_url()
    if not url:
        return False
    try:
        import psycopg2  # noqa: F401

        return True
    except ImportError:
        return False


# --- I/O: PostgreSQL operations ---


def pg_connect():
    """Connect to PostgreSQL."""
    import psycopg2

    return psycopg2.connect(get_postgres_url())


def pg_get_latest_id() -> str | None:
    """Get latest handoff ID from PostgreSQL."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute("SELECT id::text FROM handoffs ORDER BY indexed_at DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def pg_get_handoff(handoff_id: str) -> tuple | None:
    """Get handoff by ID from PostgreSQL. Exact match first, then prefix."""
    conn = pg_connect()
    cur = conn.cursor()
    # Try exact match first
    cur.execute(
        "SELECT id::text, session_name, goal FROM handoffs WHERE id::text = %s",
        (handoff_id,),
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return row
    # Fall back to prefix match, reject ambiguous
    cur.execute(
        "SELECT id::text, session_name, goal FROM handoffs WHERE id::text LIKE %s",
        (f"{handoff_id}%",),
    )
    rows = cur.fetchall()
    conn.close()
    if len(rows) == 1:
        return rows[0]
    return None


def pg_update_outcome(handoff_id: str, outcome: str, notes: str) -> bool:
    """Update handoff outcome in PostgreSQL (exact ID match only)."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute(
        "UPDATE handoffs SET outcome = %s, outcome_notes = %s WHERE id::text = %s",
        (outcome, notes, handoff_id),
    )
    updated = cur.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def pg_list_recent() -> list:
    """List recent handoffs from PostgreSQL."""
    conn = pg_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id::text, session_name, goal FROM handoffs"
        " ORDER BY indexed_at DESC LIMIT 10"
    )
    rows = cur.fetchall()
    conn.close()
    return rows


# --- I/O: SQLite operations ---


def sqlite_connect():
    """Connect to SQLite."""
    db_path = get_sqlite_path()
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def sqlite_get_latest_id() -> str | None:
    """Get latest handoff ID from SQLite."""
    conn = sqlite_connect()
    if not conn:
        return None
    cursor = conn.execute(
        "SELECT id FROM handoffs ORDER BY indexed_at DESC LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def sqlite_get_handoff(handoff_id: str) -> tuple | None:
    """Get handoff by ID from SQLite. Exact match first, then prefix."""
    conn = sqlite_connect()
    if not conn:
        return None
    # Try exact match first
    cursor = conn.execute(
        "SELECT id, session_name, task_summary FROM handoffs WHERE id = ?",
        (handoff_id,),
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return row
    # Fall back to prefix match, reject ambiguous
    cursor = conn.execute(
        "SELECT id, session_name, task_summary FROM handoffs WHERE id LIKE ?",
        (f"{handoff_id}%",),
    )
    rows = cursor.fetchall()
    conn.close()
    if len(rows) == 1:
        return rows[0]
    return None


def sqlite_update_outcome(handoff_id: str, outcome: str, notes: str) -> bool:
    """Update handoff outcome in SQLite (exact ID match only)."""
    conn = sqlite_connect()
    if not conn:
        return False
    cursor = conn.execute(
        "UPDATE handoffs SET outcome = ?, outcome_notes = ?, confidence = 'HIGH'"
        " WHERE id = ?",
        (outcome, notes, handoff_id),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def sqlite_list_recent() -> list:
    """List recent handoffs from SQLite."""
    conn = sqlite_connect()
    if not conn:
        return []
    cursor = conn.execute(
        "SELECT id, session_name, task_summary FROM handoffs"
        " ORDER BY indexed_at DESC LIMIT 10"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


# --- I/O: Unified interface ---


def get_latest_id() -> str | None:
    """Get latest handoff ID from database."""
    if use_postgres():
        return pg_get_latest_id()
    return sqlite_get_latest_id()


def get_handoff(handoff_id: str) -> tuple | None:
    """Get handoff by ID."""
    if use_postgres():
        return pg_get_handoff(handoff_id)
    return sqlite_get_handoff(handoff_id)


def update_outcome(handoff_id: str, outcome: str, notes: str) -> bool:
    """Update handoff outcome."""
    if use_postgres():
        return pg_update_outcome(handoff_id, outcome, notes)
    return sqlite_update_outcome(handoff_id, outcome, notes)


def list_recent() -> list:
    """List recent handoffs."""
    if use_postgres():
        return pg_list_recent()
    return sqlite_list_recent()


# --- I/O: CLI entrypoint ---


def main():
    """CLI entrypoint — thin I/O shell over pure functions."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Mode: get-latest-id
    if args.get_latest_id:
        latest = get_latest_id()
        if latest:
            print(latest)
            return 0
        print("No handoffs found", file=sys.stderr)
        return 1

    # Mode: mark handoff
    if not args.outcome:
        parser.error("--outcome is required unless using --get-latest-id")

    # Resolve handoff ID (lazy: only query latest when --latest is set)
    latest_id = get_latest_id() if args.latest else None
    handoff_id, error = resolve_handoff_id(
        args.handoff, args.latest, latest_id
    )
    if error:
        print(error, file=sys.stderr)
        return 1

    db_type = "PostgreSQL" if use_postgres() else "SQLite"

    handoff = get_handoff(handoff_id)
    if not handoff:
        print(format_error_not_found(handoff_id, db_type, list_recent()), file=sys.stderr)
        return 1

    # Use the full resolved ID from SELECT, not the user-supplied prefix
    resolved_id = handoff[0]
    if not update_outcome(resolved_id, args.outcome, args.notes):
        print(f"Error: Failed to update handoff: {resolved_id}", file=sys.stderr)
        return 1

    print(format_confirmation(handoff, args.outcome, args.notes, db_type))
    return 0


if __name__ == "__main__":
    exit(main())
