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

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import faulthandler
faulthandler.enable(file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"), all_threads=True)

# Load env
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
opc_env = Path(__file__).parent.parent.parent / ".env"
if opc_env.exists():
    load_dotenv(opc_env, override=True)


def get_pg_url():
    return (
        os.environ.get("CONTINUOUS_CLAUDE_DB_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql://claude:claude_dev@localhost:5432/continuous_claude"
    )


def decode_project_path(dir_name: str) -> str:
    """Convert -Users-stephenfeather-Dev-foo to /Users/stephenfeather/Dev/foo.

    Claude Code encodes paths by replacing / with -, but directory names
    can also contain hyphens. We resolve ambiguity by checking which
    candidate path actually exists on the filesystem.
    """
    # Strip leading dash
    encoded = dir_name.lstrip("-")
    parts = encoded.split("-")

    # Try all possible splits, preferring longer path segments
    # Use DFS to find a valid filesystem path
    def resolve(idx: int, current: str) -> str | None:
        if idx == len(parts):
            return current
        # Try consuming multiple parts joined by '-' (greedy, longest first)
        for end in range(len(parts), idx, -1):
            segment = "-".join(parts[idx:end])
            candidate = f"{current}/{segment}"
            # For intermediate segments, check dir exists
            # For final segment, accept it (might be the project dir)
            if end == len(parts):
                if os.path.isdir(candidate):
                    return candidate
            elif os.path.isdir(candidate):
                result = resolve(end, candidate)
                if result:
                    return result
        return None

    result = resolve(0, "")
    if result:
        return result

    # Fallback: naive replacement (may be wrong for hyphenated dirs)
    return "/" + encoded.replace("-", "/")


def find_unregistered_sessions(after_date: str | None = None) -> list[dict]:
    """Find JSONL files with no matching database session."""
    import psycopg2

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

    # Find all JSONL files (skip subagent transcripts)
    sessions = []
    for jsonl_path in projects_dir.glob("*/*.jsonl"):
        if "subagents" in str(jsonl_path):
            continue

        uuid = jsonl_path.stem

        # Skip subagent transcripts (agent-aXXX pattern)
        if uuid.startswith("agent-"):
            continue

        # Skip daemon extraction sessions (first line contains
        # "Extract learnings from session")
        try:
            with open(jsonl_path) as f:
                first_line = f.readline(500)
            if "Extract learnings from session" in first_line:
                continue
        except Exception:
            pass

        mtime = datetime.fromtimestamp(jsonl_path.stat().st_mtime)

        # Filter by date if specified
        if after_date:
            cutoff = datetime.strptime(after_date, "%Y-%m-%d")
            if mtime < cutoff:
                continue

        # Check if already registered (by UUID or any ID containing this UUID)
        if uuid in registered_ids:
            continue

        # Decode project path from parent directory name
        project_dir = jsonl_path.parent.name
        project_path = decode_project_path(project_dir)

        # Get file size for sorting (larger = more content = more learnings)
        size = jsonl_path.stat().st_size

        sessions.append({
            "uuid": uuid,
            "project": project_path,
            "mtime": mtime,
            "size": size,
            "jsonl_path": str(jsonl_path),
        })

    # Sort by mtime ascending (oldest first)
    sessions.sort(key=lambda s: s["mtime"])
    return sessions


def insert_sessions(sessions: list[dict], dry_run: bool = False):
    """Insert session records into PostgreSQL."""
    if dry_run:
        print(f"\nDry run: would insert {len(sessions)} sessions:\n")
        for s in sessions:
            size_kb = s["size"] / 1024
            print(f"  {s['uuid'][:8]}...  {s['mtime'].strftime('%Y-%m-%d %H:%M')}  "
                  f"{size_kb:6.0f}KB  {s['project'].split('/')[-1]}")
        return

    import psycopg2

    conn = psycopg2.connect(get_pg_url())
    cur = conn.cursor()

    # Use PID range 900000+ to avoid collision with real PIDs
    base_pid = 900000

    inserted = 0
    for i, s in enumerate(sessions):
        fake_pid = base_pid + i
        # Set last_heartbeat to file mtime (in the past) so daemon sees it as stale
        try:
            cur.execute("""
                INSERT INTO sessions (id, project, working_on, started_at, last_heartbeat, pid)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                s["uuid"],
                s["project"],
                "backfill",
                s["mtime"],
                s["mtime"],  # Already in the past = immediately stale
                fake_pid,
            ))
            inserted += 1
        except Exception as e:
            print(f"  Error inserting {s['uuid']}: {e}")

    conn.commit()
    conn.close()
    print(f"Inserted {inserted} sessions. Daemon will process them over the next "
          f"~{inserted // 4 + 1} minutes (4 concurrent extractions).")


def main():
    parser = argparse.ArgumentParser(description="Backfill unregistered sessions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be inserted")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Sessions to insert (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Insert all unregistered sessions")
    parser.add_argument("--after", type=str, default=None,
                        help="Only after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    sessions = find_unregistered_sessions(after_date=args.after)

    if not sessions:
        print("No unregistered sessions found.")
        return 0

    print(f"Found {len(sessions)} unregistered sessions")

    if args.dry_run:
        insert_sessions(sessions, dry_run=True)
        return 0

    batch = sessions if args.all else sessions[:args.batch_size]
    print(f"Inserting batch of {len(batch)} "
          f"(of {len(sessions)} total)")
    insert_sessions(batch)

    remaining = len(sessions) - len(batch)
    if remaining > 0:
        print(f"\n{remaining} sessions remaining. "
              "Run again to insert the next batch.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
