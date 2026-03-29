#!/usr/bin/env python3
"""
One-time backfill: Archive existing JSONL files to S3.

Finds all .jsonl files in ~/.claude/projects/ and archives them to S3
using the same compress+upload logic as the memory daemon.

Usage:
    uv run python scripts/core/backfill_archive.py --dry-run   # preview
    uv run python scripts/core/backfill_archive.py              # execute
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

import faulthandler
faulthandler.enable(file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"), all_threads=True)

# Load env (same as daemon)
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
opc_env = Path(__file__).parent.parent.parent / ".env"
if opc_env.exists():
    load_dotenv(opc_env, override=True)


def get_postgres_url():
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")


def mark_archived_in_db(session_id: str, archive_path: str):
    """Mark session as archived in PostgreSQL."""
    url = get_postgres_url()
    if not url:
        return
    try:
        import psycopg2
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        # Try UUID match first, then check if any s- session used this JSONL
        cur.execute(
            "UPDATE sessions SET archived_at = NOW(), archive_path = %s "
            "WHERE id = %s AND archived_at IS NULL",
            (archive_path, session_id),
        )
        updated = cur.rowcount
        # Also stamp learnings
        if updated > 0:
            cur.execute(
                "UPDATE archival_memory SET metadata = metadata || "
                "jsonb_build_object('archive_path', %s) "
                "WHERE session_id = %s AND (metadata->>'archive_path') IS NULL",
                (archive_path, session_id),
            )
        conn.commit()
        conn.close()
        return updated
    except Exception as e:
        print(f"  DB error: {e}")
        return 0


def archive_jsonl(jsonl_path: Path, bucket: str, dry_run: bool = False):
    """Compress and upload a single JSONL to S3."""
    project_name = jsonl_path.parent.name
    session_id = jsonl_path.stem
    s3_key = f"s3://{bucket}/sessions/{project_name}/{session_id}.jsonl.zst"
    zst_path = jsonl_path.with_suffix(".jsonl.zst")

    if dry_run:
        size_mb = jsonl_path.stat().st_size / (1024 * 1024)
        print(f"  [DRY RUN] {jsonl_path.name} ({size_mb:.1f}MB) -> {s3_key}")
        return True

    try:
        # Compress
        result = subprocess.run(
            ["zstd", "-q", "--rm", str(jsonl_path)],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  zstd failed: {result.stderr.decode()}")
            return False

        # Upload
        result = subprocess.run(
            ["aws", "s3", "cp", str(zst_path), s3_key, "--quiet"],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  S3 upload failed: {result.stderr.decode()}")
            # Restore original
            subprocess.run(
                ["zstd", "-d", "-q", "--rm", str(zst_path)],
                capture_output=True, timeout=300,
            )
            return False

        # Clean up local compressed file
        zst_path.unlink(missing_ok=True)

        # Mark in DB
        mark_archived_in_db(session_id, s3_key)

        print(f"  Archived {session_id} -> {s3_key}")
        return True

    except subprocess.TimeoutExpired:
        print(f"  Timeout for {session_id}")
        if zst_path.exists() and not jsonl_path.exists():
            subprocess.run(
                ["zstd", "-d", "-q", "--rm", str(zst_path)],
                capture_output=True, timeout=300,
            )
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    dry_run = "--dry-run" in sys.argv

    bucket = os.environ.get("CLAUDE_SESSION_ARCHIVE_BUCKET")
    if not bucket:
        print("CLAUDE_SESSION_ARCHIVE_BUCKET not set")
        sys.exit(1)

    config_dir = Path(
        os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))
    )
    jsonl_dir = config_dir / "projects"

    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(minutes=10)
    all_jsonls = sorted(jsonl_dir.glob("*/*.jsonl"), key=lambda x: x.stat().st_mtime)
    jsonls = [
        f for f in all_jsonls
        if datetime.fromtimestamp(f.stat().st_mtime) < cutoff
    ]
    skipped = len(all_jsonls) - len(jsonls)
    print(f"Found {len(all_jsonls)} JSONL files, skipping {skipped} modified in last 10 min")
    print(f"Archiving {len(jsonls)} files")

    if dry_run:
        total_mb = sum(f.stat().st_size for f in jsonls) / (1024 * 1024)
        print(f"Total size: {total_mb:.1f}MB")
        print()

    success = 0
    failed = 0
    for jsonl_path in jsonls:
        if archive_jsonl(jsonl_path, bucket, dry_run):
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} archived, {failed} failed")


if __name__ == "__main__":
    main()
