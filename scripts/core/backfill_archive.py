#!/usr/bin/env python3
"""
One-time backfill: Archive existing JSONL files to S3.

Finds all .jsonl files in ~/.claude/projects/ and archives them to S3
using the same compress+upload logic as the memory daemon.

Usage:
    uv run python scripts/core/backfill_archive.py --dry-run   # preview
    uv run python scripts/core/backfill_archive.py              # execute
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2

# Add repository root to path
_repo_root = str(Path(__file__).parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.core.config.models import ArchivalConfig  # noqa: E402
from scripts.core.log_safety import safe  # noqa: E402

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def get_pg_url() -> str | None:
    """Resolve PostgreSQL URL from environment."""
    return (
        os.environ.get("CONTINUOUS_CLAUDE_DB_URL")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("POSTGRES_URL")
    )


def build_s3_key(bucket: str, project_name: str, session_id: str) -> str:
    """Build the full S3 key for an archived session."""
    return f"s3://{bucket}/sessions/{project_name}/{session_id}.jsonl.zst"


def build_zst_path(jsonl_path: Path) -> Path:
    """Compute the .jsonl.zst path for a given JSONL file."""
    return jsonl_path.with_suffix(".jsonl.zst")


def format_dry_run_info(jsonl_path: Path, size_bytes: int) -> dict:
    """Build a dry-run info dict from a JSONL path and its size."""
    return {
        "path": str(jsonl_path),
        "name": jsonl_path.name,
        "size_mb": size_bytes / (1024 * 1024),
        "project": jsonl_path.parent.name,
        "session_id": jsonl_path.stem,
    }


def filter_recent_jsonls(files: list[dict], cutoff: datetime) -> list[dict]:
    """Return files with mtime before cutoff. Does not mutate input."""
    return [f for f in files if f["mtime"] < cutoff]


def build_archive_summary(
    total: int, skipped: int, success: int, failed: int
) -> dict:
    """Build a summary dict of the archive run."""
    return {"total": total, "skipped": skipped, "archived": success, "failed": failed}


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------


def compress_file(jsonl_path: Path, timeout: int) -> subprocess.CompletedProcess:
    """Compress a JSONL file with zstd."""
    return subprocess.run(
        ["zstd", "-q", "--rm", str(jsonl_path)],
        capture_output=True,
        timeout=timeout,
    )


def upload_to_s3(zst_path: Path, s3_key: str, timeout: int) -> subprocess.CompletedProcess:
    """Upload a .zst file to S3."""
    return subprocess.run(
        ["aws", "s3", "cp", str(zst_path), s3_key, "--quiet"],
        capture_output=True,
        timeout=timeout,
    )


def decompress_file(zst_path: Path, timeout: int) -> subprocess.CompletedProcess:
    """Decompress a .zst file back to JSONL (restore on failure)."""
    return subprocess.run(
        ["zstd", "-d", "-q", "--rm", str(zst_path)],
        capture_output=True,
        timeout=timeout,
    )


def mark_archived_in_db(
    session_id: str, archive_path: str, url: str | None
) -> int | None:
    """Mark session as archived in PostgreSQL. Returns rows updated."""
    if not url:
        return None
    conn = None
    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        cur.execute(
            "UPDATE sessions SET archived_at = NOW(), archive_path = %s "
            "WHERE id = %s AND archived_at IS NULL",
            (archive_path, session_id),
        )
        updated = cur.rowcount
        if updated > 0:
            cur.execute(
                "UPDATE archival_memory SET metadata = COALESCE(metadata, '{}'::jsonb) || "
                "jsonb_build_object('archive_path', %s) "
                "WHERE session_id = %s AND (metadata->>'archive_path') IS NULL",
                (archive_path, session_id),
            )
        conn.commit()
        return updated
    except Exception as e:
        print(f"  DB error: {type(e).__name__}")
        return None
    finally:
        if conn:
            conn.close()


def find_archivable_jsonls(config_dir: Path) -> list[dict]:
    """Discover JSONL files in projects dir, returning dicts with path + mtime + size."""
    projects_dir = config_dir / "projects"
    if not projects_dir.exists():
        return []
    resolved_root = projects_dir.resolve()
    archivable = []
    for f in projects_dir.glob("*/*.jsonl"):
        if f.is_symlink():
            continue
        try:
            f.resolve().relative_to(resolved_root)
        except ValueError:
            continue
        try:
            st = f.stat()
        except FileNotFoundError:
            continue
        archivable.append(
            {"path": f, "mtime": datetime.fromtimestamp(st.st_mtime), "size": st.st_size}
        )
    return sorted(archivable, key=lambda x: x["mtime"])


def _safe_restore(zst_path: Path, timeout: int) -> bool:
    """Restore .jsonl from .zst, warning if restore fails."""
    try:
        result = decompress_file(zst_path, timeout)
        if result.returncode != 0:
            print(f"  WARNING: Failed to restore {safe(zst_path)} — file may be orphaned")
            return False
        return True
    except Exception as e:
        print(f"  WARNING: Restore error for {safe(zst_path)}: {safe(e)}")
        return False


def archive_jsonl(
    jsonl_path: Path,
    bucket: str,
    cfg: ArchivalConfig,
    db_url: str | None,
    dry_run: bool = False,
) -> bool:
    """Thin orchestrator: compress, upload, mark in DB."""
    project_name = jsonl_path.parent.name
    session_id = jsonl_path.stem
    s3_key = build_s3_key(bucket, project_name, session_id)
    zst_path = build_zst_path(jsonl_path)

    if dry_run:
        info = format_dry_run_info(jsonl_path, jsonl_path.stat().st_size)
        print(f"  [DRY RUN] {info['name']} ({info['size_mb']:.1f}MB) -> {s3_key}")
        return True

    try:
        result = compress_file(jsonl_path, cfg.compress_timeout)
        if result.returncode != 0:
            print(f"  zstd failed: {safe(result.stderr.decode(errors='replace'))}")
            return False

        result = upload_to_s3(zst_path, s3_key, cfg.upload_timeout)
        if result.returncode != 0:
            print(f"  S3 upload failed: {safe(result.stderr.decode(errors='replace'))}")
            _safe_restore(zst_path, cfg.compress_timeout)
            return False

        zst_path.unlink(missing_ok=True)

        # DB mark is best-effort after successful S3 upload (matches daemon behavior)
        if db_url:
            db_result = mark_archived_in_db(session_id, s3_key, db_url)
            if db_result is None:
                print(f"  Warning: DB error for {safe(session_id)} (S3 upload succeeded)")
            elif db_result == 0:
                print(f"  Note: No DB row updated for {safe(session_id)} (not tracked or archived)")

        print(f"  Archived {safe(session_id)} -> {s3_key}")
        return True

    except subprocess.TimeoutExpired:
        print(f"  Timeout for {safe(session_id)}")
        if zst_path.exists() and not jsonl_path.exists():
            _safe_restore(zst_path, cfg.compress_timeout)
        return False
    except Exception as e:
        print(f"  Error: {safe(e)}")
        return False


# ---------------------------------------------------------------------------
# Bootstrap and main
# ---------------------------------------------------------------------------

_faulthandler_log_file = None


def _bootstrap() -> None:
    """Initialize faulthandler and load .env files. Called only from main()."""
    global _faulthandler_log_file
    from dotenv import load_dotenv

    log_path = os.path.expanduser("~/.claude/logs/opc_crash.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _faulthandler_log_file = open(log_path, "a")  # noqa: SIM115
    faulthandler.enable(file=_faulthandler_log_file, all_threads=True)

    global_env = Path.home() / ".claude" / ".env"
    if global_env.exists():
        load_dotenv(global_env)
    opc_env = Path(__file__).parent.parent.parent / ".env"
    if opc_env.exists():
        load_dotenv(opc_env, override=True)


def main() -> int:
    """CLI entry point for backfill_archive."""
    parser = argparse.ArgumentParser(description="Archive JSONL files to S3")
    parser.add_argument("--dry-run", action="store_true", help="Preview without archiving")
    args = parser.parse_args()

    _bootstrap()

    from scripts.core.config import get_config

    cfg = get_config().archival

    bucket = os.environ.get("CLAUDE_SESSION_ARCHIVE_BUCKET")
    if not bucket:
        print("CLAUDE_SESSION_ARCHIVE_BUCKET not set")
        return 1

    config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    db_url = get_pg_url()

    all_files = find_archivable_jsonls(config_dir)
    cutoff = datetime.now() - timedelta(minutes=cfg.skip_recent_minutes)
    files = filter_recent_jsonls(all_files, cutoff)
    skipped = len(all_files) - len(files)

    print(
        f"Found {len(all_files)} JSONL files, "
        f"skipping {skipped} modified in last {cfg.skip_recent_minutes} min"
    )
    print(f"Archiving {len(files)} files")

    if args.dry_run:
        total_mb = sum(f["size"] for f in files) / (1024 * 1024)
        print(f"Total size: {total_mb:.1f}MB\n")

    success = 0
    failed = 0
    for f in files:
        if archive_jsonl(f["path"], bucket, cfg, db_url, dry_run=args.dry_run):
            success += 1
        else:
            failed += 1

    summary = build_archive_summary(len(all_files), skipped, success, failed)
    print(f"\nDone: {summary['archived']} archived, {summary['failed']} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
