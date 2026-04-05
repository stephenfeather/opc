"""Backfill learnings from S3-archived session JSONL files.

For each archived session:
1. List S3-archived JSONLs and filter by project
2. Look up the s-* session ID in PostgreSQL via transcript_path UUID match
3. Skip sessions already in backfill_log
4. Download + decompress JSONL from S3
5. Launch the memory-extractor agent (claude -p) to extract and store learnings
6. Log result to backfill_log table

Usage:
    uv run python scripts/core/backfill_learnings.py --dry-run       # preview
    uv run python scripts/core/backfill_learnings.py --limit 10      # first 10
    uv run python scripts/core/backfill_learnings.py --project opc   # filter by project
    uv run python scripts/core/backfill_learnings.py --workers 3     # parallel
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def get_pg_url() -> str | None:
    """Resolve PostgreSQL URL from environment with fallback chain."""
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or None


def get_s3_bucket() -> str | None:
    """Get S3 archive bucket from environment."""
    return os.environ.get("CLAUDE_SESSION_ARCHIVE_BUCKET") or None


def parse_s3_listing(
    stdout: str, bucket: str, project_filter: str | None
) -> list[dict]:
    """Parse ``aws s3 ls --recursive`` output into session dicts.

    Returns a new list of ``{s3_key, uuid, project}`` dicts.
    The ``s3_key`` is a full ``s3://bucket/key`` URI.
    """
    if not stdout.strip():
        return []

    sessions: list[dict] = []
    for line in stdout.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        key = parts[3]
        if not key.endswith(".jsonl.zst"):
            continue

        filename = key.split("/")[-1]
        uuid = filename.replace(".jsonl.zst", "")
        path_parts = key.split("/")
        project = path_parts[1] if len(path_parts) >= 3 else "unknown"

        if project_filter and project_filter not in project:
            continue

        sessions.append({
            "s3_key": f"s3://{bucket}/{key}",
            "uuid": uuid,
            "project": project,
        })

    return sessions


def build_extraction_cmd(
    jsonl_path: Path,
    session_id: str,
    agent_prompt: str,
    model: str,
    max_turns: int,
) -> list[str]:
    """Build the ``claude -p`` command argv. Pure — no subprocess call."""
    return [
        "claude",
        "-p",
        "--model",
        model,
        "--dangerously-skip-permissions",
        "--allowedTools",
        "Bash,Read",
        "--max-turns",
        str(max_turns),
        "--append-system-prompt",
        agent_prompt,
        f"Extract learnings from session {session_id}. JSONL path: {jsonl_path}",
    ]


_ALLOW_ENV_PREFIXES = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_",
    "TERM",
    "SHELL",
    "TMPDIR",
    "XDG_",
    "CLAUDE_",
    "UV_",
    "PYTHONPATH",
    "VIRTUAL_ENV",
)


def build_extraction_env(project_dir: str | None) -> dict[str, str]:
    """Build allowlisted environment dict for extraction subprocess.

    Only passes env vars matching known-safe prefixes. This prevents
    leaking DB credentials, API keys, or tokens to the LLM process.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if any(k.startswith(prefix) for prefix in _ALLOW_ENV_PREFIXES)
    }
    env["CLAUDE_MEMORY_EXTRACTION"] = "1"
    if project_dir:
        env["CLAUDE_PROJECT_DIR"] = project_dir
    return env


def parse_extraction_output(stdout: str) -> dict[str, int]:
    """Extract learnings/duplicates counts from extraction stdout."""
    result: dict[str, int] = {"learnings": 0, "duplicates": 0}
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Learnings stored:"):
            try:
                result["learnings"] = int(stripped.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif stripped.startswith("Duplicates skipped:"):
            try:
                result["duplicates"] = int(stripped.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
    return result


def classify_session(
    uuid: str, session_id: str | None, skip_no_db: bool
) -> tuple[str | None, str]:
    """Determine effective session_id and skip reason.

    Returns ``(effective_session_id, skip_reason)``.
    Empty skip_reason means proceed.
    """
    if session_id is not None:
        return session_id, ""
    if skip_no_db:
        return None, "no DB row (--skip-no-db)"
    return uuid, ""


def format_dry_run_line(session: dict) -> str:
    """Format a single session for dry-run output."""
    sid = session.get("session_id", session.get("uuid", "?"))
    uuid = session.get("uuid", "?")
    project = session.get("project", "?")
    return f"  {sid} <- {uuid[:8]}... ({project})"


def format_summary(
    processed: int, learnings: int, dupes: int, errors: int, elapsed: float
) -> str:
    """Format final summary text."""
    return (
        f"Sessions processed: {processed}\n"
        f"Learnings stored: {learnings}\n"
        f"Duplicates skipped: {dupes}\n"
        f"Errors: {errors}\n"
        f"Elapsed: {elapsed:.1f}s"
    )


def select_batch(sessions: list[dict], limit: int) -> list[dict]:
    """Select a batch of sessions. Returns a new list; does not mutate input."""
    if limit <= 0:
        return list(sessions)
    return list(sessions[:limit])


def strip_yaml_frontmatter(content: str) -> str:
    """Remove YAML frontmatter (``---...---``) from content."""
    if not content or not content.startswith("---"):
        return content
    parts = content.split("---", 2)
    return parts[2].strip() if len(parts) >= 3 else content


def _positive_int(value: str) -> int:
    """Argparse type for positive integers."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return n


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Write timestamped log message."""
    print(f"[backfill] {msg}", flush=True)


_faulthandler_log_file = None


def _bootstrap() -> None:
    """Initialize faulthandler and load .env files. Called only from main()."""
    global _faulthandler_log_file  # noqa: PLW0603
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


def list_s3_keys(bucket: str) -> str:
    """Run ``aws s3 ls`` and return raw stdout. Returns empty string on error."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", f"s3://{bucket}/sessions/", "--recursive"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            _log(f"S3 list failed: {result.stderr}")
            return ""
        return result.stdout
    except subprocess.TimeoutExpired:
        _log("S3 list timed out")
        return ""


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _is_valid_uuid(value: str) -> bool:
    """Validate that a string matches UUID format (lowercase hex)."""
    return bool(_UUID_RE.match(value))


def lookup_session_id(uuid: str, conn) -> str | None:
    """Look up s-* session ID from DB using JSONL UUID from transcript_path."""
    if not _is_valid_uuid(uuid):
        _log(f"Invalid UUID format, skipping lookup: {uuid[:40]}")
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM sessions WHERE transcript_path LIKE %s LIMIT 1",
            (f"%{uuid}%",),
        )
        row = cur.fetchone()
        return row[0] if row else None
    except Exception as e:
        _log(f"DB lookup failed for {uuid}: {e}")
        conn.rollback()
        return None


def is_session_extracted(uuid: str, conn) -> bool:
    """Check if this session UUID has a terminal-success or in-progress row.

    Suppresses retries for ``ok`` and ``in_progress`` statuses. Failed,
    timed-out, or other non-success rows are retryable.
    """
    skip_statuses = {"ok", "in_progress"}
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT status FROM backfill_log WHERE s3_uuid = %s",
            (uuid,),
        )
        row = cur.fetchone()
        return row is not None and row[0] in skip_statuses
    except Exception:
        return False


def claim_session(uuid: str, session_id: str, project: str, conn) -> bool:
    """Atomically claim a session for extraction via ``in_progress`` row.

    Returns True if the claim was acquired (row inserted). Returns False
    if another process already holds the claim (``ok`` or ``in_progress``).
    Failed/timed-out rows are overwritten to allow retries.
    """
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO backfill_log (s3_uuid, session_id, project, status)
               VALUES (%s, %s, %s, 'in_progress')
               ON CONFLICT (s3_uuid) DO UPDATE
                   SET status = 'in_progress', processed_at = NOW()
                   WHERE backfill_log.status NOT IN ('ok', 'in_progress')
               RETURNING s3_uuid""",
            (uuid, session_id, project),
        )
        claimed = cur.fetchone() is not None
        conn.commit()
        return claimed
    except Exception as e:
        conn.rollback()
        _log(f"WARNING: failed to claim {uuid}: {e}")
        return False


def log_extraction_result(result: dict, conn) -> None:
    """Insert extraction result into backfill_log table."""
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO backfill_log (s3_uuid, session_id, project, status,
               learnings_stored, file_size_bytes)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (s3_uuid) DO UPDATE SET
                   status = EXCLUDED.status,
                   learnings_stored = EXCLUDED.learnings_stored,
                   file_size_bytes = EXCLUDED.file_size_bytes,
                   processed_at = NOW()""",
            (
                result["uuid"],
                result["session_id"],
                result["project"],
                result["status"],
                result.get("learnings", 0),
                result.get("file_size", 0),
            ),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        _log(f"WARNING: failed to log result for {result.get('uuid')}: {e}")


def download_and_decompress(
    s3_key: str, tmpdir: Path, uuid: str
) -> Path | None:
    """Download from S3 and decompress. Returns JSONL path or None on failure."""
    zst_path = tmpdir / f"{uuid}.jsonl.zst"
    jsonl_path = tmpdir / f"{uuid}.jsonl"

    dl = subprocess.run(
        ["aws", "s3", "cp", s3_key, str(zst_path), "--quiet"],
        capture_output=True,
        timeout=120,
    )
    if dl.returncode != 0:
        _log(f"Download failed for {uuid}: {dl.stderr.decode()[:200]}")
        return None

    dc = subprocess.run(
        ["zstd", "-d", str(zst_path), "-o", str(jsonl_path), "-q"],
        capture_output=True,
        timeout=60,
    )
    if dc.returncode != 0:
        _log(f"Decompress failed for {uuid}: {dc.stderr.decode()[:200]}")
        return None

    return jsonl_path


def run_extraction(
    jsonl_path: Path,
    session_id: str,
    agent_prompt: str,
    model: str,
    max_turns: int,
    timeout: int,
    project_dir: str | None,
) -> dict:
    """Run headless extraction subprocess. Returns result dict with status."""
    cmd = build_extraction_cmd(jsonl_path, session_id, agent_prompt, model, max_turns)
    env = build_extraction_env(project_dir)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return {"status": "extraction_timeout", "learnings": 0, "duplicates": 0}

    if proc.returncode != 0:
        return {
            "status": "extraction_failed",
            "learnings": 0,
            "duplicates": 0,
            "error": (proc.stderr or "")[:500],
        }

    counts = parse_extraction_output(proc.stdout)
    return {"status": "ok", **counts}


def load_agent_prompt(agent_file: Path | None = None) -> str:
    """Load memory-extractor agent body, stripping YAML frontmatter."""
    if agent_file is None:
        agent_file = Path.home() / ".claude" / "agents" / "memory-extractor.md"
    if not agent_file.exists():
        return "Extract learnings from this session. Store each with store_learning.py."
    content = agent_file.read_text()
    return strip_yaml_frontmatter(content)


def _process_one(
    session: dict,
    agent_prompt: str,
    model: str,
    max_turns: int,
    timeout: int,
) -> dict:
    """Process a single session: download and extract. Returns result dict.

    Does NOT write to the database — callers serialize DB writes on the
    main thread to avoid sharing a connection across worker threads.
    """
    uuid = session["uuid"]
    session_id = session["session_id"]
    s3_key = session["s3_key"]
    project = session["project"]

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = download_and_decompress(s3_key, Path(tmpdir), uuid)
        if jsonl_path is None:
            return {
                "uuid": uuid,
                "session_id": session_id,
                "project": project,
                "status": "download_failed",
                "learnings": 0,
            }

        file_size = jsonl_path.stat().st_size
        extraction = run_extraction(
            jsonl_path, session_id, agent_prompt, model, max_turns, timeout, project
        )
        return {
            "uuid": uuid,
            "session_id": session_id,
            "project": project,
            "file_size": file_size,
            **extraction,
        }


def main() -> int:
    """CLI entry point for backfill_learnings."""
    parser = argparse.ArgumentParser(
        description="Backfill learnings from S3-archived sessions"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without extracting")
    parser.add_argument(
        "--limit", type=_positive_int, default=0, help="Max sessions (0=all)"
    )
    parser.add_argument("--project", type=str, default="", help="Filter by project substring")
    parser.add_argument(
        "--workers", type=_positive_int, default=3, help="Parallel workers (default: 3)"
    )
    parser.add_argument(
        "--skip-no-db", action="store_true", help="Skip sessions with no DB row"
    )
    args = parser.parse_args()
    _bootstrap()

    bucket = get_s3_bucket()
    if not bucket:
        _log("ERROR: CLAUDE_SESSION_ARCHIVE_BUCKET not set")
        return 1

    # Connect to PostgreSQL (required for non-dry runs)
    pg_url = get_pg_url()
    conn = None
    if pg_url:
        try:
            import psycopg2

            conn = psycopg2.connect(pg_url)
            _log("Connected to PostgreSQL")
        except Exception as e:
            _log(f"WARNING: PostgreSQL unavailable: {e}")

    if not conn and not args.dry_run:
        _log("ERROR: PostgreSQL required for non-dry runs (session ID resolution + backfill_log)")
        return 1

    # List and parse S3 sessions
    raw_listing = list_s3_keys(bucket)
    sessions = parse_s3_listing(raw_listing, bucket, args.project or None)
    _log(f"Found {len(sessions)} archived sessions")

    # Resolve session IDs and filter
    to_process: list[dict] = []
    skipped_extracted = 0
    skipped_no_db = 0

    for session in sessions:
        uuid = session["uuid"]
        db_session_id = lookup_session_id(uuid, conn) if conn else None
        effective_id, skip_reason = classify_session(uuid, db_session_id, args.skip_no_db)

        if skip_reason:
            skipped_no_db += 1
            continue

        if conn and is_session_extracted(uuid, conn):
            skipped_extracted += 1
            continue

        session["session_id"] = effective_id
        to_process.append(session)

    _log(
        f"To process: {len(to_process)} | "
        f"Already extracted: {skipped_extracted} | "
        f"Skipped (no DB): {skipped_no_db}"
    )

    batch = select_batch(to_process, args.limit)
    if args.limit:
        _log(f"Limited to {len(batch)} sessions")

    if args.dry_run:
        _log("DRY RUN — sessions that would be processed:")
        for s in batch:
            print(format_dry_run_line(s))
        return 0

    # Load config
    try:
        from scripts.core.config import get_config

        cfg = get_config().daemon
        model = cfg.extraction_model
        max_turns = cfg.extraction_max_turns
        timeout = cfg.extraction_timeout
    except Exception:
        model = "sonnet"
        max_turns = 15
        timeout = 300

    agent_prompt = load_agent_prompt()

    # Process sessions
    total_learnings = 0
    total_dupes = 0
    errors = 0
    start = time.time()

    # Claim sessions atomically before submitting to workers
    claimed_batch = [
        s for s in batch
        if claim_session(s["uuid"], s["session_id"], s["project"], conn)
    ]
    if len(claimed_batch) < len(batch):
        _log(f"Claimed {len(claimed_batch)} of {len(batch)} (rest already in progress)")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_one, s, agent_prompt, model, max_turns, timeout
            ): s
            for s in claimed_batch
        }
        for i, future in enumerate(as_completed(futures), 1):
            s = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "status": "exception",
                    "error": str(e),
                    "uuid": s.get("uuid", "?"),
                    "session_id": s.get("session_id", "?"),
                    "project": s.get("project", "?"),
                }
                errors += 1

            status = result.get("status", "unknown")
            learned = result.get("learnings", 0)
            dupes = result.get("duplicates", 0)
            total_learnings += learned
            total_dupes += dupes

            # Serialize DB writes on main thread (no shared conn across workers)
            if conn:
                log_extraction_result(result, conn)

            if status != "ok":
                errors += 1
                _log(
                    f"[{i}/{len(batch)}] FAIL {s.get('session_id', '?')} "
                    f"status={status} error={result.get('error', '')[:100]}"
                )
            else:
                _log(
                    f"[{i}/{len(batch)}] OK {s.get('session_id', '?')} "
                    f"learnings={learned} dupes={dupes}"
                )

    elapsed = time.time() - start
    _log(f"\n{format_summary(len(claimed_batch), total_learnings, total_dupes, errors, elapsed)}")

    if conn:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
