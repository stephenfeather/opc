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
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path so `scripts.*` imports work
# when launched via `uv run` (which doesn't add cwd to sys.path)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import faulthandler

# Module-level handle kept alive for faulthandler's lifetime. Populated by
# _enable_faulthandler() at main() entry — NOT at import time. Issue #55/#57
# track the refactor that removed module-level faulthandler.enable() calls
# across the codebase to stop leaking an fd per import.
_faulthandler_log_file = None


def _enable_faulthandler() -> None:
    """Enable faulthandler with best-effort crash logging.

    Idempotent — safe to call multiple times. Falls back to stderr if the
    crash log path cannot be created. Called only from main() / CLI entry
    points, never at module import, to avoid fd leaks when other scripts
    import memory_daemon for its helpers.
    """
    global _faulthandler_log_file  # noqa: PLW0603
    if _faulthandler_log_file is not None:
        return
    try:
        log_path = Path.home() / ".claude" / "logs" / "opc_crash.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _faulthandler_log_file = open(log_path, "a")  # noqa: SIM115
        faulthandler.enable(file=_faulthandler_log_file, all_threads=True)
    except OSError:
        # Best effort: if we cannot open the crash log, still enable
        # faulthandler against stderr so crashes produce *some* diagnostic.
        faulthandler.enable(all_threads=True)


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


# Re-exports from memory_daemon_extractors (D12: shims per step)
from scripts.core.memory_daemon_extractors import (  # noqa: E402
    is_extraction_blocked as _is_extraction_blocked,
    extract_memories_impl as _extract_memories_impl,
    archive_session_jsonl as _archive_session_jsonl_impl,
    calibrate_session_confidence as _calibrate_session_confidence_impl,
    extract_and_store_workflows as _extract_and_store_workflows_impl,
    generate_mini_handoff as _generate_mini_handoff_impl,
    count_session_rejections as _count_session_rejections_impl,
)

# Re-exports from memory_daemon_core (D12: shims per step)
from scripts.core.memory_daemon_core import (  # noqa: E402
    StaleSession,
    _ALLOWED_EXTRACTION_MODELS,
    _normalize_project,
    strip_yaml_frontmatter,
)

# Re-exports from memory_daemon_db (D12: shims per step)
from scripts.core.memory_daemon_db import (  # noqa: E402
    get_postgres_url,
    use_postgres,
    pg_connect,
    get_sqlite_path,
    pg_ensure_column,
    sqlite_ensure_table,
    ensure_schema,
    pg_get_stale_sessions as _pg_get_stale_sessions_impl,
    sqlite_get_stale_sessions as _sqlite_get_stale_sessions_impl,
    get_stale_sessions as _get_stale_sessions_impl,
    pg_mark_extracting,
    pg_mark_extracted,
    pg_mark_extraction_failed as _pg_mark_extraction_failed_impl,
    pg_mark_archived,
    mark_archived,
    pg_mark_session_exited,
    sqlite_mark_extracting,
    sqlite_mark_extracted,
    sqlite_mark_extraction_failed as _sqlite_mark_extraction_failed_impl,
    sqlite_mark_session_exited,
    pg_recover_stalled_extractions,
    sqlite_recover_stalled_extractions,
    recover_stalled_extractions,
    mark_extracting,
    mark_extracted,
    mark_extraction_failed as _mark_extraction_failed_impl,
    mark_session_exited,
    count_session_learnings as _count_session_learnings_db,
    seed_last_pattern_run as _seed_last_pattern_run_db,
)

# Config from opc.toml [daemon] — read at call time, not import time (D3)
from scripts.core.config import get_config as _get_config
_daemon_cfg = _get_config().daemon

PID_FILE = Path.home() / ".claude" / "memory-daemon.pid"
LOG_FILE = Path.home() / ".claude" / "memory-daemon.log"


# All config-derived values are read at call time via properties on _daemon_cfg.
# These uppercase names are kept for backward compat but now delegate to live config.
def _poll_interval() -> int:
    return _daemon_cfg.poll_interval


def _max_concurrent() -> int:
    return _daemon_cfg.max_concurrent_extractions


def _extraction_timeout() -> int:
    return _daemon_cfg.extraction_timeout


def _harvest_grace_period() -> int:
    return _daemon_cfg.harvest_grace_period


def _pattern_detection_interval() -> float:
    return _daemon_cfg.pattern_detection_interval_hours * 3600


# ---------------------------------------------------------------------------
# DaemonState (D14: single source of truth for mutable daemon state)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402


@dataclass
class DaemonState:
    """All mutable state for the daemon process.

    Replaces module-level globals (active_extractions, pending_queue,
    _pattern_proc, _last_pattern_run). Created once in daemon_loop(),
    stored in _daemon_state for accessor helpers.
    """

    active_extractions: dict = field(default_factory=dict)
    pending_queue: list = field(default_factory=list)
    pattern_proc: subprocess.Popen | None = None
    last_pattern_run: float = 0.0


def create_daemon_state() -> DaemonState:
    """Factory: create a fresh DaemonState with empty collections."""
    return DaemonState()


# Module-level state pointer (set in daemon_loop before while True,
# lazy-initialized by accessors for backward compat outside daemon context)
_daemon_state: DaemonState | None = None


def _ensure_daemon_state() -> DaemonState:
    """Return _daemon_state, lazy-initializing if needed.

    This allows callers outside daemon_loop (tests, one-off extract_memories
    calls) to work without requiring daemon_loop setup first.
    """
    global _daemon_state
    if _daemon_state is None:
        _daemon_state = create_daemon_state()
    return _daemon_state


def get_active_extractions() -> dict:
    """Return the live active_extractions dict from daemon state."""
    return _ensure_daemon_state().active_extractions


def get_pending_queue() -> list:
    """Return the live pending_queue list from daemon state."""
    return _ensure_daemon_state().pending_queue




def _reserve_low_fds(
    *,
    os_open_fn=os.open,
    os_close_fn=os.close,
) -> None:
    """Reserve fds 0/1/2 with /dev/null before any long-lived file handle opens.

    Addresses Issue #99 Round 1 HIGH finding: the rotating log file handler
    is opened at module import via ``_logger = _setup_logging()``. If the
    process is launched with any of fd 0/1/2 already closed, the kernel
    assigns that slot to the log file. ``_setup_daemon_fds()`` later dup2s
    ``/dev/null`` onto 0/1/2, which would silently redirect the log handle
    to ``/dev/null`` in exactly the degraded-stdio scenario this patch
    exists to harden.

    This function opens ``/dev/null`` until it is handed back an fd > 2, at
    which point it knows 0/1/2 are all held (either by whatever they were
    before, or by fresh ``/dev/null`` handles we just installed). The low
    fds returned are intentionally NOT closed — they stay alive as guards
    until ``_setup_daemon_fds()`` dup2s over them, which implicitly closes
    them.

    Tolerant of ``OSError`` so a fd-exhausted process still imports
    successfully.
    """
    while True:
        try:
            fd = os_open_fn(os.devnull, os.O_RDWR)
        except OSError:
            return
        if fd > 2:
            os_close_fn(fd)
            return
        # fd in (0, 1, 2): leave it open as a guard; loop to fill remaining slots.


_reserve_low_fds()


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
    """Wrapper: delegates to memory_daemon_db.seed_last_pattern_run."""
    return _seed_last_pattern_run_db()


def log(msg: str):
    """Write timestamped log message via rotating file handler."""
    try:
        _logger.info(msg)
    except Exception:
        pass  # Don't crash on log failures


# Debug mode: off by default, enabled via --debug CLI flag or
# MEMORY_DAEMON_DEBUG=1 env var. main() also exports the env var when the
# CLI flag is set so child subprocesses (pattern_batch, claude -p extraction)
# inherit the setting through os.environ.
DEBUG: bool = os.environ.get("MEMORY_DAEMON_DEBUG", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def debug(msg: str) -> None:
    """Log a diagnostic message when DEBUG mode is enabled.

    No-op when DEBUG is False so gated call sites cost nothing in normal
    operation. Output is prefixed with ``[DEBUG]`` and routed through the
    same rotating log file as ``log()``.
    """
    if DEBUG:
        log(f"[DEBUG] {msg}")


# Database operations - PostgreSQL
# (get_postgres_url, use_postgres, pg_connect moved to memory_daemon_db.py)


# (pg_ensure_column moved to memory_daemon_db.py)
# (pg_mark_*, sqlite_mark_*, recovery, dispatchers moved to memory_daemon_db.py)
# Config-injecting wrappers for functions that took module-level globals (D3):


def pg_get_stale_sessions() -> list:
    """Wrapper: reads config from live _daemon_cfg at call time (D3)."""
    return _pg_get_stale_sessions_impl(
        stale_threshold=_daemon_cfg.stale_threshold,
        max_retries=_daemon_cfg.max_retries,
        harvest_grace_period=_daemon_cfg.harvest_grace_period,
    )


def sqlite_get_stale_sessions() -> list:
    """Wrapper: reads config from live _daemon_cfg at call time (D3)."""
    return _sqlite_get_stale_sessions_impl(
        stale_threshold=_daemon_cfg.stale_threshold,
        max_retries=_daemon_cfg.max_retries,
        harvest_grace_period=_daemon_cfg.harvest_grace_period,
    )


def get_stale_sessions() -> list:
    """Wrapper: reads config from live _daemon_cfg at call time (D3)."""
    return _get_stale_sessions_impl(
        stale_threshold=_daemon_cfg.stale_threshold,
        max_retries=_daemon_cfg.max_retries,
        harvest_grace_period=_daemon_cfg.harvest_grace_period,
    )


def pg_mark_extraction_failed(session_id: str, *, last_error: str | None = None):
    """Wrapper: reads max_retries from live config at call time (D3)."""
    _pg_mark_extraction_failed_impl(
        session_id, max_retries=_daemon_cfg.max_retries, last_error=last_error
    )


def sqlite_mark_extraction_failed(session_id: str, *, last_error: str | None = None):
    """Wrapper: reads max_retries from live config at call time (D3)."""
    _sqlite_mark_extraction_failed_impl(
        session_id, max_retries=_daemon_cfg.max_retries, last_error=last_error
    )


def mark_extraction_failed(session_id: str, *, last_error: str | None = None):
    """Wrapper: reads max_retries from live config at call time (D3)."""
    _mark_extraction_failed_impl(
        session_id, max_retries=_daemon_cfg.max_retries, last_error=last_error
    )


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


def extract_memories(
    session_id: str,
    project_dir: str,
    transcript_path: str | None = None,
) -> bool:
    """D1 wrapper: resolves collaborators and delegates to extractors."""
    return _extract_memories_impl(
        session_id=session_id,
        project_dir=project_dir,
        transcript_path=transcript_path,
        active_extractions=get_active_extractions(),
        subprocess_popen=subprocess.Popen,
        is_blocked_fn=_is_extraction_blocked,
        mark_extracted_fn=mark_extracted,
        mark_failed_fn=mark_extraction_failed,
        log_fn=log,
        daemon_cfg=_daemon_cfg,
        allowed_models=_ALLOWED_EXTRACTION_MODELS,
        strip_frontmatter_fn=strip_yaml_frontmatter,
    )


def archive_session_jsonl(session_id: str, jsonl_path: Path | None = None):
    """Wrapper: delegates to memory_daemon_extractors.archive_session_jsonl."""
    _archive_session_jsonl_impl(
        session_id, jsonl_path, log_fn=log, mark_archived_fn=mark_archived,
    )


def _calibrate_session_confidence(session_id: str):
    """Wrapper: delegates to memory_daemon_extractors."""
    _calibrate_session_confidence_impl(session_id, log_fn=log)


def _extract_and_store_workflows(session_id: str, jsonl_path: Path, project: str | None):
    """Wrapper: delegates to memory_daemon_extractors."""
    _extract_and_store_workflows_impl(
        session_id, jsonl_path, project,
        log_fn=log, normalize_project_fn=_normalize_project,
    )


def _generate_mini_handoff(session_id: str, jsonl_path: Path, project: str | None):
    """Wrapper: delegates to memory_daemon_extractors."""
    _generate_mini_handoff_impl(session_id, jsonl_path, project, log_fn=log)


def _count_session_learnings(session_id: str) -> int | None:
    """Wrapper: delegates to memory_daemon_db.count_session_learnings."""
    return _count_session_learnings_db(session_id)


def _count_session_rejections(session_id: str) -> int | None:
    """Wrapper: delegates to memory_daemon_extractors.count_session_rejections."""
    return _count_session_rejections_impl(session_id)


def reap_completed_extractions():
    """Check for completed extraction processes and remove from active set."""
    ae = get_active_extractions()
    completed = []
    for pid, (session_id, proc, jsonl_path, project, _start) in list(ae.items()):
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
            # Safe to read(): proc.poll() already returned, so child has exited
            # and the write end of the pipe is closed. No deadlock risk.
            stderr_text = ""
            if proc.stderr:
                try:
                    if exit_code != 0:
                        raw = proc.stderr.read(65536)  # bounded read, ~64KB max
                        stderr_text = raw.decode("utf-8", errors="replace").strip()[-500:]
                        if stderr_text:
                            log(f"  stderr: {stderr_text}")
                finally:
                    proc.stderr.close()

            if exit_code == 0:
                mark_extracted(session_id)
                _calibrate_session_confidence(session_id)
                _extract_and_store_workflows(session_id, jsonl_path, project)
                _generate_mini_handoff(session_id, jsonl_path, project)
                archive_session_jsonl(session_id, jsonl_path)
            else:
                mark_extraction_failed(
                    session_id, last_error=stderr_text or None
                )

    for pid in completed:
        del ae[pid]
        # Note: Popen.poll() already calls waitpid internally, so the
        # child is fully reaped. No explicit os.waitpid needed here.

    return len(completed)


def watchdog_stuck_extractions():
    """Kill extraction subprocesses that exceed EXTRACTION_TIMEOUT."""
    ae = get_active_extractions()
    now = time.time()
    killed = []
    for pid, (session_id, proc, jsonl_path, project, start_time) in list(ae.items()):
        elapsed = now - start_time
        if elapsed > _extraction_timeout():
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
        del ae[pid]

    if killed:
        log(f"Watchdog: killed {len(killed)} stuck extractions")
    return len(killed)


def process_pending_queue():
    """Spawn extractions from queue if under concurrency limit."""
    ae = get_active_extractions()
    pq = get_pending_queue()
    spawned = 0
    while pq and len(ae) < _max_concurrent():
        session_id, project, transcript_path = pq.pop(0)
        log(f"Dequeuing {session_id} (project={project or 'unknown'}, "
            f"queue remaining: {len(pq)})")
        extract_memories(session_id, project, transcript_path)
        spawned += 1
    return spawned


def queue_or_extract(
    session_id: str,
    project: str,
    transcript_path: str | None = None,
):
    """Queue extraction if at limit, otherwise extract immediately."""
    ae = get_active_extractions()
    pq = get_pending_queue()
    if len(ae) >= _max_concurrent():
        pq.append((session_id, project, transcript_path))
        log(f"Queued {session_id} (active={len(ae)}, "
            f"queue={len(pq)})")
    else:
        extract_memories(session_id, project, transcript_path)


# ---------------------------------------------------------------------------
# Pattern detection (non-blocking subprocess)
# ---------------------------------------------------------------------------

def _run_pattern_detection_batch():
    """Launch pattern detection as a non-blocking subprocess.

    Uses Popen to avoid blocking the daemon loop (detection can take
    minutes on large datasets). Only one detection run at a time.
    Operates on _daemon_state fields (D14).
    """
    state = _ensure_daemon_state()
    # Don't start if already running
    if state.pattern_proc is not None and state.pattern_proc.poll() is None:
        debug("Pattern detection already running, skipping tick")
        return
    try:
        project_root = Path(__file__).parent.parent.parent
        log("Starting pattern detection batch...")
        # stdin=DEVNULL is defense-in-depth against fd-0 corruption from
        # broken daemonization (Issue #99). The primary fix is
        # _setup_daemon_fds, but explicit DEVNULL here prevents any future
        # regression in the parent's fd handling from cascading into a
        # pattern_batch init_sys_streams EBADF crash.
        # Popen inherits os.environ by default, so MEMORY_DAEMON_DEBUG
        # propagates to the child for any consumer that reads it. But
        # pattern_batch.py itself only raises its log level when passed
        # --verbose, so we also inject that flag when DEBUG is on to make
        # --debug materially affect child verbosity (not just env cosmetics).
        pb_argv = [sys.executable, "-m", "scripts.core.pattern_batch"]
        pb_stderr = subprocess.PIPE
        debug_stderr_handle = None
        if DEBUG:
            pb_argv.append("--verbose")
            # Round 2 Codex finding: verbose pattern_batch writes DEBUG-level
            # logs to stderr via logging.basicConfig. With stderr=PIPE, the
            # pipe buffer (~64KB) can fill up and deadlock the child on
            # write — poll() stays stuck at None, state.pattern_proc is
            # permanently occupied, and future pattern runs are suppressed.
            # Mitigation: redirect stderr to an append-mode log file so the
            # child can write freely. Operators tail the file; the daemon
            # does not read it.
            try:
                verbose_log = Path.home() / ".claude" / "logs" / "pattern_batch_verbose.log"
                verbose_log.parent.mkdir(parents=True, exist_ok=True)
                debug_stderr_handle = open(verbose_log, "ab")  # noqa: SIM115
                pb_stderr = debug_stderr_handle
            except OSError as e:
                # If we cannot open the log file, fall back to DEVNULL —
                # discarding verbose output is safer than deadlocking.
                log(f"Pattern detection verbose log open failed: {e}")
                pb_stderr = subprocess.DEVNULL
        state.pattern_proc = subprocess.Popen(
            pb_argv,
            cwd=str(project_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=pb_stderr,
        )
        # Stash the verbose-log file handle on the proc object so
        # _check_pattern_detection can close it when reaping. Using an
        # attribute on the proc ties the handle's lifetime to the process.
        if debug_stderr_handle is not None:
            state.pattern_proc._debug_stderr_handle = debug_stderr_handle
        state.last_pattern_run = time.time()
        # Gate .pid access behind DEBUG so test mocks without a .pid
        # attribute don't explode, and so release-mode callers pay nothing
        # for the attribute lookup.
        if DEBUG:
            debug(f"pattern_batch spawned pid={state.pattern_proc.pid}")
    except Exception as e:
        log(f"Pattern detection launch error: {e}")


def _check_pattern_detection():
    """Check if background pattern detection finished.

    Called from daemon loop each iteration. Operates on _daemon_state (D14).
    """
    state = _ensure_daemon_state()
    if state.pattern_proc is None:
        return
    rc = state.pattern_proc.poll()
    if rc is not None:
        if rc == 0:
            stdout_stream = state.pattern_proc.stdout
            stdout = stdout_stream.read().decode() if stdout_stream is not None else ""
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
            # Guard against stderr being None, which happens when the child
            # was spawned in DEBUG mode with stderr redirected to a file
            # (rather than subprocess.PIPE). The operator can tail
            # ~/.claude/logs/pattern_batch_verbose.log for the full output.
            stderr_stream = state.pattern_proc.stderr
            if stderr_stream is not None:
                stderr = stderr_stream.read().decode()[:200]
            else:
                stderr = "(see ~/.claude/logs/pattern_batch_verbose.log)"
            log(f"Pattern detection failed (rc={rc}): {stderr}")
        # Close the DEBUG-mode stderr log handle if we stashed one at spawn.
        debug_stderr = getattr(state.pattern_proc, "_debug_stderr_handle", None)
        if debug_stderr is not None:
            try:
                debug_stderr.close()
            except Exception:
                pass
        state.pattern_proc = None


def daemon_tick() -> None:
    """Execute one iteration of the daemon loop.

    Reads/writes _daemon_state (the single source of truth).
    try/except and time.sleep stay in daemon_loop, NOT here.
    """
    # Reap completed, kill stuck, then process pending queue
    reap_completed_extractions()
    watchdog_stuck_extractions()
    process_pending_queue()

    # Find stale sessions (SQL excludes those within grace period)
    stale_rows = get_stale_sessions()
    if stale_rows:
        # Convert raw tuples to StaleSession NamedTuples
        from scripts.core.memory_daemon_core import filter_truly_stale_sessions

        stale_sessions = [
            StaleSession(
                id=row[0],
                project=row[1],
                transcript_path=row[2] if len(row) > 2 else None,
                pid=row[3] if len(row) > 3 else None,
                exited_at=row[4] if len(row) > 4 else None,
            )
            for row in stale_rows
        ]

        truly_stale, newly_dead_ids, still_alive = filter_truly_stale_sessions(
            stale_sessions, is_alive=_is_process_alive
        )

        # Log still-alive sessions — noisy per-iteration, gated to debug.
        for s in still_alive:
            debug(f"Skipping {s.id}: process {s.pid} still alive")

        # Mark newly-dead sessions (grace period starts)
        for sid in newly_dead_ids:
            mark_session_exited(sid)
            log(f"Skipping {sid}: marked exited, "
                f"grace period {_harvest_grace_period()}s")

        # Extract truly stale sessions
        if truly_stale:
            summary = ", ".join(
                f"{s.id}({s.project or '?'})" for s in truly_stale
            )
            log(f"Found {len(truly_stale)} stale sessions: "
                f"{summary}")
            for s in truly_stale:
                mark_extracting(s.id)
                queue_or_extract(
                    s.id, s.project or "", s.transcript_path
                )

    # Pattern detection: check completion, trigger if due
    # Only runs on PostgreSQL — pattern_batch.py requires asyncpg
    _check_pattern_detection()
    if use_postgres():
        elapsed = time.time() - _ensure_daemon_state().last_pattern_run
        if elapsed > _pattern_detection_interval():
            _run_pattern_detection_batch()


def daemon_loop():
    """Main daemon loop: init, then tick + sleep forever."""
    global _daemon_state

    db_type = "PostgreSQL" if use_postgres() else "SQLite"
    log(f"Memory daemon v{DAEMON_VERSION} started "
        f"(using {db_type}, "
        f"max_concurrent={_max_concurrent()})")
    ensure_schema()
    recover_stalled_extractions()

    # Reuse existing DaemonState if lazily initialized, else create (D14).
    # Prior bug: daemon_loop() replaced a lazily-created state, losing
    # pre-loop extractions — log the branch taken so future regressions
    # are visible in --debug runs.
    if _daemon_state is None:
        debug("daemon_loop: creating fresh DaemonState (no pre-loop init)")
        _daemon_state = create_daemon_state()
    else:
        debug(
            f"daemon_loop: reusing lazily-created DaemonState "
            f"(active={len(_daemon_state.active_extractions)}, "
            f"queued={len(_daemon_state.pending_queue)})"
        )
    _daemon_state.last_pattern_run = _seed_last_pattern_run()
    if _daemon_state.last_pattern_run:
        log(f"Seeded last pattern run from DB: "
            f"{datetime.fromtimestamp(_daemon_state.last_pattern_run).isoformat()}")

    while True:
        try:
            daemon_tick()
        except Exception as e:
            log(f"Error in daemon loop: {e}")
            if DEBUG:
                import traceback

                log(f"[DEBUG] traceback:\n{traceback.format_exc()}")
        time.sleep(_poll_interval())


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


def _setup_daemon_fds(
    *,
    os_open_fn=os.open,
    os_dup2_fn=os.dup2,
    os_close_fn=os.close,
) -> None:
    """Redirect kernel fds 0/1/2 to /dev/null for safe daemonization.

    Correct replacement for the previous ``sys.std*.close()`` pattern, which
    left kernel fd-table holes at 0/1/2 and corrupted any subsequent
    ``subprocess.Popen()`` that did not pass ``stdin=subprocess.DEVNULL``.

    Bug fingerprint (see Issue #99):
        Fatal Python error: init_sys_streams: can't initialize sys standard streams
        OSError: [Errno 9] Bad file descriptor

    observed in pattern_batch children because ``pipe()`` during Popen setup
    reused the closed fd 0/1 slots, leaving the child with a dangling std fd.

    The os.* functions are injected for testability — callers in production
    use the real syscalls; tests pass mocks to verify the syscall sequence
    without touching real file descriptors.
    """
    devnull_fd = os_open_fn(os.devnull, os.O_RDWR)
    os_dup2_fn(devnull_fd, 0)
    os_dup2_fn(devnull_fd, 1)
    os_dup2_fn(devnull_fd, 2)
    # Only close if /dev/null landed above fd 2. If it landed at 0/1/2
    # (because those fds were already closed pre-call), closing would
    # re-open the hole we are trying to fix.
    if devnull_fd > 2:
        os_close_fn(devnull_fd)


def _run_as_daemon():
    """Run the daemon loop (called by subprocess on Windows, directly after fork on Unix)."""
    # Write PID file
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    # Redirect standard fds to /dev/null. Must NOT sys.std*.close() — that
    # leaves kernel fd-table holes which corrupt child subprocess stdio
    # (observed as pattern_batch init_sys_streams crashes, Issue #99).
    _setup_daemon_fds()

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
                interval_h = int(_pattern_detection_interval()) // 3600
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
    # Enable crash diagnostics before doing anything else so even an early
    # import-time failure in a dependency still produces a traceback.
    _enable_faulthandler()

    parser = argparse.ArgumentParser(description="Global Memory Extraction Daemon")
    parser.add_argument("command", nargs="?", choices=["start", "stop", "status"], help="Command")
    parser.add_argument("--daemon-subprocess", action="store_true",
                        help="Internal: run as daemon subprocess (Windows)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose diagnostic logging. Also exports "
                             "MEMORY_DAEMON_DEBUG=1 so child subprocesses "
                             "(pattern_batch, claude -p extraction) inherit it.")
    args = parser.parse_args()

    # Honour --debug before anything else so all subsequent log calls and
    # spawned children see the elevated verbosity.
    if args.debug:
        global DEBUG
        DEBUG = True
        os.environ["MEMORY_DAEMON_DEBUG"] = "1"

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
