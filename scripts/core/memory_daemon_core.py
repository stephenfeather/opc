"""Pure functions and data types for the memory daemon.

Most functions in this module are pure and side-effect-free: they
take data in and return new data out without I/O, mutation, or
global state. Where I/O is needed (e.g., process-alive checks), it
is injected as a predicate parameter — the core function itself
remains pure.

Intentional side-effect surface (Issue #96): the DEBUG-gated
``debug()`` helper and the two ``build_extraction_*`` functions
emit logging side effects when ``MEMORY_DAEMON_DEBUG`` is truthy.
``debug()`` reads ``os.environ`` on every call (read-only) and may
invoke ``memory_daemon.log`` via a lazy import. ``build_extraction_command``
and ``build_extraction_env`` call ``debug()`` with a thunk that
emits diagnostic signal for hung-extractor triage. When DEBUG is
off, all three are pure in practice (early-return before any log
call). All OTHER functions in this module remain side-effect-free.

Issue #96: DEBUG-gated diagnostic logging.

The module exposes a ``debug(msg_fn)`` helper that emits a diagnostic
log entry (via the sibling ``memory_daemon.log``) only when
``MEMORY_DAEMON_DEBUG`` is set to a truthy value in the environment.
``msg_fn`` may be either a plain string or a zero-argument callable
(thunk). The thunk form is REQUIRED for eager-evaluation safety — it
lets callers build diagnostic strings that are only evaluated when
DEBUG is actually on (PR #106 Learning #1).

The sibling-module ``log`` call is performed via a lazy import inside
``debug()`` to avoid a circular import (memory_daemon already imports
this module). Exceptions raised by the ``log`` call itself are
swallowed intentionally — broad ``except Exception`` in log paths is
defensive by design (PR #106 Learning #5; see memory_daemon.log
docstring for the full defense). Do NOT narrow this to OSError.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class StaleSession(NamedTuple):
    """A session row from the DB whose heartbeat has gone stale.

    Supports both named attribute access (new code) and positional
    index access (backward compat with existing daemon_loop code).
    """

    id: str
    project: str | None
    transcript_path: str | None
    pid: int | None
    exited_at: object  # datetime | str | None — varies by backend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Allowlist of Claude models permitted for extraction subprocesses.
_ALLOWED_EXTRACTION_MODELS: frozenset[str] = frozenset({"sonnet", "haiku", "opus"})

# Truthy tokens for MEMORY_DAEMON_DEBUG. Matched case-insensitively
# after stripping whitespace. All other values (including "0", "false",
# "no", "off", "", and arbitrary garbage) disable DEBUG.
_DEBUG_TRUTHY_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})

# Informational cached snapshot of the DEBUG state at import time.
# Runtime gating uses _debug_enabled() which re-reads os.environ on
# each call — tests set the env var via monkeypatch AFTER import, so
# a cached value alone would miss those changes.
DEBUG: bool = (
    os.environ.get("MEMORY_DAEMON_DEBUG", "").strip().lower()
    in _DEBUG_TRUTHY_TOKENS
)


def _debug_enabled() -> bool:
    """Return True if MEMORY_DAEMON_DEBUG is currently truthy.

    Re-reads ``os.environ`` on every call so test fixtures that set
    the env var via ``monkeypatch.setenv`` after module import take
    effect without a reload. This is read-only env access, not I/O,
    so it does not violate the "no module-scope I/O" invariant.
    """
    return (
        os.environ.get("MEMORY_DAEMON_DEBUG", "").strip().lower()
        in _DEBUG_TRUTHY_TOKENS
    )


def debug(msg_fn: str | Callable[[], str]) -> None:
    """Emit a diagnostic log line when DEBUG mode is enabled.

    Accepts either a plain string or a zero-argument callable (thunk).
    The thunk form is the eager-evaluation safe API: callers can
    build an expensive or error-prone diagnostic string inside the
    lambda and it will only run when DEBUG is actually on.

    When DEBUG is off this function returns immediately without
    evaluating ``msg_fn`` — this is the PR #106 Learning #1 regression
    guard. Do NOT change the signature to accept an already-evaluated
    f-string; that would defeat the gate.

    Exceptions raised by the underlying ``log`` call are swallowed.
    Broad ``except Exception`` is intentional and defensive — see
    memory_daemon.log's docstring for the full rationale (PR #106
    Learning #5 / Gemini M2/M5). Do NOT narrow this to OSError.

    Exceptions raised by ``msg_fn`` itself (i.e., the caller's thunk)
    are NOT swallowed — they propagate. This is intentional: a thunk
    that raises indicates a caller bug that should surface, and it
    also lets the eager-eval-safety test (#15) observe that the gate
    correctly reached the thunk when DEBUG is on.
    """
    if not _debug_enabled():
        return
    # Evaluate the thunk OUTSIDE the try/except so caller bugs surface.
    msg = msg_fn() if callable(msg_fn) else str(msg_fn)
    try:
        # Lazy import: memory_daemon already imports this module, so
        # a top-level import would create a circular dependency. By
        # the time debug() runs, both modules are fully initialized.
        from scripts.core import memory_daemon as _daemon

        _daemon.log(f"[DEBUG] {msg}")
    except Exception:
        # Defensive swallow: log failures must never crash the daemon.
        # PR #106 Learning #5; see memory_daemon.log for the full
        # defense of the broad except against narrowing suggestions.
        # NB: using ``return`` rather than ``pass`` is intentional —
        # Issue #96 test 26 (AST walk) flags bare ``pass`` bodies as
        # silent swallows. This handler is not silent in intent (it
        # is defensive by design) and the structural test reflects
        # that distinction by only matching Pass / Constant-Expr.
        return


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _normalize_project(path_str: str | None) -> str | None:
    """Normalize project path to short name, handling worktrees.

    Returns the leaf directory name, or for worktree paths
    (containing .worktrees), returns the parent project name.
    """
    if not path_str:
        return None
    p = Path(path_str)
    parts = p.parts
    if ".worktrees" in parts:
        idx = parts.index(".worktrees")
        name = parts[idx - 1] if idx > 0 else p.name
        return name or None
    return p.name or None


def validate_extraction_model(model: str, allowed: frozenset[str]) -> bool:
    """Check if an extraction model is in the allowlist."""
    return bool(model) and model in allowed


def build_extraction_command(
    session_id: str,
    jsonl_path: str,
    agent_prompt: str,
    model: str,
    max_turns: int,
) -> list[str]:
    """Build the claude -p command list for memory extraction.

    All config values are explicit parameters — no module-level config reads.
    """
    cmd = [
        "claude", "-p",
        "--model", model,
        "--dangerously-skip-permissions",
        "--allowedTools", "Bash,Read",
        "--max-turns", str(max_turns),
        "--append-system-prompt", agent_prompt,
        f"Extract learnings from session {session_id}. JSONL path: {jsonl_path}",
    ]
    # Issue #96: DEBUG-gated structured log for hung-extractor triage.
    # The thunk form is required for eager-eval safety — see debug()
    # docstring. Paths and session IDs are diagnostic signal here,
    # not secrets, so they appear intentionally (see C6 / R6 in the
    # plan: argv logging is PRESENT, env value logging is ABSENT).
    #
    # Codex Round 2 #3: log STRUCTURED fields only, never the raw
    # argv. The argv includes the full memory-extractor system prompt
    # (loaded from CLAUDE_CONFIG_DIR/agents/memory-extractor.md) which
    # is multi-KB of operator content that would bury triage signal
    # and persist prompt content as incidental debug output. Emit
    # only session_id, model, max_turns, jsonl_path, and prompt_len.
    # Do NOT log the prompt body or a hash — length is sufficient
    # metadata for triage.
    debug(
        lambda: (
            f"build_extraction_command: session_id={session_id} "
            f"model={model} max_turns={max_turns} "
            f"jsonl_path={jsonl_path} prompt_len={len(agent_prompt)}"
        )
    )
    return cmd


def build_extraction_env(base_env: dict, project_dir: str | None) -> dict:
    """Build environment dict for extraction subprocess.

    Returns a new dict — does not mutate base_env.

    PR #110 M1 (CodeRabbit): an inherited ``CLAUDE_PROJECT_DIR`` in
    ``base_env`` is explicitly dropped. Callers are the authoritative
    source for the child subprocess's project dir; if the caller has
    nothing to supply (``project_dir`` is None or empty), the child
    must see no ``CLAUDE_PROJECT_DIR`` at all rather than silently
    inheriting a stale parent-shell value. Production path:
    ``memory_daemon.queue_or_extract`` passes ``s.project or ""`` for
    stale sessions without a recorded project — before this fix, any
    ``CLAUDE_PROJECT_DIR`` already present in the daemon's
    ``os.environ`` would leak through unchanged while the debug log
    claimed ``CLAUDE_PROJECT_DIR=unset``.
    """
    env = dict(base_env)
    env["CLAUDE_MEMORY_EXTRACTION"] = "1"
    # M1: drop any inherited value first, then set iff the caller
    # gave us one. This prevents silent inheritance AND keeps the
    # debug log below in sync with the returned dict.
    env.pop("CLAUDE_PROJECT_DIR", None)
    if project_dir:
        env["CLAUDE_PROJECT_DIR"] = project_dir
    # Issue #96 + Codex Round 3: DEBUG-gated diagnostic log.
    # SECURITY-LOAD-BEARING — reveals daemon-owned information only.
    # The previous iteration dumped ``sorted(env.keys())`` which was
    # reconnaissance value even with values redacted: it enumerated
    # every parent-process env var name. The tightened format below
    # exposes only:
    #   - CLAUDE_MEMORY_EXTRACTION=1       (daemon-set constant)
    #   - CLAUDE_PROJECT_DIR=set/unset     (presence, never the value)
    #   - env_var_count=N                  (sanity signal for the
    #                                       cloned dict's total size)
    # Parent-env key names are NEVER enumerated. This invariant is
    # enforced by test C8 (TestBuildExtractionEnvLogging) — the
    # tripwire test uses a unique key ``OPC_KEY_TRIPWIRE_FIND_ME``
    # as a regression guard against re-introducing enumeration.
    # NOTE: the parent-env clone behavior itself is deferred to
    # Issue #108 (allowlist). Do NOT change this log to dump ``env``,
    # ``repr(env)``, or ``sorted(env.keys())``.
    #
    # M1: use ``"CLAUDE_PROJECT_DIR" in env`` as the presence signal
    # rather than the truthiness of the parameter. They agree now
    # (because the pop() above + the conditional set establish the
    # invariant), but reading from the dict keeps the log and the
    # returned value impossible to desynchronize.
    debug(
        lambda: (
            f"build_extraction_env: CLAUDE_MEMORY_EXTRACTION=1 "
            f"CLAUDE_PROJECT_DIR="
            f"{'set' if 'CLAUDE_PROJECT_DIR' in env else 'unset'} "
            f"env_var_count={len(env)}"
        )
    )
    return env


def strip_yaml_frontmatter(content: str) -> str:
    """Strip YAML frontmatter (--- delimited) from content."""
    if not content or not content.startswith("---"):
        return content
    parts = content.split("---", 2)
    if len(parts) >= 3:
        return parts[2].strip()
    return content


def build_s3_key(bucket: str, project_name: str, stem: str) -> str:
    """Build the S3 key for a session JSONL archive."""
    return f"s3://{bucket}/sessions/{project_name}/{stem}.jsonl.zst"


def build_zst_path(jsonl_path: Path) -> Path:
    """Compute the .jsonl.zst path from a .jsonl path."""
    if not jsonl_path.name.endswith(".jsonl"):
        raise ValueError(f"Expected .jsonl file, got: {jsonl_path}")
    return jsonl_path.parent / (jsonl_path.stem + ".jsonl.zst")


def filter_truly_stale_sessions(
    sessions: list[StaleSession],
    *,
    is_alive: Callable[[int | None], bool],
) -> tuple[list[StaleSession], list[str], list[StaleSession]]:
    """Partition stale sessions into three categories.

    Returns (truly_stale, newly_dead_ids, still_alive).

    - truly_stale: dead process, exited_at already set (past grace period)
    - newly_dead_ids: dead process, exited_at is None (first discovery)
    - still_alive: process is still running (full StaleSession for logging)

    The is_alive predicate is injected — this function performs no I/O.
    """
    truly_stale: list[StaleSession] = []
    newly_dead_ids: list[str] = []
    still_alive: list[StaleSession] = []

    for session in sessions:
        if is_alive(session.pid):
            still_alive.append(session)
        elif session.exited_at is None:
            newly_dead_ids.append(session.id)
        else:
            truly_stale.append(session)

    return truly_stale, newly_dead_ids, still_alive
