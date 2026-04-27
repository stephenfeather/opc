#!/usr/bin/env python3
"""Single-session memory extraction CLI (issue #128).

Run the memory-daemon's extraction pipeline against ONE session for testing
or debugging, without involving the daemon scheduler.

Usage:
    uv run python scripts/core/extract_session.py --session-id <uuid>

The CLI reuses ``build_extraction_env`` and ``build_extraction_command`` from
``scripts.core.memory_daemon_core`` unmodified — wire-compatibility with the
daemon path is the whole point of this tool. The daemon owns session
lifecycle state (``backfill_extracted_at`` etc.); this CLI never marks
sessions extracted, retries on failure, or persists state. It runs the
subprocess once, streams its stderr, and reports the delta count of
archival_memory rows.

Concurrency note:
    All async DB calls share a single event loop driven by ``asyncio.run``
    at the entry point. ``postgres_pool.get_pool`` caches an asyncpg pool
    bound to the loop it was created on, so calling ``asyncio.run`` more
    than once would land on a closed loop and raise. ``run_main`` is async
    end-to-end for that reason — see PR #129 cycle-1 review (BUG 1).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from scripts.core.config.handlers import get_config
from scripts.core.db.postgres_pool import get_pool
from scripts.core.memory_daemon_core import (
    _ALLOWED_EXTRACTION_MODELS,
    build_extraction_command,
    build_extraction_env,
    strip_yaml_frontmatter,
    validate_extraction_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Substring redaction list (case-insensitive). Any env key containing one of
# these tokens has its value masked in --verbose output.
_SECRET_KEY_TOKENS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "DATABASE_URL",
    "DSN",
    "AUTH",
    "CREDENTIAL",
    "PRIVATE",
    "CERT",
)

_REDACTED = "***REDACTED***"

# Allowlist of env-key prefixes the operator is permitted to see (with
# values still subject to the secret-token redaction above) when --verbose
# is passed. Anything outside this allowlist is summarized rather than
# enumerated, to avoid leaking the full parent process environment.
# See PR #129 cycle-1 review (DESIGN 3).
_VERBOSE_ENV_PREFIX_ALLOWLIST: tuple[str, ...] = (
    "CLAUDE_",
    "OPC_",
    "PYTHONPATH",
)

# Threshold for redacting long argv tokens in dry-run output. The
# memory-extractor system prompt is multi-KB and would bury triage signal
# without a cap. See PR #129 cycle-1 review (DESIGN 2).
_DRY_RUN_MAX_TOKEN_CHARS = 256

# Fallback prompt used when CLAUDE_CONFIG_DIR/agents/memory-extractor.md is
# absent. INVARIANT: this string must remain byte-identical to the daemon's
# fallback in ``memory_daemon_extractors.extract_memories_impl`` so dev/test
# behavior matches production. If the daemon's fallback changes, update this
# constant in lock-step. See PR #129 cycle-1 review (CodeRabbit drift note).
_FALLBACK_AGENT_PROMPT = (
    "Extract learnings from this Claude Code session.\n"
    "Look for decisions, what worked, what failed, and patterns discovered.\n"
    "Store each learning using store_learning.py with appropriate type and tags."
)


# ---------------------------------------------------------------------------
# Pure helpers — argparse types
# ---------------------------------------------------------------------------


def _strict_positive_int(value: str) -> int:
    """argparse type: require an integer >= 1.

    Used for ``--max-turns`` so the daemon does not spawn an extraction with
    zero turns budgeted.
    """
    try:
        n = int(value)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from e
    if n < 1:
        raise argparse.ArgumentTypeError(f"value must be >= 1, got {n}")
    return n


def _nonneg_int(value: str) -> int:
    """argparse type: require an integer >= 0.

    Used for ``--timeout`` where 0 is reserved as "no timeout" — see
    ``parse_args`` for the policy. Negative values are rejected because
    they would crash ``proc.wait`` with a confusing low-level ValueError.
    """
    try:
        n = int(value)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from e
    if n < 0:
        raise argparse.ArgumentTypeError(f"value must be >= 0, got {n}")
    return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Pure aside from argparse error handling."""
    parser = argparse.ArgumentParser(
        prog="extract_session",
        description=(
            "Run the memory-daemon extraction pipeline against a single "
            "session. For testing/debugging only — never marks sessions "
            "extracted (the daemon owns that lifecycle)."
        ),
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="UUID of the session to extract (validated before any DB query)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved cmd + env summary; do not spawn the subprocess",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override extraction_model. Validated against the daemon's "
            f"allowlist {sorted(_ALLOWED_EXTRACTION_MODELS)}."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=_strict_positive_int,
        default=None,
        help="Override extraction_max_turns (must be >= 1)",
    )
    parser.add_argument(
        "--timeout",
        type=_nonneg_int,
        default=None,
        help=(
            "Subprocess timeout in seconds (must be >= 0; 0 is treated as "
            "no timeout, same as omitting the flag). Negative values are "
            "rejected at parse time."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Show a summary of env vars exported to the subprocess: keys "
            "matching CLAUDE_/OPC_/PYTHONPATH are listed (secrets redacted), "
            "everything else is just counted. Full env enumeration is "
            "intentionally not supported — see DESIGN 3 in PR #129."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pure helpers — validation + presentation
# ---------------------------------------------------------------------------


def validate_session_id(value: str) -> str:
    """Reject anything that does not parse as a UUID.

    Defense in depth: the parameterized DB query already prevents SQL
    injection, but rejecting bad UUIDs early avoids spurious DB round-trips
    and keeps the error message clear.
    """
    if not value:
        raise ValueError("session-id must not be empty")
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as e:
        raise ValueError(f"invalid uuid: {value!r}") from e
    return value


def redact_env(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``env`` with secret-bearing values replaced.

    A key is treated as a secret if any token in ``_SECRET_KEY_TOKENS``
    appears (case-insensitively) in its name. The original dict is not
    modified.
    """
    redacted: dict[str, str] = {}
    for key, val in env.items():
        upper = key.upper()
        if any(token in upper for token in _SECRET_KEY_TOKENS):
            redacted[key] = _REDACTED
        else:
            redacted[key] = val
    return redacted


def summarize_env_for_verbose(
    env: dict[str, str],
) -> tuple[dict[str, str], int]:
    """Project an env dict to the keys safe to enumerate in --verbose output.

    Returns a (visible, hidden_count) pair. Visible keys match an allowlist
    prefix and have their values redacted via :func:`redact_env`. Hidden keys
    are not enumerated — only their count is reported. This avoids leaking
    the full parent-process environment even when values are safe.
    """
    redacted = redact_env(env)
    visible: dict[str, str] = {}
    hidden = 0
    for key, val in redacted.items():
        if any(key.startswith(prefix) for prefix in _VERBOSE_ENV_PREFIX_ALLOWLIST):
            visible[key] = val
        else:
            hidden += 1
    return visible, hidden


def redact_long_tokens_for_dry_run(
    cmd: list[str], *, max_chars: int = _DRY_RUN_MAX_TOKEN_CHARS
) -> list[str]:
    """Replace argv tokens longer than ``max_chars`` with a length placeholder.

    The memory-extractor system prompt is multi-KB and would bury triage
    signal in ``--dry-run`` output. Tokens above the threshold are replaced
    with ``<token: N chars>``. The original list is not modified.
    """
    return [
        f"<token: {len(t)} chars>" if len(t) > max_chars else t
        for t in cmd
    ]


def load_agent_prompt() -> str:
    """Read the memory-extractor agent prompt from CLAUDE_CONFIG_DIR.

    Mirrors the daemon's behavior in ``extract_memories_impl``: read the
    file under ``CLAUDE_CONFIG_DIR/agents/memory-extractor.md``, strip YAML
    frontmatter if present, fall back to ``_FALLBACK_AGENT_PROMPT`` (kept
    in sync with the daemon's fallback).
    """
    config_dir = Path(
        os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))
    )
    agent_file = config_dir / "agents" / "memory-extractor.md"
    if agent_file.exists():
        return strip_yaml_frontmatter(agent_file.read_text())
    return _FALLBACK_AGENT_PROMPT


# ---------------------------------------------------------------------------
# I/O helpers (DB)
# ---------------------------------------------------------------------------


async def fetch_session_row(session_id: str) -> dict | None:
    """Fetch the sessions row for ``session_id``. Returns None if missing."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, project, transcript_path "
            "FROM sessions WHERE id = $1",
            session_id,
        )
    if row is None:
        return None
    return dict(row) if not isinstance(row, dict) else row


async def count_session_memories(session_id: str) -> int:
    """Count archival_memory rows currently attributed to ``session_id``."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        value = await conn.fetchval(
            "SELECT COUNT(*) FROM archival_memory WHERE session_id = $1",
            session_id,
        )
    return int(value or 0)


# ---------------------------------------------------------------------------
# Subprocess helpers (sync — invoked via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _stream_stderr(proc: subprocess.Popen) -> None:
    """Relay subprocess stderr to the parent's stderr line-by-line.

    Intended to run in a daemon thread so it does not block ``proc.wait``.
    Without ``text=True`` ``proc.stderr`` always yields ``bytes``, so the
    ``.decode`` call is straightforward — the previous AttributeError catch
    was dead code (PR #129 cycle-1 review, DESIGN 4).
    """
    if proc.stderr is None:
        return
    for raw in proc.stderr:
        sys.stderr.write(raw.decode("utf-8", errors="replace"))
        sys.stderr.flush()


def _spawn_and_collect_sync(
    cmd: list[str],
    env: dict[str, str],
    timeout: int | None,
) -> int:
    """Run the subprocess, drain stderr in a background thread, return exit.

    Returns the subprocess exit code, or ``124`` on timeout (matching the
    GNU coreutils convention used elsewhere in the daemon). Drains stderr
    in a daemon thread so ``proc.wait(timeout=...)`` is not blocked by a
    child that holds stderr open after producing output (PR #129 cycle-1
    review, BUG 2).

    All Popen state is held inside a ``with`` block for deterministic
    PIPE/handle cleanup even on the timeout path.
    """
    effective_timeout: int | None = timeout if timeout else None

    with subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    ) as proc:
        stderr_thread = threading.Thread(
            target=_stream_stderr,
            args=(proc,),
            daemon=True,
            name="extract-session-stderr",
        )
        stderr_thread.start()

        try:
            exit_code = proc.wait(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            stderr_thread.join(timeout=2)
            print(
                f"error: subprocess timed out after {effective_timeout}s; killed",
                file=sys.stderr,
            )
            return 124

        # Drain remaining stderr before exit so late lines are not lost.
        stderr_thread.join(timeout=5)
    return exit_code


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _resolve_model_and_turns(args: argparse.Namespace) -> tuple[str, int]:
    """Resolve model + max_turns from CLI args, falling back to config."""
    cfg = get_config()
    daemon_cfg = cfg.daemon
    model = args.model or daemon_cfg.extraction_model
    max_turns = (
        args.max_turns
        if args.max_turns is not None
        else daemon_cfg.extraction_max_turns
    )
    return model, max_turns


def _print_dry_run(
    cmd: list[str], env: dict[str, str], *, verbose: bool
) -> None:
    print("DRY RUN — subprocess will not be spawned")
    print("cmd:")
    for token in redact_long_tokens_for_dry_run(cmd):
        print(f"  {token}")
    if verbose:
        _print_verbose_env(env)
    else:
        marker = env.get("CLAUDE_MEMORY_EXTRACTION", "<unset>")
        print(f"env: CLAUDE_MEMORY_EXTRACTION={marker}")


def _print_verbose_env(env: dict[str, str]) -> None:
    """Print the verbose-mode env summary (allowlist + hidden count)."""
    visible, hidden = summarize_env_for_verbose(env)
    print(
        f"env: {len(visible)} visible (allowlisted), "
        f"{hidden} additional vars not enumerated. "
        f"Visible keys (secrets redacted):"
    )
    for key, val in sorted(visible.items()):
        print(f"  {key}={val}")


async def run_main(argv: list[str] | None = None) -> int:
    """Top-level orchestrator (async). Returns the process exit code.

    Driven by a single ``asyncio.run`` at the entry point so the cached
    asyncpg pool stays bound to a single live event loop. Subprocess
    spawning is delegated to :func:`_spawn_and_collect_sync` via
    :func:`asyncio.to_thread` so the loop is not blocked.
    """
    args = parse_args(argv)

    try:
        session_id = validate_session_id(args.session_id)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    model, max_turns = _resolve_model_and_turns(args)
    if not validate_extraction_model(model, _ALLOWED_EXTRACTION_MODELS):
        print(
            f"error: invalid model {model!r}; "
            f"must be one of {sorted(_ALLOWED_EXTRACTION_MODELS)}",
            file=sys.stderr,
        )
        return 1

    row = await fetch_session_row(session_id)
    if row is None:
        print(
            f"error: no session found with id {session_id}",
            file=sys.stderr,
        )
        return 1

    transcript_path = row.get("transcript_path")
    if not transcript_path:
        print(
            f"error: session {session_id} has no transcript_path; "
            f"cannot extract.",
            file=sys.stderr,
        )
        return 1

    transcript = Path(transcript_path)
    if not transcript.exists():
        print(
            f"error: transcript file missing on disk: {transcript_path}",
            file=sys.stderr,
        )
        return 1

    project_dir = row.get("project") or ""
    agent_prompt = load_agent_prompt()

    env = build_extraction_env(dict(os.environ), project_dir)
    cmd = build_extraction_command(
        session_id=session_id,
        jsonl_path=str(transcript),
        agent_prompt=agent_prompt,
        model=model,
        max_turns=max_turns,
    )

    if args.dry_run:
        _print_dry_run(cmd, env, verbose=args.verbose)
        return 0

    if args.verbose:
        _print_verbose_env(env)

    pre_count = await count_session_memories(session_id)
    print(
        f"starting extraction: session_id={session_id} "
        f"model={model} max_turns={max_turns} "
        f"timeout={args.timeout if args.timeout else 'none'}",
    )

    exit_code = await asyncio.to_thread(
        _spawn_and_collect_sync, cmd, env, args.timeout
    )

    post_count = await count_session_memories(session_id)
    delta = max(0, post_count - pre_count)
    print(
        f"subprocess exited with code {exit_code}; "
        f"new_memories={delta} (delta {pre_count} -> {post_count})",
    )
    return exit_code


def main() -> int:
    return asyncio.run(run_main(None))


if __name__ == "__main__":
    sys.exit(main())
