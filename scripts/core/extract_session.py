#!/usr/bin/env python3
"""Single-session memory extraction CLI (issue #128).

Run the memory-daemon's extraction pipeline against ONE session for testing
or debugging, without involving the daemon scheduler.

Usage:
    uv run python scripts/core/extract_session.py --session-id <uuid>

The CLI reuses ``build_extraction_env`` and ``build_extraction_command`` from
``scripts.core.memory_daemon_core`` unmodified — wire-compatibility with the
daemon path is the whole point of this tool. The daemon's lifecycle bookkeeping
(``backfill_extracted_at`` etc.) is intentionally untouched: this CLI does not
mark sessions extracted, retry on failure, or persist state. It just runs the
subprocess once, streams its stderr, and reports the delta.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
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

# Conservative redaction list — substring match (case-insensitive). Anything
# whose key contains one of these tokens is redacted in --verbose output.
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

_FALLBACK_AGENT_PROMPT = (
    "Extract learnings from this Claude Code session.\n"
    "Look for decisions, what worked, what failed, and patterns discovered.\n"
    "Store each learning using store_learning.py with appropriate type and tags."
)


# ---------------------------------------------------------------------------
# Pure helpers
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
        raise argparse.ArgumentTypeError(
            f"value must be >= 1, got {n}"
        )
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
        raise argparse.ArgumentTypeError(
            f"value must be >= 0, got {n}"
        )
    return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Pure aside from argparse error handling."""
    parser = argparse.ArgumentParser(
        prog="extract_session",
        description=(
            "Run the memory-daemon extraction pipeline against a single "
            "session. For testing/debugging only — does not modify session "
            "lifecycle state."
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
        "--no-mark-extracted",
        action="store_true",
        default=True,
        help=(
            "Do not mark the session as extracted. This is the default and "
            "the only supported mode — the daemon owns lifecycle state. The "
            "flag exists to make the intent explicit on the command line."
        ),
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
        help="Print env vars exported to the subprocess (secrets redacted)",
    )
    return parser.parse_args(argv)


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


def load_agent_prompt() -> str:
    """Read the memory-extractor agent prompt from CLAUDE_CONFIG_DIR.

    Mirrors the daemon's behavior in ``extract_memories_impl``: read the
    file under ``CLAUDE_CONFIG_DIR/agents/memory-extractor.md``, strip YAML
    frontmatter if present, fall back to a built-in prompt otherwise.
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
    # asyncpg.Record supports dict-like access; coerce for plain-dict tests.
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
# Orchestrator
# ---------------------------------------------------------------------------


def _resolve_model_and_turns(
    args: argparse.Namespace,
) -> tuple[str, int]:
    """Resolve model + max_turns from CLI args, falling back to config."""
    cfg = get_config()
    daemon_cfg = cfg.daemon
    model = args.model or daemon_cfg.extraction_model
    max_turns = args.max_turns if args.max_turns is not None else (
        daemon_cfg.extraction_max_turns
    )
    return model, max_turns


def _print_dry_run(
    cmd: list[str], env: dict[str, str], *, verbose: bool
) -> None:
    print("DRY RUN — subprocess will not be spawned")
    print("cmd:")
    for token in cmd:
        print(f"  {token}")
    if verbose:
        print("env (secrets redacted):")
        for key, val in sorted(redact_env(env).items()):
            print(f"  {key}={val}")
    else:
        # Always show the daemon-set extraction marker so the operator can
        # confirm wiring without --verbose.
        marker = env.get("CLAUDE_MEMORY_EXTRACTION", "<unset>")
        print(f"env: CLAUDE_MEMORY_EXTRACTION={marker}")


def _stream_stderr(proc: subprocess.Popen) -> None:
    """Relay subprocess stderr to the parent's stderr line-by-line."""
    if proc.stderr is None:
        return
    for raw in proc.stderr:
        try:
            line = raw.decode("utf-8", errors="replace")
        except AttributeError:
            line = str(raw)
        sys.stderr.write(line)
        sys.stderr.flush()


def run_main(argv: list[str] | None = None) -> int:
    """Top-level orchestrator. Returns the process exit code."""
    args = parse_args(argv)

    try:
        session_id = validate_session_id(args.session_id)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # Resolve and validate model BEFORE any DB query so a typo fails fast.
    model, max_turns = _resolve_model_and_turns(args)
    if not validate_extraction_model(model, _ALLOWED_EXTRACTION_MODELS):
        print(
            f"error: invalid model {model!r}; "
            f"must be one of {sorted(_ALLOWED_EXTRACTION_MODELS)}",
            file=sys.stderr,
        )
        return 1

    row = asyncio.run(fetch_session_row(session_id))
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
        print("env (secrets redacted):")
        for key, val in sorted(redact_env(env).items()):
            print(f"  {key}={val}")

    pre_count = asyncio.run(count_session_memories(session_id))
    print(
        f"starting extraction: session_id={session_id} "
        f"model={model} max_turns={max_turns} "
        f"timeout={args.timeout if args.timeout else 'none'}",
    )

    return _spawn_and_collect(
        cmd, env, args.timeout, session_id, pre_count
    )


def _spawn_and_collect(
    cmd: list[str],
    env: dict[str, str],
    timeout: int | None,
    session_id: str,
    pre_count: int,
) -> int:
    """Spawn the subprocess, stream stderr, then report exit + delta count.

    The Popen handle is held inside a ``with`` block so its PIPEs and the
    process itself are released deterministically on every exit path, even
    when the kill-on-timeout branch fires. The inner ``wait(timeout=5)``
    after kill is retained — the ``with`` is belt-and-suspenders cleanup
    on top, not a replacement.
    """
    # Treat timeout == 0 as "no timeout", consistent with --help wording.
    effective_timeout: int | None = timeout if timeout else None

    with subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    ) as proc:
        # Drain stderr in the foreground. For an interactive single-session
        # tool a serial drain is sufficient — we are not multiplexing
        # multiple extractions like the daemon does.
        _stream_stderr(proc)

        try:
            exit_code = proc.wait(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            print(
                f"error: subprocess timed out after {effective_timeout}s; killed",
                file=sys.stderr,
            )
            return 124

    post_count = asyncio.run(count_session_memories(session_id))
    delta = max(0, post_count - pre_count)
    print(
        f"subprocess exited with code {exit_code}; "
        f"new_memories={delta} (delta {pre_count} -> {post_count})",
    )
    return exit_code


def main() -> int:
    return run_main(None)


if __name__ == "__main__":
    sys.exit(main())
