"""Pure functions and data types for the memory daemon.

All functions in this module are side-effect-free: they take data in
and return new data out without I/O, mutation, or global state.

Where I/O is needed (e.g., process-alive checks), it is injected as
a predicate parameter — the core function itself remains pure.
"""

from __future__ import annotations

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
    project: str
    transcript_path: str | None
    pid: int | None
    exited_at: object  # datetime | str | None — varies by backend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Allowlist of Claude models permitted for extraction subprocesses.
_ALLOWED_EXTRACTION_MODELS: frozenset[str] = frozenset({"sonnet", "haiku", "opus"})


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
    return [
        "claude", "-p",
        "--model", model,
        "--dangerously-skip-permissions",
        "--allowedTools", "Bash,Read",
        "--max-turns", str(max_turns),
        "--append-system-prompt", agent_prompt,
        f"Extract learnings from session {session_id}. JSONL path: {jsonl_path}",
    ]


def build_extraction_env(base_env: dict, project_dir: str | None) -> dict:
    """Build environment dict for extraction subprocess.

    Returns a new dict — does not mutate base_env.
    """
    env = dict(base_env)
    env["CLAUDE_MEMORY_EXTRACTION"] = "1"
    if project_dir:
        env["CLAUDE_PROJECT_DIR"] = project_dir
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
    return jsonl_path.with_suffix(".jsonl.zst")


def filter_truly_stale_sessions(
    sessions: list[StaleSession],
    *,
    is_alive: Callable[[int | None], bool],
) -> tuple[list[StaleSession], list[str], list[str]]:
    """Partition stale sessions into three categories.

    Returns (truly_stale, newly_dead_ids, still_alive_ids).

    - truly_stale: dead process, exited_at already set (past grace period)
    - newly_dead_ids: dead process, exited_at is None (first discovery)
    - still_alive_ids: process is still running

    The is_alive predicate is injected — this function performs no I/O.
    """
    truly_stale: list[StaleSession] = []
    newly_dead_ids: list[str] = []
    still_alive_ids: list[str] = []

    for session in sessions:
        if is_alive(session.pid):
            still_alive_ids.append(session.id)
        elif session.exited_at is None:
            newly_dead_ids.append(session.id)
        else:
            truly_stale.append(session)

    return truly_stale, newly_dead_ids, still_alive_ids
