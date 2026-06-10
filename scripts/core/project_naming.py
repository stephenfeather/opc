"""Canonical project naming for the memory system (issue #130, audit fix 3).

The archival_memory.project column accumulated 40 fragmented values: case
variants, flattened path artifacts, and alias pairs. This module is the
single source of truth for project canonicalization, applied at:

- store time (store_learning.py)
- recall-context time (recall_learnings.make_recall_context)
- one-time data migration (scripts/migrations/normalize_project_values.py)

Pure functions only — no I/O.
"""

from __future__ import annotations

import re
from pathlib import PurePath

# The daemon's explicit "could not resolve" sentinel. Preserved verbatim so
# unresolved rows stay identifiable instead of masquerading as a project.
UNRESOLVED_SENTINEL = "_unresolved"

# Flattened absolute paths leak in as project values when a path with '/'
# replaced by '-' is stored wholesale (e.g. Claude project-dir encoding).
_FLATTENED_HOME_RE = re.compile(r"^-users-[^-]+-", re.IGNORECASE)

# Known alias -> canonical collapses, applied AFTER lowercasing. Every
# target must be a fixed point of canonicalize_project (tested). Keys come
# from the issue #130 audit of distinct archival_memory.project values;
# extend deliberately, never programmatically.
PROJECT_ALIASES: dict[str, str] = {
    # ~/Operations/DigitalOcean — basename is the canonical name
    "operations-digitalocean": "digitalocean",
    # ~/Development/2026-calebs-hospital — year prefix is part of the dir
    "calebs-hospital": "2026-calebs-hospital",
}


def canonicalize_project(raw: str | None) -> str | None:
    """Normalize a raw project value to its canonical form.

    Lowercases, strips whitespace and flattened-home-path prefixes, then
    collapses known aliases. Returns None for empty input. Idempotent.
    """
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if value == UNRESOLVED_SENTINEL:
        return value
    value = value.lower()
    value = _FLATTENED_HOME_RE.sub("", value).strip("/").strip()
    if not value:
        # e.g. '-users-bob-' strips to nothing, or a bare '/' — an empty
        # string would violate the fixed-point contract (aegis MEDIUM-1).
        return None
    return PROJECT_ALIASES.get(value, value)


def project_from_path(path_str: str | None) -> str | None:
    """Derive the canonical project name from a filesystem path.

    Worktree-aware: '/repo/.worktrees/branch' and
    '/repo/.claude/worktrees/branch' both resolve to 'repo', so writes
    from worktree sessions don't fragment project values (issue #130).
    """
    if not path_str:
        return None
    parts = PurePath(path_str).parts
    if ".worktrees" in parts:
        idx = parts.index(".worktrees")
        name = parts[idx - 1] if idx > 0 else parts[-1]
        return canonicalize_project(name)
    if "worktrees" in parts:
        idx = parts.index("worktrees")
        if idx > 0 and parts[idx - 1] == ".claude":
            name = parts[idx - 2] if idx > 1 else parts[-1]
            return canonicalize_project(name)
    return canonicalize_project(parts[-1] if parts else None)


def resolve_project_for_store(
    explicit: str | None, *, env_project_dir: str,
) -> str | None:
    """Resolve the project value to store: explicit arg wins, else the
    project directory (worktree-aware); both canonicalized."""
    if explicit:
        return canonicalize_project(explicit)
    return project_from_path(env_project_dir or None)
