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
    value = _FLATTENED_HOME_RE.sub("", value)
    return PROJECT_ALIASES.get(value, value)


def resolve_project_for_store(
    explicit: str | None, *, env_project_dir: str,
) -> str | None:
    """Resolve the project value to store: explicit arg wins, else the
    basename of the project directory; both canonicalized."""
    if explicit:
        return canonicalize_project(explicit)
    basename = env_project_dir.rstrip("/").rsplit("/", 1)[-1]
    return canonicalize_project(basename)
