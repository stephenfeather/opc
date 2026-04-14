#!/usr/bin/env python3
"""Backfill NULL project values in archival_memory.

Three-tier approach:
  Tier 1: Cross-reference sessions table (covers ~86% of NULLs)
  Tier 2: Infer from session_id naming patterns and tags
  Tier 3: Flag remaining as '_unresolved'

Usage:
    uv run python scripts/migrations/backfill_project_column.py --dry-run
    uv run python scripts/migrations/backfill_project_column.py --tier 1
    uv run python scripts/migrations/backfill_project_column.py --tier all
    uv run python scripts/migrations/backfill_project_column.py --tier all --verbose
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg2

# Path normalization: map full session project paths to short names
PATH_TO_PROJECT: dict[str, str] = {
    "opc": "opc",
    "Continuous-Claude-v3/opc": "opc",
    "binbrain-ios": "binbrain-ios",
    "binbrain": "binbrain",
    "Pharmacokinetics-Grapher": "pharmacokinetics-grapher",
    "agentic-work": "agentic-work",
    "fa-wpmcp": "fa-wpmcp",
    "fa-wpmcp-integration-tests": "fa-wpmcp",
    "sermon-browser": "sermon-browser",
    "opc-memory-mcp": "opc-memory-mcp",
    "ai-agents": "ai-agents",
    ".dotfiles/claude": "opc",  # OPC infrastructure
}

# Session ID prefix patterns → project (for Tier 2 orphans)
SESSION_PREFIX_TO_PROJECT: dict[str, str] = {
    "binbrain": "binbrain",
    "sermon-browser": "sermon-browser",
    "3d-print-logger": "3d-print-logger",
    "fa-toolkit": "fa-toolkit",
    "fa-wpmcp": "fa-wpmcp",
    "inventory-feeds": "inventory-feeds",
    "featherarms": "featherarms",
    "pharmacokinetics": "pharmacokinetics-grapher",
    "pk-grapher": "pharmacokinetics-grapher",
    "ai-docs-indexer": "ai-docs-indexer",
    "ai-agents": "ai-agents",
    "agentic-work": "agentic-work",
    "calebs-hospital": "calebs-hospital",
    "2026-calebs-hospital": "calebs-hospital",
    "catatonia": "calebs-hospital",
    "xcopri": "calebs-hospital",
    "wp-cli": "fa-wpmcp",
    "wordpress": "fa-wpmcp",
    "testing-wp": "fa-wpmcp",
    "wp69": "fa-wpmcp",
    "docker-healthcheck": "featherarms",
    "nginx-alpine": "featherarms",
    "tauri": "binbrain",
    "task-4-multiDose": "pharmacokinetics-grapher",
    "task-5-localStorage": "pharmacokinetics-grapher",
    "validate-task-3-pk": "pharmacokinetics-grapher",
    "task-5-plan-validation": "pharmacokinetics-grapher",
}

# Tags that strongly signal a project (Tier 2 fallback)
TAG_TO_PROJECT: list[tuple[set[str], str]] = [
    # Most specific first
    ({"3d-print-logger"}, "3d-print-logger"),
    ({"fa-toolkit"}, "fa-toolkit"),
    ({"xctest", "swift"}, "binbrain-ios"),
    ({"ios", "swift"}, "binbrain-ios"),
    ({"binbrain"}, "binbrain"),
    ({"pharmacokinetics"}, "pharmacokinetics-grapher"),
    ({"alexa"}, "alexa-skill"),
    ({"skill-builder"}, "alexa-skill"),
    ({"sermon"}, "sermon-browser"),
]

# Session IDs that map to OPC (infrastructure/tooling sessions)
OPC_SESSION_PATTERNS = {
    "braintrust",
    "btql",
    "cross-session-patterns",
    "learning-classification",
    "crash-recovery",
    "daemon",
    "dedup",
    "mcp-integration",
    "mcp-session",
    "mcp-test",
    "mcp-type",
    "mcp-comment",
    "mcp-response",
    "opengrep",
    "fork-to-spawn",
    "gemini-model",
    "git-worktree",
    "dotfiles",
    "cache-mcp",
    "fix-daemon",
    "ability-",
    "user-preferences",
    "user-fact",
    "production-debug",
    "blog-technical",
    "quickstart",
    "validate-agent",
    "hook",
}


def normalize_session_path(path: str) -> str | None:
    """Normalize a sessions.project path to a short project name."""
    if not path:
        return None

    p = Path(path)
    parts = p.parts

    # Try matching known suffixes (most specific first)
    path_str = str(p)
    for suffix, project in sorted(PATH_TO_PROJECT.items(), key=lambda x: -len(x[0])):
        if path_str.endswith(suffix):
            return project

    # Ambiguous top-level paths
    if path_str in ("/Users/stephenfeather", "/Users/stephenfeather/Development"):
        return "_ambiguous"

    # Default: lowercase basename
    return parts[-1].lower() if parts else None


def infer_from_session_id(session_id: str) -> str | None:
    """Infer project from session_id naming convention."""
    sid_lower = session_id.lower()

    # Check OPC patterns
    for pattern in OPC_SESSION_PATTERNS:
        if sid_lower.startswith(pattern):
            return "opc"

    # Check prefix mappings (longest prefix first)
    for prefix, project in sorted(
        SESSION_PREFIX_TO_PROJECT.items(), key=lambda x: -len(x[0])
    ):
        if sid_lower.startswith(prefix):
            return project

    # Manual/generic session IDs — can't infer
    if sid_lower.startswith("manual-") or len(session_id) == 8:
        return None

    return None


def infer_from_tags(tags: list[str]) -> str | None:
    """Infer project from tag combinations."""
    tag_set = set(t.lower() for t in tags)
    for required_tags, project in TAG_TO_PROJECT:
        if required_tags.issubset(tag_set):
            return project
    return None


def get_connection() -> psycopg2.extensions.connection:
    """Get database connection using env-driven URL (#62)."""
    db_url = (
        os.environ.get("CONTINUOUS_CLAUDE_DB_URL")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("OPC_POSTGRES_URL")
    )
    if not db_url:
        raise ValueError(
            "Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL (preferred), "
            "DATABASE_URL, or OPC_POSTGRES_URL. "
            "For local Docker dev, run "
            "`docker compose -f docker/docker-compose.yml up -d` "
            "and export the credentials from docker/.env before "
            "invoking this migration."
        )
    return psycopg2.connect(db_url)


def show_current_state(conn: psycopg2.extensions.connection) -> None:
    """Print current project distribution."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(project, 'NULL') AS proj, COUNT(*) "
            "FROM archival_memory GROUP BY project ORDER BY COUNT(*) DESC"
        )
        rows = cur.fetchall()

    print("\n=== Current project distribution ===")
    total = sum(r[1] for r in rows)
    for proj, count in rows:
        pct = count / total * 100
        print(f"  {proj:30s} {count:5d}  ({pct:.1f}%)")
    print(f"  {'TOTAL':30s} {total:5d}")


def run_tier1(
    conn: psycopg2.extensions.connection, dry_run: bool, verbose: bool
) -> int:
    """Tier 1: Backfill from sessions table join."""
    print("\n--- Tier 1: Sessions table cross-reference ---")

    with conn.cursor() as cur:
        # Get all NULL-project learnings that have a session match
        cur.execute("""
            SELECT am.id, am.session_id, s.project AS session_project
            FROM archival_memory am
            JOIN sessions s ON am.session_id = s.id
            WHERE am.project IS NULL
              AND s.project IS NOT NULL
        """)
        rows = cur.fetchall()

    if not rows:
        print("  No learnings to backfill via sessions table.")
        return 0

    # Build update map
    updates: dict[str, list[str]] = {}  # project_name -> [ids]
    skipped = 0
    for am_id, session_id, session_project in rows:
        project = normalize_session_path(session_project)
        if project and project != "_ambiguous":
            updates.setdefault(project, []).append(am_id)
        else:
            skipped += 1
            if verbose:
                print(f"  SKIP {am_id[:8]}  session={session_id}  path={session_project}")

    total_updates = sum(len(ids) for ids in updates.values())
    print(f"  Found {total_updates} learnings to update ({skipped} skipped/ambiguous)")

    for project, ids in sorted(updates.items(), key=lambda x: -len(x[1])):
        print(f"    {project:30s} {len(ids):5d}")

    if dry_run:
        print("  [DRY RUN] No changes applied.")
        return total_updates

    with conn.cursor() as cur:
        for project, ids in updates.items():
            cur.execute(
                "UPDATE archival_memory SET project = %s"
                " WHERE id = ANY(%s::uuid[]) AND project IS NULL",
                (project, ids),
            )
            if verbose:
                print(f"  Updated {cur.rowcount} rows → {project}")
    conn.commit()
    print(f"  Applied {total_updates} updates.")
    return total_updates


def run_tier2(
    conn: psycopg2.extensions.connection, dry_run: bool, verbose: bool
) -> int:
    """Tier 2: Infer from session_id patterns and tags."""
    print("\n--- Tier 2: Session ID + tag heuristics ---")

    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, session_id, metadata->'tags' AS tags_json
            FROM archival_memory
            WHERE project IS NULL
        """)
        rows = cur.fetchall()

    if not rows:
        print("  No remaining NULL-project learnings.")
        return 0

    updates: dict[str, list[str]] = {}
    unresolved = []

    for am_id, session_id, tags_json in rows:
        tags = []
        if tags_json:
            try:
                import json
                tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
            except (json.JSONDecodeError, TypeError):
                pass

        # Try session_id inference first
        project = infer_from_session_id(session_id)

        # Fall back to tag inference
        if not project:
            project = infer_from_tags(tags)

        if project:
            updates.setdefault(project, []).append(am_id)
            if verbose:
                print(f"  MATCH {am_id[:8]}  session={session_id}  → {project}")
        else:
            unresolved.append((am_id, session_id))
            if verbose:
                print(f"  UNRESOLVED {am_id[:8]}  session={session_id}  tags={tags[:3]}")

    total_updates = sum(len(ids) for ids in updates.values())
    print(f"  Found {total_updates} learnings to update ({len(unresolved)} unresolved)")

    for project, ids in sorted(updates.items(), key=lambda x: -len(x[1])):
        print(f"    {project:30s} {len(ids):5d}")

    if dry_run:
        print("  [DRY RUN] No changes applied.")
        if unresolved:
            print(f"\n  Unresolved session_ids ({len(unresolved)}):")
            for am_id, sid in unresolved:
                print(f"    {sid}")
        return total_updates

    with conn.cursor() as cur:
        for project, ids in updates.items():
            cur.execute(
                "UPDATE archival_memory SET project = %s"
                " WHERE id = ANY(%s::uuid[]) AND project IS NULL",
                (project, ids),
            )
    conn.commit()
    print(f"  Applied {total_updates} updates.")
    return total_updates


def run_tier3(
    conn: psycopg2.extensions.connection, dry_run: bool, verbose: bool
) -> int:
    """Tier 3: Flag remaining NULLs as _unresolved."""
    print("\n--- Tier 3: Flag remaining as _unresolved ---")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM archival_memory WHERE project IS NULL")
        count = cur.fetchone()[0]

    if count == 0:
        print("  No remaining NULL-project learnings.")
        return 0

    print(f"  {count} learnings will be flagged as '_unresolved'")

    if dry_run:
        print("  [DRY RUN] No changes applied.")
        return count

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE archival_memory SET project = '_unresolved' WHERE project IS NULL"
        )
        updated = cur.rowcount
    conn.commit()
    print(f"  Flagged {updated} learnings as '_unresolved'.")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill project column in archival_memory")
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which tier(s) to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying the database",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-row decisions",
    )
    args = parser.parse_args()

    conn = get_connection()
    try:
        show_current_state(conn)

        tiers = args.tier
        total = 0

        if tiers in ("1", "all"):
            total += run_tier1(conn, args.dry_run, args.verbose)

        if tiers in ("2", "all"):
            total += run_tier2(conn, args.dry_run, args.verbose)

        if tiers in ("3", "all"):
            total += run_tier3(conn, args.dry_run, args.verbose)

        show_current_state(conn)

        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Total: {total} learnings processed.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
