"""One-time migration: normalize archival_memory.project values (issue #130).

The #130 audit found 40 distinct project values fragmented by case
variants, flattened path artifacts, and alias pairs. This script collapses
them to the canonical forms defined by scripts.core.project_naming.

Dry-run by default; pass --apply to write. Sessions table is intentionally
out of scope (its values feed session bookkeeping, not recall).

Usage:
    uv run python scripts/migrations/normalize_project_values.py            # dry-run
    uv run python scripts/migrations/normalize_project_values.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Repo-pattern bootstrap for direct file invocation (memory_daemon.py:43-47)
project_dir = os.environ.get(
    "CLAUDE_PROJECT_DIR", str(Path(__file__).parent.parent.parent)
)
sys.path.insert(0, project_dir)

from scripts.core.project_naming import canonicalize_project  # noqa: E402


def build_normalization_plan(
    distinct_values: list[str],
) -> list[tuple[str, str]]:
    """Map each stored project value to its canonical form.

    Returns only the values that actually change, as (old, new) pairs,
    sorted for stable output. Pure function — testable without a DB.
    """
    plan: list[tuple[str, str]] = []
    for value in distinct_values:
        canonical = canonicalize_project(value)
        if canonical is not None and canonical != value:
            plan.append((value, canonical))
    return sorted(plan)


# Phase 2: align metadata->'project' with the (now canonical) column —
# rewriting stale keys AND creating missing ones (backfilled rows have a
# column value but no metadata key, and the reranker historically reads
# metadata['project']). Idempotent: re-running matches zero rows.
METADATA_SYNC_SQL = """
    UPDATE archival_memory
    SET metadata = jsonb_set(
        COALESCE(metadata, '{}'::jsonb), '{project}', to_jsonb(project)
    )
    WHERE project IS NOT NULL
      AND metadata->>'project' IS DISTINCT FROM project
"""

METADATA_SYNC_COUNT_SQL = """
    SELECT count(*) FROM archival_memory
    WHERE project IS NOT NULL
      AND metadata->>'project' IS DISTINCT FROM project
"""


async def run(apply: bool) -> int:
    """Execute the migration. Returns process exit code."""
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        # Single transaction: a failure mid-rewrite rolls back atomically
        # instead of leaving values half-normalized (aegis recommendation).
        rows = await conn.fetch(
            "SELECT DISTINCT project FROM archival_memory WHERE project IS NOT NULL"
        )
        plan = build_normalization_plan([r["project"] for r in rows])

        if plan:
            print(f"{'Applying' if apply else 'Dry-run:'} {len(plan)} value rewrites:")
            total = 0
            for old, new in plan:
                if apply:
                    result = await conn.execute(
                        "UPDATE archival_memory SET project = $1 WHERE project = $2",
                        new, old,
                    )
                    count = int(result.split()[-1])
                else:
                    count = await conn.fetchval(
                        "SELECT count(*) FROM archival_memory WHERE project = $1",
                        old,
                    )
                total += count
                print(f"  {old!r} -> {new!r}  ({count} rows)")
            print(f"{'Updated' if apply else 'Would update'} {total} rows total.")
        else:
            print("All project column values already canonical.")

        # Phase 2: metadata->'project' lockstep sync (see METADATA_SYNC_SQL)
        if apply:
            result = await conn.execute(METADATA_SYNC_SQL)
            meta_count = int(result.split()[-1])
            print(f"Synced metadata.project on {meta_count} rows.")
        else:
            meta_count = await conn.fetchval(METADATA_SYNC_COUNT_SQL)
            print(f"Would sync metadata.project on {meta_count} rows.")

        if not apply:
            print("Re-run with --apply to write changes.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes (default is dry-run)",
    )
    args = parser.parse_args()
    return asyncio.run(run(apply=args.apply))


if __name__ == "__main__":
    sys.exit(main())
