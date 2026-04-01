#!/usr/bin/env python3
"""Proactive memory push — surface high-value, never-recalled learnings.

Targets two pools of learnings that would otherwise never be seen:
1. Stale high-confidence learnings for the current project (recall_count=0)
2. Pattern representatives from anti_pattern / problem_solution clusters

Designed to be called by session-start-memory-push.ts hook at session start.

USAGE:
    # JSON output for hook consumption
    uv run python scripts/core/push_learnings.py --project opc --k 5 --json

    # Dry run (don't update recall_count)
    uv run python scripts/core/push_learnings.py --project opc --json --no-record

    # Custom truncation length
    uv run python scripts/core/push_learnings.py --project opc --json --max-chars 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def get_stale_learnings(
    project: str, k: int, *, conn: Any = None
) -> list[dict[str, Any]]:
    """Fetch never-recalled, high/medium confidence learnings for a project."""
    if conn is None:
        from scripts.core.db.postgres_pool import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            return await get_stale_learnings(project, k, conn=conn)
    rows = await conn.fetch(
        """
        SELECT id, content, metadata, created_at, recall_count
        FROM archival_memory
        WHERE metadata->>'type' = 'session_learning'
          AND superseded_by IS NULL
          AND recall_count = 0
          AND (metadata->>'confidence') IN ('high', 'medium')
          AND project = $1
        ORDER BY
          CASE metadata->>'confidence' WHEN 'high' THEN 0 ELSE 1 END,
          created_at DESC
        LIMIT $2
        """,
        project,
        k,
    )
    return [_row_to_dict(row) for row in rows]


async def get_pattern_representatives(
    k: int, *, conn: Any = None
) -> list[dict[str, Any]]:
    """Fetch never-recalled pattern representatives from high-value clusters."""
    if conn is None:
        from scripts.core.db.postgres_pool import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            return await get_pattern_representatives(k, conn=conn)
    rows = await conn.fetch(
        """
        SELECT a.id, a.content, a.metadata, a.created_at, a.recall_count,
               dp.pattern_type, dp.label AS pattern_label,
               dp.confidence AS pattern_confidence
        FROM detected_patterns dp
        JOIN archival_memory a ON a.id = dp.representative_id
        WHERE dp.superseded_at IS NULL
          AND dp.pattern_type IN ('anti_pattern', 'problem_solution')
          AND a.recall_count = 0
          AND a.superseded_by IS NULL
        ORDER BY dp.confidence DESC, a.created_at DESC
        LIMIT $1
        """,
        k,
    )
    results = []
    for row in rows:
        d = _row_to_dict(row)
        d["pattern_label"] = row.get("pattern_label")
        raw_conf = row.get("pattern_confidence")
        d["pattern_confidence"] = float(raw_conf) if raw_conf is not None else None
        results.append(d)
    return results


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an asyncpg Record to a plain dict."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return {
        "id": str(row["id"]),
        "content": row["content"],
        "metadata": metadata,
        "created_at": row["created_at"],
        "recall_count": row.get("recall_count", 0) or 0,
        "learning_type": metadata.get("learning_type", "UNKNOWN"),
        "confidence": metadata.get("confidence", "medium"),
        "pattern_label": None,
        "pattern_confidence": None,
    }


def merge_candidates(
    pattern_results: list[dict[str, Any]],
    stale_results: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Merge pattern reps (priority) with stale learnings, dedup by ID, cap at k."""
    seen_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for r in pattern_results + stale_results:
        if len(candidates) >= k:
            break
        rid = r["id"]
        if rid not in seen_ids:
            seen_ids.add(rid)
            candidates.append(r)
        if len(candidates) >= k:
            break
    return candidates


def truncate_content(content: str, max_chars: int) -> str:
    """Truncate content to max_chars, appending '...' if trimmed."""
    # Collapse to single line for compact display
    one_line = " ".join(
        line.strip() for line in content.split("\n") if line.strip()
    )
    if len(one_line) <= max_chars:
        return one_line
    return one_line[:max_chars] + "..."


def format_results(
    candidates: list[dict[str, Any]], project: str, max_chars: int
) -> dict[str, Any]:
    """Build the JSON output structure."""
    from scripts.core.recall_formatters import get_api_version

    results = []
    for c in candidates:
        results.append({
            "id": c["id"],
            "content": truncate_content(c["content"], max_chars),
            "learning_type": c["learning_type"],
            "confidence": c["confidence"],
            "pattern_label": c.get("pattern_label"),
            "created_at": (
                c["created_at"].isoformat()
                if hasattr(c["created_at"], "isoformat")
                else str(c["created_at"])
            ),
        })
    return {
        "version": get_api_version(),
        "push_source": "session_start",
        "project": project,
        "results": results,
    }


def write_cache_file(output: dict[str, Any]) -> None:
    """Write push results to .claude/cache/memory-push.json for compaction survival."""
    cache_dir = Path.home() / ".claude" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "memory-push.json"
    cache_file.write_text(json.dumps(output, default=str))


async def get_push_candidates(
    project: str, k: int = 5
) -> list[dict[str, Any]]:
    """Main entry: fetch and merge push candidates from PostgreSQL."""
    from scripts.core.db.postgres_pool import get_pool
    from scripts.core.recall_learnings import get_backend

    if get_backend() != "postgres":
        return []

    pool = await get_pool()
    async with pool.acquire() as conn:
        stale = await get_stale_learnings(project, k, conn=conn)
        try:
            patterns = await get_pattern_representatives(k, conn=conn)
        except Exception:
            logger.debug("detected_patterns table not available", exc_info=True)
            patterns = []

    return merge_candidates(patterns, stale, k)


async def main() -> int:
    """CLI entry point for push_learnings."""
    parser = argparse.ArgumentParser(
        description="Proactive memory push — surface never-recalled high-value learnings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project",
        default=Path(os.environ["CLAUDE_PROJECT_DIR"]).name if os.environ.get("CLAUDE_PROJECT_DIR") else None,
        help="Project name to filter by (default: auto-detect from CLAUDE_PROJECT_DIR)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Max number of learnings to push (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON (for hook consumption)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Don't update recall_count (dry run / testing)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=150,
        help="Max characters per learning content (default: 150)",
    )
    args = parser.parse_args()

    if not args.project:
        if args.json_output:
            print(json.dumps({"error": "No project specified", "results": []}))
        else:
            print("Error: --project required (or set CLAUDE_PROJECT_DIR)", file=sys.stderr)
        return 1

    try:
        candidates = await get_push_candidates(args.project, args.k)
    except Exception as e:
        if args.json_output:
            print(json.dumps({"error": str(e), "results": []}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    if not candidates:
        empty = {
            "push_source": "session_start",
            "project": args.project,
            "results": [],
        }
        # Always overwrite cache so stale data from previous runs isn't reused
        write_cache_file(empty)
        if args.json_output:
            print(json.dumps(empty))
        return 0

    # Record recall to break the death spiral (unless dry run)
    if not args.no_record:
        from scripts.core.recall_learnings import record_recall
        await record_recall([c["id"] for c in candidates])

    output = format_results(candidates, args.project, args.max_chars)

    # Write cache file for compaction survival
    write_cache_file(output)

    if args.json_output:
        print(json.dumps(output, default=str))
    else:
        print(f"Pushing {len(candidates)} learnings for project '{args.project}':")
        for i, c in enumerate(candidates, 1):
            label = f" (Pattern: {c['pattern_label']})" if c.get("pattern_label") else ""
            print(f"  {i}. [{c['learning_type']}|{c['confidence']}] "
                  f"{truncate_content(c['content'], args.max_chars)}{label}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
