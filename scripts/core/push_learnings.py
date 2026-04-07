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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import asyncpg

from scripts.core.db.postgres_pool import get_pool
from scripts.core.recall_formatters import get_api_version
from scripts.core.recall_learnings import get_backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL query builders (pure)
# ---------------------------------------------------------------------------


def build_stale_query_params(project: str, k: int) -> tuple[str, list[Any]]:
    """Build SQL and params for fetching stale learnings."""
    sql = """
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
    """
    return sql, [project, k]


def build_pattern_query_params(k: int) -> tuple[str, list[Any]]:
    """Build SQL and params for fetching pattern representatives."""
    sql = """
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
    """
    return sql, [k]


# ---------------------------------------------------------------------------
# Row converters (pure)
# ---------------------------------------------------------------------------


def _row_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an asyncpg Record to a plain dict."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    if not isinstance(metadata, Mapping):
        metadata = {}
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


def parse_pattern_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a pattern query row to a candidate dict with pattern fields."""
    d = _row_to_dict(row)
    d["pattern_label"] = row.get("pattern_label")
    raw_conf = row.get("pattern_confidence")
    d["pattern_confidence"] = float(raw_conf) if raw_conf is not None else None
    return d


# ---------------------------------------------------------------------------
# Pure domain functions
# ---------------------------------------------------------------------------


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
    return candidates


def truncate_content(content: str, max_chars: int) -> str:
    """Truncate content to max_chars, appending '...' if trimmed."""
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
    results = [
        {
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
        }
        for c in candidates
    ]
    return {
        "version": get_api_version(),
        "push_source": "session_start",
        "project": project,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI helpers (pure)
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse CLI arguments into a config dict."""
    parser = argparse.ArgumentParser(
        description="Proactive memory push — surface never-recalled high-value learnings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project",
        default=(
            Path(os.environ["CLAUDE_PROJECT_DIR"]).name
            if os.environ.get("CLAUDE_PROJECT_DIR")
            else None
        ),
        help="Project name to filter by (default: auto-detect from CLAUDE_PROJECT_DIR)",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Max number of learnings to push (default: 5)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON (for hook consumption)",
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="Don't update recall_count (dry run / testing)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=150,
        help="Max characters per learning content (default: 150)",
    )
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1")
    if args.max_chars < 1:
        parser.error("--max-chars must be >= 1")
    return {
        "project": args.project,
        "k": args.k,
        "json_output": args.json_output,
        "no_record": args.no_record,
        "max_chars": args.max_chars,
    }


def build_cli_output(
    candidates: list[dict[str, Any]],
    project: str,
    *,
    max_chars: int,
    json_output: bool,
) -> str:
    """Build CLI output string from candidates (JSON or text)."""
    if not candidates:
        if json_output:
            empty = {"push_source": "session_start", "project": project, "results": []}
            return json.dumps(empty)
        return ""

    output = format_results(candidates, project, max_chars)
    if json_output:
        return json.dumps(output, default=str)

    lines = [f"Pushing {len(candidates)} learnings for project '{project}':"]
    for i, c in enumerate(candidates, 1):
        label = f" (Pattern: {c['pattern_label']})" if c.get("pattern_label") else ""
        lines.append(
            f"  {i}. [{c['learning_type']}|{c['confidence']}] "
            f"{truncate_content(c['content'], max_chars)}{label}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O boundary
# ---------------------------------------------------------------------------


def write_cache_file(output: dict[str, Any]) -> None:
    """Write push results to .claude/cache/memory-push.json for compaction survival."""
    cache_dir = Path.home() / ".claude" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "memory-push.json"
    cache_file.write_text(json.dumps(output, default=str))


async def get_stale_learnings(
    project: str, k: int, *, conn: Any = None
) -> list[dict[str, Any]]:
    """Fetch never-recalled, high/medium confidence learnings for a project."""
    if conn is None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await get_stale_learnings(project, k, conn=conn)
    sql, params = build_stale_query_params(project, k)
    rows = await conn.fetch(sql, *params)
    return [_row_to_dict(row) for row in rows]


async def get_pattern_representatives(
    k: int, *, conn: Any = None
) -> list[dict[str, Any]]:
    """Fetch never-recalled pattern representatives from high-value clusters."""
    if conn is None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await get_pattern_representatives(k, conn=conn)
    sql, params = build_pattern_query_params(k)
    rows = await conn.fetch(sql, *params)
    return [parse_pattern_row(row) for row in rows]


async def get_push_candidates(
    project: str, k: int = 5
) -> list[dict[str, Any]]:
    """Main entry: fetch and merge push candidates from PostgreSQL."""
    if get_backend() != "postgres":
        return []

    pool = await get_pool()
    async with pool.acquire() as conn:
        stale = await get_stale_learnings(project, k, conn=conn)
        try:
            patterns = await get_pattern_representatives(k, conn=conn)
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("detected_patterns table not available", exc_info=True)
            patterns = []

    return merge_candidates(patterns, stale, k)


async def main() -> int:
    """CLI entry point for push_learnings."""
    config = parse_args()
    project = config["project"]
    json_output = config["json_output"]

    if not project:
        if json_output:
            print(json.dumps({"error": "No project specified", "results": []}))
        else:
            print("Error: --project required (or set CLAUDE_PROJECT_DIR)", file=sys.stderr)
        return 1

    try:
        candidates = await get_push_candidates(project, config["k"])
    except Exception as e:
        logger.debug("push_learnings error", exc_info=True)
        error_envelope: dict[str, Any] = {
            "error": type(e).__name__, "push_source": "session_start",
            "project": project, "results": [],
        }
        try:
            write_cache_file(error_envelope)
        except OSError:
            logger.debug("Failed to write error cache file", exc_info=True)
        if json_output:
            print(json.dumps(error_envelope))
        else:
            print(f"Error: {type(e).__name__}: see logs for details", file=sys.stderr)
        return 1

    # Record recall BEFORE writing cache to prevent duplicate pushes on crash.
    # If record_recall succeeds but cache write fails, worst case is a missed
    # push (safe). If cache writes first but record_recall crashes, the same
    # learnings get re-pushed indefinitely (the death spiral this script prevents).
    if candidates and not config["no_record"]:
        from scripts.core.recall_learnings import record_recall
        await record_recall([c["id"] for c in candidates])

    # Build and persist cache data (always, even when empty)
    if not candidates:
        cache_data: dict[str, Any] = {
            "push_source": "session_start", "project": project, "results": [],
        }
    else:
        cache_data = format_results(candidates, project, config["max_chars"])

    try:
        write_cache_file(cache_data)
    except OSError:
        logger.debug("Failed to write cache file", exc_info=True)

    output_str = build_cli_output(
        candidates, project,
        max_chars=config["max_chars"], json_output=json_output,
    )
    if output_str:
        print(output_str)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
