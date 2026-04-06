#!/usr/bin/env python3
"""Store and query feedback on memory learning usefulness.

USAGE:
    # Store feedback
    uv run python scripts/core/memory_feedback.py store \
        --learning-id <uuid> --helpful --session-id <sid>

    # Store negative feedback
    uv run python scripts/core/memory_feedback.py store \
        --learning-id <uuid> --not-helpful --session-id <sid>

    # Get feedback summary (JSON)
    uv run python scripts/core/memory_feedback.py summary

    # Get feedback for a specific learning
    uv run python scripts/core/memory_feedback.py get --learning-id <uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv

# Bootstrap: load env vars and ensure scripts.core imports resolve.
_global_env = Path.home() / ".claude" / ".env"
if _global_env.exists():
    load_dotenv(_global_env)
load_dotenv()

_repo_root = str(Path(__file__).parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.core.db.postgres_pool import close_pool, get_connection  # noqa: E402

# ---------------------------------------------------------------------------
# Pure functions — row formatting
# ---------------------------------------------------------------------------


def format_feedback_row(row: dict[str, Any]) -> dict[str, Any]:
    """Transform a single DB row into a serializable feedback dict."""
    return {
        "id": str(row["id"]),
        "session_id": row["session_id"],
        "helpful": row["helpful"],
        "context": row["context"],
        "source": row["source"],
        "created_at": row["created_at"].isoformat(),
    }


def aggregate_feedback(
    rows: list[dict[str, Any]], learning_id: str
) -> dict[str, Any]:
    """Aggregate raw feedback rows into a summary for one learning."""
    feedback = [format_feedback_row(r) for r in rows]
    helpful_count = sum(1 for f in feedback if f["helpful"])
    return {
        "learning_id": learning_id,
        "total_feedback": len(feedback),
        "helpful_count": helpful_count,
        "not_helpful_count": len(feedback) - helpful_count,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Pure functions — summary formatting
# ---------------------------------------------------------------------------


def compute_helpfulness_rate(total: int, helpful: int) -> float:
    """Compute helpfulness percentage, returning 0.0 when total is zero."""
    if total == 0:
        return 0.0
    return round(helpful / total * 100, 1)


def format_top_entries(
    rows: list[dict[str, Any]], count_key: str
) -> list[dict[str, Any]]:
    """Format top-N learning rows, truncating content to 120 chars."""
    return [
        {
            "learning_id": str(r["learning_id"]),
            "content": r["content"][:120],
            count_key: r[count_key],
        }
        for r in rows
    ]


def format_summary_result(
    totals: dict[str, Any],
    top_helpful_rows: list[dict[str, Any]],
    top_not_helpful_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the feedback summary dict from query results."""
    total = totals["total"]
    helpful = totals["helpful"]
    return {
        "total_feedback": total,
        "helpful_count": helpful,
        "not_helpful_count": totals["not_helpful"],
        "unique_learnings_rated": totals["unique_learnings"],
        "helpfulness_rate": compute_helpfulness_rate(total, helpful),
        "top_helpful": format_top_entries(top_helpful_rows, "helpful_count"),
        "top_not_helpful": format_top_entries(top_not_helpful_rows, "not_helpful_count"),
    }


def empty_summary() -> dict[str, Any]:
    """Return a fresh empty summary dict (no shared mutable state)."""
    return {
        "total_feedback": 0,
        "helpful_count": 0,
        "not_helpful_count": 0,
        "unique_learnings_rated": 0,
        "helpfulness_rate": 0.0,
        "top_helpful": [],
        "top_not_helpful": [],
    }


# ---------------------------------------------------------------------------
# I/O wrappers — thin async functions delegating to pure logic
# ---------------------------------------------------------------------------

_TABLE_EXISTS_SQL = "SELECT to_regclass('public.memory_feedback') IS NOT NULL"

_LEARNING_EXISTS_SQL = (
    "SELECT EXISTS(SELECT 1 FROM archival_memory WHERE id = $1::uuid)"
)

_UPSERT_SQL = """
INSERT INTO memory_feedback (learning_id, session_id, helpful, context, source)
VALUES ($1::uuid, $2, $3, $4, $5)
ON CONFLICT (learning_id, session_id)
DO UPDATE SET helpful = EXCLUDED.helpful,
              context = EXCLUDED.context,
              source = EXCLUDED.source,
              created_at = NOW()
RETURNING id, created_at
"""

_TOTALS_SQL = """
SELECT
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE helpful) AS helpful,
    COUNT(*) FILTER (WHERE NOT helpful) AS not_helpful,
    COUNT(DISTINCT learning_id) AS unique_learnings
FROM memory_feedback
"""

_TOP_HELPFUL_SQL = """
SELECT mf.learning_id, am.content, COUNT(*) AS helpful_count
FROM memory_feedback mf
JOIN archival_memory am ON am.id = mf.learning_id
WHERE mf.helpful = true
GROUP BY mf.learning_id, am.content
ORDER BY helpful_count DESC
LIMIT 5
"""

_TOP_NOT_HELPFUL_SQL = """
SELECT mf.learning_id, am.content, COUNT(*) AS not_helpful_count
FROM memory_feedback mf
JOIN archival_memory am ON am.id = mf.learning_id
WHERE mf.helpful = false
GROUP BY mf.learning_id, am.content
ORDER BY not_helpful_count DESC
LIMIT 5
"""


async def store_feedback(
    learning_id: str,
    helpful: bool,
    session_id: str,
    context: str = "",
    source: str = "manual",
) -> dict[str, Any]:
    """Store feedback for a learning. Upserts on (learning_id, session_id)."""
    async with get_connection() as conn:
        if not await conn.fetchval(_TABLE_EXISTS_SQL):
            return {
                "success": False,
                "error": "memory_feedback table not found — run the migration first",
            }
        if not await conn.fetchval(_LEARNING_EXISTS_SQL, learning_id):
            return {"success": False, "error": f"Learning {learning_id} not found"}
        row = await conn.fetchrow(
            _UPSERT_SQL, learning_id, session_id, helpful, context or None, source
        )
    return {
        "success": True,
        "feedback_id": str(row["id"]),
        "learning_id": learning_id,
        "helpful": helpful,
        "created_at": row["created_at"].isoformat(),
    }


async def get_feedback_for_learning(learning_id: str) -> dict[str, Any]:
    """Get all feedback for a specific learning."""
    async with get_connection() as conn:
        rows = await conn.fetch(
            "SELECT id, session_id, helpful, context, source, created_at "
            "FROM memory_feedback WHERE learning_id = $1::uuid ORDER BY created_at DESC",
            learning_id,
        )
    return aggregate_feedback(rows, learning_id)


async def get_feedback_summary() -> dict[str, Any]:
    """Get aggregate feedback statistics."""
    async with get_connection() as conn:
        if not await conn.fetchval(_TABLE_EXISTS_SQL):
            return empty_summary()
        totals = await conn.fetchrow(_TOTALS_SQL)
        top_helpful = await conn.fetch(_TOP_HELPFUL_SQL)
        top_not_helpful = await conn.fetch(_TOP_NOT_HELPFUL_SQL)
    return format_summary_result(totals, top_helpful, top_not_helpful)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _uuid_arg(value: str) -> str:
    """Validate that value is a well-formed UUID."""
    try:
        UUID(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid UUID: {value}") from exc
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Memory feedback CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    store_p = sub.add_parser("store", help="Store feedback for a learning")
    store_p.add_argument(
        "--learning-id", type=_uuid_arg, required=True, help="UUID of the learning"
    )
    store_p.add_argument("--session-id", default="cli", help="Session identifier")
    store_p.add_argument("--context", default="", help="Why it was/wasn't helpful")
    store_p.add_argument(
        "--source",
        default="manual",
        choices=["manual", "hook", "auto"],
        help="Feedback source",
    )
    helpful_group = store_p.add_mutually_exclusive_group(required=True)
    helpful_group.add_argument("--helpful", action="store_true", dest="helpful")
    helpful_group.add_argument("--not-helpful", action="store_true", dest="not_helpful")

    get_p = sub.add_parser("get", help="Get feedback for a learning")
    get_p.add_argument(
        "--learning-id", type=_uuid_arg, required=True, help="UUID of the learning"
    )

    sub.add_parser("summary", help="Get feedback summary statistics")

    return parser


async def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "store":
            result = await store_feedback(
                learning_id=args.learning_id,
                helpful=args.helpful,
                session_id=args.session_id,
                context=args.context,
                source=args.source,
            )
        elif args.command == "get":
            result = await get_feedback_for_learning(args.learning_id)
        elif args.command == "summary":
            result = await get_feedback_summary()
        else:
            parser.print_help()
            return

        print(json.dumps(result, indent=2, default=str))
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
