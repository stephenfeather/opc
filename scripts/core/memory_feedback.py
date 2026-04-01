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

from dotenv import load_dotenv

global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
load_dotenv()

repo_root = str(Path(__file__).parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.core.db.postgres_pool import close_pool, get_connection  # noqa: E402


async def store_feedback(
    learning_id: str,
    helpful: bool,
    session_id: str,
    context: str = "",
    source: str = "manual",
) -> dict[str, Any]:
    """Store feedback for a learning. Upserts on (learning_id, session_id)."""
    async with get_connection() as conn:
        # Verify learning exists
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM archival_memory WHERE id = $1::uuid)",
            learning_id,
        )
        if not exists:
            return {"success": False, "error": f"Learning {learning_id} not found"}

        row = await conn.fetchrow(
            """
            INSERT INTO memory_feedback (learning_id, session_id, helpful, context, source)
            VALUES ($1::uuid, $2, $3, $4, $5)
            ON CONFLICT (learning_id, session_id)
            DO UPDATE SET helpful = EXCLUDED.helpful,
                          context = EXCLUDED.context,
                          source = EXCLUDED.source,
                          created_at = NOW()
            RETURNING id, created_at
            """,
            learning_id, session_id, helpful, context or None, source,
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
            """
            SELECT id, session_id, helpful, context, source, created_at
            FROM memory_feedback
            WHERE learning_id = $1::uuid
            ORDER BY created_at DESC
            """,
            learning_id,
        )
    feedback = [
        {
            "id": str(r["id"]),
            "session_id": r["session_id"],
            "helpful": r["helpful"],
            "context": r["context"],
            "source": r["source"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]
    helpful_count = sum(1 for f in feedback if f["helpful"])
    return {
        "learning_id": learning_id,
        "total_feedback": len(feedback),
        "helpful_count": helpful_count,
        "not_helpful_count": len(feedback) - helpful_count,
        "feedback": feedback,
    }


async def get_feedback_summary() -> dict[str, Any]:
    """Get aggregate feedback statistics."""
    async with get_connection() as conn:
        # Check if table exists
        table_exists = await conn.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'memory_feedback'
            )
            """
        )
        if not table_exists:
            return {
                "total_feedback": 0,
                "helpful_count": 0,
                "not_helpful_count": 0,
                "unique_learnings_rated": 0,
                "helpfulness_rate": 0.0,
                "top_helpful": [],
                "top_not_helpful": [],
            }

        totals = await conn.fetchrow(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE helpful) AS helpful,
                COUNT(*) FILTER (WHERE NOT helpful) AS not_helpful,
                COUNT(DISTINCT learning_id) AS unique_learnings
            FROM memory_feedback
            """
        )

        top_helpful = await conn.fetch(
            """
            SELECT mf.learning_id, am.content, COUNT(*) AS helpful_count
            FROM memory_feedback mf
            JOIN archival_memory am ON am.id = mf.learning_id
            WHERE mf.helpful = true
            GROUP BY mf.learning_id, am.content
            ORDER BY helpful_count DESC
            LIMIT 5
            """,
        )

        top_not_helpful = await conn.fetch(
            """
            SELECT mf.learning_id, am.content, COUNT(*) AS not_helpful_count
            FROM memory_feedback mf
            JOIN archival_memory am ON am.id = mf.learning_id
            WHERE mf.helpful = false
            GROUP BY mf.learning_id, am.content
            ORDER BY not_helpful_count DESC
            LIMIT 5
            """,
        )

    total = totals["total"]
    helpful = totals["helpful"]
    return {
        "total_feedback": total,
        "helpful_count": helpful,
        "not_helpful_count": totals["not_helpful"],
        "unique_learnings_rated": totals["unique_learnings"],
        "helpfulness_rate": round(helpful / total * 100, 1) if total > 0 else 0.0,
        "top_helpful": [
            {
                "learning_id": str(r["learning_id"]),
                "content": r["content"][:120],
                "helpful_count": r["helpful_count"],
            }
            for r in top_helpful
        ],
        "top_not_helpful": [
            {
                "learning_id": str(r["learning_id"]),
                "content": r["content"][:120],
                "not_helpful_count": r["not_helpful_count"],
            }
            for r in top_not_helpful
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory feedback CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    store_p = sub.add_parser("store", help="Store feedback for a learning")
    store_p.add_argument("--learning-id", required=True, help="UUID of the learning")
    store_p.add_argument("--session-id", default="cli", help="Session identifier")
    store_p.add_argument("--context", default="", help="Why it was/wasn't helpful")
    store_p.add_argument("--source", default="manual", help="Feedback source")
    helpful_group = store_p.add_mutually_exclusive_group(required=True)
    helpful_group.add_argument("--helpful", action="store_true", dest="helpful")
    helpful_group.add_argument("--not-helpful", action="store_true", dest="not_helpful")

    get_p = sub.add_parser("get", help="Get feedback for a learning")
    get_p.add_argument("--learning-id", required=True, help="UUID of the learning")

    sub.add_parser("summary", help="Get feedback summary statistics")

    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "store":
        helpful = args.helpful if hasattr(args, "helpful") else not args.not_helpful
        result = await store_feedback(
            learning_id=args.learning_id,
            helpful=helpful,
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
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
