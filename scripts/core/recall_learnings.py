#!/usr/bin/env python3
"""Semantic recall of session learnings from archival_memory.

Searches the archival_memory table for session_learning entries
using vector similarity search.

USAGE:
    # Simple search (top 5 results, local embeddings)
    uv run python scripts/recall_learnings.py --query "authentication patterns"

    # More results
    uv run python scripts/recall_learnings.py --query "database schema" --k 10

    # Voyage embeddings (higher quality, requires VOYAGE_API_KEY)
    uv run python scripts/recall_learnings.py --query "errors" --provider voyage

    # Structured output (grouped by learning type)
    uv run python scripts/recall_learnings.py --query "hooks" --structured

Workflow:
    Query -> Embed (Local/Voyage) -> Vector Search (pgvector) -> Return

Environment:
    VOYAGE_API_KEY - For Voyage embeddings (optional)
    PostgreSQL with pgvector extension
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

faulthandler.enable(file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"), all_threads=True)

# Load .env files
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
load_dotenv()

# Add project root to path for imports (opc/)
project_dir = os.environ.get("CLAUDE_PROJECT_DIR", str(Path(__file__).parent.parent.parent))
sys.path.insert(0, project_dir)

# Re-export search backends for backward compatibility (tests import these directly)
from scripts.core.recall_backends import (  # noqa: E402, F401
    search_learnings_hybrid_rrf,
    search_learnings_postgres,
    search_learnings_sqlite,
    search_learnings_text_only_postgres,
)

# Re-export formatters
from scripts.core.recall_formatters import (  # noqa: E402, F401
    format_human_output,
    format_json_output,
    format_result_preview,
    group_by_type,
)


async def record_recall(result_ids: list[str]) -> None:
    """Update last_recalled and recall_count for recalled learnings.

    Batch-updates all returned results in a single query.
    Fails silently to avoid breaking recall if columns don't exist yet.
    """
    if not result_ids:
        return

    backend = get_backend()
    if backend != "postgres":
        return

    try:
        from scripts.core.db.postgres_pool import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE archival_memory
                SET last_recalled = NOW(),
                    recall_count = recall_count + 1
                WHERE id = ANY($1::uuid[])
                """,
                result_ids,
            )
    except Exception:
        # Graceful degradation: don't break recall if columns missing or DB error
        pass


async def enrich_with_pattern_strength(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add pattern_strength and pattern_tags to recall results.

    Queries detected_patterns + pattern_members via LEFT JOIN.
    Gracefully returns results unchanged if tables don't exist.
    """
    if not results or get_backend() != "postgres":
        return results

    result_ids = [r["id"] for r in results if r.get("id")]
    if not result_ids:
        return results

    try:
        from scripts.core.db.postgres_pool import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT pm.memory_id,
                       MAX(dp.confidence * GREATEST(1.0 - COALESCE(pm.distance, 0), 0))
                           AS pattern_strength,
                       ARRAY_AGG(DISTINCT unnested_tag)
                           AS pattern_tags
                FROM pattern_members pm
                JOIN detected_patterns dp ON dp.id = pm.pattern_id
                LEFT JOIN LATERAL unnest(dp.tags) AS unnested_tag
                    ON true
                WHERE dp.superseded_at IS NULL
                  AND pm.memory_id = ANY($1::uuid[])
                GROUP BY pm.memory_id
                """,
                [uuid.UUID(rid) for rid in result_ids],
            )

        # Build lookup
        lookup: dict[str, dict] = {}
        for row in rows:
            mid = str(row["memory_id"])
            lookup[mid] = {
                "pattern_strength": float(row["pattern_strength"] or 0.0),
                "pattern_tags": row["pattern_tags"] or [],
            }

        # Enrich results in place
        for result in results:
            rid = result.get("id")
            if rid and rid in lookup:
                result["pattern_strength"] = lookup[rid]["pattern_strength"]
                result["pattern_tags"] = lookup[rid]["pattern_tags"]

    except (ImportError, OSError, ConnectionError) as e:
        # Expected: tables don't exist, DB unreachable, pool import fails
        logger.debug("Pattern enrichment unavailable: %s", e)
    except Exception as e:
        # Unexpected: surface for debugging but don't break recall
        logger.warning("Pattern enrichment error: %s", e, exc_info=True)

    return results


def get_backend() -> str:
    """Determine which backend to use (sqlite or postgres)."""
    # Check explicit env var first
    backend = os.environ.get("AGENTICA_MEMORY_BACKEND", "").lower()
    if backend in ("sqlite", "postgres"):
        return backend

    # Check if CONTINUOUS_CLAUDE_DB_URL or DATABASE_URL is set (canonical first)
    if os.environ.get("CONTINUOUS_CLAUDE_DB_URL") or os.environ.get("DATABASE_URL"):
        return "postgres"

    # Default to sqlite for simplicity
    return "sqlite"


async def search_learnings(
    query: str,
    k: int = 5,
    provider: str = "local",
    text_fallback: bool = True,
    similarity_threshold: float = 0.2,
    recency_weight: float = 0.0,
) -> list[dict[str, Any]]:
    """Search archival_memory for session learnings.

    Automatically selects SQLite (BM25) or PostgreSQL (vector) based on environment.

    Args:
        query: Search query for semantic matching
        k: Number of results to return
        provider: Embedding provider ("local" or "voyage") - PostgreSQL only
        text_fallback: If True, use text search when no embeddings exist
        similarity_threshold: Minimum similarity score (default 0.2 filters garbage)
        recency_weight: Weight for recency boost (0.0-1.0). 0=no boost, 0.3=30% recency

    Returns:
        List of matching learnings with similarity scores
    """
    if not query.strip():
        return []

    backend = get_backend()

    if backend == "sqlite":
        results = await search_learnings_sqlite(query, k)
    else:
        results = await search_learnings_postgres(query, k, provider, text_fallback, similarity_threshold, recency_weight)

    return results


async def main() -> int:
    """Run semantic recall on session learnings."""
    parser = argparse.ArgumentParser(
        description="Semantic recall of session learnings from archival_memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="Search query for semantic matching",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "voyage"],
        default="local",
        help="Embedding provider (default: local)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for programmatic use)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Use text search only (faster, no embeddings)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.2,
        help="Minimum similarity threshold (default: 0.2, filters low-quality results)",
    )
    parser.add_argument(
        "--vector-only",
        action="store_true",
        help="Use vector-only search (disables hybrid RRF, enables recency)",
    )
    parser.add_argument(
        "--recency",
        "-r",
        type=float,
        default=0.1,
        help="Recency weight for vector-only mode (0.0-1.0, default: 0.1)",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Boost results matching these tags via reranker (space-separated)",
    )
    parser.add_argument(
        "--tags-strict",
        action="store_true",
        help="Hard-filter to results sharing at least one tag with --tags",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Bypass contextual re-ranking (use raw retrieval scores)",
    )
    parser.add_argument(
        "--project",
        help="Project context for re-ranking (default: auto-detect from CLAUDE_PROJECT_DIR)",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Group results by learning_type in output",
    )

    args = parser.parse_args()

    # Adaptive over-fetch: retrieve more candidates when reranking will trim
    fetch_k = args.k if args.no_rerank else max(3 * args.k, 50)

    # JSON mode: suppress human-readable output
    if not args.json:
        print(f'Recalling learnings for: "{args.query}"')
        print(f"Provider: {args.provider}")
        print()

    try:
        backend = get_backend()

        if backend == "sqlite":
            # SQLite only supports text search (no pgvector)
            if not args.text_only and not args.json:
                print("  (SQLite backend - using text search)")
            results = await search_learnings_sqlite(args.query, fetch_k)
        elif args.text_only:
            # Fast text-only search (no embeddings)
            results = await search_learnings_text_only_postgres(args.query, fetch_k)
        elif args.vector_only:
            # When reranking, suppress SQL-level recency blend to avoid
            # double-counting (the reranker applies its own recency signal).
            sql_recency = 0.0 if not args.no_rerank else args.recency
            results = await search_learnings(
                query=args.query,
                k=fetch_k,
                provider=args.provider,
                similarity_threshold=args.threshold,
                recency_weight=sql_recency,
            )
        else:
            # Default: Hybrid RRF search (text + vector combined)
            results = await search_learnings_hybrid_rrf(
                query=args.query,
                k=fetch_k,
                provider=args.provider,
                similarity_threshold=args.threshold * 0.01,  # RRF scores are ~0.01-0.03 range
            )
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e), "results": []}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    # Enrich with pattern strength for reranker (graceful if tables missing)
    if not args.no_rerank and backend == "postgres":
        results = await enrich_with_pattern_strength(results)

    # Hard-filter by tags BEFORE reranking (so reranker sees filtered pool)
    if args.tags_strict and args.tags:
        tag_set = set(args.tags)
        results = [
            r for r in results
            if set(r.get("metadata", {}).get("tags", [])) & tag_set
        ]

    # Apply contextual re-ranking
    if not args.no_rerank:
        from scripts.core.reranker import RecallContext, rerank

        # Determine retrieval mode from backend and flags
        if backend == "sqlite":
            retrieval_mode = "sqlite"
        elif args.text_only:
            retrieval_mode = "text"
        elif args.vector_only:
            retrieval_mode = "vector"
        else:
            retrieval_mode = "hybrid_rrf"

        ctx = RecallContext(
            project=args.project or os.environ.get("CLAUDE_PROJECT_DIR", "").rsplit("/", 1)[-1] or None,
            tags_hint=args.tags,
            retrieval_mode=retrieval_mode,
        )
        results = rerank(results, ctx, k=args.k)

    # Record recall ONLY for final results (after rerank trims)
    await record_recall([r["id"] for r in results])

    # Output results
    if args.json:
        print(format_json_output(results, structured=args.structured))
    else:
        print(format_human_output(results, structured=args.structured))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
