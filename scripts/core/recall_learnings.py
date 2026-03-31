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


def format_result_preview(content: str, max_length: int = 200) -> str:
    """Format content for display, truncating if needed.

    Args:
        content: Full content string
        max_length: Maximum characters before truncation

    Returns:
        Content string, truncated with ... if over max_length
    """
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


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


async def search_learnings_text_only_postgres(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Fast text-only search for PostgreSQL using full-text search.

    Uses tsvector/tsquery with GIN index. Automatic stopword handling.
    Falls back to ILIKE if tsquery fails (e.g., all stopwords).
    """
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()

    async with pool.acquire() as conn:
        # Try full-text search using plainto_tsquery (flexible OR semantics)
        # Strip meta-words and normalize for better matching
        meta_words = {'help', 'want', 'need', 'show', 'tell', 'find', 'look', 'please', 'with', 'for'}
        clean_query = query.lower().replace('-', ' ')  # "multi-terminal" -> "multi terminal"
        clean_query = ' '.join(w for w in clean_query.split() if w not in meta_words)
        if not clean_query.strip():
            clean_query = query  # Fallback to original if all stripped

        # Build OR-based query: "session affinity terminal" -> 'session' | 'affinity' | 'terminal'
        # This matches documents containing ANY of the terms, ranked by how many match
        words = [w for w in clean_query.split() if len(w) > 2]
        if not words:
            words = clean_query.split()[:1] or [query.split()[0]]
        or_query = ' | '.join(words)

        try:
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    session_id,
                    content,
                    metadata,
                    created_at,
                    ts_rank(to_tsvector('english', content), to_tsquery('english', $1)) as similarity
                FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'
                    AND superseded_by IS NULL
                    AND to_tsvector('english', content) @@ to_tsquery('english', $1)
                ORDER BY similarity DESC, created_at DESC
                LIMIT $2
                """,
                or_query,
                k,
            )
        except Exception:
            # Fallback: superseded_by column doesn't exist yet (pre-migration)
            logger.debug("Chain filter fallback in text_only_postgres FTS", exc_info=True)
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    session_id,
                    content,
                    metadata,
                    created_at,
                    ts_rank(to_tsvector('english', content), to_tsquery('english', $1)) as similarity
                FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'
                    AND to_tsvector('english', content) @@ to_tsquery('english', $1)
                ORDER BY similarity DESC, created_at DESC
                LIMIT $2
                """,
                or_query,
                k,
            )

        # Fallback to ILIKE if no FTS results (query was all stopwords)
        if not rows:
            # Extract first word for simple substring match
            first_word = query.split()[0] if query.split() else query
            try:
                rows = await conn.fetch(
                    """
                    SELECT
                        id,
                        session_id,
                        content,
                        metadata,
                        created_at,
                        0.1 as similarity
                    FROM archival_memory
                    WHERE metadata->>'type' = 'session_learning'
                        AND superseded_by IS NULL
                        AND content ILIKE '%' || $1 || '%'
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    first_word,
                    k,
                )
            except Exception:
                logger.debug("Chain filter fallback in text_only_postgres ILIKE", exc_info=True)
                rows = await conn.fetch(
                    """
                    SELECT
                        id,
                        session_id,
                        content,
                        metadata,
                        created_at,
                        0.1 as similarity
                    FROM archival_memory
                    WHERE metadata->>'type' = 'session_learning'
                        AND content ILIKE '%' || $1 || '%'
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    first_word,
                    k,
                )

    results = []
    for row in rows:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        results.append({
            "id": str(row["id"]),
            "session_id": row["session_id"],
            "content": row["content"],
            "metadata": metadata,
            "created_at": row["created_at"],
            "similarity": float(row["similarity"]),  # Use actual ts_rank score
        })

    return results


async def search_learnings_sqlite(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Search learnings using SQLite FTS5 (BM25 ranking).

    Cross-session search - finds learnings from ALL sessions.

    Args:
        query: Search query
        k: Number of results to return

    Returns:
        List of matching learnings with BM25 scores
    """
    import re
    import sqlite3

    # Global SQLite path
    db_path = Path.home() / ".claude" / "cache" / "memory.db"

    if not db_path.exists():
        return []

    # Prepare FTS query (OR-join words for broader matching)
    words = re.findall(r"\w+", query.lower())
    fts_query = " OR ".join(words) if words else query

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(
            """
            SELECT
                a.id,
                a.session_id,
                a.content,
                a.metadata_json,
                a.created_at,
                bm25(archival_fts) as rank
            FROM archival_memory a
            JOIN archival_fts f ON a.rowid = f.rowid
            WHERE archival_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, k),
        )
        rows = cursor.fetchall()

        results = []
        for row in rows:
            # BM25 returns negative scores (lower = better)
            # Normalize to 0.0-1.0 range
            raw_rank = row["rank"] if row["rank"] else 0
            normalized_score = min(1.0, max(0.0, -raw_rank / 25.0))

            metadata = {}
            if row["metadata_json"]:
                try:
                    metadata = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    pass

            results.append({
                "id": row["id"] or "",
                "session_id": row["session_id"] or "unknown",
                "content": row["content"] or "",
                "metadata": metadata,
                "created_at": datetime.fromtimestamp(row["created_at"]) if row["created_at"] else None,
                "similarity": normalized_score,
            })

        return results
    finally:
        conn.close()


async def search_learnings_hybrid_rrf(
    query: str,
    k: int = 5,
    provider: str = "local",
    rrf_k: int = 60,
    similarity_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Hybrid RRF search combining text and vector rankings.

    Uses Reciprocal Rank Fusion:
        score = 1/(k + rank_fts) + 1/(k + rank_vector)

    Args:
        query: Search query
        k: Number of results
        provider: Embedding provider
        rrf_k: RRF constant (default 60)
        similarity_threshold: Minimum RRF score to include

    Returns:
        List of learnings with RRF scores
    """
    from scripts.core.db.embedding_service import EmbeddingService
    from scripts.core.db.postgres_pool import get_pool, init_pgvector

    pool = await get_pool()

    # Generate query embedding
    embedder = EmbeddingService(provider=provider)
    try:
        query_embedding = await embedder.embed(query, input_type="query")
    finally:
        await embedder.aclose()

    def _build_rrf_cte(*, chain_filter: bool) -> str:
        chain_clause = "\n                AND superseded_by IS NULL" if chain_filter else ""
        return f"""
            WITH fts_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(
                            to_tsvector('english', content),
                            plainto_tsquery('english', $1)
                        ) DESC
                    ) as fts_rank
                FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'{chain_clause}
                AND to_tsvector('english', content) @@ plainto_tsquery('english', $1)
            ),
            vector_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> $2::vector) as vec_rank
                FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'{chain_clause}
                AND embedding IS NOT NULL
            ),
            combined AS (
                SELECT
                    COALESCE(f.id, v.id) as id,
                    COALESCE(1.0 / ($3 + f.fts_rank), 0) +
                    COALESCE(1.0 / ($3 + v.vec_rank), 0) as rrf_score,
                    f.fts_rank,
                    v.vec_rank
                FROM fts_ranked f
                FULL OUTER JOIN vector_ranked v ON f.id = v.id
            )"""

    _RRF_CTE = _build_rrf_cte(chain_filter=True)
    _RRF_CTE_PLAIN = _build_rrf_cte(chain_filter=False)

    _BOOSTED_TAIL = """
            SELECT
                a.id,
                a.session_id,
                a.content,
                a.metadata,
                a.created_at,
                a.recall_count,
                a.last_recalled,
                c.rrf_score +
                    CASE WHEN COALESCE(a.recall_count, 0) = 0 THEN 0
                    ELSE log(2.0, 1 + COALESCE(a.recall_count, 0)) * 0.002
                    END as boosted_score,
                c.rrf_score as raw_rrf_score,
                c.fts_rank,
                c.vec_rank
            FROM combined c
            JOIN archival_memory a ON a.id = c.id
            ORDER BY boosted_score DESC
            LIMIT $4
            """

    _PLAIN_TAIL = """
            SELECT
                a.id,
                a.session_id,
                a.content,
                a.metadata,
                a.created_at,
                c.rrf_score,
                c.fts_rank,
                c.vec_rank
            FROM combined c
            JOIN archival_memory a ON a.id = c.id
            ORDER BY c.rrf_score DESC
            LIMIT $4
            """

    # Build queries: chain-filtered first, plain fallback
    _BOOSTED_SELECT = _RRF_CTE + _BOOSTED_TAIL
    _PLAIN_SELECT = _RRF_CTE + _PLAIN_TAIL
    _PLAIN_SELECT_NO_CHAIN = _RRF_CTE_PLAIN + _PLAIN_TAIL

    has_decay_columns = True
    async with pool.acquire() as conn:
        await init_pgvector(conn)
        query_args = (query, str(query_embedding), rrf_k, k * 2)

        try:
            rows = await conn.fetch(_BOOSTED_SELECT, *query_args)
        except Exception:
            # Fallback: decay or chain columns don't exist yet
            logger.debug("RRF boosted+chain fallback", exc_info=True)
            has_decay_columns = False
            try:
                rows = await conn.fetch(_PLAIN_SELECT, *query_args)
            except Exception:
                logger.debug("RRF plain+chain fallback", exc_info=True)
                rows = await conn.fetch(
                    _PLAIN_SELECT_NO_CHAIN, *query_args
                )

    results = []
    for row in rows:
        if has_decay_columns:
            score = float(row["boosted_score"])
        else:
            score = float(row["rrf_score"])

        if similarity_threshold > 0 and score < similarity_threshold:
            continue

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        result = {
            "id": str(row["id"]),
            "session_id": row["session_id"],
            "content": row["content"],
            "metadata": metadata,
            "created_at": row["created_at"],
            "similarity": score,
            "fts_rank": row["fts_rank"],
            "vec_rank": row["vec_rank"],
        }

        if has_decay_columns:
            result["raw_rrf_score"] = float(row["raw_rrf_score"])
            result["recall_count"] = row["recall_count"] or 0
            result["last_recalled"] = row["last_recalled"]

        results.append(result)

        if len(results) >= k:
            break

    return results


async def search_learnings_postgres(
    query: str,
    k: int = 5,
    provider: str = "local",
    text_fallback: bool = True,
    similarity_threshold: float = 0.0,
    recency_weight: float = 0.0,
) -> list[dict[str, Any]]:
    """Search learnings using PostgreSQL (vector similarity or text fallback).

    Args:
        query: Search query for semantic matching
        k: Number of results to return
        provider: Embedding provider ("local" or "voyage")
        text_fallback: If True, use text search when no embeddings exist
        similarity_threshold: Minimum similarity score (0.0-1.0) to include results
        recency_weight: Weight for recency boost (0.0-1.0). 0=no boost, 0.3=30% recency

    Returns:
        List of matching learnings with similarity scores
    """
    from scripts.core.db.embedding_service import EmbeddingService
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()

    # First check if any learnings have embeddings
    async with pool.acquire() as conn:
        count_row = await conn.fetchrow(
            """
            SELECT COUNT(*) as cnt FROM archival_memory
            WHERE metadata->>'type' = 'session_learning'
                AND embedding IS NOT NULL
            """
        )
        has_embeddings = count_row["cnt"] > 0

    if has_embeddings:
        # Vector similarity search
        embedder = EmbeddingService(provider=provider)
        try:
            query_embedding = await embedder.embed(query, input_type="query")
        finally:
            await embedder.aclose()

        async with pool.acquire() as conn:
            from scripts.core.db.postgres_pool import init_pgvector
            await init_pgvector(conn)

            if recency_weight > 0:
                # Combined score: (1-recency_weight)*similarity + recency_weight*recency
                # Recency is normalized: 1.0 for newest, 0.0 for 30 days old or older
                _recency_sql = """
                    WITH scored AS (
                        SELECT
                            id,
                            session_id,
                            content,
                            metadata,
                            created_at,
                            1 - (embedding <=> $1::vector) as similarity,
                            GREATEST(0, 1.0 - EXTRACT(EPOCH FROM NOW() - created_at) / (30 * 86400)) as recency
                        FROM archival_memory
                        WHERE metadata->>'type' = 'session_learning'
                            AND embedding IS NOT NULL
                            {chain_filter}
                    )
                    SELECT
                        id, session_id, content, metadata, created_at, similarity, recency,
                        (1.0 - $3::float) * similarity + $3::float * recency as combined_score
                    FROM scored
                    ORDER BY combined_score DESC
                    LIMIT $2
                    """
                try:
                    rows = await conn.fetch(
                        _recency_sql.format(chain_filter="AND superseded_by IS NULL"),
                        str(query_embedding), k, recency_weight,
                    )
                except Exception:
                    logger.debug("Chain filter fallback in postgres recency", exc_info=True)
                    rows = await conn.fetch(
                        _recency_sql.format(chain_filter=""),
                        str(query_embedding), k, recency_weight,
                    )
            else:
                _vector_sql = """
                    SELECT
                        id,
                        session_id,
                        content,
                        metadata,
                        created_at,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM archival_memory
                    WHERE metadata->>'type' = 'session_learning'
                        AND embedding IS NOT NULL
                        {chain_filter}
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """
                try:
                    rows = await conn.fetch(
                        _vector_sql.format(chain_filter="AND superseded_by IS NULL"),
                        str(query_embedding), k,
                    )
                except Exception:
                    logger.debug("Chain filter fallback in postgres vector", exc_info=True)
                    rows = await conn.fetch(
                        _vector_sql.format(chain_filter=""),
                        str(query_embedding), k,
                    )
    elif text_fallback:
        # Fallback to text search (ILIKE) when no embeddings
        _text_sql = """
                SELECT
                    id,
                    session_id,
                    content,
                    metadata,
                    created_at,
                    0.5 as similarity
                FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'
                    AND content ILIKE '%' || $1 || '%'
                    {chain_filter}
                ORDER BY created_at DESC
                LIMIT $2
                """
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    _text_sql.format(chain_filter="AND superseded_by IS NULL"),
                    query, k,
                )
            except Exception:
                logger.debug("Chain filter fallback in postgres text", exc_info=True)
                rows = await conn.fetch(
                    _text_sql.format(chain_filter=""),
                    query, k,
                )
    else:
        return []

    results = []
    for row in rows:
        row_dict = dict(row)  # Convert Record to dict for easier access

        # Use combined_score if available (recency boost), otherwise similarity
        if "combined_score" in row_dict:
            score = float(row_dict["combined_score"]) if row_dict["combined_score"] else 0.0
        else:
            score = float(row_dict["similarity"]) if row_dict["similarity"] else 0.0

        # Skip results below threshold (only for vector search, not text fallback)
        if similarity_threshold > 0 and score < similarity_threshold:
            continue

        metadata = row_dict["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        result = {
            "id": str(row_dict["id"]),
            "session_id": row_dict["session_id"],
            "content": row_dict["content"],
            "metadata": metadata,
            "created_at": row_dict["created_at"],
            "similarity": score,
        }

        # Include raw similarity and recency if available
        if "recency" in row_dict:
            result["raw_similarity"] = float(row_dict["similarity"]) if row_dict["similarity"] else 0.0
            result["recency"] = float(row_dict["recency"]) if row_dict["recency"] else 0.0

        results.append(result)

    return results


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

    # JSON output mode
    if args.json:
        json_results = []
        for result in results:
            created_at = result["created_at"]
            if isinstance(created_at, datetime):
                created_str = created_at.isoformat()
            else:
                created_str = str(created_at)

            json_result = {
                "score": result.get("final_score", result["similarity"]),
                "raw_score": result["similarity"],
                "session_id": result["session_id"],
                "content": result["content"],
                "created_at": created_str,
            }
            if "rerank_details" in result:
                json_result["rerank_details"] = result["rerank_details"]
            json_results.append(json_result)
        print(json.dumps({"results": json_results}))
        return 0

    # Human-readable output
    if not results:
        print("No matching learnings found.")
        return 0

    print(f"Found {len(results)} matching learnings:")
    print()

    for i, result in enumerate(results, 1):
        score = result.get("final_score", result["similarity"])
        content_preview = format_result_preview(result["content"], max_length=300)
        session_id = result["session_id"]
        created_at = result["created_at"]

        # Format timestamp
        if isinstance(created_at, datetime):
            created_str = created_at.strftime("%Y-%m-%d %H:%M")
        else:
            created_str = str(created_at)[:16]

        print(f"{i}. [{score:.3f}] Session: {session_id} ({created_str})")
        print(f"   {content_preview}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
