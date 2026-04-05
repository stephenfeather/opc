"""Search backends for recall learnings.

Contains the four search implementations:
- text-only postgres (FTS)
- SQLite (FTS5/BM25)
- hybrid RRF (text + vector fusion)
- postgres vector (cosine similarity with optional recency)

Each returns list[dict] with keys: id, session_id, content, metadata, created_at, similarity.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.core.config import get_config as _get_config

logger = logging.getLogger(__name__)


def sanitize_tsquery_words(words: list[str]) -> list[str]:
    """Strip tsquery metacharacters from words to prevent injection.

    Removes: ! & | ( ) < > : *  and any non-alphanumeric characters.
    Filters out words that become too short (<=2 chars) after sanitization.
    """
    result = []
    for w in words:
        clean = re.sub(r"[^a-zA-Z0-9]", "", w)
        if len(clean) > 2:
            result.append(clean)
    return result

_recall_cfg = _get_config().recall


async def search_learnings_text_only_postgres(
    query: str, k: int = _recall_cfg.default_k,
) -> list[dict[str, Any]]:
    """Fast text-only search for PostgreSQL using full-text search.

    Uses tsvector/tsquery with GIN index. Automatic stopword handling.
    Falls back to ILIKE if tsquery fails (e.g., all stopwords).
    """
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()

    async with pool.acquire() as conn:
        # Try full-text search using plainto_tsquery (flexible OR semantics)
        # Strip meta-words and normalize for better matching
        from scripts.core.query_expansion import STOPWORDS

        meta_words = STOPWORDS
        clean_query = query.lower().replace('-', ' ')  # "multi-terminal" -> "multi terminal"
        clean_query = ' '.join(w for w in clean_query.split() if w not in meta_words)
        if not clean_query.strip():
            clean_query = query  # Fallback to original if all stripped

        # Build OR-based query: "session affinity terminal" -> 'session' | 'affinity' | 'terminal'
        # This matches documents containing ANY of the terms, ranked by how many match
        # Sanitize to strip tsquery metacharacters (!, &, |, <->, etc.)
        words = sanitize_tsquery_words(clean_query.split())
        if not words:
            fallback = re.sub(r"[^a-zA-Z0-9]", "", query.split()[0]) if query.strip() else ""
            words = [fallback] if fallback and len(fallback) > 2 else [fallback or "a"]
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


async def search_learnings_sqlite(
    query: str, k: int = _recall_cfg.default_k,
) -> list[dict[str, Any]]:
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
            normalized_score = min(1.0, max(0.0, -raw_rank / _recall_cfg.bm25_normalization_divisor))

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
    k: int = _recall_cfg.default_k,
    provider: str = "local",
    rrf_k: int = _recall_cfg.rrf_k,
    similarity_threshold: float = 0.0,
    expand: bool = True,
    max_expansion_terms: int = _recall_cfg.max_expansion_terms,
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
        expand: If True, expand query with TF-IDF related terms
        max_expansion_terms: Number of expansion terms to add

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

    # Query expansion: find related terms via TF-IDF over vector neighbors
    text_query = query
    use_tsquery = False
    if expand:
        try:
            from scripts.core.query_expansion import expand_query

            expanded = await expand_query(
                query,
                query_embedding,
                max_expansion_terms=max_expansion_terms,
            )
            if expanded != query:
                text_query = expanded
                use_tsquery = True
                logger.debug("Expanded query: %r -> %r", query, expanded)
        except Exception:
            logger.debug("Query expansion failed, using original", exc_info=True)

    def _build_rrf_cte(*, chain_filter: bool, use_tsquery: bool = False) -> str:
        chain_clause = "\n                AND superseded_by IS NULL" if chain_filter else ""
        tsquery_fn = "to_tsquery" if use_tsquery else "plainto_tsquery"
        return f"""
            WITH fts_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(
                            to_tsvector('english', content),
                            {tsquery_fn}('english', $1)
                        ) DESC
                    ) as fts_rank
                FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'{chain_clause}
                AND to_tsvector('english', content) @@ {tsquery_fn}('english', $1)
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

    _RRF_CTE = _build_rrf_cte(chain_filter=True, use_tsquery=use_tsquery)
    _RRF_CTE_PLAIN = _build_rrf_cte(chain_filter=False, use_tsquery=use_tsquery)

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
                    ELSE log(2.0, 1 + COALESCE(a.recall_count, 0)) * {_recall_cfg.recall_boost_multiplier}
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
        query_args = (text_query, str(query_embedding), rrf_k, k * 2)

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

        # Fallback: if to_tsquery failed (malformed expansion), retry with plainto_tsquery
        if not rows and use_tsquery:
            logger.debug("Expanded tsquery returned no results, falling back to plainto_tsquery")
            plain_cte = _build_rrf_cte(chain_filter=True, use_tsquery=False)
            plain_args = (query, str(query_embedding), rrf_k, k * 2)
            try:
                if has_decay_columns:
                    rows = await conn.fetch(plain_cte + _BOOSTED_TAIL, *plain_args)
                else:
                    rows = await conn.fetch(plain_cte + _PLAIN_TAIL, *plain_args)
            except Exception:
                logger.debug("plainto_tsquery fallback also failed", exc_info=True)

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
    k: int = _recall_cfg.default_k,
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
