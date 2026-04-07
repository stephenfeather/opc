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
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.core.config import get_config as _get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions — no I/O, no side effects
# ---------------------------------------------------------------------------


def sanitize_tsquery_words(words: list[str]) -> list[str]:
    """Strip tsquery metacharacters from words to prevent injection.

    Removes: ! & | ( ) < > : *  and any non-alphanumeric characters.
    Filters out words that become too short (<=2 chars) after sanitization.
    """
    return [
        clean
        for w in words
        if len(clean := re.sub(r"[^a-zA-Z0-9]", "", w)) > 2
    ]


def clean_query_text(query: str, stopwords: set[str]) -> str:
    """Normalize query and remove stopwords.

    Lowercases, replaces hyphens with spaces, and filters out stopwords.
    Falls back to original query if all words are stopwords.
    """
    normalized = query.lower().replace("-", " ")
    cleaned = " ".join(w for w in normalized.split() if w not in stopwords)
    return cleaned if cleaned.strip() else query


def build_fallback_words(query: str) -> list[str]:
    """Produce a fallback word list when sanitization yields nothing.

    Extracts the first word, strips non-alphanumeric chars.
    Returns ["a"] as last resort for empty queries.
    """
    if not query.strip():
        return ["a"]
    first_word = query.split()[0]
    fallback = re.sub(r"[^a-zA-Z0-9]", "", first_word)
    return [fallback] if fallback else ["a"]


def build_or_query(query: str, stopwords: set[str]) -> str:
    """Build an OR-joined tsquery string from a raw query.

    Pipeline: clean → split → sanitize → OR-join, with fallback.
    """
    cleaned = clean_query_text(query, stopwords)
    words = sanitize_tsquery_words(cleaned.split())
    if not words:
        words = build_fallback_words(query)
    return " | ".join(words)


def normalize_bm25_score(raw_rank: float | None, divisor: float) -> float:
    """Convert raw BM25 rank (negative = better) to 0.0-1.0 range."""
    rank = raw_rank if raw_rank is not None else 0.0
    return min(1.0, max(0.0, -rank / divisor))


def format_row_metadata(metadata: Any) -> dict[str, Any]:
    """Parse metadata from a DB row, handling str, dict, or None."""
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Malformed metadata JSON: %r", metadata[:200])
            return {}
    return {}


def format_text_result(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a text/vector search DB row to a result dict."""
    return {
        "id": str(row["id"]),
        "session_id": row["session_id"],
        "content": row["content"],
        "metadata": format_row_metadata(row["metadata"]),
        "created_at": row["created_at"],
        "similarity": float(row["similarity"]),
    }


def format_sqlite_result(
    row: Mapping[str, Any], *, divisor: float,
) -> dict[str, Any]:
    """Convert a SQLite FTS5 row to a result dict with normalized BM25 score."""
    metadata = {}
    if row["metadata_json"]:
        try:
            metadata = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "id": row["id"] or "",
        "session_id": row["session_id"] or "unknown",
        "content": row["content"] or "",
        "metadata": metadata,
        "created_at": (
            datetime.fromtimestamp(row["created_at"]) if row["created_at"] else None
        ),
        "similarity": normalize_bm25_score(row["rank"], divisor),
    }


def format_rrf_result(
    row: Mapping[str, Any], *, has_decay: bool,
) -> dict[str, Any]:
    """Convert an RRF search row to a result dict."""
    score = float(row["boosted_score"]) if has_decay else float(row["rrf_score"])

    result: dict[str, Any] = {
        "id": str(row["id"]),
        "session_id": row["session_id"],
        "content": row["content"],
        "metadata": format_row_metadata(row["metadata"]),
        "created_at": row["created_at"],
        "similarity": score,
        "fts_rank": row["fts_rank"],
        "vec_rank": row["vec_rank"],
    }

    if has_decay:
        result["raw_rrf_score"] = float(row["raw_rrf_score"])
        result["recall_count"] = row["recall_count"] or 0
        result["last_recalled"] = row["last_recalled"]

    return result


def format_vector_result(
    row: Mapping[str, Any],
    *,
    similarity_threshold: float = 0.0,
) -> dict[str, Any] | None:
    """Convert a vector search row to a result dict. Returns None if below threshold."""
    row_dict = dict(row)

    if "combined_score" in row_dict and row_dict["combined_score"] is not None:
        score = float(row_dict["combined_score"])
    else:
        score = float(row_dict["similarity"]) if row_dict["similarity"] else 0.0

    if similarity_threshold > 0 and score < similarity_threshold:
        return None

    result: dict[str, Any] = {
        "id": str(row_dict["id"]),
        "session_id": row_dict["session_id"],
        "content": row_dict["content"],
        "metadata": format_row_metadata(row_dict["metadata"]),
        "created_at": row_dict["created_at"],
        "similarity": score,
    }

    if "recency" in row_dict:
        result["raw_similarity"] = (
            float(row_dict["similarity"]) if row_dict["similarity"] else 0.0
        )
        result["recency"] = float(row_dict["recency"]) if row_dict["recency"] else 0.0

    return result


def build_rrf_cte(*, chain_filter: bool, use_tsquery: bool = False) -> str:
    """Build the SQL CTE for RRF (Reciprocal Rank Fusion) queries."""
    chain_clause = (
        "\n                AND superseded_by IS NULL" if chain_filter else ""
    )
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


# ---------------------------------------------------------------------------
# I/O handlers — async functions that interact with databases
# ---------------------------------------------------------------------------

_recall_cfg = _get_config().recall


async def search_learnings_text_only_postgres(
    query: str, k: int = _recall_cfg.default_k,
) -> list[dict[str, Any]]:
    """Fast text-only search for PostgreSQL using full-text search.

    Uses tsvector/tsquery with GIN index. Automatic stopword handling.
    Falls back to ILIKE if tsquery fails (e.g., all stopwords).
    """
    from scripts.core.db.postgres_pool import get_pool
    from scripts.core.query_expansion import STOPWORDS

    or_query = build_or_query(query, STOPWORDS)

    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT
                    id, session_id, content, metadata, created_at,
                    ts_rank(to_tsvector('english', content),
                            to_tsquery('english', $1)) as similarity
                FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'
                    AND superseded_by IS NULL
                    AND to_tsvector('english', content) @@ to_tsquery('english', $1)
                ORDER BY similarity DESC, created_at DESC
                LIMIT $2
                """,
                or_query, k,
            )
        except Exception:
            logger.debug("Chain filter fallback in text_only_postgres FTS", exc_info=True)
            rows = await conn.fetch(
                """
                SELECT
                    id, session_id, content, metadata, created_at,
                    ts_rank(to_tsvector('english', content),
                            to_tsquery('english', $1)) as similarity
                FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'
                    AND to_tsvector('english', content) @@ to_tsquery('english', $1)
                ORDER BY similarity DESC, created_at DESC
                LIMIT $2
                """,
                or_query, k,
            )

        if not rows:
            first_word = query.split()[0] if query.split() else query
            try:
                rows = await conn.fetch(
                    """
                    SELECT
                        id, session_id, content, metadata, created_at,
                        0.1 as similarity
                    FROM archival_memory
                    WHERE metadata->>'type' = 'session_learning'
                        AND superseded_by IS NULL
                        AND content ILIKE '%' || $1 || '%'
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    first_word, k,
                )
            except Exception:
                logger.debug("Chain filter fallback in text_only_postgres ILIKE", exc_info=True)
                rows = await conn.fetch(
                    """
                    SELECT
                        id, session_id, content, metadata, created_at,
                        0.1 as similarity
                    FROM archival_memory
                    WHERE metadata->>'type' = 'session_learning'
                        AND content ILIKE '%' || $1 || '%'
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    first_word, k,
                )

    return [format_text_result(row) for row in rows]


async def search_learnings_sqlite(
    query: str, k: int = _recall_cfg.default_k,
) -> list[dict[str, Any]]:
    """Search learnings using SQLite FTS5 (BM25 ranking).

    Cross-session search - finds learnings from ALL sessions.
    """
    import sqlite3

    db_path = Path.home() / ".claude" / "cache" / "memory.db"
    if not db_path.exists():
        return []

    words = re.findall(r"\w+", query.lower())
    fts_query = " OR ".join(words) if words else query

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        try:
            cursor = conn.execute(
                """
                SELECT
                    a.id, a.session_id, a.content, a.metadata_json,
                    a.created_at, bm25(archival_fts) as rank
                FROM archival_memory a
                JOIN archival_fts f ON a.rowid = f.rowid
                WHERE archival_fts MATCH ?
                    AND json_extract(a.metadata_json, '$.type') = 'session_learning'
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, k),
            )
        except Exception:
            logger.debug("SQLite json_extract not available, filtering in Python", exc_info=True)
            cursor = conn.execute(
                """
                SELECT
                    a.id, a.session_id, a.content, a.metadata_json,
                    a.created_at, bm25(archival_fts) as rank
                FROM archival_memory a
                JOIN archival_fts f ON a.rowid = f.rowid
                WHERE archival_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, k * 3),
            )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            formatted = format_sqlite_result(row, divisor=_recall_cfg.bm25_normalization_divisor)
            if formatted["metadata"].get("type") != "session_learning":
                continue
            results.append(formatted)
            if len(results) >= k:
                break
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
    """
    from scripts.core.db.embedding_service import EmbeddingService
    from scripts.core.db.postgres_pool import get_pool, init_pgvector

    pool = await get_pool()

    embedder = EmbeddingService(provider=provider)
    try:
        query_embedding = await embedder.embed(query, input_type="query")
    finally:
        await embedder.aclose()

    text_query = query
    use_tsquery = False
    if expand:
        try:
            from scripts.core.query_expansion import expand_query

            expanded = await expand_query(
                query, query_embedding,
                max_expansion_terms=max_expansion_terms,
            )
            if expanded != query:
                text_query = expanded
                use_tsquery = True
                logger.debug("Expanded query: %r -> %r", query, expanded)
        except Exception:
            logger.debug("Query expansion failed, using original", exc_info=True)

    rrf_cte = build_rrf_cte(chain_filter=True, use_tsquery=use_tsquery)
    rrf_cte_plain = build_rrf_cte(chain_filter=False, use_tsquery=use_tsquery)

    boosted_tail = """
            SELECT
                a.id, a.session_id, a.content, a.metadata, a.created_at,
                a.recall_count, a.last_recalled,
                c.rrf_score +
                    CASE WHEN COALESCE(a.recall_count, 0) = 0 THEN 0
                    ELSE log(2.0, 1 + COALESCE(a.recall_count, 0)) * $5
                    END as boosted_score,
                c.rrf_score as raw_rrf_score, c.fts_rank, c.vec_rank
            FROM combined c
            JOIN archival_memory a ON a.id = c.id
            ORDER BY boosted_score DESC
            LIMIT $4
            """

    plain_tail = """
            SELECT
                a.id, a.session_id, a.content, a.metadata, a.created_at,
                c.rrf_score, c.fts_rank, c.vec_rank
            FROM combined c
            JOIN archival_memory a ON a.id = c.id
            ORDER BY c.rrf_score DESC
            LIMIT $4
            """

    has_decay_columns = True
    async with pool.acquire() as conn:
        await init_pgvector(conn)
        boost = _recall_cfg.recall_boost_multiplier
        query_args = (text_query, str(query_embedding), rrf_k, k * 2, boost)

        try:
            rows = await conn.fetch(rrf_cte + boosted_tail, *query_args)
        except Exception:
            logger.debug("RRF boosted+chain fallback", exc_info=True)
            has_decay_columns = False
            try:
                rows = await conn.fetch(rrf_cte + plain_tail, *query_args)
            except Exception:
                logger.debug("RRF plain+chain fallback", exc_info=True)
                rows = await conn.fetch(rrf_cte_plain + plain_tail, *query_args)

        if not rows and use_tsquery:
            logger.debug("Expanded tsquery returned no results, falling back to plainto_tsquery")
            plain_cte = build_rrf_cte(chain_filter=True, use_tsquery=False)
            plain_args = (query, str(query_embedding), rrf_k, k * 2, boost)
            try:
                tail = boosted_tail if has_decay_columns else plain_tail
                rows = await conn.fetch(plain_cte + tail, *plain_args)
            except Exception:
                logger.debug("plainto_tsquery fallback also failed", exc_info=True)

    results = []
    for row in rows:
        result = format_rrf_result(row, has_decay=has_decay_columns)
        if similarity_threshold > 0 and result["similarity"] < similarity_threshold:
            continue
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
    """Search learnings using PostgreSQL (vector similarity or text fallback)."""
    from scripts.core.db.embedding_service import EmbeddingService
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()

    async with pool.acquire() as conn:
        try:
            count_row = await conn.fetchrow(
                """
                SELECT COUNT(*) as cnt FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'
                    AND embedding IS NOT NULL
                    AND superseded_by IS NULL
                """
            )
        except Exception:
            logger.debug("Chain filter fallback in embedding probe", exc_info=True)
            count_row = await conn.fetchrow(
                """
                SELECT COUNT(*) as cnt FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'
                    AND embedding IS NOT NULL
                """
            )
        has_embeddings = count_row["cnt"] > 0

    if has_embeddings:
        embedder = EmbeddingService(provider=provider)
        try:
            query_embedding = await embedder.embed(query, input_type="query")
        except Exception:
            logger.warning(
                "Embedding generation failed, falling back to text search",
                exc_info=True,
            )
            if text_fallback:
                has_embeddings = False
            else:
                return []
        finally:
            await embedder.aclose()

    if has_embeddings:
        async with pool.acquire() as conn:
            from scripts.core.db.postgres_pool import init_pgvector
            await init_pgvector(conn)

            if recency_weight > 0:
                _recency_sql = """
                    WITH scored AS (
                        SELECT
                            id, session_id, content, metadata, created_at,
                            1 - (embedding <=> $1::vector) as similarity,
                            GREATEST(0, 1.0 - EXTRACT(EPOCH FROM NOW() - created_at)
                                     / (30 * 86400)) as recency
                        FROM archival_memory
                        WHERE metadata->>'type' = 'session_learning'
                            AND embedding IS NOT NULL
                            {chain_filter}
                    )
                    SELECT
                        id, session_id, content, metadata, created_at,
                        similarity, recency,
                        (1.0 - $3::float) * similarity + $3::float * recency
                            as combined_score
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
                        id, session_id, content, metadata, created_at,
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
        _text_sql = """
                SELECT
                    id, session_id, content, metadata, created_at,
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
        formatted = format_vector_result(
            row, similarity_threshold=similarity_threshold,
        )
        if formatted is not None:
            results.append(formatted)

    return results
