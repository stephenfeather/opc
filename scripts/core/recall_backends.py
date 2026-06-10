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


def merge_project_into_metadata(
    metadata: dict[str, Any], row: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay the archival_memory.project column onto result metadata.

    The reranker's project_match signal reads metadata["project"], but the
    canonical project attribution lives in the project column — 37% of rows
    have the column set with no metadata key (issue #130). The column wins
    when present and non-null. Rows from backends without the column
    (SQLite) pass through unchanged.
    """
    try:
        project = row["project"]
    except (KeyError, IndexError):
        return metadata
    if project:
        return {**metadata, "project": project}
    return metadata


def format_text_result(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a text/vector search DB row to a result dict."""
    return {
        "id": str(row["id"]),
        "session_id": row["session_id"],
        "content": row["content"],
        "metadata": merge_project_into_metadata(
            format_row_metadata(row["metadata"]), row,
        ),
        "created_at": row["created_at"],
        "similarity": float(row["similarity"]),
    }


def format_sqlite_result(
    row: Mapping[str, Any], *, divisor: float,
) -> dict[str, Any]:
    """Convert a SQLite FTS5 row to a result dict with normalized BM25 score."""
    metadata = merge_project_into_metadata(
        format_row_metadata(row["metadata_json"]), row,
    )

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
        "metadata": merge_project_into_metadata(
            format_row_metadata(row["metadata"]), row,
        ),
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
        "metadata": merge_project_into_metadata(
            format_row_metadata(row_dict["metadata"]), row_dict,
        ),
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


def render_recall_sql(
    template: str,
    *,
    include_project: bool,
    project_expr: str = ", project",
    **fmt: str,
) -> str:
    """Render a recall SQL template, optionally selecting the project column.

    The project column comes from an additive migration
    (scripts/migrations/add_project_column.sql); pre-migration databases
    must receive project-free SQL instead of UndefinedColumnError
    (issue #130 review finding).
    """
    return template.format(
        project_col=project_expr if include_project else "", **fmt,
    )


# ---------------------------------------------------------------------------
# SQL constants — module-level templates so wiring (e.g. the project column
# required by the reranker, issue #130) is testable without a database
# connection. Render with render_recall_sql().
# ---------------------------------------------------------------------------

_TEXT_ONLY_FTS_SQL = """
    SELECT
        id, session_id, content, metadata, created_at{project_col},
        ts_rank(to_tsvector('english', content),
                to_tsquery('english', $1)) as similarity
    FROM archival_memory
    WHERE metadata->>'type' = 'session_learning'
        AND superseded_by IS NULL
        AND to_tsvector('english', content) @@ to_tsquery('english', $1)
    ORDER BY similarity DESC, created_at DESC
    LIMIT $2
    """

_TEXT_ONLY_FTS_NO_CHAIN_SQL = """
    SELECT
        id, session_id, content, metadata, created_at{project_col},
        ts_rank(to_tsvector('english', content),
                to_tsquery('english', $1)) as similarity
    FROM archival_memory
    WHERE metadata->>'type' = 'session_learning'
        AND to_tsvector('english', content) @@ to_tsquery('english', $1)
    ORDER BY similarity DESC, created_at DESC
    LIMIT $2
    """

_TEXT_ONLY_ILIKE_SQL = """
    SELECT
        id, session_id, content, metadata, created_at{project_col},
        0.1 as similarity
    FROM archival_memory
    WHERE metadata->>'type' = 'session_learning'
        AND superseded_by IS NULL
        AND content ILIKE '%' || $1 || '%'
    ORDER BY created_at DESC
    LIMIT $2
    """

_TEXT_ONLY_ILIKE_NO_CHAIN_SQL = """
    SELECT
        id, session_id, content, metadata, created_at{project_col},
        0.1 as similarity
    FROM archival_memory
    WHERE metadata->>'type' = 'session_learning'
        AND content ILIKE '%' || $1 || '%'
    ORDER BY created_at DESC
    LIMIT $2
    """

_RRF_BOOSTED_TAIL_SQL = """
    SELECT
        a.id, a.session_id, a.content, a.metadata, a.created_at{project_col},
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

_RRF_PLAIN_TAIL_SQL = """
    SELECT
        a.id, a.session_id, a.content, a.metadata, a.created_at{project_col},
        c.rrf_score, c.fts_rank, c.vec_rank
    FROM combined c
    JOIN archival_memory a ON a.id = c.id
    ORDER BY c.rrf_score DESC
    LIMIT $4
    """

_PG_RECENCY_SQL = """
    WITH scored AS (
        SELECT
            id, session_id, content, metadata, created_at{project_col},
            1 - (embedding <=> $1::vector) as similarity,
            GREATEST(0, 1.0 - EXTRACT(EPOCH FROM NOW() - created_at)
                     / (30 * 86400)) as recency
        FROM archival_memory
        WHERE metadata->>'type' = 'session_learning'
            AND embedding IS NOT NULL
            {chain_filter}
    )
    SELECT
        id, session_id, content, metadata, created_at{project_col},
        similarity, recency,
        (1.0 - $3::float) * similarity + $3::float * recency
            as combined_score
    FROM scored
    ORDER BY combined_score DESC
    LIMIT $2
    """

_PG_VECTOR_SQL = """
    SELECT
        id, session_id, content, metadata, created_at{project_col},
        1 - (embedding <=> $1::vector) as similarity
    FROM archival_memory
    WHERE metadata->>'type' = 'session_learning'
        AND embedding IS NOT NULL
        {chain_filter}
    ORDER BY embedding <=> $1::vector
    LIMIT $2
    """

_PG_TEXT_FALLBACK_SQL = """
    SELECT
        id, session_id, content, metadata, created_at{project_col},
        0.5 as similarity
    FROM archival_memory
    WHERE metadata->>'type' = 'session_learning'
        AND content ILIKE '%' || $1 || '%'
        {chain_filter}
    ORDER BY created_at DESC
    LIMIT $2
    """


# ---------------------------------------------------------------------------
# I/O handlers — async functions that interact with databases
# ---------------------------------------------------------------------------

_recall_cfg = _get_config().recall

# Cached result of the project-column capability probe. None = not yet
# probed (or last probe failed transiently — retry next call). Only
# definitive answers are cached: True on probe success, False on a
# concrete missing-column/table error or a mid-query downgrade. Lives for
# the process lifetime: a migration applied while a daemon is running is
# picked up on restart.
_project_column_cache: bool | None = None

# Probes the exact relation the recall queries hit (same search_path
# resolution), not information_schema by bare table name — schema drift
# between the two was a review round-2 finding. LIMIT 0 keeps it free.
_PROJECT_COLUMN_PROBE_SQL = "SELECT project FROM archival_memory LIMIT 0"


def _missing_relation_errors() -> tuple[type[Exception], ...]:
    """asyncpg error classes meaning the column/table definitively lacks."""
    try:
        from asyncpg.exceptions import UndefinedColumnError, UndefinedTableError

        return (UndefinedColumnError, UndefinedTableError)
    except ImportError:  # pragma: no cover - asyncpg absent (sqlite-only)
        return ()


# Matches Postgres undefined_column messages about the project column in any
# qualification: 'column "project"', 'column a.project',
# 'column archival_memory.project'.
_PROJECT_COLUMN_ERROR_RE = re.compile(
    r'column\s+"?(?:[\w]+\.)?"?project"?\s+does not exist', re.IGNORECASE,
)


def _is_project_capability_error(exc: BaseException) -> bool:
    """True only when exc says the project column (or the table) is missing.

    Other additive columns (superseded_by, recall_count, last_recalled)
    raise the same UndefinedColumnError class on mixed-schema installs and
    must keep flowing into their own chain/decay fallbacks instead of
    being misread as a project-capability miss (review round 3).
    """
    errors = _missing_relation_errors()
    if not errors or not isinstance(exc, errors):
        return False
    try:
        from asyncpg.exceptions import UndefinedTableError
    except ImportError:  # pragma: no cover - asyncpg absent
        return False
    if isinstance(exc, UndefinedTableError):
        return True
    return bool(_PROJECT_COLUMN_ERROR_RE.search(str(exc)))


def reset_project_column_cache() -> None:
    """Clear the cached capability probe (test isolation)."""
    global _project_column_cache
    _project_column_cache = None


def _set_project_column_cache_for_tests(value: bool | None) -> None:
    """Force the probe cache to a value (test-only stale-cache simulation)."""
    global _project_column_cache
    _project_column_cache = value


def mark_project_column_missing() -> None:
    """Downgrade to project-free SQL after a mid-query UndefinedColumnError."""
    global _project_column_cache
    if _project_column_cache is not False:
        logger.warning(
            "archival_memory.project disappeared mid-process; recall "
            "downgraded to no-project mode (project_match disabled, issue #130)"
        )
    _project_column_cache = False


async def project_column_available(conn: Any) -> bool:
    """Check (once per process) whether archival_memory.project exists.

    Only definitive answers are cached. A transient probe failure
    (timeout, connection reset) degrades THIS call to project-free SQL
    but leaves the cache unset so the next recall retries — one hiccup
    must not silently disable project scoping for the process lifetime.
    """
    global _project_column_cache
    if _project_column_cache is not None:
        return _project_column_cache
    try:
        await conn.fetch(_PROJECT_COLUMN_PROBE_SQL)
    except _missing_relation_errors():
        logger.warning(
            "archival_memory.project column missing — recall running in "
            "degraded no-project mode (project_match disabled, issue #130). "
            "Apply scripts/migrations/add_project_column.sql to enable."
        )
        _project_column_cache = False
        return False
    except Exception:
        logger.warning(
            "project column probe failed transiently; degrading this recall "
            "to no-project SQL and retrying the probe on the next call",
            exc_info=True,
        )
        return False
    _project_column_cache = True
    return True


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

    async def _fetch_all(conn: Any, has_project: bool) -> list[Any]:
        try:
            rows = await conn.fetch(
                render_recall_sql(_TEXT_ONLY_FTS_SQL, include_project=has_project),
                or_query, k,
            )
        except Exception as exc:
            if _is_project_capability_error(exc):
                raise
            logger.debug("Chain filter fallback in text_only_postgres FTS", exc_info=True)
            rows = await conn.fetch(
                render_recall_sql(
                    _TEXT_ONLY_FTS_NO_CHAIN_SQL, include_project=has_project,
                ),
                or_query, k,
            )

        if not rows:
            first_word = query.split()[0] if query.split() else query
            try:
                rows = await conn.fetch(
                    render_recall_sql(
                        _TEXT_ONLY_ILIKE_SQL, include_project=has_project,
                    ),
                    first_word, k,
                )
            except Exception as exc:
                if _is_project_capability_error(exc):
                    raise
                logger.debug("Chain filter fallback in text_only_postgres ILIKE", exc_info=True)
                rows = await conn.fetch(
                    render_recall_sql(
                        _TEXT_ONLY_ILIKE_NO_CHAIN_SQL, include_project=has_project,
                    ),
                    first_word, k,
                )
        return rows

    pool = await get_pool()
    async with pool.acquire() as conn:
        has_project = await project_column_available(conn)
        try:
            rows = await _fetch_all(conn, has_project)
        except _missing_relation_errors() as exc:
            # Stale/wrong capability answer (schema drift after probe):
            # downgrade and retry once with project-free SQL. Errors about
            # other columns are not a project-capability signal.
            if not has_project or not _is_project_capability_error(exc):
                raise
            mark_project_column_missing()
            rows = await _fetch_all(conn, False)

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

    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if not words:
        words = ["a"]
    fts_query = " OR ".join(f'"{w}"' for w in words)

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

    has_decay_columns = True

    async def _fetch_all(conn: Any, has_project: bool) -> list[Any]:
        nonlocal has_decay_columns
        boosted_tail = render_recall_sql(
            _RRF_BOOSTED_TAIL_SQL,
            include_project=has_project, project_expr=", a.project",
        )
        plain_tail = render_recall_sql(
            _RRF_PLAIN_TAIL_SQL,
            include_project=has_project, project_expr=", a.project",
        )
        boost = _recall_cfg.recall_boost_multiplier
        embedding_str = str(query_embedding)
        boosted_args = (text_query, embedding_str, rrf_k, k * 2, boost)
        plain_args = (text_query, embedding_str, rrf_k, k * 2)

        try:
            rows = await conn.fetch(rrf_cte + boosted_tail, *boosted_args)
        except Exception as exc:
            if _is_project_capability_error(exc):
                raise
            logger.debug("RRF boosted+chain fallback", exc_info=True)
            has_decay_columns = False
            try:
                rows = await conn.fetch(rrf_cte + plain_tail, *plain_args)
            except Exception as exc:
                if _is_project_capability_error(exc):
                    raise
                logger.debug("RRF plain+chain fallback", exc_info=True)
                rows = await conn.fetch(
                    rrf_cte_plain + plain_tail, *plain_args,
                )

        if not rows and use_tsquery:
            logger.debug("Expanded tsquery returned no results, falling back to plainto_tsquery")
            plain_cte = build_rrf_cte(chain_filter=True, use_tsquery=False)
            fb_boosted = (query, embedding_str, rrf_k, k * 2, boost)
            fb_plain = (query, embedding_str, rrf_k, k * 2)
            try:
                if has_decay_columns:
                    rows = await conn.fetch(plain_cte + boosted_tail, *fb_boosted)
                else:
                    rows = await conn.fetch(plain_cte + plain_tail, *fb_plain)
            except Exception as exc:
                if _is_project_capability_error(exc):
                    raise
                logger.debug("plainto_tsquery chain fallback", exc_info=True)
                try:
                    no_chain_cte = build_rrf_cte(
                        chain_filter=False, use_tsquery=False,
                    )
                    rows = await conn.fetch(
                        no_chain_cte + plain_tail, *fb_plain,
                    )
                except Exception as exc:
                    if _is_project_capability_error(exc):
                        raise
                    logger.debug("plainto_tsquery fallback also failed", exc_info=True)
        return rows

    async with pool.acquire() as conn:
        await init_pgvector(conn)
        has_project = await project_column_available(conn)
        try:
            rows = await _fetch_all(conn, has_project)
        except _missing_relation_errors() as exc:
            if not has_project or not _is_project_capability_error(exc):
                raise
            mark_project_column_missing()
            rows = await _fetch_all(conn, False)

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

    async def _fetch_with_chain_fallback(
        conn: Any,
        template: str,
        has_project: bool,
        args: tuple[Any, ...],
        label: str,
    ) -> list[Any]:
        def _sql(chain_filter: str) -> str:
            return render_recall_sql(
                template, include_project=has_project, chain_filter=chain_filter,
            )
        try:
            return await conn.fetch(_sql("AND superseded_by IS NULL"), *args)
        except Exception as exc:
            if _is_project_capability_error(exc):
                raise
            logger.debug("Chain filter fallback in postgres %s", label, exc_info=True)
            return await conn.fetch(_sql(""), *args)

    async def _fetch_branch(
        conn: Any, template: str, has_project: bool,
        args: tuple[Any, ...], label: str,
    ) -> list[Any]:
        try:
            return await _fetch_with_chain_fallback(
                conn, template, has_project, args, label,
            )
        except _missing_relation_errors() as exc:
            if not has_project or not _is_project_capability_error(exc):
                raise
            mark_project_column_missing()
            return await _fetch_with_chain_fallback(
                conn, template, False, args, label,
            )

    if has_embeddings:
        async with pool.acquire() as conn:
            from scripts.core.db.postgres_pool import init_pgvector
            await init_pgvector(conn)
            has_project = await project_column_available(conn)

            if recency_weight > 0:
                rows = await _fetch_branch(
                    conn, _PG_RECENCY_SQL, has_project,
                    (str(query_embedding), k, recency_weight), "recency",
                )
            else:
                rows = await _fetch_branch(
                    conn, _PG_VECTOR_SQL, has_project,
                    (str(query_embedding), k), "vector",
                )
    elif text_fallback:
        async with pool.acquire() as conn:
            has_project = await project_column_available(conn)
            rows = await _fetch_branch(
                conn, _PG_TEXT_FALLBACK_SQL, has_project, (query, k), "text",
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
