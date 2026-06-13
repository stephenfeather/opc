"""Search backends for recall learnings.

Contains the four search implementations:
- text-only postgres (FTS)
- SQLite (FTS5/BM25)
- hybrid RRF (text + vector fusion)
- postgres vector (cosine similarity with optional recency)

Each returns list[dict] with keys: id, session_id, content, metadata, created_at, similarity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.core.config import get_config as _get_config

logger = logging.getLogger(__name__)


@dataclass
class SearchCapture:
    """Mutable out-param for surfacing a backend's internal embedding (#54).

    ``search_learnings_hybrid_rrf`` already embeds the query to drive the
    vector leg of RRF. To wire the type-affinity reranker signal without a
    second embed call, the caller passes a ``SearchCapture``; the backend fills
    ``query_embedding`` and ``model_label`` after a *successful* embed. The
    degraded text-only path (missing key / embed failure) and the
    text-only/sqlite backends leave it at its ``None`` defaults, so the caller
    keeps type affinity disabled (neutral reranking) on those paths.

    Mutable by design (out-param contract); not frozen.
    """

    query_embedding: list[float] | None = None
    model_label: str | None = None

# Issue #53: hybrid recall degrades to text-only by design when the query
# embedding is unavailable (this is the contract — the memory-awareness hook
# relies on it for the missing-key case). To keep that quiet path from
# spamming hook-captured stderr, the human-readable warning is latched to at
# most once per process; logger.debug still fires on every degrade. The hook
# spawns a fresh process per prompt, so this caps it at one warning per prompt.
_EMBED_DEGRADE_WARNED = False

# Issue #53: a recall-specific deadline on the hybrid query-embed. The voyage
# provider uses httpx timeout=30.0 with retries, so a network stall would
# otherwise take ~90s+ to raise — long past the memory-awareness hook's 5s
# spawnSync budget, so the text-only fallback would never run and the recall
# output would be lost. 2.0s leaves ~3s for the text-only query + reranking +
# output. Same bounded-best-effort pattern as RECORD_RECALL_TIMEOUT in
# recall_learnings.py (issue #140); asyncio.wait_for cancels the in-flight
# embed on timeout (cancelling an httpx request is safe).
QUERY_EMBED_TIMEOUT = 2.0


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
    when non-empty; NULL and empty-string values are treated as absent
    (an empty project scores 0 in project_match regardless, and the
    issue #130 canonicalizer never stores one). When the column
    contributes, a new dict is returned; otherwise the input metadata is
    returned as-is (it is already per-row, never shared).
    """
    try:
        project = row["project"]
    except (KeyError, IndexError):
        return dict(metadata)
    if project is not None:
        return {**metadata, "project": project}
    return dict(metadata)


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


def model_filter_clause(model_label: str | None, *, param_index: int) -> str:
    """Build the optional 'AND embedding_model = $N' clause (issue #151).

    Pins recall to a single embedding space so voyage-space queries never
    cross-compare cosine distance against bge-space vectors (split-brain
    corpus). Returns "" — leaving the SQL byte-identical to the pre-#151
    path — when no label is supplied. The caller binds ``model_label`` as the
    ``param_index``-th positional arg.

    Zero-match is intentional and degrades safely: in the RRF CTE the clause
    lives on ``vector_ranked`` only, so a label that matches no rows empties
    the vector leg and the FULL OUTER JOIN keeps the text (fts) leg — recall
    becomes text-only rather than erroring.
    """
    if not model_label:
        return ""
    return f"AND embedding_model = ${param_index}"


def _build_vector_ranked_cte(
    *, chain_clause: str, vector_extra: str, candidate_param: int | None,
) -> str:
    """Render the ``vector_ranked`` CTE body (issue #153).

    With ``candidate_param`` it wraps a bounded ANN inner subquery
    (``ORDER BY embedding <=> $2::vector LIMIT $N``) so the HNSW index can
    accelerate the scan, then ranks the candidates with an outer
    ``ROW_NUMBER()``. Without it, it reproduces the legacy unbounded leg
    byte-identically (the window orders straight off the table).
    """
    if candidate_param is None:
        return f"""vector_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> $2::vector) as vec_rank
                FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'{chain_clause}{vector_extra}
                AND embedding IS NOT NULL
            )"""
    return f"""vector_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY dist) as vec_rank
                FROM (
                    SELECT
                        id,
                        embedding <=> $2::vector AS dist
                    FROM archival_memory
                    WHERE metadata->>'type' = 'session_learning'{chain_clause}{vector_extra}
                    AND embedding IS NOT NULL
                    ORDER BY embedding <=> $2::vector
                    LIMIT ${candidate_param}
                ) cand
            )"""


def build_rrf_cte(
    *,
    chain_filter: bool,
    use_tsquery: bool = False,
    project_filter: str | None = None,
    model_filter: str | None = None,
    candidate_param: int | None = None,
) -> str:
    """Build the SQL CTE for RRF (Reciprocal Rank Fusion) queries.

    ``project_filter`` is the optional fetch-time scoping predicate for
    ``--project-first`` (issue #139). It must live inside *both* ranking
    subqueries (fts_ranked and vector_ranked) so the project predicate
    shrinks the pool *before* ranking — filtering only the tail would rank
    the global pool and then drop rows, defeating the purpose. ``None``
    leaves the CTE byte-identical to today.

    ``model_filter`` is the optional ``AND embedding_model = $N`` predicate
    for single-space recall (issue #151). It lives in ``vector_ranked``
    *only* — never the FTS leg — so a label matching no rows empties the
    vector leg and the FULL OUTER JOIN degrades RRF to the text leg rather
    than erroring. ``None`` leaves the CTE byte-identical to the pre-#151
    path.

    ``candidate_param`` is the positional index of the bounded ANN candidate
    ``LIMIT`` (issue #153). When given, ``vector_ranked`` ranks a bounded
    inner subquery shaped ``... ORDER BY embedding <=> $2::vector LIMIT $N`` —
    the only shape pgvector's HNSW index accelerates — and an outer
    ``ROW_NUMBER()`` ranks the (already small) candidate set for RRF fusion.
    The filters live *inside* the inner subquery so the pool shrinks before
    the LIMIT. ``None`` preserves the legacy unbounded leg byte-identically
    for callers/tests that don't pass it.
    """
    chain_clause = (
        "\n                AND superseded_by IS NULL" if chain_filter else ""
    )
    project_clause = (
        f"\n                {project_filter}" if project_filter else ""
    )
    model_clause = (
        f"\n                {model_filter}" if model_filter else ""
    )
    # Vector leg carries both project and model predicates; fts leg only the
    # project predicate (issue #151: model filter is vector-only).
    vector_extra = f"{project_clause}{model_clause}"
    tsquery_fn = "to_tsquery" if use_tsquery else "plainto_tsquery"
    vector_ranked = _build_vector_ranked_cte(
        chain_clause=chain_clause,
        vector_extra=vector_extra,
        candidate_param=candidate_param,
    )
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
                WHERE metadata->>'type' = 'session_learning'{chain_clause}{project_clause}
                AND to_tsvector('english', content) @@ {tsquery_fn}('english', $1)
            ),
            {vector_ranked},
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
    project_filter: str | None = None,
    model_filter: str | None = None,
    **fmt: str,
) -> str:
    """Render a recall SQL template, optionally selecting the project column.

    The project column comes from an additive migration
    (scripts/migrations/add_project_column.sql); pre-migration databases
    must receive project-free SQL instead of UndefinedColumnError
    (issue #130 review finding).

    ``project_filter`` is the optional fetch-time scoping clause for
    ``--project-first`` (issue #139). It is the fully-formed predicate the
    backend computed (e.g. ``"AND LOWER(project) = $3"`` — bound last so
    the existing positional params keep their numbers). ``None`` renders
    the ``{project_filter}`` placeholder empty, leaving the SQL
    byte-identical to today.
    """
    return template.format(
        project_col=project_expr if include_project else "",
        project_filter=project_filter or "",
        model_filter=model_filter or "",
        **fmt,
    )


def project_filter_clause(
    project: str | None, *, has_project: bool, param_index: int,
) -> str:
    """Build the optional 'AND LOWER(project) = $N' scoping clause (issue #139).

    Returns "" — leaving the SQL byte-identical to the global path — when no
    project was requested or the column is unavailable (pre-migration DB).
    The caller binds ``project`` as the ``param_index``-th positional arg.

    The predicate is case-insensitive (review round 2): un-migrated DBs may
    still hold case variants like 'OPC'/'Opc'. The bind value is lowercase by
    construction (project_naming.canonicalize_project lowercases), so
    LOWER(project) = $N compares lower-to-lower and matches every case
    variant of the same project. Legacy alias/flattened-path values that the
    canonicalizer would collapse remain covered by the global fill pass and
    the project_match rerank signal; run scripts/migrations/
    normalize_project_values.py for full project-first effectiveness on
    legacy data.
    """
    if not project or not has_project:
        return ""
    return f"AND LOWER(project) = ${param_index}"


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
        {project_filter}
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
        {project_filter}
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
        {project_filter}
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
        {project_filter}
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
            {project_filter}
            {model_filter}
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
        {project_filter}
        {model_filter}
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
        {project_filter}
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


# ---------------------------------------------------------------------------
# embedding_model column capability probe (issue #151, round 1 FIX 1)
#
# The model filter (AND embedding_model = $N) and the explicit store-time
# column write both assume the embedding_model column exists. On a
# pre-migration DB it may not, so recall would raise UndefinedColumnError (a
# hard failure, not degradation) and stores would fail. Mirror the #139
# project-column probe: probe once, cache only definitive answers, and when
# the column is absent bind model_label=None (filter off, SQL byte-identical
# to the pre-#151 path). Apply scripts/migrations/add_embedding_hnsw.sql to
# add the column.
# ---------------------------------------------------------------------------

_embedding_model_column_cache: bool | None = None

_EMBEDDING_MODEL_COLUMN_PROBE_SQL = (
    "SELECT embedding_model FROM archival_memory LIMIT 0"
)

def reset_embedding_model_column_cache() -> None:
    """Clear the cached embedding_model capability probe (test isolation)."""
    global _embedding_model_column_cache
    _embedding_model_column_cache = None


def _set_embedding_model_column_cache_for_tests(value: bool | None) -> None:
    """Force the embedding_model probe cache to a value (test-only)."""
    global _embedding_model_column_cache
    _embedding_model_column_cache = value


async def embedding_model_column_available(conn: Any) -> bool:
    """Check (once per process) whether archival_memory.embedding_model exists.

    Only definitive answers are cached. A transient probe failure degrades
    THIS call to the unfiltered (pre-#151) path but leaves the cache unset so
    the next recall retries — one hiccup must not disable the model filter for
    the process lifetime.
    """
    global _embedding_model_column_cache
    if _embedding_model_column_cache is not None:
        return _embedding_model_column_cache
    try:
        await conn.fetch(_EMBEDDING_MODEL_COLUMN_PROBE_SQL)
    except _missing_relation_errors():
        logger.warning(
            "archival_memory.embedding_model column missing — recall/store "
            "running in degraded unfiltered mode (cross-space recall not "
            "prevented, issue #151). Apply "
            "scripts/migrations/add_embedding_hnsw.sql to enable."
        )
        _embedding_model_column_cache = False
        return False
    except Exception:
        logger.warning(
            "embedding_model column probe failed transiently; degrading this "
            "recall to unfiltered SQL and retrying the probe on the next call",
            exc_info=True,
        )
        return False
    _embedding_model_column_cache = True
    return True


async def search_learnings_text_only_postgres(
    query: str, k: int = _recall_cfg.default_k, *, project: str | None = None,
) -> list[dict[str, Any]]:
    """Fast text-only search for PostgreSQL using full-text search.

    Uses tsvector/tsquery with GIN index. Automatic stopword handling.
    Falls back to ILIKE if tsquery fails (e.g., all stopwords).

    ``project`` (issue #139) scopes the fetch to one project via an
    ``AND LOWER(project) = $3`` clause bound last; ``None`` (or a
    pre-migration DB lacking the column) leaves the query byte-identical
    to the global path.
    """
    from scripts.core.db.postgres_pool import get_pool
    from scripts.core.query_expansion import STOPWORDS

    or_query = build_or_query(query, STOPWORDS)

    async def _fetch_all(conn: Any, has_project: bool) -> list[Any]:
        # Project value (when scoping) is bound as $3, after $1 (query) and
        # $2 (limit) — see project_filter_clause's param_index.
        pf = project_filter_clause(project, has_project=has_project, param_index=3)
        extra = (project,) if pf else ()

        def _r(template: str) -> str:
            return render_recall_sql(
                template, include_project=has_project, project_filter=pf,
            )

        try:
            rows = await conn.fetch(_r(_TEXT_ONLY_FTS_SQL), or_query, k, *extra)
        except Exception as exc:
            if _is_project_capability_error(exc):
                raise
            logger.debug("Chain filter fallback in text_only_postgres FTS", exc_info=True)
            rows = await conn.fetch(
                _r(_TEXT_ONLY_FTS_NO_CHAIN_SQL), or_query, k, *extra,
            )

        if not rows:
            first_word = query.split()[0] if query.split() else query
            try:
                rows = await conn.fetch(
                    _r(_TEXT_ONLY_ILIKE_SQL), first_word, k, *extra,
                )
            except Exception as exc:
                if _is_project_capability_error(exc):
                    raise
                logger.debug("Chain filter fallback in text_only_postgres ILIKE", exc_info=True)
                rows = await conn.fetch(
                    _r(_TEXT_ONLY_ILIKE_NO_CHAIN_SQL), first_word, k, *extra,
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
    query: str, k: int = _recall_cfg.default_k, *, project: str | None = None,
) -> list[dict[str, Any]]:
    """Search learnings using SQLite FTS5 (BM25 ranking).

    Cross-session search - finds learnings from ALL sessions.

    The SQLite cache has no project column, so ``project`` (issue #139) is
    accepted for signature parity but ignored — this backend always
    degrades to a global fetch.
    """
    import sqlite3

    if project:
        logger.debug(
            "search_learnings_sqlite ignores project=%r: the SQLite cache "
            "has no project column; degrading to global fetch (issue #139)",
            project,
        )

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
    *,
    project: str | None = None,
    capture: SearchCapture | None = None,
) -> list[dict[str, Any]]:
    """Hybrid RRF search combining text and vector rankings.

    Uses Reciprocal Rank Fusion:
        score = 1/(k + rank_fts) + 1/(k + rank_vector)

    ``project`` (issue #139) scopes the fetch to one project. The predicate
    lives inside both CTE ranking subqueries (so it shrinks the pool before
    ranking) and is bound last; the boosted tail carries 5 base params so
    project is $6, the plain tail 4 so project is $5. ``None`` or a
    pre-migration DB leaves the SQL byte-identical to the global path.
    """
    from scripts.core.db.embedding_service import EmbeddingService
    from scripts.core.db.postgres_pool import get_pool, init_pgvector

    pool = await get_pool()

    # embedder is constructed inside the guarded try so that constructor
    # failures (e.g. VoyageEmbeddingProvider raising when VOYAGE_API_KEY is
    # absent, or a local model load error) degrade like embed() failures
    # rather than bypassing the fallback (review round 1).
    embedder: EmbeddingService | None = None
    model_label: str | None = None
    try:
        # Construction is non-network (provider __init__ only reads env), so
        # it stays outside wait_for and remains visible to the finally below.
        embedder = EmbeddingService(provider=provider)
        # Capture the embedding-space label before the finally closes the
        # embedder; bound into the recall vector leg so the query never
        # cross-compares spaces (issue #151). getattr guards custom embedders
        # that predate model_label — they degrade to the unfiltered (pre-#151)
        # path rather than erroring.
        model_label = getattr(embedder, "model_label", None)
        query_embedding = await asyncio.wait_for(
            embedder.embed(query, input_type="query"),
            timeout=QUERY_EMBED_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - degrade, do not crash recall
        # Issue #53: the memory-awareness hook now calls hybrid (no
        # --text-only). If the query-embed is unavailable (missing API key,
        # model load error, network timeout), degrade to the text-only
        # backend instead of aborting — same pool/k/project semantics and
        # an identical result shape to --text-only. exc text can embed a
        # DSN/host; recall stderr is injected into the model context by
        # hooks, so redact and never echo the query (#139 redactor; aegis).
        global _EMBED_DEGRADE_WARNED
        from scripts.core.db.postgres_pool import sanitize_log_message

        # TimeoutError (from the QUERY_EMBED_TIMEOUT wait_for) has an empty
        # str(); surface a stable reason so the message stays informative.
        reason = sanitize_log_message(str(exc)) or type(exc).__name__
        logger.debug(
            "hybrid query-embed failed (provider %r); degrading to text-only",
            provider,
            exc_info=True,
        )
        if not _EMBED_DEGRADE_WARNED:
            # Latch the human-readable warning to once per process so the
            # --project-first second pass (and repeated recalls) stay quiet.
            _EMBED_DEGRADE_WARNED = True
            print(
                "warning: hybrid recall degraded to text-only "
                f"(provider {provider!r} unavailable or timed out: "
                f"{reason}); "
                "set the provider API key or pass --text-only to silence.",
                file=sys.stderr,
            )
        return await search_learnings_text_only_postgres(
            query, k, project=project,
        )
    finally:
        if embedder is not None:
            # A raise here would override the pending fallback return
            # (FIX A): cleanup failure must never abort recall. Swallow it
            # to the debug log only.
            try:
                await embedder.aclose()
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                logger.debug("embedder close failed", exc_info=True)

    # FIX 1 (issue #151): probe the embedding_model column once up front. On a
    # pre-migration DB drop the model filter everywhere downstream (query
    # expansion neighbors and both vector legs) so recall degrades to the
    # byte-identical pre-#151 SQL instead of raising UndefinedColumnError. The
    # probe result is cached, so the later pool.acquire() reuses it for free.
    if model_label is not None:
        async with pool.acquire() as probe_conn:
            if not await embedding_model_column_available(probe_conn):
                model_label = None

    # Issue #54: surface the embedding (and its now-resolved model-space label)
    # for the type-affinity reranker signal. Only reached after a successful
    # embed — the degraded text-only return above never gets here, so a
    # degraded recall leaves the caller's capture at its None defaults.
    # model_label may be None here (pre-#151 DB without the embedding_model
    # column); the caller must then skip model-filtered centroids rather than
    # cosine across embedding spaces.
    #
    # Finding 3 (round 1): under --project-first the SAME capture is passed to
    # both fetch passes. Only fill an empty capture so the first successful
    # embed wins and the second pass (identical query) never clobbers it.
    if capture is not None and capture.query_embedding is None:
        capture.query_embedding = query_embedding
        capture.model_label = model_label

    text_query = query
    use_tsquery = False
    if expand:
        try:
            from scripts.core.query_expansion import expand_query

            expanded = await expand_query(
                query, query_embedding,
                max_expansion_terms=max_expansion_terms,
                model_label=model_label,
            )
            if expanded != query:
                text_query = expanded
                use_tsquery = True
                logger.debug("Expanded query: %r -> %r", query, expanded)
        except Exception:
            logger.debug("Query expansion failed, using original", exc_info=True)

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

        # Project scoping (issue #139): the predicate lives in the CTE
        # subqueries, bound after the tail's params. The boosted tail's
        # last positional is $5 (project -> $6); the plain tail's is $4
        # (project -> $5). When unscoped, both filters are "" and the
        # *_scoped CTEs equal the plain CTEs — byte-identical default path.
        boosted_pf = project_filter_clause(
            project, has_project=has_project, param_index=6,
        )
        plain_pf = project_filter_clause(
            project, has_project=has_project, param_index=5,
        )

        # Model filter (issue #151) is bound *after* the project value so the
        # #139 positional binds keep their numbers. Boosted base = $5, plain
        # base = $4; project (when active) consumes the next slot, then the
        # model label takes the last slot: boosted -> $7/$6, plain -> $6/$5.
        boosted_model_idx = 7 if boosted_pf else 6
        plain_model_idx = 6 if plain_pf else 5
        boosted_mf = model_filter_clause(model_label, param_index=boosted_model_idx)
        plain_mf = model_filter_clause(model_label, param_index=plain_model_idx)

        # Bounded ANN candidate LIMIT (issue #153) binds *last*, after project
        # and model, mirroring #139/#151 so $1..$5 keep their numbers. Boosted
        # base = $5, plain base = $4; project (when active) +1, model +1, then
        # the candidate count takes the final slot.
        boosted_candidate_idx = 6 + bool(boosted_pf) + bool(boosted_mf)
        plain_candidate_idx = 5 + bool(plain_pf) + bool(plain_mf)
        candidate_count = k * _recall_cfg.vector_candidate_multiplier

        def _cte(
            *, chain: bool, ts: bool, pf: str, mf: str, candidate_param: int,
        ) -> str:
            return build_rrf_cte(
                chain_filter=chain, use_tsquery=ts,
                project_filter=pf, model_filter=mf,
                candidate_param=candidate_param,
            )

        # Append project then model values, matching the param ordering above,
        # then the candidate count as the final positional arg.
        boosted_extra = (
            *((project,) if boosted_pf else ()),
            *((model_label,) if boosted_mf else ()),
        )
        plain_extra = (
            *((project,) if plain_pf else ()),
            *((model_label,) if plain_mf else ()),
        )
        boosted_args = (
            text_query, embedding_str, rrf_k, k * 2, boost,
            *boosted_extra, candidate_count,
        )
        plain_args = (
            text_query, embedding_str, rrf_k, k * 2,
            *plain_extra, candidate_count,
        )

        cte_boosted = _cte(
            chain=True, ts=use_tsquery, pf=boosted_pf, mf=boosted_mf,
            candidate_param=boosted_candidate_idx,
        )
        cte_plain_args = _cte(
            chain=True, ts=use_tsquery, pf=plain_pf, mf=plain_mf,
            candidate_param=plain_candidate_idx,
        )
        cte_plain_nochain = _cte(
            chain=False, ts=use_tsquery, pf=plain_pf, mf=plain_mf,
            candidate_param=plain_candidate_idx,
        )

        try:
            rows = await conn.fetch(cte_boosted + boosted_tail, *boosted_args)
        except Exception as exc:
            if _is_project_capability_error(exc):
                raise
            logger.debug("RRF boosted+chain fallback", exc_info=True)
            has_decay_columns = False
            try:
                rows = await conn.fetch(cte_plain_args + plain_tail, *plain_args)
            except Exception as exc:
                if _is_project_capability_error(exc):
                    raise
                logger.debug("RRF plain+chain fallback", exc_info=True)
                rows = await conn.fetch(
                    cte_plain_nochain + plain_tail, *plain_args,
                )

        if not rows and use_tsquery:
            logger.debug("Expanded tsquery returned no results, falling back to plainto_tsquery")
            fb_boosted = (
                query, embedding_str, rrf_k, k * 2, boost,
                *boosted_extra, candidate_count,
            )
            fb_plain = (
                query, embedding_str, rrf_k, k * 2,
                *plain_extra, candidate_count,
            )
            fb_cte_boosted = _cte(
                chain=True, ts=False, pf=boosted_pf, mf=boosted_mf,
                candidate_param=boosted_candidate_idx,
            )
            fb_cte_plain = _cte(
                chain=True, ts=False, pf=plain_pf, mf=plain_mf,
                candidate_param=plain_candidate_idx,
            )
            try:
                if has_decay_columns:
                    rows = await conn.fetch(fb_cte_boosted + boosted_tail, *fb_boosted)
                else:
                    rows = await conn.fetch(fb_cte_plain + plain_tail, *fb_plain)
            except Exception as exc:
                if _is_project_capability_error(exc):
                    raise
                logger.debug("plainto_tsquery chain fallback", exc_info=True)
                try:
                    no_chain_cte = _cte(
                        chain=False, ts=False, pf=plain_pf, mf=plain_mf,
                        candidate_param=plain_candidate_idx,
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
    *,
    project: str | None = None,
) -> list[dict[str, Any]]:
    """Search learnings using PostgreSQL (vector similarity or text fallback).

    ``project`` (issue #139) scopes the fetch to one project. The clause is
    bound as the last positional arg (its index is derived from each
    branch's base arg count); ``None`` or a pre-migration DB leaves the SQL
    byte-identical to the global path.
    """
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

    # Captured before aclose() so the vector/recency branches can pin the
    # embedding space (issue #151). None for the text-fallback branch.
    model_label: str | None = None
    if has_embeddings:
        embedder = EmbeddingService(provider=provider)
        try:
            model_label = getattr(embedder, "model_label", None)
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
        model_filter_label: str | None = None,
    ) -> list[Any]:
        pf = project_filter_clause(
            project, has_project=has_project, param_index=len(args) + 1,
        )
        scoped_args = (*args, project) if pf else args
        # Model filter binds after the project value (issue #151), so its
        # index follows the (possibly already appended) project arg.
        mf = model_filter_clause(
            model_filter_label, param_index=len(scoped_args) + 1,
        )
        if mf:
            scoped_args = (*scoped_args, model_filter_label)

        def _sql(chain_filter: str) -> str:
            return render_recall_sql(
                template,
                include_project=has_project,
                chain_filter=chain_filter,
                project_filter=pf,
                model_filter=mf,
            )
        try:
            return await conn.fetch(_sql("AND superseded_by IS NULL"), *scoped_args)
        except Exception as exc:
            if _is_project_capability_error(exc):
                raise
            logger.debug("Chain filter fallback in postgres %s", label, exc_info=True)
            return await conn.fetch(_sql(""), *scoped_args)

    async def _fetch_branch(
        conn: Any, template: str, has_project: bool,
        args: tuple[Any, ...], label: str,
        model_filter_label: str | None = None,
    ) -> list[Any]:
        try:
            return await _fetch_with_chain_fallback(
                conn, template, has_project, args, label, model_filter_label,
            )
        except _missing_relation_errors() as exc:
            if not has_project or not _is_project_capability_error(exc):
                raise
            mark_project_column_missing()
            return await _fetch_with_chain_fallback(
                conn, template, False, args, label, model_filter_label,
            )

    if has_embeddings:
        async with pool.acquire() as conn:
            from scripts.core.db.postgres_pool import init_pgvector
            await init_pgvector(conn)
            has_project = await project_column_available(conn)
            # FIX 1 (issue #151): no model filter on a pre-migration DB —
            # degrade to the byte-identical pre-#151 SQL, never raise.
            if model_label is not None and not await embedding_model_column_available(conn):
                model_label = None

            if recency_weight > 0:
                rows = await _fetch_branch(
                    conn, _PG_RECENCY_SQL, has_project,
                    (str(query_embedding), k, recency_weight), "recency",
                    model_label,
                )
            else:
                rows = await _fetch_branch(
                    conn, _PG_VECTOR_SQL, has_project,
                    (str(query_embedding), k), "vector",
                    model_label,
                )

            # FIX 4 (round 2, issue #151): a model-filtered vector query can
            # match zero rows during a split corpus (the query's space has no
            # rows yet). resolve_search_params() sets text_fallback=True even
            # for --vector-only, so honor it: when the vector/recency fetch is
            # empty AND text_fallback is on, run the ILIKE text fallback rather
            # than returning nothing. The text fallback never touches
            # embedding_model, so it is safe across spaces.
            if not rows and text_fallback:
                has_project = await project_column_available(conn)
                rows = await _fetch_branch(
                    conn, _PG_TEXT_FALLBACK_SQL, has_project,
                    (query, k), "text",
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
