"""Pure functions for memory service SQL queries and data formatting.

Contains no I/O — all database interaction stays in memory_service_pg.py.
Functions here build SQL strings, format result rows, pad embeddings,
and generate IDs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

# Active-row predicate: excludes superseded learnings from search results.
# The superseded_by column may not exist on pre-migration databases.
# Callers should set include_active_filter=False when schema detection
# indicates the column is absent.
ACTIVE_ROW_FILTER = "superseded_by IS NULL"

# ==================== ID Generation ====================


def generate_memory_id() -> str:
    """Generate a UUID string for memory ID.

    Returns:
        UUID string suitable for PostgreSQL UUID column.
    """
    return str(uuid4())


# ==================== Embedding Normalization ====================


def pad_embedding(embedding: list[float], target_dim: int = 1024) -> list[float]:
    """Pad or truncate embedding to target dimension.

    Args:
        embedding: Original embedding vector.
        target_dim: Target dimension (default 1024 for bge-large-en-v1.5).

    Returns:
        List of floats with exactly target_dim elements.
    """
    vec = np.array(embedding)
    if len(vec) >= target_dim:
        return vec[:target_dim].tolist()
    return np.pad(vec, (0, target_dim - len(vec)), mode="constant").tolist()


# ==================== Row Formatting ====================


def format_archival_row(
    row: Mapping[str, Any],
    extra_fields: list[str] | None = None,
    float_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a database row to a result dict with parsed metadata.

    Args:
        row: Database row (dict-like).
        extra_fields: Additional fields to include from the row.
        float_fields: Fields to explicitly convert to float (e.g., Decimal → float).

    Returns:
        Formatted result dict.
    """
    float_set = frozenset(float_fields or [])
    raw_meta = row["metadata"]
    if raw_meta is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw_meta, str):
        metadata = json.loads(raw_meta)
    else:
        metadata = raw_meta
    result: dict[str, Any] = {
        "id": str(row["id"]),
        "content": row["content"],
        "metadata": metadata,
        "created_at": row["created_at"],
    }
    for field in extra_fields or []:
        value = row[field]
        result[field] = float(value) if field in float_set else value
    return result


def format_rows(
    rows: list[Mapping[str, Any]],
    extra_fields: list[str] | None = None,
    float_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Format multiple database rows.

    Args:
        rows: List of database rows.
        extra_fields: Additional fields per row.
        float_fields: Fields to convert to float.

    Returns:
        List of formatted result dicts.
    """
    return [
        format_archival_row(row, extra_fields=extra_fields, float_fields=float_fields)
        for row in rows
    ]


# ==================== Date Filter Building ====================


def build_date_conditions(
    start_date: datetime | None,
    end_date: datetime | None,
    param_start_idx: int,
) -> tuple[list[str], list[datetime], int]:
    """Build SQL WHERE conditions for date filtering.

    Args:
        start_date: Optional lower bound (inclusive).
        end_date: Optional upper bound (inclusive).
        param_start_idx: Next available $N parameter index.

    Returns:
        Tuple of (conditions, params, next_param_idx).
    """
    conditions: list[str] = []
    params: list[datetime] = []
    idx = param_start_idx

    if start_date is not None:
        conditions.append(f"created_at >= ${idx}")
        params.append(start_date)
        idx += 1

    if end_date is not None:
        conditions.append(f"created_at <= ${idx}")
        params.append(end_date)
        idx += 1

    return conditions, params, idx


# ==================== SQL Query Builders ====================


def build_text_search_sql(
    session_id: str,
    agent_id: str | None,
    query: str,
    limit: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    include_active_filter: bool = True,
) -> tuple[str, list[Any]]:
    """Build SQL for full-text search on archival memory.

    Args:
        session_id: Session ID for scoping.
        agent_id: Optional agent ID.
        query: FTS query string.
        limit: Max results.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        include_active_filter: Whether to add superseded_by IS NULL filter.
            Set False when the column doesn't exist (pre-migration).

    Returns:
        Tuple of (sql_string, params_list).
    """
    conditions = [
        "session_id = $1",
        "agent_id IS NOT DISTINCT FROM $2",
        "to_tsvector('english', content) @@ plainto_tsquery('english', $3)",
    ]
    if include_active_filter:
        conditions.append(ACTIVE_ROW_FILTER)
    params: list[Any] = [session_id, agent_id, query]

    date_conds, date_params, next_idx = build_date_conditions(
        start_date, end_date, param_start_idx=4
    )
    conditions.extend(date_conds)
    params.extend(date_params)
    params.append(limit)

    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT
            id, content, metadata, created_at,
            ts_rank(
                to_tsvector('english', content),
                plainto_tsquery('english', $3)
            ) as rank
        FROM archival_memory
        WHERE {where_clause}
        ORDER BY rank DESC
        LIMIT ${next_idx}
    """
    return sql, params


def build_vector_search_sql(
    session_id: str,
    agent_id: str | None,
    query_embedding: list[float],
    limit: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    include_active_filter: bool = True,
) -> tuple[str, list[Any]]:
    """Build SQL for vector similarity search on archival memory.

    Args:
        session_id: Session ID for scoping.
        agent_id: Optional agent ID.
        query_embedding: Query embedding (already padded).
        limit: Max results.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        include_active_filter: Whether to add superseded_by IS NULL filter.

    Returns:
        Tuple of (sql_string, params_list).
    """
    conditions = [
        "session_id = $1",
        "agent_id IS NOT DISTINCT FROM $2",
        "embedding IS NOT NULL",
    ]
    if include_active_filter:
        conditions.append(ACTIVE_ROW_FILTER)
    params: list[Any] = [session_id, agent_id, query_embedding]

    date_conds, date_params, next_idx = build_date_conditions(
        start_date, end_date, param_start_idx=4
    )
    conditions.extend(date_conds)
    params.extend(date_params)
    params.append(limit)

    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT
            id, content, metadata, created_at,
            1 - (embedding <=> $3::vector) as similarity
        FROM archival_memory
        WHERE {where_clause}
        ORDER BY embedding <=> $3::vector
        LIMIT ${next_idx}
    """
    return sql, params


def build_hybrid_search_sql(
    session_id: str,
    agent_id: str | None,
    text_query: str,
    query_embedding: list[float],
    limit: int,
    text_weight: float = 0.3,
    vector_weight: float = 0.7,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    include_active_filter: bool = True,
) -> tuple[str, list[Any]]:
    """Build SQL for hybrid (text + vector) search.

    Args:
        session_id: Session ID for scoping.
        agent_id: Optional agent ID.
        text_query: FTS query.
        query_embedding: Query embedding (already padded).
        limit: Max results.
        text_weight: Weight for text search score.
        vector_weight: Weight for vector similarity.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        include_active_filter: Whether to add superseded_by IS NULL filter.

    Returns:
        Tuple of (sql_string, params_list).
    """
    conditions = [
        "session_id = $1",
        "agent_id IS NOT DISTINCT FROM $2",
        (
            "(to_tsvector('english', content) @@ plainto_tsquery('english', $3)"
            " OR embedding IS NOT NULL)"
        ),
    ]
    if include_active_filter:
        conditions.append(ACTIVE_ROW_FILTER)
    params: list[Any] = [
        session_id, agent_id, text_query, query_embedding,
        text_weight, vector_weight,
    ]

    date_conds, date_params, next_idx = build_date_conditions(
        start_date, end_date, param_start_idx=7
    )
    conditions.extend(date_conds)
    params.extend(date_params)
    params.append(limit)

    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT
            id, content, metadata, created_at,
            ts_rank(
                to_tsvector('english', content),
                plainto_tsquery('english', $3)
            ) as text_rank,
            CASE
                WHEN embedding IS NOT NULL
                THEN 1 - (embedding <=> $4::vector)
                ELSE 0
            END as similarity,
            (
                $5 * COALESCE(ts_rank(
                    to_tsvector('english', content),
                    plainto_tsquery('english', $3)
                ), 0) +
                $6 * CASE
                    WHEN embedding IS NOT NULL
                    THEN 1 - (embedding <=> $4::vector)
                    ELSE 0
                END
            ) as combined_score
        FROM archival_memory
        WHERE {where_clause}
        ORDER BY combined_score DESC
        LIMIT ${next_idx}
    """
    return sql, params


# ==================== Recall Formatting ====================


def filter_core_by_query(core: dict[str, str], query: str) -> dict[str, str]:
    """Filter core memory keys matching the query (case-insensitive).

    Args:
        core: All core memory key-value pairs.
        query: Search query.

    Returns:
        Dict of matching key-value pairs.
    """
    query_lower = query.lower()
    return {
        key: value
        for key, value in core.items()
        if query_lower in key.lower() or key.lower() in query_lower
    }


def format_recall_text(
    core_matches: dict[str, str],
    archival_results: list[dict[str, Any]],
) -> str:
    """Format recall results into a combined text string.

    Args:
        core_matches: Matching core memory key-value pairs.
        archival_results: Archival search results (dicts with 'content' key).

    Returns:
        Combined recall text, or "No relevant memories found." if empty.
    """
    parts: list[str] = []
    for key, value in core_matches.items():
        parts.append(f"[Core/{key}]: {value}")
    for result in archival_results:
        parts.append(f"[Archival]: {result['content']}")
    return "\n".join(parts) if parts else "No relevant memories found."


def format_context_string(
    core: dict[str, str],
    archival_contents: list[str],
) -> str:
    """Format core + archival data into a context string for prompt injection.

    Args:
        core: All core memory key-value pairs.
        archival_contents: List of archival content strings.

    Returns:
        Formatted context string.
    """
    lines = ["## Core Memory"]
    if core:
        lines.extend(f"**{key}:** {value}" for key, value in core.items())
    else:
        lines.append("(empty)")

    lines.append("")
    lines.append("## Recent Archival Memory")
    if archival_contents:
        lines.extend(f"- {content}" for content in archival_contents)
    else:
        lines.append("(empty)")

    return "\n".join(lines)
