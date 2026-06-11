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
import re
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

_FAULT_LOG_FILE = None

# Hard upper bound on best-effort recall-event logging (issue #140). Well under
# the memory-awareness hook's 5s spawnSync budget (spawnSync waits for process
# EXIT, so a slow DB write would otherwise burn the whole budget). Telemetry is
# best-effort and cancellation-safe: on timeout the asyncio.wait_for cancels
# record_recall and the pool.acquire() context manager releases the connection.
RECORD_RECALL_TIMEOUT = 2.0


def _enable_faulthandler() -> None:
    """Enable faulthandler without breaking imports if log dir is missing."""
    global _FAULT_LOG_FILE
    log_path = Path.home() / ".claude" / "logs" / "opc_crash.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _FAULT_LOG_FILE = log_path.open("a")
        faulthandler.enable(file=_FAULT_LOG_FILE, all_threads=True)
    except OSError:
        faulthandler.enable(all_threads=True)


_enable_faulthandler()

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
    format_json_full_output,
    format_json_output,
    format_result_preview,
    group_by_type,
)

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def compute_fetch_k(k: int, *, no_rerank: bool) -> int:
    """Compute how many candidates to fetch from the backend.

    When reranking is enabled, over-fetch 3x (minimum 50) so the reranker
    has enough candidates to trim.
    """
    if no_rerank:
        return k
    return max(3 * k, 50)


def merge_project_first(
    own: list[dict[str, Any]],
    global_: list[dict[str, Any]],
    fetch_k: int,
) -> list[dict[str, Any]]:
    """Merge a project-scoped pass with the global pass, own-project first.

    Two-pass fetch for ``--project-first`` (issue #139). Reserves a global
    quota so an own-project pass that returns ``fetch_k`` rows cannot starve
    global candidates entirely — important with ``--no-rerank`` where
    ``fetch_k == k`` would otherwise make recall project-only (review round
    2). Composition:

    1. Own rows lead, capped at ``ceil(fetch_k / 2)`` (the own quota).
    2. Global rows then fill the remaining slots, deduped by ``id`` against
       the own rows already chosen.
    3. Any slots global could not fill are backfilled from the leftover own
       rows (those beyond the quota), keeping the result own-before-global
       ordered.

    Deduped by ``id`` (the own copy wins on collision); truncated to
    ``fetch_k`` so the rerank pool matches the single-pass path. Pure: inputs
    are not mutated.
    """
    own_quota = fetch_k - (fetch_k // 2)  # ceil(fetch_k / 2)

    seen: set[str] = set()
    lead: list[dict[str, Any]] = []  # own rows within the quota
    own_leftover: list[dict[str, Any]] = []  # own rows beyond the quota
    for row in own:
        rid = row.get("id")
        if rid is not None and rid in seen:
            continue
        if rid is not None:
            seen.add(rid)
        if len(lead) < own_quota:
            lead.append(row)
        else:
            own_leftover.append(row)

    # Global rows fill the slots the own quota did not consume.
    global_fill: list[dict[str, Any]] = []
    global_budget = fetch_k - len(lead)
    for row in global_:
        if len(global_fill) >= global_budget:
            break
        rid = row.get("id")
        if rid is not None and rid in seen:
            continue
        if rid is not None:
            seen.add(rid)
        global_fill.append(row)

    # Backfill any remaining slots from leftover own rows (own-first ordered).
    remaining = fetch_k - len(lead) - len(global_fill)
    backfill = own_leftover[:remaining] if remaining > 0 else []

    return [*lead, *backfill, *global_fill][:fetch_k]


def determine_retrieval_mode(
    backend: str, *, text_only: bool, vector_only: bool
) -> str:
    """Map backend + flags to a retrieval mode string for the reranker."""
    if backend == "sqlite":
        return "sqlite"
    if text_only:
        return "text"
    if vector_only:
        return "vector"
    return "hybrid_rrf"


def filter_by_tags(
    results: list[dict[str, Any]],
    tags: list[str] | None,
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    """Hard-filter results to those sharing at least one tag.

    Returns results unchanged when strict is False, tags is None, or tags is empty.
    Does not mutate the input list.
    """
    if not strict or not tags:
        return results
    tag_set = set(tags)
    return [r for r in results if set(r.get("metadata", {}).get("tags") or []) & tag_set]


def build_pattern_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a {memory_id: {pattern_strength, pattern_tags}} lookup from DB rows."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        mid = str(row["memory_id"])
        lookup[mid] = {
            "pattern_strength": float(row["pattern_strength"] or 0.0),
            "pattern_tags": row["pattern_tags"] or [],
        }
    return lookup


def apply_pattern_enrichment(
    results: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a new list with pattern_strength/pattern_tags merged from lookup.

    Does not mutate the input results.
    """
    enriched: list[dict[str, Any]] = []
    for result in results:
        rid = result.get("id")
        if rid and rid in lookup:
            enriched.append({**result, **lookup[rid]})
        else:
            enriched.append(result)
    return enriched


# Cap on the query string length we will pass to kg_extractor. Extraction
# uses regex patterns whose worst-case time grows with input size, and the
# reranker's kg_overlap signal gains nothing from matching against a
# megabytes-long prompt. 4096 bytes comfortably covers every realistic
# recall query while bounding regex CPU on hostile/accidental input.
# See aegis audit finding LOW-1.
_KG_QUERY_EXTRACTION_MAX_CHARS = 4096


def make_recall_context(
    project: str | None,
    tags: list[str] | None,
    retrieval_mode: str,
    query: str | None = None,
) -> Any:
    """Construct a RecallContext for the reranker.

    When ``query`` is provided and backend is postgres, populates
    ``query_entities`` via ``kg_extractor.extract_entities`` for the
    ``kg_overlap`` signal. Non-fatal: any extractor failure yields no
    query entities.
    """
    from scripts.core.reranker import RecallContext

    query_entities: list[dict] | None = None
    if query and get_backend() == "postgres":
        # Cap the input fed to the extractor. This is defense-in-depth against
        # regex CPU blow-up from oversized queries; no production path in this
        # codebase sends queries larger than a few hundred chars.
        extract_input = query[:_KG_QUERY_EXTRACTION_MAX_CHARS]
        try:
            from scripts.core.kg_extractor import extract_entities

            extracted = extract_entities(extract_input)
            query_entities = [
                {"name": e.name, "type": e.entity_type} for e in extracted
            ] or None
        except Exception as e:
            logger.debug("Query-side entity extraction failed: %s", e)

    from scripts.core.project_naming import canonicalize_project

    return RecallContext(
        project=canonicalize_project(project),
        tags_hint=tags,
        retrieval_mode=retrieval_mode,
        query_entities=query_entities,
    )


def resolve_project_scope(
    *,
    project_first: bool,
    project: str | None,
    project_dir: str | None,
) -> str | None:
    """Resolve the project to fetch-scope to for ``--project-first``.

    Returns ``None`` (no scoping, global recall) when the flag is off or no
    project can be resolved. Explicit ``--project`` wins; otherwise the
    project is auto-detected from ``CLAUDE_PROJECT_DIR`` (worktree-aware).
    The value is canonicalized so the SQL bind matches stored project
    values (issue #139, mirrors make_recall_context's read boundary).
    """
    if not project_first:
        return None
    from scripts.core.project_naming import canonicalize_project, project_from_path

    if project:
        return canonicalize_project(project)
    return project_from_path(project_dir or None)


def resolve_caller_project(project: str | None, project_dir: str | None) -> str | None:
    """Resolve the caller's canonical project for recall-event logging (#140).

    Explicit ``--project`` wins (canonicalized); otherwise the project is
    derived from ``project_dir`` (worktree-aware ``project_from_path``).
    Returns ``None`` when neither yields a project. Unlike
    ``resolve_project_scope``, this resolves unconditionally (no opt-in flag)
    so every recall event can record who called it.
    """
    from scripts.core.project_naming import canonicalize_project, project_from_path

    return canonicalize_project(project) or project_from_path(project_dir or None)


def resolve_search_params(
    *,
    backend: str,
    text_only: bool,
    vector_only: bool,
    query: str,
    fetch_k: int,
    provider: str,
    threshold: float,
    recency: float,
    no_rerank: bool,
    no_expand: bool,
    expand_terms: int,
    rebuild_idf: bool,
    project_scope: str | None = None,
) -> dict[str, Any]:
    """Resolve CLI flags into a search parameter dict.

    All returned dicts include ``mode``, ``query``, ``k``, and
    ``project_scope`` (the canonical project to fetch-scope to under
    ``--project-first``, or ``None`` for the default global pass — issue
    #139).

    Additional keys depend on the selected mode:
    - ``sqlite``: no additional keys.
    - ``text_only``: no additional keys.
    - ``vector``: adds ``provider``, ``similarity_threshold``,
      ``recency_weight``, and ``text_fallback``.
    - ``hybrid_rrf``: adds ``provider``, ``similarity_threshold``, ``expand``,
      ``max_expansion_terms``, and ``rebuild_idf``.
    """
    if backend == "sqlite":
        return {
            "mode": "sqlite",
            "query": query,
            "k": fetch_k,
            "project_scope": project_scope,
        }

    if text_only:
        return {
            "mode": "text_only",
            "query": query,
            "k": fetch_k,
            "project_scope": project_scope,
        }

    if vector_only:
        sql_recency = 0.0 if not no_rerank else recency
        return {
            "mode": "vector",
            "query": query,
            "k": fetch_k,
            "provider": provider,
            "similarity_threshold": threshold,
            "recency_weight": sql_recency,
            "text_fallback": True,
            "project_scope": project_scope,
        }

    # Default: hybrid RRF — no recency_weight; temporal relevance is handled
    # by recall_count boost in the SQL query, not a separate weight param.
    return {
        "mode": "hybrid_rrf",
        "query": query,
        "k": fetch_k,
        "provider": provider,
        "similarity_threshold": threshold * 0.01,  # RRF scores are ~0.01-0.03 range
        "expand": not no_expand,
        "max_expansion_terms": expand_terms,
        "rebuild_idf": rebuild_idf,
        "project_scope": project_scope,
    }


def select_output(
    *, json_flag: bool, json_full: bool
) -> str:
    """Choose output format: 'json_full', 'json', or 'human'."""
    if json_full:
        return "json_full"
    if json_flag:
        return "json"
    return "human"


# ---------------------------------------------------------------------------
# Environment helpers (reads os.environ — not pure)
# ---------------------------------------------------------------------------


def get_backend() -> str:
    """Determine which backend to use (sqlite or postgres)."""
    backend = os.environ.get("AGENTICA_MEMORY_BACKEND", "").lower()
    if backend in ("sqlite", "postgres"):
        return backend

    if os.environ.get("CONTINUOUS_CLAUDE_DB_URL") or os.environ.get("DATABASE_URL"):
        return "postgres"

    return "sqlite"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


# Source labels are validated here (writer-side) instead of with a DB CHECK
# constraint: recall_log is append-only and a CHECK violation would abort the
# whole INSERT, silently dropping the entire log row (losing the recall event).
# Validating in Python lets us drop just the bad label to NULL and still log.
_SOURCE_LABEL_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def _sanitize_source(source: str | None) -> str | None:
    """Validate a recall ``source`` label (issue #140).

    Labels are documented as fixed call-site identifiers (``hook``/``mcp``/
    ``cli``), never prompt-derived text. A value that does not match
    ``^[a-z][a-z0-9_-]{0,31}$`` is dropped to ``None`` so arbitrary text can't
    leak into the append-only log. Pure function.
    """
    if source is None:
        return None
    if _SOURCE_LABEL_RE.fullmatch(source):
        return source
    logger.debug("invalid recall source label dropped")
    return None


async def record_recall(
    result_ids: list[str],
    caller_project: str | None = None,
    source: str | None = None,
) -> None:
    """Update last_recalled/recall_count and log the recall event (issue #140).

    Batch-updates the recalled rows in a single ``UPDATE ... RETURNING id,
    project`` (point-in-time truth, zero extra round trips), then best-effort
    appends one ``recall_log`` row with parallel ``recalled_ids`` /
    ``recalled_projects`` arrays plus the caller's project and source label.

    A **zero-result** recall (empty ``result_ids``) skips the counter UPDATE
    (nothing to update) but STILL logs a recall_log row with empty arrays and
    ``result_count = 0`` — empty results are the signature of over-restrictive
    project scoping (#130), so they must be observable.

    The INSERT is wrapped in its own try/except so a pre-migration DB (no
    ``recall_log`` table) still gets the counter UPDATE committed — both run
    as separate autocommitted statements in the same acquire, NOT a CTE or
    transaction. If ``archival_memory.project`` itself is missing (temporal-
    decay columns applied but not add_project_column.sql), the RETURNING fetch
    raises ``UndefinedColumnError``; we fall back to the original counter-only
    UPDATE so counters never silently stop, and skip the recall_log INSERT.
    The whole body never raises (recall must not break).

    ``caller_project`` and ``source`` are accepted for SQLite parity but
    unused there: SQLite has no project/log columns, so it skips entirely.
    """
    if get_backend() != "postgres":
        logger.debug(
            "record_recall: sqlite backend, skipping recall-event logging"
        )
        return

    source = _sanitize_source(source)

    try:
        from asyncpg.exceptions import UndefinedColumnError

        from scripts.core.db.postgres_pool import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            # Zero-result recall: no rows to bump, but log the event so
            # over-restrictive scoping (#130) is visible.
            if not result_ids:
                await _insert_recall_log(conn, caller_project, [], [], source)
                return

            try:
                rows = await conn.fetch(
                    """
                    UPDATE archival_memory
                    SET last_recalled = NOW(),
                        recall_count = recall_count + 1
                    WHERE id = ANY($1::uuid[])
                    RETURNING id, project
                    """,
                    result_ids,
                )
            except UndefinedColumnError:
                # archival_memory.project absent: keep counters working and
                # skip recall_log entirely (point-in-time projects unknown).
                logger.debug(
                    "archival_memory.project missing; "
                    "counters updated, recall_log skipped"
                )
                await conn.execute(
                    """
                    UPDATE archival_memory
                    SET last_recalled = NOW(),
                        recall_count = recall_count + 1
                    WHERE id = ANY($1::uuid[])
                    """,
                    result_ids,
                )
                return

            # ids were supplied but matched nothing (e.g. concurrent deletion):
            # skip the INSERT — this is a stale-id event, not a zero-result one.
            if not rows:
                return

            recalled_ids = [str(r["id"]) for r in rows]
            recalled_projects = [r["project"] for r in rows]
            await _insert_recall_log(
                conn, caller_project, recalled_ids, recalled_projects, source
            )
    except Exception:
        # Best-effort telemetry must never break recall, but failures (e.g.
        # pool acquisition) should still be observable at debug level.
        logger.debug("record_recall failed", exc_info=True)


async def _insert_recall_log(
    conn: Any,
    caller_project: str | None,
    recalled_ids: list[str],
    recalled_projects: list[str | None],
    source: str | None,
) -> None:
    """Best-effort append one recall_log row (issue #140).

    Runs as a separate autocommitted statement after the counter UPDATE (NOT a
    CTE or transaction) so a pre-migration DB lacking ``recall_log`` fails here
    alone — the failure is swallowed (debug log) and the counter UPDATE stands.
    """
    try:
        await conn.execute(
            """
            INSERT INTO recall_log (
                caller_project, recalled_ids, recalled_projects,
                result_count, source
            )
            VALUES ($1, $2, $3, $4, $5)
            """,
            caller_project,
            recalled_ids,
            recalled_projects,
            len(recalled_ids),
            source,
        )
    except Exception:
        logger.debug("recall_log insert failed", exc_info=True)


async def _fetch_pattern_rows(result_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch pattern strength/tags from PostgreSQL. Caller must ensure Postgres backend."""
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT pm.memory_id,
                   MAX(dp.confidence * GREATEST(1.0 - COALESCE(pm.distance, 0), 0))
                       AS pattern_strength,
                   ARRAY_AGG(DISTINCT unnested_tag) FILTER (WHERE unnested_tag IS NOT NULL)
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
    return [dict(row) for row in rows]


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
        rows = await _fetch_pattern_rows(result_ids)
        lookup = build_pattern_lookup(rows)
        return apply_pattern_enrichment(results, lookup)
    except (ImportError, OSError, ConnectionError) as e:
        logger.debug("Pattern enrichment unavailable: %s", e)
    except Exception as e:
        logger.warning("Pattern enrichment error: %s", e, exc_info=True)

    return results


# ---------------------------------------------------------------------------
# Knowledge graph enrichment (Phase 3, read-side)
# ---------------------------------------------------------------------------

KG_MAX_EDGES_PER_MEMORY = 50
"""Safety cap on edges returned per memory in kg_context. Typical usage is
< 10; the cap prevents pathological payload bloat on high-connectivity
learnings. When exceeded, the top-N by weight are kept and a warning logs."""


async def _fetch_kg_rows(result_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch entities and edges for each memory_id in one round trip.

    Caller must ensure Postgres backend. Returns rows shaped as:
    {id: UUID, kg_entities: list[dict], kg_edges: list[dict]}.
    """
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        # Edges are capped at the SQL layer via a correlated LATERAL
        # subquery with ORDER BY + LIMIT so the database only materializes
        # and transfers up to KG_MAX_EDGES_PER_MEMORY edges per memory.
        # build_kg_lookup still caps on the Python side as defense-in-depth
        # (catches SQL ordering drift or callers that bypass this query).
        # Entity output carries both 'canonical' (matching key) and 'name'
        # (display_name) so consumers can ID-match without losing casing.
        rows = await conn.fetch(
            """
            SELECT m.memory_id AS id,
                   ARRAY_AGG(DISTINCT jsonb_build_object(
                     'id', e.id::text,
                     'name', e.display_name,
                     'canonical', e.name,
                     'type', e.entity_type,
                     'mention_count', e.mention_count
                   )) AS kg_entities,
                   COALESCE((
                     SELECT ARRAY_AGG(jsonb_build_object(
                       'source', se.display_name,
                       'target', te.display_name,
                       'relation', ed.relation,
                       'weight', ed.weight
                     ))
                     FROM (
                       SELECT ed.*
                       FROM kg_edges ed
                       WHERE ed.memory_id = m.memory_id
                       ORDER BY ed.weight DESC
                       LIMIT $2
                     ) ed
                     JOIN kg_entities se ON se.id = ed.source_id
                     JOIN kg_entities te ON te.id = ed.target_id
                   ), ARRAY[]::jsonb[]) AS kg_edges
            FROM kg_entity_mentions m
            JOIN kg_entities e ON e.id = m.entity_id
            WHERE m.memory_id = ANY($1::uuid[])
            GROUP BY m.memory_id
            """,
            [uuid.UUID(rid) for rid in result_ids],
            KG_MAX_EDGES_PER_MEMORY,
        )
    return [dict(row) for row in rows]


def build_kg_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a {memory_id: {entities, edges}} lookup from fetched rows.

    Edges exceeding KG_MAX_EDGES_PER_MEMORY are capped to the top-N by
    weight (descending) and a warning is logged with the overflow count.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        mid = str(row["id"])
        # asyncpg auto-decodes top-level jsonb columns to dicts, but elements
        # inside a jsonb[] array come back as JSON-encoded strings. Decode
        # defensively so we accept both shapes.
        entities = [
            json.loads(e) if isinstance(e, str) else e
            for e in (row.get("kg_entities") or [])
        ]
        edges = [
            json.loads(e) if isinstance(e, str) else e
            for e in (row.get("kg_edges") or [])
        ]
        if len(edges) > KG_MAX_EDGES_PER_MEMORY:
            total = len(edges)
            edges = sorted(edges, key=lambda e: e.get("weight", 0.0), reverse=True)
            edges = edges[:KG_MAX_EDGES_PER_MEMORY]
            logger.warning(
                "kg_context edges capped for memory %s: %d edges truncated to %d",
                mid, total, KG_MAX_EDGES_PER_MEMORY,
            )
        else:
            edges = sorted(edges, key=lambda e: e.get("weight", 0.0), reverse=True)
        lookup[mid] = {"entities": entities, "edges": edges}
    return lookup


def apply_kg_enrichment(
    results: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a new list with kg_context merged in for matching results.

    Does not mutate input. Results with no matching entry in lookup are
    returned unchanged (no kg_context key set).
    """
    enriched: list[dict[str, Any]] = []
    for result in results:
        rid = result.get("id")
        if rid and str(rid) in lookup:
            enriched.append({**result, "kg_context": lookup[str(rid)]})
        else:
            enriched.append(result)
    return enriched


async def enrich_with_kg_context(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add kg_context to recall results from kg_entities / kg_edges tables.

    Gracefully degrades only on expected availability failures (missing
    module, missing DB, network). Genuine defects -- bad SQL, row-shape
    drift, missing table on a configured-Postgres install -- are allowed
    to propagate so they surface rather than silently changing ranking.
    """
    if not results or get_backend() != "postgres":
        return results

    result_ids = [str(r["id"]) for r in results if r.get("id")]
    if not result_ids:
        return results

    try:
        import asyncpg.exceptions as _pg_exc
    except ImportError:
        _pg_exc = None  # type: ignore[assignment]

    try:
        rows = await _fetch_kg_rows(result_ids)
        lookup = build_kg_lookup(rows)
        return apply_kg_enrichment(results, lookup)
    except (ImportError, OSError, ConnectionError) as e:
        logger.debug("KG enrichment unavailable: %s", e)
        return results
    except Exception as e:
        # Re-raise unless the failure is specifically an asyncpg
        # InterfaceError (connection-state, pool exhaustion, or protocol
        # issues that indicate a transiently unavailable DB). Broader
        # asyncpg errors -- SQL syntax errors, UndefinedTableError,
        # DataError, any PostgresError subclass -- continue to propagate
        # so real defects surface instead of masquerading as degraded
        # mode. InterfaceError was chosen narrowly over asyncpg.Error
        # for exactly that reason.
        if _pg_exc is not None and isinstance(e, _pg_exc.InterfaceError):
            logger.debug("KG enrichment unavailable (asyncpg): %s", e)
            return results
        raise


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
    """
    if not query.strip():
        return []

    backend = get_backend()

    if backend == "sqlite":
        return await search_learnings_sqlite(query, k)
    return await search_learnings_postgres(
        query, k, provider, text_fallback, similarity_threshold, recency_weight
    )


async def _dispatch_search(
    params: dict[str, Any], *, project: str | None = None,
) -> list[dict[str, Any]]:
    """Dispatch a search call based on resolved params dict.

    ``project`` (issue #139) optionally fetch-scopes the backend query.
    ``None`` (the default) leaves every backend call byte-identical to the
    pre-#139 path.
    """
    mode = params["mode"]
    # Only forward project= when scoping is active so the default path stays
    # byte-identical to the pre-#139 call signatures.
    project_kw: dict[str, Any] = {"project": project} if project is not None else {}
    if mode == "sqlite":
        return await search_learnings_sqlite(
            params["query"], params["k"], **project_kw,
        )
    if mode == "text_only":
        return await search_learnings_text_only_postgres(
            params["query"], params["k"], **project_kw,
        )
    if mode == "vector":
        return await search_learnings_postgres(
            params["query"],
            params["k"],
            params["provider"],
            text_fallback=params.get("text_fallback", True),
            similarity_threshold=params["similarity_threshold"],
            recency_weight=params["recency_weight"],
            **project_kw,
        )
    # hybrid_rrf
    if params.get("rebuild_idf"):
        from scripts.core.query_expansion import get_idf_index

        await get_idf_index(force_rebuild=True)
    return await search_learnings_hybrid_rrf(
        query=params["query"],
        k=params["k"],
        provider=params["provider"],
        similarity_threshold=params["similarity_threshold"],
        expand=params.get("expand", True),
        max_expansion_terms=params.get("max_expansion_terms", 5),
        **project_kw,
    )


async def _dispatch_search_project_first(
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Two-pass fetch for ``--project-first`` (issue #139).

    Pass 1 scopes the backend to ``params["project_scope"]``; pass 2 fetches
    globally. The passes are merged own-project-first, deduped by id and
    truncated to ``params["k"]`` (the over-fetched rerank pool size) by
    ``merge_project_first``. The reranker's project_match signal still runs
    downstream — this only changes the fetch pool composition.

    Each pass is isolated (review round 2): a transient failure in one pass
    must not discard the other pass's results. A scoped-pass failure degrades
    to global-only; a global-pass failure (after the scoped pass succeeded)
    returns the scoped rows alone; both failing re-raises the global error,
    preserving the default path's error behavior. The degraded pass is named
    on stderr. Tradeoff: for vector/hybrid modes the embedding/expansion is
    computed once per pass (twice total) — accepted for now.
    """
    scope = params.get("project_scope")

    from scripts.core.db.postgres_pool import sanitize_log_message

    own: list[dict[str, Any]] = []
    own_failed = False
    try:
        own = await _dispatch_search(params, project=scope)
    except Exception as exc:  # noqa: BLE001 - degrade, do not crash recall
        own_failed = True
        # exc text can embed a DSN/host/path; recall stderr is injected into
        # the model context by hooks, so redact before printing and keep the
        # full traceback in the debug log only (aegis MEDIUM-2).
        logger.debug("--project-first scoped pass failed", exc_info=True)
        print(
            "warning: --project-first scoped pass failed "
            f"({sanitize_log_message(str(exc))}); "
            "continuing with global results only.",
            file=sys.stderr,
        )

    try:
        global_ = await _dispatch_search(params, project=None)
    except Exception as exc:  # noqa: BLE001 - degrade if the scoped pass held
        if own_failed:
            # Both passes failed — surface the error like the default path.
            raise
        logger.debug("--project-first global pass failed", exc_info=True)
        print(
            "warning: --project-first global pass failed "
            f"({sanitize_log_message(str(exc))}); "
            "returning project-scoped results only.",
            file=sys.stderr,
        )
        return own

    return merge_project_first(own, global_, params["k"])


def _format_output(
    results: list[dict[str, Any]], *, output_mode: str, structured: bool
) -> str:
    """Format results for the chosen output mode."""
    if output_mode == "json_full":
        return format_json_full_output(results)
    if output_mode == "json":
        return format_json_output(results, structured=structured)
    return format_human_output(results, structured=structured)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Semantic recall of session learnings from archival_memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--provider", choices=["local", "voyage"], default="local", help="Embedding provider"
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--json-full", action="store_true", help="Full JSON metadata output")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--text-only", action="store_true", help="Text search only (no embeddings)"
    )
    mode_group.add_argument("--vector-only", action="store_true", help="Vector-only search")
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.2, help="Minimum similarity (default: 0.2)"
    )
    parser.add_argument(
        "--recency", "-r", type=float, default=0.1, help="Recency weight (default: 0.1)"
    )
    parser.add_argument("--tags", nargs="+", help="Boost results matching these tags")
    parser.add_argument("--tags-strict", action="store_true", help="Hard-filter by tags")
    parser.add_argument("--no-rerank", action="store_true", help="Bypass re-ranking")
    parser.add_argument("--no-expand", action="store_true", help="Disable TF-IDF query expansion")
    parser.add_argument("--expand-terms", type=int, default=5, help="Expansion terms (default: 5)")
    parser.add_argument("--rebuild-idf", action="store_true", help="Force rebuild IDF index")
    parser.add_argument("--project", help="Project context for re-ranking")
    parser.add_argument(
        "--project-first",
        action="store_true",
        help=(
            "Fetch own-project rows first, then fill globally (issue #139). "
            "Opt-in; degrades to global recall if no project resolves."
        ),
    )
    parser.add_argument("--structured", action="store_true", help="Group results by type")
    parser.add_argument(
        "--source",
        help=(
            "Short caller label for recall-event logging (e.g. hook|mcp|cli). "
            "Label-only -- never prompt-derived text. Must match "
            "^[a-z][a-z0-9_-]{0,31}$; invalid labels are dropped to NULL "
            "(issue #140)."
        ),
    )
    return parser


async def main() -> int:
    """Run semantic recall on session learnings."""
    args = _build_arg_parser().parse_args()

    output_mode = select_output(json_flag=args.json, json_full=args.json_full)
    fetch_k = compute_fetch_k(args.k, no_rerank=args.no_rerank)

    if output_mode == "human":
        print(f'Recalling learnings for: "{args.query}"')
        print(f"Provider: {args.provider}")
        print()

    backend = get_backend()

    # Fetch-time project scoping (issue #139). Opt-in: resolve a project for
    # --project-first, warn-and-degrade to global recall when none resolves.
    project_scope = resolve_project_scope(
        project_first=args.project_first,
        project=args.project,
        project_dir=os.environ.get("CLAUDE_PROJECT_DIR") or None,
    )
    if args.project_first and project_scope is None:
        print(
            "warning: --project-first set but no project could be resolved; "
            "falling back to global recall (pass --project or set "
            "CLAUDE_PROJECT_DIR).",
            file=sys.stderr,
        )

    # Resolve and dispatch search
    try:
        params = resolve_search_params(
            backend=backend,
            text_only=args.text_only,
            vector_only=args.vector_only,
            query=args.query,
            fetch_k=fetch_k,
            provider=args.provider,
            threshold=args.threshold,
            recency=args.recency,
            no_rerank=args.no_rerank,
            no_expand=args.no_expand,
            expand_terms=args.expand_terms,
            rebuild_idf=args.rebuild_idf,
            project_scope=project_scope,
        )

        if backend == "sqlite" and output_mode == "human" and not args.text_only:
            print("  (SQLite backend - using text search)")

        if project_scope is not None:
            results = await _dispatch_search_project_first(params)
        else:
            results = await _dispatch_search(params)
    except Exception as e:
        if output_mode != "human":
            from scripts.core.recall_formatters import get_api_version

            print(json.dumps({"version": get_api_version(), "error": str(e), "results": []}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    # Pattern enrichment
    if (not args.no_rerank or args.json_full) and backend == "postgres":
        results = await enrich_with_pattern_strength(results)
        results = await enrich_with_kg_context(results)

    # Tag filtering
    results = filter_by_tags(results, args.tags, strict=args.tags_strict)

    # Reranking
    if not args.no_rerank:
        retrieval_mode = determine_retrieval_mode(
            backend, text_only=args.text_only, vector_only=args.vector_only
        )
        from scripts.core.project_naming import project_from_path

        ctx = make_recall_context(
            project=args.project
            or project_from_path(os.environ.get("CLAUDE_PROJECT_DIR") or None),
            tags=args.tags,
            retrieval_mode=retrieval_mode,
            query=args.query,
        )
        from scripts.core.reranker import rerank

        results = rerank(results, ctx, k=args.k)

    # Output FIRST: best-effort recall logging must never delay user-visible
    # output under the memory-awareness hook's 5s spawn timeout (issue #140).
    print(_format_output(results, output_mode=output_mode, structured=args.structured))

    # Record recall (skip benchmarking mode). Bounded by RECORD_RECALL_TIMEOUT
    # so a slow DB write can't burn the hook's spawn budget; telemetry is
    # best-effort and cancellation-safe (issue #140).
    if not args.json_full:
        caller_project = resolve_caller_project(
            args.project, os.environ.get("CLAUDE_PROJECT_DIR")
        )
        try:
            await asyncio.wait_for(
                record_recall(
                    [r["id"] for r in results],
                    caller_project=caller_project,
                    source=args.source,
                ),
                timeout=RECORD_RECALL_TIMEOUT,
            )
        except TimeoutError:
            logger.debug("record_recall timed out; recall event dropped")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
