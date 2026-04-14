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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

_FAULT_LOG_FILE = None


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

    return RecallContext(
        project=project,
        tags_hint=tags,
        retrieval_mode=retrieval_mode,
        query_entities=query_entities,
    )


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
) -> dict[str, Any]:
    """Resolve CLI flags into a search parameter dict.

    All returned dicts include ``mode``, ``query``, and ``k``.

    Additional keys depend on the selected mode:
    - ``sqlite``: no additional keys.
    - ``text_only``: no additional keys.
    - ``vector``: adds ``provider``, ``similarity_threshold``,
      ``recency_weight``, and ``text_fallback``.
    - ``hybrid_rrf``: adds ``provider``, ``similarity_threshold``, ``expand``,
      ``max_expansion_terms``, and ``rebuild_idf``.
    """
    if backend == "sqlite":
        return {"mode": "sqlite", "query": query, "k": fetch_k}

    if text_only:
        return {"mode": "text_only", "query": query, "k": fetch_k}

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


async def record_recall(result_ids: list[str]) -> None:
    """Update last_recalled and recall_count for recalled learnings.

    Batch-updates all returned results in a single query.
    Fails silently to avoid breaking recall if columns don't exist yet.
    """
    if not result_ids:
        return

    if get_backend() != "postgres":
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
        pass


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
        entities = list(row.get("kg_entities") or [])
        edges = list(row.get("kg_edges") or [])
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


async def _dispatch_search(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch a search call based on resolved params dict."""
    mode = params["mode"]
    if mode == "sqlite":
        return await search_learnings_sqlite(params["query"], params["k"])
    if mode == "text_only":
        return await search_learnings_text_only_postgres(params["query"], params["k"])
    if mode == "vector":
        return await search_learnings_postgres(
            params["query"],
            params["k"],
            params["provider"],
            text_fallback=params.get("text_fallback", True),
            similarity_threshold=params["similarity_threshold"],
            recency_weight=params["recency_weight"],
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
    )


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
    parser.add_argument("--structured", action="store_true", help="Group results by type")
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
        )

        if backend == "sqlite" and output_mode == "human" and not args.text_only:
            print("  (SQLite backend - using text search)")

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
        ctx = make_recall_context(
            project=args.project
            or os.environ.get("CLAUDE_PROJECT_DIR", "").rsplit("/", 1)[-1]
            or None,
            tags=args.tags,
            retrieval_mode=retrieval_mode,
            query=args.query,
        )
        from scripts.core.reranker import rerank

        results = rerank(results, ctx, k=args.k)

    # Record recall (skip benchmarking mode)
    if not args.json_full:
        await record_recall([r["id"] for r in results])

    # Output
    print(_format_output(results, output_mode=output_mode, structured=args.structured))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
