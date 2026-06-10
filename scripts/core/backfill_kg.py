"""Backfill knowledge graph for existing archival_memory rows (#124).

Walks archival_memory rows that have no kg_entity_mentions yet and runs the
same extraction + storage path used by store_learning._try_index_kg.

Idempotent and resumable: store_entities_and_edges dedupes on
(memory_id, entity_id) and (source_id, target_id, relation, memory_id); the
fetch query excludes memories that already have a mention row and memories
durably marked kg_backfill=no_entities in metadata, so partial runs resume
cleanly and re-running converges to a no-op. Rows that error stay eligible
and are retried on the next run (exit code 2 signals partial failure).

Systemic failures abort promptly instead of erroring once per row (#131):
infrastructure errors (connection loss, pool timeouts) propagate out of the
per-row handler, and a circuit breaker trips after --max-consecutive-errors
consecutive per-row errors. Both abort paths exit 3; unprocessed rows stay
eligible for the next run.

Usage:
    uv run python scripts/core/backfill_kg.py --dry-run         # counts only
    uv run python scripts/core/backfill_kg.py                   # full backfill
    uv run python scripts/core/backfill_kg.py --limit 100       # first 100
    uv run python scripts/core/backfill_kg.py --since 2026-01-01
    uv run python scripts/core/backfill_kg.py --memory-id <uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `scripts.*` imports work
# when launched via `uv run python scripts/core/backfill_kg.py`
# (which doesn't add cwd to sys.path) — memory_daemon.py pattern
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import asyncpg

from scripts.core.db.postgres_pool import close_pool, get_pool
from scripts.core.kg_extractor import (
    extract_entities,
    extract_relations,
    store_entities_and_edges,
)
from scripts.core.log_safety import safe
from scripts.core.store_learning import detect_backend

# Per-row cap before regex extraction: a single oversized adversarial
# transcript row must not stall the whole backfill. Extraction is heuristic;
# the leading portion of a learning carries the salient entities.
MAX_CONTENT_CHARS = 100_000

# Issue #131: systemic failures must abort the run promptly instead of being
# logged once per row across the whole backlog. Explicit allowlist of
# connectivity/availability failures — deliberately NOT the generic
# asyncpg.InterfaceError, whose subtree includes deterministic client API
# misuse (e.g. ClientConfigurationError) that must stay on the per-row error
# path (review round 2):
# - PostgresConnectionError: server-side connection loss
#   (ConnectionDoesNotExistError, ConnectionFailureError, ...)
# - InvalidAuthorizationSpecificationError: bad/revoked credentials at
#   connect time (covers InvalidPasswordError)
# - CannotConnectNowError: server starting up / not accepting connections
# - InsufficientResourcesError: server out of capacity
#   (covers TooManyConnectionsError)
# - OSError: socket-level failures (ConnectionError subclasses)
# - TimeoutError: pool-acquire timeouts (asyncio.TimeoutError alias)
INFRA_ERRORS: tuple[type[BaseException], ...] = (
    asyncpg.PostgresConnectionError,
    asyncpg.InvalidAuthorizationSpecificationError,
    asyncpg.CannotConnectNowError,
    asyncpg.InsufficientResourcesError,
    OSError,
    TimeoutError,
)

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_fetch_query(
    limit: int | None = None,
    since: datetime | None = None,
    memory_id: str | None = None,
    count_only: bool = False,
    after: tuple[datetime, str] | None = None,
    project: str | None = None,
    recheck_no_entities: bool = False,
) -> tuple[str, list[Any]]:
    """Build the SQL to select memories that still need KG indexing.

    Returns (sql, params) with consecutively numbered placeholders.
    Eligibility excludes memories that already have a kg_entity_mentions row
    and memories durably marked kg_backfill in metadata (zero-entity rows),
    so partial runs resume cleanly and reruns converge to a no-op.

    A targeted ``memory_id`` run bypasses the kg_backfill marker so a row
    previously marked no_entities can be reprocessed (repair path).
    ``recheck_no_entities`` does the same for bulk runs, letting an
    extractor upgrade revisit previously marked rows.

    ``after`` is a (created_at, id) keyset cursor: combined with the
    deterministic ORDER BY it pages through the backlog without loading it
    all at once, and advances past rows that errored mid-run.
    """
    # Invariant: f-string parts below are compile-time constants or in-code
    # integers (placeholder numbers) only; every user/data value binds via $N
    select = "SELECT count(*)" if count_only else "SELECT id, content, created_at"
    sql = (
        f"{select} FROM archival_memory m "
        "WHERE NOT EXISTS ("
        "SELECT 1 FROM kg_entity_mentions km WHERE km.memory_id = m.id)"
    )
    if memory_id is None and not recheck_no_entities:
        sql += " AND (m.metadata->>'kg_backfill') IS NULL"
    params: list[Any] = []
    if since is not None:
        params.append(since)
        sql += f" AND m.created_at >= ${len(params)}"
    if memory_id is not None:
        params.append(memory_id)
        sql += f" AND m.id = ${len(params)}::uuid"
    if project is not None:
        params.append(project)
        sql += f" AND m.project = ${len(params)}"
    if after is not None:
        params.extend([after[0], after[1]])
        sql += f" AND (m.created_at, m.id) > (${len(params) - 1}, ${len(params)}::uuid)"
    if not count_only:
        sql += " ORDER BY m.created_at, m.id"
    if limit is not None:
        params.append(limit)
        sql += f" LIMIT ${len(params)}"
    return sql, params


def format_summary(stats: dict[str, int]) -> str:
    """Format the final run summary line."""
    return (
        f"Backfill complete: {stats['processed']} processed, "
        f"{stats['indexed']} indexed, {stats['no_entities']} no_entities, "
        f"{stats['errors']} errors"
    )


def _positive_int(value: str) -> int:
    """Argparse type for positive integers."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def _iso_datetime(value: str) -> datetime:
    """Argparse type for ISO-8601 dates/datetimes.

    Naive inputs are normalized to UTC so the cutoff against the
    TIMESTAMPTZ created_at column does not depend on runtime timezone.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value!r}") from e
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _uuid_str(value: str) -> str:
    """Argparse type validating UUID format, returning the original string."""
    try:
        uuid.UUID(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid UUID: {value!r}") from e
    return value


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill knowledge graph for existing archival_memory rows."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report eligible memory counts without writing",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Process at most N memories",
    )
    parser.add_argument(
        "--since",
        type=_iso_datetime,
        default=None,
        help="Only memories created at or after this ISO date",
    )
    parser.add_argument(
        "--memory-id",
        type=_uuid_str,
        default=None,
        help="Backfill a single still-unindexed memory by UUID. Bypasses the "
        "no_entities marker; rows that already have kg_entity_mentions are "
        "skipped. Mutually exclusive with --since/--project/--limit.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Only memories tagged with this project (default: all projects; "
        "the KG is a global graph by design)",
    )
    parser.add_argument(
        "--recheck-no-entities",
        action="store_true",
        help="Re-include rows previously marked kg_backfill=no_entities "
        "(use after extractor upgrades)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=500,
        help="DB page size and progress-logging batch (default 500)",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=_positive_int,
        default=10,
        help="Abort the run (exit 3) after this many consecutive per-row "
        "errors — a systemic-failure circuit breaker (default 10). Raise it "
        "to push past a contiguous block of poison rows; error rows stay "
        "eligible and are never marked",
    )
    args = parser.parse_args(argv)
    if args.memory_id is not None and (
        args.since is not None or args.project is not None or args.limit is not None
    ):
        parser.error("--memory-id cannot be combined with --since/--project/--limit")
    return args


# ---------------------------------------------------------------------------
# Async I/O
# ---------------------------------------------------------------------------


async def backfill_one(memory_id: str, content: str) -> dict[str, Any]:
    """Extract and store KG rows for one memory.

    Returns a status dict; regular exceptions are converted into
    ``{"status": "error", "error": ...}``.
    """
    try:
        # Truncate once so entity spans and relation extraction index into
        # the same string
        content = content[:MAX_CONTENT_CHARS]
        entities = extract_entities(content)
        if not entities:
            return {"status": "no_entities"}
        relations = extract_relations(content, entities)
        stats = await store_entities_and_edges(memory_id, entities, relations)
        return {"status": "indexed", "stats": stats}
    except INFRA_ERRORS:
        # Systemic failure (DB down, pool closing): abort, don't retry rows
        raise
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


async def mark_no_entities(pool: Any, memory_ids: list[str]) -> None:
    """Durably mark zero-entity memories so reruns skip them."""
    if not memory_ids:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE archival_memory "
            "SET metadata = COALESCE(metadata, '{}'::jsonb) "
            '|| \'{"kg_backfill": "no_entities"}\'::jsonb '
            "WHERE id = ANY($1::uuid[])",
            memory_ids,
        )


async def _fetch_page(
    pool: Any,
    args: argparse.Namespace,
    after: tuple[datetime, str] | None,
    page_size: int,
) -> list[Any]:
    """Fetch one keyset page of eligible memories."""
    sql, params = build_fetch_query(
        limit=page_size,
        since=args.since,
        memory_id=args.memory_id,
        after=after,
        project=args.project,
        recheck_no_entities=args.recheck_no_entities,
    )
    async with pool.acquire() as conn:
        return await conn.fetch(sql, *params)


async def run_backfill(args: argparse.Namespace) -> int:
    """Run the backfill per parsed CLI args.

    Exit codes: 0 = success, 1 = unusable backend, 2 = some rows failed
    (failed rows stay eligible and are retried on the next run), 3 = aborted
    on a systemic failure — an infrastructure error or the consecutive-error
    circuit breaker (#131).
    """
    backend = detect_backend(dict(os.environ), fallback="sqlite")
    if backend != "postgres":
        _log(
            "KG backfill requires the postgres backend; "
            "set DATABASE_URL or CONTINUOUS_CLAUDE_DB_URL"
        )
        return 1

    stats = {"processed": 0, "indexed": 0, "no_entities": 0, "errors": 0}
    after: tuple[datetime, str] | None = None
    remaining = args.limit
    consecutive_errors = 0
    no_entity_ids: list[str] = []
    pool = None
    try:
        pool = await get_pool()

        if args.dry_run:
            sql, params = build_fetch_query(
                since=args.since,
                memory_id=args.memory_id,
                count_only=True,
                project=args.project,
                recheck_no_entities=args.recheck_no_entities,
            )
            async with pool.acquire() as conn:
                eligible = await conn.fetchval(sql, *params)
            _log(f"Dry run: {eligible} memories eligible for KG backfill")
            if args.limit is not None:
                _log(f"--limit {args.limit} would cap this run at " f"{min(eligible, args.limit)}")
            return 0

        while True:
            page_size = args.batch_size if remaining is None else min(args.batch_size, remaining)
            if page_size <= 0:
                break
            rows = await _fetch_page(pool, args, after, page_size)
            if not rows:
                break

            no_entity_ids: list[str] = []
            for row in rows:
                result = await backfill_one(str(row["id"]), row["content"])
                stats["processed"] += 1
                status = result["status"]
                stats["errors" if status == "error" else status] += 1
                if status == "error":
                    # safe() at the log site, after the raw [:200] slice in
                    # backfill_one: exception text can embed semi-trusted
                    # memory content (issue #104 log-injection class)
                    _log(f"error indexing {row['id']}: {safe(result['error'])}")
                    consecutive_errors += 1
                    if consecutive_errors >= args.max_consecutive_errors:
                        # Issue #131: every row failing in a row is a
                        # systemic failure even when no INFRA_ERRORS type
                        # surfaced (e.g. schema mismatch) — stop the run
                        await mark_no_entities(pool, no_entity_ids)
                        _log(
                            f"aborting: {consecutive_errors} consecutive "
                            "errors (circuit breaker); if these are isolated "
                            "poison rows rerun with a higher "
                            "--max-consecutive-errors to push past them"
                        )
                        print(format_summary(stats), flush=True)
                        return 3
                else:
                    consecutive_errors = 0
                    if status == "no_entities":
                        no_entity_ids.append(str(row["id"]))
            await mark_no_entities(pool, no_entity_ids)

            last = rows[-1]
            after = (last["created_at"], str(last["id"]))
            if remaining is not None:
                remaining -= len(rows)
            _log(f"progress: {stats['processed']} processed")
            if len(rows) < page_size:
                break
    except INFRA_ERRORS as e:
        # Issue #131: systemic failure — abort promptly instead of logging
        # once per remaining row. Errored/unprocessed rows stay eligible.
        if pool is not None and no_entity_ids:
            # Best-effort: keep durable markers for rows already classified
            # this page when the pool still answers (e.g. the abort came
            # from a transient acquire timeout). Re-marking a previous
            # page's ids is an idempotent UPDATE.
            with contextlib.suppress(*INFRA_ERRORS):
                await mark_no_entities(pool, no_entity_ids)
        _log(f"aborting: infrastructure error: {safe(str(e)[:200])}")
        print(format_summary(stats), flush=True)
        return 3

    print(format_summary(stats), flush=True)
    return 2 if stats["errors"] else 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Write a timestamped progress message to stdout."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _bootstrap() -> None:
    """Load .env files (global ~/.claude/.env, then project .env). Called from main()."""
    from pathlib import Path

    from dotenv import load_dotenv

    global_env = Path.home() / ".claude" / ".env"
    if global_env.exists():
        load_dotenv(global_env)
    opc_env = Path(__file__).parent.parent.parent / ".env"
    if opc_env.exists():
        load_dotenv(opc_env, override=True)


async def _main_async(argv: Sequence[str]) -> int:
    """Run the CLI and always close the connection pool on the way out."""
    _bootstrap()
    args = parse_args(argv)
    try:
        return await run_backfill(args)
    finally:
        await close_pool()


def main() -> None:
    """CLI entry point."""
    sys.exit(asyncio.run(_main_async(sys.argv[1:])))


if __name__ == "__main__":
    main()
