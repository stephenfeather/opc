"""Backfill knowledge graph for existing archival_memory rows (#124).

Walks archival_memory rows that have no kg_entity_mentions yet and runs the
same extraction + storage path used by store_learning._try_index_kg.

Idempotent and resumable: store_entities_and_edges dedupes on
(memory_id, entity_id) and (source_id, target_id, relation, memory_id), and
the fetch query excludes memories that already have a mention row, so partial
runs resume cleanly and re-running is a no-op.

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
import os
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

from scripts.core.db.postgres_pool import get_pool
from scripts.core.kg_extractor import (
    extract_entities,
    extract_relations,
    store_entities_and_edges,
)
from scripts.core.store_learning import detect_backend

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_fetch_query(
    limit: int | None = None,
    since: datetime | None = None,
    memory_id: str | None = None,
    count_only: bool = False,
) -> tuple[str, list[Any]]:
    """Build the SQL to select memories with no KG mentions yet.

    Returns (sql, params) with consecutively numbered placeholders.
    The NOT EXISTS filter makes partial runs resumable.
    """
    select = "SELECT count(*)" if count_only else "SELECT id, content"
    sql = (
        f"{select} FROM archival_memory m "
        "WHERE NOT EXISTS ("
        "SELECT 1 FROM kg_entity_mentions km WHERE km.memory_id = m.id)"
    )
    params: list[Any] = []
    if since is not None:
        params.append(since)
        sql += f" AND m.created_at >= ${len(params)}"
    if memory_id is not None:
        params.append(memory_id)
        sql += f" AND m.id = ${len(params)}::uuid"
    if not count_only:
        sql += " ORDER BY m.created_at"
    if limit is not None:
        params.append(limit)
        sql += f" LIMIT ${len(params)}"
    return sql, params


def chunked(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive lists of at most ``size`` items."""
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


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
    """Argparse type for ISO-8601 dates/datetimes."""
    try:
        return datetime.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value!r}") from e


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
        help="Backfill a single memory by UUID",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=500,
        help="Progress-logging batch size (default 500)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Async I/O
# ---------------------------------------------------------------------------


async def backfill_one(memory_id: str, content: str) -> dict[str, Any]:
    """Extract and store KG rows for one memory. Never raises."""
    try:
        entities = extract_entities(content)
        if not entities:
            return {"status": "no_entities"}
        relations = extract_relations(content, entities)
        stats = await store_entities_and_edges(memory_id, entities, relations)
        return {"status": "indexed", "stats": stats}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


async def run_backfill(args: argparse.Namespace) -> int:
    """Run the backfill per parsed CLI args. Returns a process exit code."""
    backend = detect_backend(dict(os.environ), fallback="sqlite")
    if backend != "postgres":
        _log(
            "KG backfill requires the postgres backend; "
            "set DATABASE_URL or CONTINUOUS_CLAUDE_DB_URL"
        )
        return 1

    pool = await get_pool()

    if args.dry_run:
        sql, params = build_fetch_query(
            since=args.since, memory_id=args.memory_id, count_only=True
        )
        async with pool.acquire() as conn:
            eligible = await conn.fetchval(sql, *params)
        _log(f"Dry run: {eligible} memories eligible for KG backfill")
        if args.limit is not None:
            _log(f"--limit {args.limit} would cap this run at {min(eligible, args.limit)}")
        return 0

    sql, params = build_fetch_query(
        limit=args.limit, since=args.since, memory_id=args.memory_id
    )
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    total = len(rows)
    _log(f"{total} memories to index")
    stats = {"processed": 0, "indexed": 0, "no_entities": 0, "errors": 0}
    for batch in chunked(rows, args.batch_size):
        for row in batch:
            result = await backfill_one(str(row["id"]), row["content"])
            stats["processed"] += 1
            status = result["status"]
            stats["errors" if status == "error" else status] += 1
            if status == "error":
                _log(f"error indexing {row['id']}: {result['error']}")
        _log(f"progress: {stats['processed']}/{total}")
    print(format_summary(stats), flush=True)
    return 0


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


def main() -> None:
    """CLI entry point."""
    _bootstrap()
    args = parse_args(sys.argv[1:])
    sys.exit(asyncio.run(run_backfill(args)))


if __name__ == "__main__":
    main()
