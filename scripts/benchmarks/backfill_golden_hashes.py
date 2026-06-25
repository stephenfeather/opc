"""One-time backfill: re-anchor an existing golden set from UUIDs to content hashes.

Older golden sets (``scripts/benchmarks/rerank_queries.json``) pin relevance with
``golden_ids`` — raw ``archival_memory`` UUIDs that only mean anything on the DB
they were curated against. This script looks up each id's stored
``content_hash`` and writes a parallel ``golden_hashes`` list, so the benchmark
can judge relevance by content (surviving a fresh DB). Run once after curating or
when migrating an old query file.

Usage:
    uv run python scripts/benchmarks/backfill_golden_hashes.py
    uv run python scripts/benchmarks/backfill_golden_hashes.py --queries custom.json
    uv run python scripts/benchmarks/backfill_golden_hashes.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from scripts.core.db.postgres_pool import get_pool

DEFAULT_QUERIES = Path("scripts/benchmarks/rerank_queries.json")


async def lookup_hashes(ids: list[str]) -> dict[str, str]:
    """Map archival_memory id -> content_hash for the given ids.

    Missing/NULL content_hash rows are simply absent from the result.
    """
    if not ids:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id::text AS id, content_hash "
            "FROM archival_memory "
            "WHERE id = ANY($1::uuid[]) AND content_hash IS NOT NULL",
            ids,
        )
    return {r["id"]: r["content_hash"] for r in rows}


async def backfill(queries_path: Path, dry_run: bool = False) -> int:
    """Add golden_hashes to every query that has golden_ids. Returns exit code."""
    query_data = json.loads(queries_path.read_text())
    queries = query_data["queries"]

    # Gather every golden id across all queries, resolve in one round-trip.
    all_ids = sorted({gid for q in queries for gid in q.get("golden_ids", []) if gid})
    id_to_hash = await lookup_hashes(all_ids)

    total_resolved = 0
    total_missing = 0
    for q in queries:
        gids = [gid for gid in q.get("golden_ids", []) if gid]
        if not gids:
            continue
        hashes = [id_to_hash[gid] for gid in gids if gid in id_to_hash]
        missing = [gid for gid in gids if gid not in id_to_hash]
        total_resolved += len(hashes)
        total_missing += len(missing)
        q["golden_hashes"] = hashes
        status = f"{len(hashes)}/{len(gids)} anchored"
        if missing:
            status += f" — {len(missing)} unresolved (not in this DB)"
        print(f"  {q['id']}: {status}")

    print("=" * 60)
    print(f"Resolved {total_resolved} hashes; {total_missing} ids unresolved.")
    if total_missing:
        print(
            "Unresolved ids are not in the current DB — re-bootstrap those "
            "queries against this DB, or accept the partial anchor.",
            file=sys.stderr,
        )

    if dry_run:
        print("Dry run — not writing.")
        return 0

    queries_path.write_text(json.dumps(query_data, indent=2) + "\n")
    print(f"Updated {queries_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill golden_hashes from golden_ids for a benchmark query file"
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without writing the file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.queries.exists():
        print(f"Query file not found: {args.queries}", file=sys.stderr)
        sys.exit(1)
    sys.exit(asyncio.run(backfill(args.queries, dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
