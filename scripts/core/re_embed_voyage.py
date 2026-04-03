#!/usr/bin/env python3
"""Re-embed archival_memory learnings from BGE to Voyage embeddings.

Converts all stored learnings from local BGE (BAAI/bge-large-en-v1.5) to
Voyage embeddings (voyage-code-3 by default). Both are 1024-dim so no schema
change is needed for the embedding column.

Uses an embedding_model column as a progress queue — idempotent and restartable.
Any row not yet marked as the target model gets processed. If an API call fails
after retries, that row is skipped and will be picked up on the next run.

Usage:
    uv run python scripts/core/re_embed_voyage.py
    uv run python scripts/core/re_embed_voyage.py --model voyage-3
    uv run python scripts/core/re_embed_voyage.py --batch-size 32 --dry-run

Environment:
    VOYAGE_API_KEY:  Required. Voyage AI API key.
    DATABASE_URL:    PostgreSQL connection string (defaults to dev docker).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv

# Load ~/.claude/.env first, then opc/.env
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
load_dotenv()

# Add project root to path
project_dir = os.environ.get("CLAUDE_PROJECT_DIR", str(Path(__file__).parent.parent.parent))
sys.path.insert(0, project_dir)

import faulthandler

from scripts.core.db.embedding_service import EmbeddingError, VoyageEmbeddingProvider  # noqa: E402
from scripts.core.db.postgres_pool import close_pool, get_connection  # noqa: E402

faulthandler.enable(file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"), all_threads=True)

from scripts.core.config import get_config as _get_config

BATCH_SIZE = _get_config().embedding.re_embed_batch_size
TARGET_MODEL = "voyage-code-3"


async def ensure_embedding_model_column() -> None:
    """Add embedding_model column if it doesn't exist, and backfill NULLs."""
    async with get_connection() as conn:
        await conn.execute("""
            ALTER TABLE archival_memory
            ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'bge'
        """)
        # Backfill any rows that pre-date the column (NULL → 'bge')
        status = await conn.execute(
            "UPDATE archival_memory SET embedding_model = 'bge' WHERE embedding_model IS NULL"
        )
        # status is e.g. "UPDATE 42"
        updated = int(status.split()[-1])
        if updated:
            print(f"  Backfilled {updated} rows with embedding_model = 'bge'")


async def count_pending(target_model: str) -> tuple[int, int]:
    """Return (pending, total) row counts."""
    async with get_connection() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM archival_memory")
        done = await conn.fetchval(
            "SELECT COUNT(*) FROM archival_memory WHERE embedding_model = $1",
            target_model,
        )
    return total - done, total


async def fetch_batch(target_model: str, batch_size: int, offset: int) -> list[dict]:
    """Fetch a batch of rows that still need re-embedding."""
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content
            FROM archival_memory
            WHERE embedding_model != $1 OR embedding_model IS NULL
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
            """,
            target_model,
            batch_size,
            offset,
        )
    return [{"id": row["id"], "content": row["content"]} for row in rows]


async def update_batch(rows: list[dict], embeddings: list[list[float]], target_model: str) -> None:
    """Write new embeddings and mark rows as converted."""
    async with get_connection() as conn:
        async with conn.transaction():
            for row, embedding in zip(rows, embeddings):
                await conn.execute(
                    """
                    UPDATE archival_memory
                    SET embedding = $1, embedding_model = $2
                    WHERE id = $3
                    """,
                    embedding,
                    target_model,
                    row["id"],
                )


async def run(model: str, batch_size: int, dry_run: bool) -> None:
    print(f"Re-embedding archival_memory → {model}")
    print(f"  Batch size: {batch_size}  |  Dry run: {dry_run}")
    print()

    # Step 1: Migration — add column if needed
    print("Step 1: Ensuring embedding_model column exists...")
    await ensure_embedding_model_column()

    # Step 2: Count work
    pending, total = await count_pending(model)
    print(f"Step 2: {pending} rows to convert, {total - pending} already done, {total} total")
    print()

    if pending == 0:
        print("Nothing to do — all rows already use", model)
        return

    if dry_run:
        print("Dry run complete. Run without --dry-run to apply changes.")
        return

    # Step 3: Initialize Voyage provider
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY is not set. Add it to ~/.claude/.env or opc/.env")
        sys.exit(1)

    provider = VoyageEmbeddingProvider(model=model, api_key=api_key)

    # Step 4: Process in batches
    converted = 0
    failed_ids: list[str] = []
    start_time = time.monotonic()

    # Use offset=0 always — after each successful batch the rows are marked
    # as the target model and won't appear in the next fetch.
    # On failure, those rows stay unconverted and appear again in offset=0 effectively.
    # We track how many consecutive batches returned 0 rows to detect completion.
    empty_batches = 0

    print("Step 3: Converting...")
    while True:
        rows = await fetch_batch(model, batch_size, offset=0)
        if not rows:
            break

        batch_num = converted // batch_size + 1
        pct = (converted / pending * 100) if pending else 100
        elapsed = time.monotonic() - start_time
        print(f"  Batch {batch_num}: {len(rows)} rows  ({converted}/{pending} done, {pct:.0f}%,  {elapsed:.0f}s elapsed)")

        texts = [row["content"] for row in rows]
        try:
            embeddings = await provider.embed_batch(texts)
            await update_batch(rows, embeddings, model)
            converted += len(rows)
        except EmbeddingError as e:
            ids = [str(row["id"]) for row in rows]
            failed_ids.extend(ids)
            print(f"  WARNING: Batch failed — {len(rows)} rows skipped. Will retry next run.")
            print(f"    Error: {str(e)[:200]}")
            # Shift offset past the failed batch so we don't loop forever on the same rows
            # We re-fetch with the failed IDs excluded next iteration by marking them
            # temporarily — instead we just break out and report. User can re-run.
            # For a cleaner approach: mark failed rows with a temporary 'bge-failed' model.
            async with get_connection() as conn:
                async with conn.transaction():
                    for row_id in ids:
                        await conn.execute(
                            "UPDATE archival_memory SET embedding_model = 'bge-failed' WHERE id = $1",
                            UUID(row_id),
                        )
            print(f"    Marked {len(ids)} rows as 'bge-failed' — re-run to retry them.")

        # Small pause between batches to be kind to the API
        await asyncio.sleep(0.2)

    # Step 5: Summary
    elapsed = time.monotonic() - start_time
    print()
    print("=" * 50)
    print(f"Done in {elapsed:.1f}s")
    print(f"  Converted: {converted}")
    print(f"  Failed:    {len(failed_ids)}")

    if failed_ids:
        print()
        print("To retry failed rows, run:")
        print("  uv run python scripts/core/re_embed_voyage.py --retry-failed")
        print()
        print("Or reset them manually:")
        print("  docker exec continuous-claude-postgres psql -U claude -d continuous_claude -c \\")
        print("    \"UPDATE archival_memory SET embedding_model = 'bge' WHERE embedding_model = 'bge-failed';\"")

    await provider.aclose()
    await close_pool()


async def reset_failed() -> None:
    """Reset bge-failed rows back to bge so they're retried."""
    async with get_connection() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM archival_memory WHERE embedding_model = 'bge-failed'"
        )
        if count == 0:
            print("No failed rows to reset.")
            return
        await conn.execute(
            "UPDATE archival_memory SET embedding_model = 'bge' WHERE embedding_model = 'bge-failed'"
        )
        print(f"Reset {count} failed rows back to 'bge' — ready to retry.")
    await close_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed archival_memory learnings to Voyage")
    parser.add_argument(
        "--model",
        default=TARGET_MODEL,
        choices=["voyage-3", "voyage-3-large", "voyage-code-3"],
        help="Voyage model to use (default: voyage-code-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Rows per API call (default: {BATCH_SIZE}, max: 128)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows and validate setup without making API calls",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset bge-failed rows back to bge so they get retried",
    )
    args = parser.parse_args()

    if args.retry_failed:
        asyncio.run(reset_failed())
    else:
        asyncio.run(run(model=args.model, batch_size=args.batch_size, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
