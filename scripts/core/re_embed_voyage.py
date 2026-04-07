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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
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

faulthandler.enable(
    file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"),
    all_threads=True,
)

from scripts.core.config import get_config as _get_config

BATCH_SIZE = _get_config().embedding.re_embed_batch_size
TARGET_MODEL = "voyage-code-3"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchResult:
    """Immutable result from processing a single batch."""

    converted: int
    failed_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def classify_pending(total: int, done: int) -> tuple[int, int]:
    """Return (pending, already_done) from total and done counts."""
    return total - done, done


def build_batch_texts(rows: list[dict[str, Any]]) -> list[str]:
    """Extract content strings from row dicts."""
    return [row["content"] for row in rows]


def format_progress_line(
    batch_num: int,
    batch_len: int,
    converted: int,
    pending: int,
    elapsed: float,
) -> str:
    """Format a single batch progress line."""
    pct = (converted / pending * 100) if pending else 100
    return (
        f"  Batch {batch_num}: {batch_len} rows  "
        f"({converted}/{pending} done, {pct:.0f}%,  {elapsed:.0f}s elapsed)"
    )


def format_summary(converted: int, failed_ids: list[str], elapsed: float) -> str:
    """Format the final run summary."""
    lines = [
        "",
        "=" * 50,
        f"Done in {elapsed:.1f}s",
        f"  Converted: {converted}",
        f"  Failed:    {len(failed_ids)}",
    ]
    if failed_ids:
        lines += [
            "",
            "To retry failed rows, run:",
            "  uv run python scripts/core/re_embed_voyage.py --retry-failed",
            "",
            "Or reset them manually:",
            "  docker exec continuous-claude-postgres psql -U claude -d continuous_claude -c \\",
            "    \"UPDATE archival_memory SET embedding_model = 'bge'"
            " WHERE embedding_model = 'bge-failed';\"",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O boundary functions
# ---------------------------------------------------------------------------


async def ensure_embedding_model_column() -> None:
    """Add embedding_model column if it doesn't exist, and backfill NULLs."""
    async with get_connection() as conn:
        await conn.execute("""
            ALTER TABLE archival_memory
            ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'bge'
        """)
        status = await conn.execute(
            "UPDATE archival_memory SET embedding_model = 'bge' WHERE embedding_model IS NULL"
        )
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
    return classify_pending(total, done)


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


async def update_batch(
    rows: list[dict], embeddings: list[list[float]], target_model: str
) -> None:
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


async def mark_failed_rows(row_ids: list[str]) -> None:
    """Mark rows as 'bge-failed' so they can be retried later."""
    async with get_connection() as conn:
        async with conn.transaction():
            for row_id in row_ids:
                await conn.execute(
                    "UPDATE archival_memory SET embedding_model = 'bge-failed' WHERE id = $1",
                    UUID(row_id),
                )


async def process_single_batch(
    rows: list[dict],
    provider: Any,
    target_model: str,
    update_fn: Callable[..., Awaitable[None]],
    mark_failed_fn: Callable[..., Awaitable[None]] | None = None,
) -> BatchResult:
    """Embed and update a single batch, returning an immutable result."""
    texts = build_batch_texts(rows)
    try:
        embeddings = await provider.embed_batch(texts)
        await update_fn(rows, embeddings, target_model)
        return BatchResult(converted=len(rows))
    except EmbeddingError:
        ids = [str(row["id"]) for row in rows]
        if mark_failed_fn is not None:
            await mark_failed_fn(ids)
        return BatchResult(converted=0, failed_ids=ids)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run(model: str, batch_size: int, dry_run: bool) -> None:
    """Orchestrate the re-embedding pipeline."""
    print(f"Re-embedding archival_memory → {model}")
    print(f"  Batch size: {batch_size}  |  Dry run: {dry_run}")
    print()

    print("Step 1: Ensuring embedding_model column exists...")
    await ensure_embedding_model_column()

    pending, already_done = await count_pending(model)
    total = pending + already_done
    print(f"Step 2: {pending} rows to convert, {already_done} already done, {total} total")
    print()

    if pending == 0:
        print("Nothing to do — all rows already use", model)
        return

    if dry_run:
        print("Dry run complete. Run without --dry-run to apply changes.")
        return

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY is not set. Add it to ~/.claude/.env or opc/.env")
        sys.exit(1)

    provider = VoyageEmbeddingProvider(model=model, api_key=api_key)
    converted = 0
    all_failed: list[str] = []
    start_time = time.monotonic()

    print("Step 3: Converting...")
    while True:
        rows = await fetch_batch(model, batch_size, offset=0)
        if not rows:
            break

        batch_num = converted // batch_size + 1
        elapsed = time.monotonic() - start_time
        print(format_progress_line(batch_num, len(rows), converted, pending, elapsed))

        result = await process_single_batch(
            rows=rows,
            provider=provider,
            target_model=model,
            update_fn=update_batch,
            mark_failed_fn=mark_failed_rows,
        )
        converted += result.converted
        all_failed.extend(result.failed_ids)

        if result.failed_ids:
            print(f"  WARNING: Batch failed — {len(result.failed_ids)} rows skipped.")
            print("    Marked as 'bge-failed' — re-run to retry them.")

        await asyncio.sleep(0.2)

    elapsed = time.monotonic() - start_time
    print(format_summary(converted, all_failed, elapsed))

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
            "UPDATE archival_memory SET embedding_model = 'bge'"
            " WHERE embedding_model = 'bge-failed'"
        )
        print(f"Reset {count} failed rows back to 'bge' — ready to retry.")
    await close_pool()


def main() -> None:
    """CLI entry point."""
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
