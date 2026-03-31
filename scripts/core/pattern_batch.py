#!/usr/bin/env python3
"""Cross-session pattern detection batch runner.

Orchestrates the full pipeline: load data from PostgreSQL, run pattern
detection, write results back. Can be run as CLI or imported.

Usage:
    uv run python scripts/core/pattern_batch.py                 # Full run
    uv run python scripts/core/pattern_batch.py --dry-run       # Analyze only
    uv run python scripts/core/pattern_batch.py --report        # Last run results
    uv run python scripts/core/pattern_batch.py --min-size 3    # Smaller clusters
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load env files
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
load_dotenv()

# Add project to path
project_dir = os.environ.get("CLAUDE_PROJECT_DIR", str(Path(__file__).parent.parent.parent))
sys.path.insert(0, project_dir)

from scripts.core.pattern_detector import (  # noqa: E402
    DetectedPattern,
    Learning,
    detect_patterns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

async def _get_pool():
    """Get asyncpg connection pool."""
    from scripts.core.db.postgres_pool import get_pool
    return await get_pool()


async def _ensure_tables(pool) -> bool:
    """Ensure detected_patterns and pattern_members tables exist.

    Returns True if tables exist (or were created), False on error.
    """
    migration_path = Path(__file__).parent.parent / "migrations" / "add_detected_patterns.sql"
    if not migration_path.exists():
        logger.error("Migration file not found: %s", migration_path)
        return False

    sql = migration_path.read_text()
    try:
        async with pool.acquire() as conn:
            await conn.execute(sql)
        return True
    except Exception as e:
        logger.error("Failed to ensure tables: %s", e)
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

async def load_learnings(pool) -> list[Learning]:
    """Load all active learnings with embeddings from PostgreSQL.

    Filters:
    - superseded_by IS NULL (active only)
    - embedding IS NOT NULL
    - Excludes SYNTHESIZED_PATTERN to prevent feedback loops
    """
    query = """
        SELECT id, content, embedding, metadata, session_id, created_at
        FROM archival_memory
        WHERE superseded_by IS NULL
          AND embedding IS NOT NULL
          AND (metadata->>'learning_type') IS DISTINCT FROM 'SYNTHESIZED_PATTERN'
        ORDER BY created_at DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    learnings = []
    for row in rows:
        raw = row["metadata"]
        meta = raw if isinstance(raw, dict) else json.loads(raw or "{}")

        # Parse embedding from pgvector format
        emb = row["embedding"]
        if isinstance(emb, str):
            emb = np.fromstring(emb.strip("[]"), sep=",", dtype=np.float32)
        elif isinstance(emb, (list, tuple)):
            emb = np.array(emb, dtype=np.float32)
        elif isinstance(emb, np.ndarray):
            emb = emb.astype(np.float32)
        else:
            continue  # skip if we can't parse embedding

        if len(emb) == 0:
            continue

        learnings.append(Learning(
            id=str(row["id"]),
            content=row["content"] or "",
            embedding=emb,
            learning_type=meta.get("learning_type", "UNKNOWN"),
            tags=meta.get("tags", []) or [],
            session_id=meta.get("session_id", str(row["session_id"] or "")),
            context=meta.get("context", ""),
            created_at=row["created_at"] or datetime.now(UTC),
            confidence=meta.get("confidence", "medium"),
        ))

    return learnings


async def load_tags_for_learnings(
    pool,
    learning_ids: list[str],
) -> dict[str, list[str]]:
    """Batch-load tags from memory_tags table.

    Supplements metadata['tags'] which may be stale or incomplete.
    Returns {memory_id: [tag1, tag2, ...]}.
    """
    if not learning_ids:
        return {}

    query = """
        SELECT memory_id, tag
        FROM memory_tags
        WHERE memory_id = ANY($1::uuid[])
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, [uuid.UUID(lid) for lid in learning_ids])

    tags: dict[str, list[str]] = {}
    for row in rows:
        mid = str(row["memory_id"])
        tags.setdefault(mid, []).append(row["tag"])

    return tags


# ---------------------------------------------------------------------------
# Result writing
# ---------------------------------------------------------------------------

async def write_patterns(
    pool,
    patterns: list[DetectedPattern],
    run_id: str,
) -> int:
    """Write detected patterns to database.

    Uses pg_advisory_xact_lock to prevent concurrent runs from racing.
    Supersedes previous run's patterns before inserting new ones.

    Returns count of patterns written.
    """
    if not patterns:
        return 0

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Advisory lock to prevent concurrent pattern detection runs
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext('pattern_detection'))"
            )

            # Supersede all patterns from previous runs
            await conn.execute(
                """
                UPDATE detected_patterns
                SET superseded_at = NOW()
                WHERE superseded_at IS NULL
                """
            )

            # Insert new patterns and their members
            for pattern in patterns:
                pattern_id = await conn.fetchval(
                    """
                    INSERT INTO detected_patterns
                        (pattern_type, label, representative_id, tags,
                         session_count, confidence, metadata, run_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::uuid)
                    RETURNING id
                    """,
                    pattern.pattern_type,
                    pattern.label,
                    uuid.UUID(pattern.representative_id),
                    pattern.tags,
                    pattern.session_count,
                    pattern.confidence,
                    json.dumps(pattern.metadata),
                    uuid.UUID(run_id),
                )

                # Insert members
                if pattern.member_ids:
                    member_records = [
                        (
                            pattern_id,
                            uuid.UUID(mid),
                            pattern.distances.get(mid),
                        )
                        for mid in pattern.member_ids
                    ]
                    await conn.executemany(
                        """
                        INSERT INTO pattern_members (pattern_id, memory_id, distance)
                        VALUES ($1, $2, $3)
                        """,
                        member_records,
                    )

    return len(patterns)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

async def get_last_run_report(pool) -> str | None:
    """Generate a report from the most recent detection run."""
    async with pool.acquire() as conn:
        # Find the latest run_id
        row = await conn.fetchrow(
            """
            SELECT run_id, created_at, COUNT(*) as pattern_count
            FROM detected_patterns
            WHERE superseded_at IS NULL
            GROUP BY run_id, created_at
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        if not row:
            return "No pattern detection runs found."

        run_id = row["run_id"]
        created_at = row["created_at"]

        # Get patterns for this run
        patterns = await conn.fetch(
            """
            SELECT dp.*, COUNT(pm.memory_id) as member_count
            FROM detected_patterns dp
            LEFT JOIN pattern_members pm ON pm.pattern_id = dp.id
            WHERE dp.run_id = $1 AND dp.superseded_at IS NULL
            GROUP BY dp.id
            ORDER BY dp.confidence DESC
            """,
            run_id,
        )

    lines = [
        "=" * 50,
        "Pattern Detection Report",
        f"Run: {run_id} | Date: {created_at}",
        f"Patterns: {len(patterns)}",
        "=" * 50,
        "",
    ]

    # Group by type
    by_type: dict[str, list] = {}
    for p in patterns:
        by_type.setdefault(p["pattern_type"], []).append(p)

    for ptype, group in sorted(by_type.items()):
        lines.append(f"{ptype.upper()} ({len(group)} detected)")
        lines.append("-" * 40)
        for i, p in enumerate(group, 1):
            tags_str = ", ".join(p["tags"][:5]) if p["tags"] else "none"
            lines.append(f"  {i}. \"{p['label']}\" (confidence: {p['confidence']:.2f})")
            lines.append(f"     {p['member_count']} learnings across {p['session_count']} sessions")
            lines.append(f"     Tags: {tags_str}")
            lines.append("")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pattern_detection(
    min_cluster_size: int = 5,
    min_samples: int = 3,
    min_confidence: float = 0.3,
    dry_run: bool = False,
) -> dict:
    """Run full pattern detection pipeline.

    Returns summary dict with run stats.
    """
    start = time.monotonic()
    run_id = str(uuid.uuid4())

    pool = await _get_pool()

    # Ensure tables exist
    if not dry_run:
        if not await _ensure_tables(pool):
            return {"success": False, "error": "Failed to create tables"}

    # Load data
    logger.info("Loading learnings...")
    learnings = await load_learnings(pool)
    logger.info("Loaded %d learnings", len(learnings))

    if not learnings:
        return {
            "success": True,
            "run_id": run_id,
            "learnings_analyzed": 0,
            "patterns_detected": 0,
            "patterns_by_type": {},
            "duration_seconds": time.monotonic() - start,
            "dry_run": dry_run,
        }

    # Enrich tags from memory_tags table
    logger.info("Loading tags...")
    learning_ids = [lrn.id for lrn in learnings]
    db_tags = await load_tags_for_learnings(pool, learning_ids)
    for lrn in learnings:
        extra_tags = db_tags.get(lrn.id, [])
        if extra_tags:
            lrn.tags = list(set(lrn.tags + extra_tags))

    # Run detection
    logger.info("Running pattern detection...")
    patterns = detect_patterns(
        learnings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        min_confidence=min_confidence,
    )
    logger.info("Detected %d patterns", len(patterns))

    # Count by type
    by_type: dict[str, int] = {}
    for p in patterns:
        by_type[p.pattern_type] = by_type.get(p.pattern_type, 0) + 1

    # Write results
    written = 0
    if not dry_run and patterns:
        logger.info("Writing patterns to database...")
        written = await write_patterns(pool, patterns, run_id)
        logger.info("Wrote %d patterns", written)

    duration = time.monotonic() - start

    summary = {
        "success": True,
        "run_id": run_id,
        "learnings_analyzed": len(learnings),
        "patterns_detected": len(patterns),
        "patterns_by_type": by_type,
        "written": written,
        "duration_seconds": round(duration, 2),
        "dry_run": dry_run,
    }

    # Print pattern summaries for dry-run
    if dry_run and patterns:
        print("\n--- Dry Run Results ---")
        for i, p in enumerate(patterns, 1):
            print(f"\n{i}. [{p.pattern_type}] {p.label}")
            members = len(p.member_ids)
            sess = p.session_count
            print(f"   Confidence: {p.confidence:.2f} | Members: {members} | Sessions: {sess}")
            print(f"   Tags: {', '.join(p.tags[:5])}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-session pattern detection batch job"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only, don't write to database",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print last run's results",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Minimum cluster size (default: 5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="HDBSCAN min_samples (default: 3)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum pattern confidence (default: 0.3)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.report:
        async def _report():
            pool = await _get_pool()
            report = await get_last_run_report(pool)
            print(report)

        asyncio.run(_report())
        return

    result = asyncio.run(run_pattern_detection(
        min_cluster_size=args.min_size,
        min_samples=args.min_samples,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
    ))

    print(json.dumps(result, indent=2))

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
