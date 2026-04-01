#!/usr/bin/env python3
"""Memory system health and quality metrics.

Queries PostgreSQL to produce a metrics report covering learnings,
confidence distribution, classification breakdown, extraction stats,
tag usage, and temporal trends.

USAGE:
    # JSON output (default)
    uv run python scripts/core/memory_metrics.py

    # Human-readable output
    uv run python scripts/core/memory_metrics.py --human

    # Filter by date range
    uv run python scripts/core/memory_metrics.py --period 2026-03-01:2026-03-31

Environment:
    DATABASE_URL or CONTINUOUS_CLAUDE_DB_URL - PostgreSQL connection string
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

faulthandler.enable(
    file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"),
    all_threads=True,
)

# Load .env files
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
load_dotenv()

# Add project root to path
project_dir = os.environ.get("CLAUDE_PROJECT_DIR", str(Path(__file__).parent.parent.parent))
sys.path.insert(0, project_dir)

from scripts.core.db.postgres_pool import close_pool, get_connection  # noqa: E402

VERSION = "0.7.3"


# ---------------------------------------------------------------------------
# Period parsing
# ---------------------------------------------------------------------------

def parse_period(period_str: str | None) -> tuple[datetime | None, datetime | None]:
    """Parse a period string like '2026-03-01:2026-03-31' into (start, end).

    Returns (None, None) if period_str is None.
    """
    if not period_str:
        return None, None
    parts = period_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid period format: {period_str!r}. Expected YYYY-MM-DD:YYYY-MM-DD")
    start = datetime.strptime(parts[0].strip(), "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(parts[1].strip(), "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, microsecond=999999, tzinfo=UTC
    )
    return start, end


# ---------------------------------------------------------------------------
# Metric query functions
# ---------------------------------------------------------------------------

async def get_totals(conn: Any, start: datetime | None, end: datetime | None) -> dict:
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE superseded_by IS NULL) AS active,
            COUNT(*) FILTER (WHERE superseded_by IS NOT NULL) AS superseded,
            COUNT(*) AS total
        FROM archival_memory
        WHERE ($1::timestamptz IS NULL OR created_at >= $1)
          AND ($2::timestamptz IS NULL OR created_at <= $2)
        """,
        start, end,
    )
    return {
        "active_learnings": row["active"],
        "superseded_learnings": row["superseded"],
        "total_learnings": row["total"],
    }


async def get_per_session_stats(conn: Any, start: datetime | None, end: datetime | None) -> dict:
    recent = await conn.fetchrow(
        """
        WITH recent_sessions AS (
            SELECT session_id, COUNT(*) AS cnt
            FROM archival_memory
            WHERE superseded_by IS NULL
              AND ($1::timestamptz IS NULL OR created_at >= $1)
              AND ($2::timestamptz IS NULL OR created_at <= $2)
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
            LIMIT 10
        )
        SELECT
            COALESCE(ROUND(AVG(cnt)::numeric, 1), 0) AS avg,
            COALESCE(MIN(cnt), 0) AS min,
            COALESCE(MAX(cnt), 0) AS max
        FROM recent_sessions
        """,
        start, end,
    )
    overall = await conn.fetchrow(
        """
        SELECT
            COALESCE(ROUND(AVG(cnt)::numeric, 1), 0) AS avg,
            COUNT(*) AS total_sessions
        FROM (
            SELECT session_id, COUNT(*) AS cnt
            FROM archival_memory
            WHERE superseded_by IS NULL
              AND ($1::timestamptz IS NULL OR created_at >= $1)
              AND ($2::timestamptz IS NULL OR created_at <= $2)
            GROUP BY session_id
        ) sub
        """,
        start, end,
    )
    return {
        "recent_10_sessions": {
            "average": float(recent["avg"]),
            "min": int(recent["min"]),
            "max": int(recent["max"]),
        },
        "overall": {
            "average": float(overall["avg"]),
            "total_sessions_with_learnings": int(overall["total_sessions"]),
        },
    }


async def get_confidence_distribution(
    conn: Any, start: datetime | None, end: datetime | None
) -> dict:
    rows = await conn.fetch(
        """
        SELECT
            COALESCE(metadata->>'confidence', 'unset') AS level,
            COUNT(*) AS count
        FROM archival_memory
        WHERE superseded_by IS NULL
          AND ($1::timestamptz IS NULL OR created_at >= $1)
          AND ($2::timestamptz IS NULL OR created_at <= $2)
        GROUP BY level
        """,
        start, end,
    )
    total = sum(r["count"] for r in rows)
    result = {}
    for r in rows:
        result[r["level"]] = {
            "count": r["count"],
            "pct": round(100.0 * r["count"] / total, 1) if total else 0.0,
        }
    return result


async def get_classification_distribution(
    conn: Any, start: datetime | None, end: datetime | None
) -> dict:
    rows = await conn.fetch(
        """
        SELECT
            COALESCE(metadata->>'learning_type', 'unset') AS learning_type,
            COUNT(*) AS count
        FROM archival_memory
        WHERE superseded_by IS NULL
          AND ($1::timestamptz IS NULL OR created_at >= $1)
          AND ($2::timestamptz IS NULL OR created_at <= $2)
        GROUP BY learning_type
        ORDER BY count DESC
        """,
        start, end,
    )
    total = sum(r["count"] for r in rows)
    result = {}
    for r in rows:
        result[r["learning_type"]] = {
            "count": r["count"],
            "pct": round(100.0 * r["count"] / total, 1) if total else 0.0,
        }
    return result


async def get_dedup_stats(conn: Any) -> dict:
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE content_hash IS NOT NULL) AS with_hash,
            COUNT(*) FILTER (WHERE content_hash IS NULL) AS without_hash,
            COUNT(*) AS total
        FROM archival_memory
        """
    )
    total = row["total"]
    return {
        "learnings_with_content_hash": row["with_hash"],
        "learnings_without_content_hash": row["without_hash"],
        "hash_coverage_pct": round(100.0 * row["with_hash"] / total, 1) if total else 0.0,
        "note": "Dedup rejections are not persisted; hash_coverage indicates dedup eligibility",
    }


async def get_extraction_stats(conn: Any) -> dict:
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) AS total_sessions,
            COUNT(*) FILTER (WHERE extraction_status = 'extracted'
                                OR memory_extracted_at IS NOT NULL) AS extracted,
            COUNT(*) FILTER (WHERE extraction_status = 'pending') AS pending,
            COUNT(*) FILTER (WHERE extraction_status = 'failed') AS failed,
            COUNT(*) FILTER (WHERE extraction_attempts > 1) AS retried
        FROM sessions
        """
    )
    total = row["total_sessions"]
    return {
        "total_sessions": total,
        "extracted": row["extracted"],
        "pending": row["pending"],
        "failed": row["failed"],
        "retried": row["retried"],
        "extraction_rate_pct": round(100.0 * row["extracted"] / total, 1) if total else 0.0,
    }


async def get_stale_learnings(
    conn: Any, start: datetime | None, end: datetime | None
) -> dict:
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE recall_count = 0 OR last_recalled IS NULL) AS never_recalled,
            COUNT(*) AS total_active
        FROM archival_memory
        WHERE superseded_by IS NULL
          AND ($1::timestamptz IS NULL OR created_at >= $1)
          AND ($2::timestamptz IS NULL OR created_at <= $2)
        """,
        start, end,
    )
    total = row["total_active"]
    return {
        "never_recalled": row["never_recalled"],
        "total_active": total,
        "never_recalled_pct": round(100.0 * row["never_recalled"] / total, 1) if total else 0.0,
    }


async def get_top_tags(conn: Any, limit: int = 10) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT tag, COUNT(*) AS count
        FROM memory_tags
        GROUP BY tag
        ORDER BY count DESC
        LIMIT $1
        """,
        limit,
    )
    return [{"tag": r["tag"], "count": r["count"]} for r in rows]


async def get_superseded_stats(conn: Any) -> dict:
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE superseded_by IS NOT NULL) AS superseded_count,
            COUNT(*) AS total
        FROM archival_memory
        """
    )
    total = row["total"]
    return {
        "superseded_count": row["superseded_count"],
        "total_learnings": total,
        "superseded_pct": (
            round(100.0 * row["superseded_count"] / total, 1) if total else 0.0
        ),
    }


async def get_temporal_stats(conn: Any) -> dict:
    row = await conn.fetchrow(
        """
        SELECT
            MIN(created_at) AS oldest,
            MAX(created_at) AS newest,
            COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') AS last_7_days,
            COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days') AS last_30_days
        FROM archival_memory
        WHERE superseded_by IS NULL
        """
    )
    return {
        "oldest_learning": row["oldest"].isoformat() if row["oldest"] else None,
        "newest_learning": row["newest"].isoformat() if row["newest"] else None,
        "last_7_days": row["last_7_days"],
        "last_30_days": row["last_30_days"],
    }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

async def collect_all_metrics(
    start: datetime | None = None, end: datetime | None = None
) -> dict:
    """Run all metric queries and return the full report dict."""
    async with get_connection() as conn:
        totals = await get_totals(conn, start, end)
        per_session = await get_per_session_stats(conn, start, end)
        confidence = await get_confidence_distribution(conn, start, end)
        classification = await get_classification_distribution(conn, start, end)
        dedup = await get_dedup_stats(conn)
        extraction = await get_extraction_stats(conn)
        stale = await get_stale_learnings(conn, start, end)
        tags = await get_top_tags(conn)
        superseded = await get_superseded_stats(conn)
        temporal = await get_temporal_stats(conn)

    period = None
    if start or end:
        period = {
            "from": start.isoformat() if start else None,
            "to": end.isoformat() if end else None,
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "period": period,
        "totals": totals,
        "per_session": per_session,
        "confidence_distribution": confidence,
        "classification_distribution": classification,
        "dedup_stats": dedup,
        "extraction_stats": extraction,
        "stale_learnings": stale,
        "top_tags": tags,
        "superseded": superseded,
        "temporal": temporal,
        "version": VERSION,
    }


# ---------------------------------------------------------------------------
# Human-readable formatter
# ---------------------------------------------------------------------------

def format_human(metrics: dict) -> str:
    """Format metrics as a human-readable report."""
    lines: list[str] = []
    lines.append(f"Memory Metrics Report  (v{metrics['version']})")
    lines.append(f"Generated: {metrics['generated_at']}")
    if metrics["period"]:
        lines.append(f"Period: {metrics['period']['from']} to {metrics['period']['to']}")
    lines.append("")

    t = metrics["totals"]
    lines.append(f"Totals:  {t['active_learnings']} active, "
                 f"{t['superseded_learnings']} superseded, "
                 f"{t['total_learnings']} total")
    lines.append("")

    ps = metrics["per_session"]
    r10 = ps["recent_10_sessions"]
    lines.append(f"Per Session (recent 10):  avg {r10['average']}, "
                 f"min {r10['min']}, max {r10['max']}")
    ov = ps["overall"]
    lines.append(f"Per Session (overall):    avg {ov['average']}, "
                 f"{ov['total_sessions_with_learnings']} sessions")
    lines.append("")

    lines.append("Confidence Distribution:")
    for level, data in metrics["confidence_distribution"].items():
        lines.append(f"  {level:8s}  {data['count']:4d}  ({data['pct']:.1f}%)")
    lines.append("")

    lines.append("Classification Distribution:")
    for lt, data in metrics["classification_distribution"].items():
        lines.append(f"  {lt:28s}  {data['count']:4d}  ({data['pct']:.1f}%)")
    lines.append("")

    d = metrics["dedup_stats"]
    lines.append(f"Dedup:  {d['hash_coverage_pct']:.1f}% hash coverage "
                 f"({d['learnings_with_content_hash']} with, "
                 f"{d['learnings_without_content_hash']} without)")
    lines.append("")

    e = metrics["extraction_stats"]
    lines.append(f"Extraction:  {e['extracted']}/{e['total_sessions']} extracted "
                 f"({e['extraction_rate_pct']:.1f}%), "
                 f"{e['pending']} pending, {e['failed']} failed, {e['retried']} retried")
    lines.append("")

    s = metrics["stale_learnings"]
    lines.append(f"Stale:  {s['never_recalled']}/{s['total_active']} never recalled "
                 f"({s['never_recalled_pct']:.1f}%)")
    lines.append("")

    lines.append("Top Tags:")
    for tag_data in metrics["top_tags"]:
        lines.append(f"  {tag_data['tag']:20s}  {tag_data['count']:4d}")
    lines.append("")

    sup = metrics["superseded"]
    lines.append(f"Superseded:  {sup['superseded_count']}/{sup['total_learnings']} "
                 f"({sup['superseded_pct']:.1f}%)")
    lines.append("")

    tmp = metrics["temporal"]
    lines.append(f"Temporal:  oldest {tmp['oldest_learning']}")
    lines.append(f"           newest {tmp['newest_learning']}")
    lines.append(f"           last 7d: {tmp['last_7_days']},  last 30d: {tmp['last_30_days']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory system health and quality metrics"
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Human-readable output (default is JSON)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="Date range filter: YYYY-MM-DD:YYYY-MM-DD",
    )
    return parser


async def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        start, end = parse_period(args.period)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        metrics = await collect_all_metrics(start, end)
    except Exception as e:
        print(f"Error collecting metrics: {e}", file=sys.stderr)
        return 1
    finally:
        await close_pool()

    if args.human:
        print(format_human(metrics))
    else:
        print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
