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
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_crash_log = Path.home() / ".claude" / "logs" / "opc_crash.log"
_crash_log.parent.mkdir(parents=True, exist_ok=True)
faulthandler.enable(file=_crash_log.open("a"), all_threads=True)

# Load .env files
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
load_dotenv()

# Add repository root to path
repo_root = str(Path(__file__).parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.core.db.postgres_pool import close_pool, get_connection  # noqa: E402


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("mcp-execution")
    except Exception:
        return "0.7.3"


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
    start_str = parts[0].strip()
    end_str = parts[1].strip()
    start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_str, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, microsecond=999999, tzinfo=UTC
    )
    if start > end:
        raise ValueError(
            f"Invalid period range: start date {start_str!r} is after end date {end_str!r}"
        )
    return start, end


# ---------------------------------------------------------------------------
# Metric query functions
# ---------------------------------------------------------------------------

async def get_totals(conn: Any, start: datetime | None, end: datetime | None) -> dict:
    """Count active, superseded, and total learnings."""
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
    """Compute learnings-per-session averages (recent 10 and overall)."""
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
    """Count learnings by confidence level (high/medium/low) with percentages."""
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
    by_level = {r["level"]: r["count"] for r in rows}
    canonical = ["high", "medium", "low"]
    result = {}
    for level in canonical:
        if level in by_level:
            cnt = by_level.pop(level)
            result[level] = {
                "count": cnt,
                "pct": round(100.0 * cnt / total, 1) if total else 0.0,
            }
    for level in sorted(by_level.keys()):
        result[level] = {
            "count": by_level[level],
            "pct": round(100.0 * by_level[level] / total, 1) if total else 0.0,
        }
    return result


async def get_classification_distribution(
    conn: Any, start: datetime | None, end: datetime | None
) -> dict:
    """Count learnings by learning_type with percentages."""
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
    """Report content-hash coverage as a dedup eligibility proxy (all-time)."""
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


async def get_embedding_coverage(conn: Any) -> dict:
    """Report what % of active learnings have valid embeddings (all-time)."""
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS with_embedding,
            COUNT(*) FILTER (WHERE embedding IS NULL) AS without_embedding,
            COUNT(*) AS total
        FROM archival_memory
        WHERE superseded_by IS NULL
        """
    )
    total = row["total"]
    return {
        "with_embedding": row["with_embedding"],
        "without_embedding": row["without_embedding"],
        "coverage_pct": round(100.0 * row["with_embedding"] / total, 1) if total else 0.0,
    }


async def get_feedback_velocity(conn: Any) -> dict:
    """Report feedback rate per week over the last 4 weeks (all-time table)."""
    table_exists = await conn.fetchval(
        "SELECT to_regclass('public.memory_feedback') IS NOT NULL"
    )
    if not table_exists:
        return {"weeks": [], "avg_per_week": 0.0}

    rows = await conn.fetch(
        """
        SELECT
            date_trunc('week', created_at)::date AS week_start,
            COUNT(*) AS feedback_count,
            COUNT(*) FILTER (WHERE helpful) AS helpful_count,
            COUNT(*) FILTER (WHERE NOT helpful) AS not_helpful_count
        FROM memory_feedback
        WHERE created_at >= NOW() - INTERVAL '4 weeks'
        GROUP BY week_start
        ORDER BY week_start
        """
    )
    weeks = [
        {
            "week": str(r["week_start"]),
            "total": r["feedback_count"],
            "helpful": r["helpful_count"],
            "not_helpful": r["not_helpful_count"],
        }
        for r in rows
    ]
    total = sum(w["total"] for w in weeks)
    window_weeks = 4  # fixed window, not len(weeks) which omits zero-feedback weeks
    return {
        "weeks": weeks,
        "avg_per_week": round(total / window_weeks, 1),
    }


async def get_supersession_candidates(conn: Any) -> dict:
    """Report aged never-recalled learnings, broken down by confidence.

    Criteria: active, never recalled, older than 30 days.
    Not a cleanup recommendation — high-confidence items may be valid long-tail knowledge.
    Use the confidence breakdown to prioritize review.
    """
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) AS candidate_count,
            COUNT(*) FILTER (
                WHERE metadata->>'confidence' = 'low'
            ) AS low_confidence,
            COUNT(*) FILTER (
                WHERE metadata->>'confidence' = 'medium'
            ) AS medium_confidence,
            COUNT(*) FILTER (
                WHERE metadata->>'confidence' = 'high'
            ) AS high_confidence
        FROM archival_memory
        WHERE superseded_by IS NULL
          AND (recall_count = 0 OR last_recalled IS NULL)
          AND created_at < NOW() - INTERVAL '30 days'
        """
    )
    return {
        "total_candidates": row["candidate_count"],
        "by_confidence": {
            "low": row["low_confidence"],
            "medium": row["medium_confidence"],
            "high": row["high_confidence"],
        },
        "criteria": "active, never recalled, older than 30 days",
    }


_RECALL_FREQ_UNAVAILABLE = {
    "recalled_learnings": 0,
    "total_active": 0,
    "recall_rate_pct": 0.0,
    "total_recall_events": 0,
    "avg_recalls_per_recalled_learning": 0.0,
    "max_recalls_single_learning": 0,
    "note": "recall_count column not available",
}


async def get_recall_frequency(conn: Any) -> dict:
    """Report recall usage across sessions (all-time).

    Uses recall_count on archival_memory. Degrades gracefully if the column
    is missing on older schema versions.
    """
    try:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE recall_count > 0) AS recalled_learnings,
                COUNT(*) AS total_active,
                COALESCE(SUM(recall_count), 0) AS total_recalls,
                COALESCE(AVG(recall_count) FILTER (WHERE recall_count > 0), 0)
                    AS avg_recalls_when_used,
                COALESCE(MAX(recall_count), 0) AS max_recalls
            FROM archival_memory
            WHERE superseded_by IS NULL
            """
        )
    except Exception as e:
        # Only degrade for schema-compatibility issues (missing column).
        # asyncpg raises UndefinedColumnError; check by name to avoid import.
        if "UndefinedColumn" in type(e).__name__ or "column" in str(e).lower():
            return {**_RECALL_FREQ_UNAVAILABLE, "note": f"schema degraded: {e}"}
        raise
    total = row["total_active"]
    recalled = row["recalled_learnings"]
    return {
        "recalled_learnings": recalled,
        "total_active": total,
        "recall_rate_pct": round(100.0 * recalled / total, 1) if total else 0.0,
        "total_recall_events": int(row["total_recalls"]),
        "avg_recalls_per_recalled_learning": round(float(row["avg_recalls_when_used"]), 1),
        "max_recalls_single_learning": int(row["max_recalls"]),
    }


async def get_type_recall_correlation(conn: Any) -> dict:
    """Compare learning_type distribution: stored vs. actually recalled.

    Degrades gracefully if recall_count column is missing.
    """
    try:
        rows = await conn.fetch(
            """
            SELECT
                COALESCE(metadata->>'learning_type', 'unset') AS learning_type,
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE recall_count > 0) AS recalled,
                COALESCE(SUM(recall_count), 0) AS total_recalls
            FROM archival_memory
            WHERE superseded_by IS NULL
            GROUP BY learning_type
            ORDER BY total DESC
            """
        )
    except Exception as e:
        if "UndefinedColumn" in type(e).__name__ or "column" in str(e).lower():
            return {}
        raise
    result = {}
    for r in rows:
        total = r["total"]
        recalled = r["recalled"]
        result[r["learning_type"]] = {
            "stored": total,
            "recalled": recalled,
            "recall_rate_pct": round(100.0 * recalled / total, 1) if total else 0.0,
            "total_recall_events": int(r["total_recalls"]),
        }
    return result


async def get_extraction_stats(conn: Any) -> dict:
    """Count sessions by extraction status (all-time)."""
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
    """Count learnings that have never been recalled."""
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
    """Return the most common tags by frequency (all-time)."""
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
    """Count superseded vs total learnings (all-time)."""
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
    """Report oldest/newest learning and recent activity counts (all-time)."""
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


async def get_feedback_stats(conn: Any) -> dict:
    """Report memory feedback statistics (all-time). Returns zeroes if table missing."""
    table_exists = await conn.fetchval(
        "SELECT to_regclass('public.memory_feedback') IS NOT NULL"
    )
    if not table_exists:
        return {
            "total_feedback": 0,
            "helpful": 0,
            "not_helpful": 0,
            "unique_learnings_rated": 0,
            "helpfulness_rate_pct": 0.0,
        }
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE helpful) AS helpful,
            COUNT(*) FILTER (WHERE NOT helpful) AS not_helpful,
            COUNT(DISTINCT learning_id) AS unique_learnings
        FROM memory_feedback
        """
    )
    total = row["total"]
    return {
        "total_feedback": total,
        "helpful": row["helpful"],
        "not_helpful": row["not_helpful"],
        "unique_learnings_rated": row["unique_learnings"],
        "helpfulness_rate_pct": round(row["helpful"] / total * 100, 1) if total > 0 else 0.0,
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
        dedup = await get_dedup_stats(conn)  # always all-time
        embedding_cov = await get_embedding_coverage(conn)  # always all-time
        extraction = await get_extraction_stats(conn)  # always all-time
        stale = await get_stale_learnings(conn, start, end)
        tags = await get_top_tags(conn)  # always all-time
        superseded = await get_superseded_stats(conn)  # always all-time
        temporal = await get_temporal_stats(conn)  # always all-time
        feedback = await get_feedback_stats(conn)  # always all-time
        feedback_vel = await get_feedback_velocity(conn)  # always all-time
        supersession_cand = await get_supersession_candidates(conn)  # always all-time
        recall_freq = await get_recall_frequency(conn)  # always all-time
        type_recall = await get_type_recall_correlation(conn)  # always all-time

    period = None
    if start or end:
        period = {
            "from": start.isoformat() if start else None,
            "to": end.isoformat() if end else None,
        }

    # Compute learnings-per-extraction from consistent all-time scope
    extracted = extraction["extracted"]
    all_time_learnings = totals["total_learnings"]
    if start or end:
        # totals is period-filtered; get all-time count for this ratio
        async with get_connection() as conn:
            all_time = await get_totals(conn, None, None)
        all_time_learnings = all_time["total_learnings"]
    extraction["learnings_per_extraction"] = (
        round(all_time_learnings / extracted, 2) if extracted else 0.0
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "period": period,
        "totals": totals,
        "per_session": per_session,
        "confidence_distribution": confidence,
        "classification_distribution": classification,
        "dedup_stats_alltime": dedup,
        "embedding_coverage_alltime": embedding_cov,
        "extraction_stats_alltime": extraction,
        "stale_learnings": stale,
        "top_tags_alltime": tags,
        "superseded_alltime": superseded,
        "temporal_alltime": temporal,
        "feedback_alltime": feedback,
        "feedback_velocity": feedback_vel,
        "supersession_candidates": supersession_cand,
        "recall_frequency": recall_freq,
        "type_recall_correlation": type_recall,
        "version": _get_version(),
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

    # --- System Health ---
    lines.append("═══ System Health ═══")
    lines.append("")

    emb = metrics.get("embedding_coverage_alltime", {})
    lines.append(f"Embedding Coverage:  {emb.get('coverage_pct', 0):.1f}% "
                 f"({emb.get('with_embedding', 0)} with, "
                 f"{emb.get('without_embedding', 0)} without)")

    d = metrics["dedup_stats_alltime"]
    lines.append(f"Dedup Hash Coverage: {d['hash_coverage_pct']:.1f}% "
                 f"({d['learnings_with_content_hash']} with, "
                 f"{d['learnings_without_content_hash']} without)")

    fb = metrics.get("feedback_alltime", {})
    total_fb = fb.get("total_feedback", 0)
    if total_fb > 0:
        lines.append(
            f"Feedback (all-time): {fb['helpful']} helpful, {fb['not_helpful']} not helpful "
            f"out of {total_fb} ({fb['helpfulness_rate_pct']:.1f}% helpful), "
            f"{fb['unique_learnings_rated']} unique learnings rated"
        )
    else:
        lines.append("Feedback (all-time): No feedback recorded yet")

    fv = metrics.get("feedback_velocity", {})
    lines.append(f"Feedback Velocity:   {fv.get('avg_per_week', 0):.1f}/week (last 4 weeks)")

    sc = metrics.get("supersession_candidates", {})
    lines.append(f"Supersession Candidates: {sc.get('total_candidates', 0)} "
                 f"(never recalled, >30d old)")
    by_conf = sc.get("by_confidence", {})
    if any(by_conf.values()):
        lines.append(f"  low: {by_conf.get('low', 0)}, "
                     f"medium: {by_conf.get('medium', 0)}, "
                     f"high: {by_conf.get('high', 0)}")
    lines.append("")

    # --- Recall Quality ---
    lines.append("═══ Recall Quality ═══")
    lines.append("")

    rf = metrics.get("recall_frequency", {})
    lines.append(f"Recall Rate:  {rf.get('recalled_learnings', 0)}/{rf.get('total_active', 0)} "
                 f"ever recalled ({rf.get('recall_rate_pct', 0):.1f}%)")
    lines.append(f"  Total recall events: {rf.get('total_recall_events', 0)}, "
                 f"avg per recalled: {rf.get('avg_recalls_per_recalled_learning', 0):.1f}, "
                 f"max: {rf.get('max_recalls_single_learning', 0)}")

    s = metrics["stale_learnings"]
    lines.append(f"Stale:  {s['never_recalled']}/{s['total_active']} never recalled "
                 f"({s['never_recalled_pct']:.1f}%)")

    trc = metrics.get("type_recall_correlation", {})
    if trc:
        lines.append("")
        lines.append("Type vs Recall:")
        lines.append(f"  {'Type':28s}  {'Stored':>6s}  {'Recalled':>8s}  {'Rate':>6s}  {'Events':>6s}")
        for lt, data in trc.items():
            lines.append(
                f"  {lt:28s}  {data['stored']:6d}  {data['recalled']:8d}  "
                f"{data['recall_rate_pct']:5.1f}%  {data['total_recall_events']:6d}"
            )
    lines.append("")

    # --- Storage & Extraction ---
    lines.append("═══ Storage & Extraction ═══")
    lines.append("")

    e = metrics["extraction_stats_alltime"]
    lines.append(f"Extraction (all-time):  {e['extracted']}/{e['total_sessions']} extracted "
                 f"({e['extraction_rate_pct']:.1f}%), "
                 f"{e['pending']} pending, {e['failed']} failed, {e['retried']} retried")
    lpe = e.get("learnings_per_extraction", 0.0)
    lines.append(f"  Learnings/Extraction:  {lpe:.2f}")
    lines.append("")

    sup = metrics["superseded_alltime"]
    lines.append(f"Superseded (all-time):  {sup['superseded_count']}/{sup['total_learnings']} "
                 f"({sup['superseded_pct']:.1f}%)")
    lines.append("")

    lines.append("Top Tags (all-time):")
    for tag_data in metrics["top_tags_alltime"]:
        lines.append(f"  {tag_data['tag']:20s}  {tag_data['count']:4d}")
    lines.append("")

    tmp = metrics["temporal_alltime"]
    lines.append(f"Temporal (all-time):  oldest {tmp['oldest_learning']}")
    lines.append(f"           newest {tmp['newest_learning']}")
    lines.append(f"           last 7d: {tmp['last_7_days']},  last 30d: {tmp['last_30_days']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Memory system health and quality metrics"
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--json",
        action="store_true",
        help="JSON output (default)",
    )
    output.add_argument(
        "--human",
        action="store_true",
        help="Human-readable output",
    )
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="Date range filter: YYYY-MM-DD:YYYY-MM-DD",
    )
    return parser


async def main() -> int:
    """Entry point: parse args, collect metrics, output results."""
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
