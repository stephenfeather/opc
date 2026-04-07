#!/usr/bin/env python3
"""Pattern detection reporting.

Generates human-readable reports from detected_patterns and pattern_members.
Can be run standalone or called from pattern_batch.py --report.

Usage:
    uv run python scripts/core/pattern_report.py              # Latest run
    uv run python scripts/core/pattern_report.py --run-id ID  # Specific run
    uv run python scripts/core/pattern_report.py --json        # JSON output
    uv run python scripts/core/pattern_report.py --summary     # One-liner
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
load_dotenv()

project_dir = os.environ.get(
    "CLAUDE_PROJECT_DIR", str(Path(__file__).parent.parent.parent)
)
sys.path.insert(0, project_dir)

logger = logging.getLogger(__name__)


async def _get_pool():
    from scripts.core.db.postgres_pool import get_pool

    return await get_pool()


# ---------------------------------------------------------------------------
# Data queries (I/O boundary)
# ---------------------------------------------------------------------------


async def _fetch_run_metadata(conn, run_id=None) -> dict | None:
    """Get metadata for a specific run or the latest active run."""
    if run_id:
        row = await conn.fetchrow(
            """
            SELECT run_id, MIN(created_at) AS created_at,
                   COUNT(*) AS pattern_count
            FROM detected_patterns
            WHERE run_id = $1
            GROUP BY run_id
            """,
            run_id,
        )
    else:
        row = await conn.fetchrow(
            """
            SELECT run_id, MIN(created_at) AS created_at,
                   COUNT(*) AS pattern_count
            FROM detected_patterns
            WHERE superseded_at IS NULL
            GROUP BY run_id
            ORDER BY MIN(created_at) DESC
            LIMIT 1
            """
        )
    if not row:
        return None
    return dict(row)


async def _fetch_patterns_with_members(conn, run_id) -> list[dict]:
    """Fetch patterns with member counts and representative content."""
    rows = await conn.fetch(
        """
        SELECT dp.*,
               COUNT(pm.memory_id) AS member_count,
               am.content AS representative_content
        FROM detected_patterns dp
        LEFT JOIN pattern_members pm ON pm.pattern_id = dp.id
        LEFT JOIN archival_memory am ON am.id = dp.representative_id
        WHERE dp.run_id = $1
        GROUP BY dp.id, am.content
        ORDER BY dp.confidence DESC
        """,
        run_id,
    )
    return [dict(r) for r in rows]


async def _fetch_total_learnings(conn) -> int:
    """Count total active learnings for context."""
    row = await conn.fetchrow(
        """
        SELECT COUNT(*) AS cnt
        FROM archival_memory
        WHERE superseded_by IS NULL
          AND embedding IS NOT NULL
        """
    )
    return row["cnt"] if row else 0


async def _fetch_total_sessions(conn) -> int:
    """Count distinct sessions for context.

    Uses the same filters as _fetch_total_learnings so both
    functions count over the same analysed population.
    """
    row = await conn.fetchrow(
        """
        SELECT COUNT(DISTINCT session_id) AS cnt
        FROM archival_memory
        WHERE superseded_by IS NULL
          AND embedding IS NOT NULL
        """
    )
    return row["cnt"] if row else 0


async def _fetch_type_breakdown(conn, run_id) -> list[dict]:
    """Fetch pattern type counts for a run."""
    rows = await conn.fetch(
        """
        SELECT pattern_type, COUNT(*) AS cnt
        FROM detected_patterns
        WHERE run_id = $1 AND superseded_at IS NULL
        GROUP BY pattern_type
        ORDER BY cnt DESC
        """,
        run_id,
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

_TYPE_LABELS: dict[str, str] = {
    "cross_project": "CROSS-PROJECT PATTERNS",
    "expertise": "EXPERTISE AREAS",
    "tool_cluster": "TOOL CLUSTERS",
    "problem_solution": "PROBLEM-SOLUTION PATTERNS",
    "anti_pattern": "ANTI-PATTERNS",
}

_TYPE_ORDER: list[str] = [
    "cross_project",
    "expertise",
    "tool_cluster",
    "problem_solution",
    "anti_pattern",
]


def _truncate(text: str, max_len: int = 80) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_age(created: datetime, *, now: datetime | None = None) -> str:
    """Format a datetime as a human-readable relative age string.

    Handles future timestamps (clock skew) by showing 'in Xm/Xh/Xd'.
    """
    now = now or datetime.now(UTC)
    delta = now - created
    total_seconds = int(delta.total_seconds())

    if total_seconds < 0:
        # Future timestamp — clock skew or backfill
        future = -total_seconds
        if future >= 86400:
            return f"in {future // 86400}d"
        if future >= 3600:
            return f"in {future // 3600}h"
        return f"in {future // 60}m"

    if total_seconds >= 86400:
        return f"{total_seconds // 86400}d ago"
    if total_seconds >= 3600:
        return f"{total_seconds // 3600}h ago"
    return f"{total_seconds // 60}m ago"


def parse_pattern_metadata(metadata: dict | str | None) -> dict:
    """Parse pattern metadata from dict or JSON string.

    Returns empty dict for None, empty strings, malformed JSON,
    or non-dict/non-string types.
    """
    if isinstance(metadata, dict):
        return metadata
    if not isinstance(metadata, str) or not metadata:
        return {}
    try:
        parsed = json.loads(metadata)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Malformed pattern metadata: %s", metadata[:80])
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Non-object pattern metadata: %s", type(parsed).__name__)
        return {}
    return parsed


def format_type_breakdown(rows: list[dict]) -> str:
    """Format type breakdown rows into a comma-separated summary."""
    if not rows:
        return ""
    return ", ".join(f"{r['cnt']} {r['pattern_type']}" for r in rows)


def _format_tags(tags: list[str] | None) -> str:
    """Format a tags list with overflow indicator."""
    tags = tags or []
    tags_str = ", ".join(tags[:5])
    if len(tags) > 5:
        tags_str += f" (+{len(tags) - 5} more)"
    return tags_str


def format_type_section(ptype: str, group: list[dict]) -> list[str]:
    """Format a single pattern type section as a list of lines.

    Pure function — returns new list instead of mutating.
    """
    header = _TYPE_LABELS.get(ptype, ptype.upper())
    lines = [
        f"{header} ({len(group)} detected)",
        "-" * 40,
    ]

    for i, pat in enumerate(group, 1):
        meta = parse_pattern_metadata(pat["metadata"])
        span = meta.get("temporal_span_days", 0)

        lines.append(
            f"  {i}. \"{pat['label']}\""
            f" (confidence: {pat['confidence']:.2f})"
        )
        lines.append(
            f"     {pat['member_count']} learnings"
            f" across {pat['session_count']} sessions"
            f" | span: {span} days"
        )
        lines.append(f"     Tags: {_format_tags(pat['tags'])}")

        rep = pat.get("representative_content")
        if rep:
            lines.append(f"     Example: {_truncate(rep)}")

        lines.append("")

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Report generation (pure orchestrators)
# ---------------------------------------------------------------------------


def _format_human(
    meta: dict,
    patterns: list[dict],
    total_learnings: int,
    total_sessions: int,
) -> str:
    lines = [
        "=" * 56,
        "  Pattern Detection Report",
        f"  Run: {meta['run_id']}",
        f"  Date: {meta['created_at']}",
        f"  Analyzed: {total_learnings:,} learnings"
        f" across {total_sessions:,} sessions",
        "=" * 56,
        "",
    ]

    if not patterns:
        lines.append("No patterns detected.")
        return "\n".join(lines)

    # Group by type
    by_type: dict[str, list[dict]] = {}
    for pat in patterns:
        by_type.setdefault(pat["pattern_type"], []).append(pat)

    for ptype in _TYPE_ORDER:
        group = by_type.pop(ptype, [])
        if not group:
            continue
        lines.extend(format_type_section(ptype, group))

    # Any remaining types not in the order
    for ptype, group in sorted(by_type.items()):
        lines.extend(format_type_section(ptype, group))

    # Summary footer
    total_members = sum(p["member_count"] for p in patterns)
    avg_conf = (
        sum(p["confidence"] for p in patterns) / len(patterns) if patterns else 0
    )
    lines.extend([
        "-" * 56,
        f"  Total: {len(patterns)} patterns covering"
        f" {total_members} learnings",
        f"  Average confidence: {avg_conf:.2f}",
        "",
    ])

    return "\n".join(lines)


def _format_json(
    meta: dict,
    patterns: list[dict],
    total_learnings: int,
    total_sessions: int,
) -> str:
    data = {
        "run_id": str(meta["run_id"]),
        "created_at": str(meta["created_at"]),
        "total_learnings": total_learnings,
        "total_sessions": total_sessions,
        "pattern_count": len(patterns),
        "patterns": [
            {
                "id": str(pat["id"]),
                "pattern_type": pat["pattern_type"],
                "label": pat["label"],
                "confidence": pat["confidence"],
                "member_count": pat["member_count"],
                "session_count": pat["session_count"],
                "tags": pat["tags"] or [],
                "temporal_span_days": parse_pattern_metadata(
                    pat["metadata"]
                ).get("temporal_span_days", 0),
                "representative_content": _truncate(
                    pat.get("representative_content", ""), 200
                ),
            }
            for pat in patterns
        ],
    }
    return json.dumps(data, indent=2)


def generate_report_from_data(
    *,
    meta: dict | None,
    patterns: list[dict],
    total_learnings: int,
    total_sessions: int,
    as_json: bool,
) -> str:
    """Pure report generation from pre-fetched data."""
    if not meta:
        return "No pattern detection runs found."
    if as_json:
        return _format_json(meta, patterns, total_learnings, total_sessions)
    return _format_human(meta, patterns, total_learnings, total_sessions)


def generate_summary_from_data(
    *,
    meta: dict | None,
    type_rows: list[dict],
    now: datetime,
) -> str:
    """Pure summary generation from pre-fetched data."""
    if not meta:
        return "Pattern detection: no runs yet"
    age_str = format_age(meta["created_at"], now=now)
    types = format_type_breakdown(type_rows)
    return f"Last pattern detection: {age_str}, {meta['pattern_count']} patterns ({types})"


# ---------------------------------------------------------------------------
# I/O orchestrators (thin wrappers)
# ---------------------------------------------------------------------------


async def generate_report(
    run_id: str | None = None,
    as_json: bool = False,
) -> str:
    """Generate a full pattern detection report.

    If run_id is None, uses the most recent active run.
    Returns formatted string (human-readable or JSON).
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        meta = await _fetch_run_metadata(conn, run_id)
        if not meta:
            return "No pattern detection runs found."

        rid = meta["run_id"]
        patterns = await _fetch_patterns_with_members(conn, rid)
        total_learnings = await _fetch_total_learnings(conn)
        total_sessions = await _fetch_total_sessions(conn)

    return generate_report_from_data(
        meta=meta,
        patterns=patterns,
        total_learnings=total_learnings,
        total_sessions=total_sessions,
        as_json=as_json,
    )


async def generate_summary() -> str:
    """One-liner summary for daemon status integration."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        meta = await _fetch_run_metadata(conn)
        if not meta:
            return "Pattern detection: no runs yet"

        rid = meta["run_id"]
        type_rows = await _fetch_type_breakdown(conn, rid)

    return generate_summary_from_data(
        meta=meta,
        type_rows=type_rows,
        now=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Pattern detection report generator"
    )
    parser.add_argument(
        "--run-id",
        help="Specific run ID (default: latest)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output as JSON",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="One-liner summary for daemon status",
    )
    args = parser.parse_args()

    if args.summary:
        print(asyncio.run(generate_summary()))
    else:
        print(asyncio.run(generate_report(
            run_id=args.run_id,
            as_json=args.as_json,
        )))


if __name__ == "__main__":
    main()
