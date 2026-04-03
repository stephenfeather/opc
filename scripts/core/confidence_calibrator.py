#!/usr/bin/env python3
"""Confidence calibration for stored learnings.

Analyzes learning content to assign calibrated confidence scores based on
specificity, actionability, evidence, and scope. Replaces the naive
"always high" default with a heuristic-based assessment.

Scoring dimensions (each 0.0-1.0):
  - Specificity: Does the learning reference concrete files, functions, errors?
  - Actionability: Does it describe what to do (not just what happened)?
  - Evidence: Does it cite commits, test results, or verified behavior?
  - Scope: Is it narrowly applicable (high) or vague/universal (low)?

Final confidence = weighted average mapped to high/medium/low.

Usage:
    # Calibrate learnings from a specific session
    uv run python scripts/core/confidence_calibrator.py --session-id abc123

    # Backfill: recalibrate all learnings missing calibration
    uv run python scripts/core/confidence_calibrator.py --backfill

    # Dry-run: show what would change without writing
    uv run python scripts/core/confidence_calibrator.py --backfill --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring patterns (module-level constants — no side effects)
# ---------------------------------------------------------------------------

# Specificity indicators: concrete references in content
_SPECIFICITY_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\b[\w/]+\.\w{1,4}(?::\d+)?\b", 0.15),        # file paths (foo.py:42)
    (r"\b(?:def|class|function|method)\s+\w+", 0.15),  # function/class names
    (r"\b[0-9a-f]{7,40}\b", 0.10),                     # git hashes
    (r"`[^`]+`", 0.10),                                 # inline code
    (r"\b(?:error|exception|traceback|TypeError|ValueError|KeyError)\b", 0.10),
    (r"\b(?:line \d+|column \d+)\b", 0.05),            # line references
    (r"(?:https?://|localhost:\d+)", 0.05),             # URLs
)

# Actionability indicators: prescriptive language
_ACTIONABILITY_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\b(?:use|always|never|must|should|prefer|avoid|ensure)\b", 0.20),
    (r"\b(?:instead of|rather than|don't|do not)\b", 0.15),
    (r"\b(?:works? (?:by|because|when)|the (?:fix|solution|approach) is)\b", 0.15),
    (r"\b(?:step \d|first|then|next|finally)\b", 0.10),
    (r"\b(?:pattern|technique|approach|strategy|workflow)\b", 0.10),
)

# Evidence indicators: verified/tested claims
_EVIDENCE_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\b(?:commit|merged|PR|pull request)\s*[#:]?\s*\w+", 0.20),
    (r"\b(?:tested|verified|confirmed|validated|proved)\b", 0.20),
    (r"\b(?:tests? pass|build succeed|works correctly)\b", 0.15),
    (r"\b\d+%|\b\d+/\d+\b", 0.10),                      # percentages/ratios
    (r"\b(?:benchmark|measured|profiled|timed)\b", 0.10),
)

# Vagueness indicators (reduce scope score)
_VAGUENESS_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\b(?:sometimes|perhaps|possibly)\b", 0.12),
    (r"\b(?:maybe|might)\b", 0.12),
    (r"\b(?:could|may)\b", 0.08),
    (r"\b(?:generally|usually|often|tends to)\b", 0.10),
    (r"\b(?:seems? to|appears? to|looks? like)\b", 0.10),
    (r"\b(?:in general|overall|broadly)\b", 0.10),
    (r"\b(?:etc|and so on|and more)\b", 0.05),
)

# Dimension weights for final score
WEIGHTS: dict[str, float] = {
    "specificity": 0.30,
    "actionability": 0.25,
    "evidence": 0.25,
    "scope": 0.20,
}


# ---------------------------------------------------------------------------
# Pure scoring functions
# ---------------------------------------------------------------------------


def _score_dimension(
    content: str, patterns: Sequence[tuple[str, float]], cap: float = 1.0
) -> float:
    """Score content against a list of (regex, weight) patterns."""
    total = 0.0
    content_lower = content.lower()
    for pattern, weight in patterns:
        if re.search(pattern, content_lower, re.IGNORECASE):
            total += weight
    return min(total, cap)


def score_specificity(content: str) -> float:
    """How specific/concrete is the learning? (0.0-1.0)"""
    return _score_dimension(content, _SPECIFICITY_PATTERNS)


def score_actionability(content: str) -> float:
    """How actionable/prescriptive is the learning? (0.0-1.0)"""
    return _score_dimension(content, _ACTIONABILITY_PATTERNS)


def score_evidence(content: str) -> float:
    """How well-evidenced is the learning? (0.0-1.0)"""
    return _score_dimension(content, _EVIDENCE_PATTERNS)


def score_scope(content: str) -> float:
    """How focused is the learning scope? (0.0-1.0, higher = more focused)."""
    base = 0.7
    vagueness = _score_dimension(content, _VAGUENESS_PATTERNS)
    word_count = len(content.split())
    length_bonus = 0.1 if word_count < 50 else (-0.1 if word_count > 200 else 0.0)
    return max(0.0, min(1.0, base - vagueness + length_bonus))


def calibrate_confidence(content: str) -> dict:
    """Compute calibrated confidence for a learning.

    Returns:
        dict with keys: confidence (str), score (float),
        dimensions (dict of dimension scores)
    """
    if not content or not content.strip():
        return {
            "confidence": "low",
            "score": 0.0,
            "dimensions": {
                "specificity": 0.0,
                "actionability": 0.0,
                "evidence": 0.0,
                "scope": 0.0,
            },
        }

    dimensions = {
        "specificity": score_specificity(content),
        "actionability": score_actionability(content),
        "evidence": score_evidence(content),
        "scope": score_scope(content),
    }

    score = sum(WEIGHTS[k] * dimensions[k] for k in WEIGHTS)

    if score >= 0.30:
        confidence = "high"
    elif score >= 0.20:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "confidence": confidence,
        "score": round(score, 3),
        "dimensions": {k: round(v, 3) for k, v in dimensions.items()},
    }


# ---------------------------------------------------------------------------
# Pure row-level calibration (no I/O)
# ---------------------------------------------------------------------------

from uuid import UUID

Row = tuple[str | UUID, str | None, dict | None]


def calibrate_rows(rows: Sequence[Row]) -> dict:
    """Calibrate a batch of (id, content, metadata) rows.

    Pure function: no database access. Returns stats and proposed changes.

    Returns:
        dict with keys:
          - stats: {total, updated, unchanged, errors}
          - changes: list of {id, old, new, score, dimensions}
    """
    stats = {"total": len(rows), "updated": 0, "unchanged": 0, "errors": 0}
    changes: list[dict] = []

    for memory_id, content, metadata in rows:
        if not content:
            stats["errors"] += 1
            continue

        result = calibrate_confidence(content)
        meta = metadata or {}
        old_confidence = meta.get("confidence")
        new_confidence = result["confidence"]

        # Row is unchanged only if label matches AND score/dimensions present
        has_full_calibration = (
            "confidence_score" in meta and "confidence_dimensions" in meta
        )
        if old_confidence == new_confidence and has_full_calibration:
            stats["unchanged"] += 1
            continue

        stats["updated"] += 1
        changes.append({
            "id": str(memory_id),
            "old": old_confidence,
            "new": new_confidence,
            "score": result["score"],
            "dimensions": result["dimensions"],
        })

    return {"stats": stats, "changes": changes}


def _confidence_patch(change: dict) -> str:
    """Build a JSON patch containing only confidence fields for JSONB merge."""
    patch = {
        "confidence": change["new"],
        "confidence_score": change["score"],
        "confidence_dimensions": change["dimensions"],
    }
    return json.dumps(patch)


# ---------------------------------------------------------------------------
# Database helpers (I/O boundary)
# ---------------------------------------------------------------------------


def _pg_connect():
    """Get a PostgreSQL connection."""
    import psycopg2

    db_url = os.environ.get("CONTINUOUS_CLAUDE_DB_URL") or os.environ.get(
        "DATABASE_URL"
    )
    if not db_url:
        raise RuntimeError("No DATABASE_URL configured")
    return psycopg2.connect(db_url)


def _ensure_calibration_column(conn) -> None:
    """Add confidence_calibrated_at column if missing."""
    cur = conn.cursor()
    cur.execute("""
        ALTER TABLE archival_memory
        ADD COLUMN IF NOT EXISTS confidence_calibrated_at TIMESTAMPTZ
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Handler functions (thin I/O wrappers around pure logic)
# ---------------------------------------------------------------------------


def calibrate_session_sync(
    session_id: str, dry_run: bool = False
) -> dict:
    """Calibrate confidence for all learnings in a session (sync).

    Returns:
        dict with keys: stats, changes
    """
    conn = _pg_connect()
    try:
        _ensure_calibration_column(conn)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, content, metadata
            FROM archival_memory
            WHERE session_id = %s AND superseded_by IS NULL
            """,
            (session_id,),
        )
        rows = cur.fetchall()

        result = calibrate_rows(rows)

        if not dry_run:
            for change in result["changes"]:
                cur.execute(
                    """
                    UPDATE archival_memory
                    SET metadata = COALESCE(metadata, '{}'::jsonb)
                                   || %s::jsonb,
                        confidence_calibrated_at = NOW()
                    WHERE id = %s
                    """,
                    (_confidence_patch(change), change["id"]),
                )
            conn.commit()
    finally:
        conn.close()

    return result


def backfill_calibration_sync(
    dry_run: bool = False, batch_size: int = 100
) -> dict:
    """Recalibrate all learnings that haven't been calibrated yet (sync).

    Returns:
        dict with total stats across all batches
    """
    conn = _pg_connect()
    try:
        _ensure_calibration_column(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, content, metadata
            FROM archival_memory
            WHERE superseded_by IS NULL
              AND confidence_calibrated_at IS NULL
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    total_stats = {
        "total": len(rows),
        "updated": 0,
        "unchanged": 0,
        "errors": 0,
    }
    all_changes: list[dict] = []

    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        batch_result = calibrate_rows(batch)

        total_stats["updated"] += batch_result["stats"]["updated"]
        total_stats["unchanged"] += batch_result["stats"]["unchanged"]
        total_stats["errors"] += batch_result["stats"]["errors"]
        all_changes.extend(batch_result["changes"])

        if not dry_run:
            conn = _pg_connect()
            try:
                cur = conn.cursor()

                for change in batch_result["changes"]:
                    cur.execute(
                        """
                        UPDATE archival_memory
                        SET metadata = COALESCE(metadata, '{}'::jsonb)
                                       || %s::jsonb,
                            confidence_calibrated_at = NOW()
                        WHERE id = %s
                        """,
                        (_confidence_patch(change), change["id"]),
                    )

                # Mark unchanged rows as calibrated too (without
                # re-running calibrate_confidence — use set difference)
                updated_ids = {
                    change["id"] for change in batch_result["changes"]
                }
                error_ids = {
                    str(mid)
                    for mid, content, _ in batch
                    if not content
                }
                unchanged_ids = [
                    str(mid)
                    for mid, _, _ in batch
                    if str(mid) not in updated_ids
                    and str(mid) not in error_ids
                ]
                for uid in unchanged_ids:
                    cur.execute(
                        """
                        UPDATE archival_memory
                        SET confidence_calibrated_at = NOW()
                        WHERE id = %s
                        """,
                        (uid,),
                    )

                conn.commit()
            finally:
                conn.close()

    return {"stats": total_stats, "changes": all_changes}


# ---------------------------------------------------------------------------
# Async wrappers (backward compatibility with memory_daemon.py)
# These will be removed when memory_daemon.py is refactored in S30.
# ---------------------------------------------------------------------------


async def calibrate_session(
    session_id: str, dry_run: bool = False
) -> dict:
    """Async wrapper around calibrate_session_sync for daemon compat."""
    import asyncio

    return await asyncio.to_thread(
        calibrate_session_sync, session_id, dry_run=dry_run
    )


async def backfill_calibration(
    dry_run: bool = False, batch_size: int = 100
) -> dict:
    """Async wrapper around backfill_calibration_sync for daemon compat."""
    import asyncio

    return await asyncio.to_thread(
        backfill_calibration_sync, dry_run=dry_run, batch_size=batch_size
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_result(result: dict, dry_run: bool) -> str:
    """Format calibration result for human-readable output."""
    stats = result["stats"]
    prefix = "[DRY RUN] " if dry_run else ""
    lines = [
        f"{prefix}Calibration complete:",
        f"  Total:     {stats['total']}",
        f"  Updated:   {stats['updated']}",
        f"  Unchanged: {stats['unchanged']}",
        f"  Errors:    {stats['errors']}",
    ]

    if result["changes"]:
        label = "Proposed" if dry_run else "Applied"
        lines.append(f"\n{label} changes:")
        for change in result["changes"][:20]:
            cid = change["id"][:8]
            old, new, sc = change["old"], change["new"], change["score"]
            lines.append(f"  {cid}: {old} -> {new} (score: {sc})")
        if len(result["changes"]) > 20:
            lines.append(
                f"  ... and {len(result['changes']) - 20} more"
            )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. Accepts argv for testability."""
    parser = argparse.ArgumentParser(
        description="Calibrate learning confidence scores",
    )
    parser.add_argument(
        "--session-id",
        help="Calibrate learnings for a specific session",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Recalibrate all uncalibrated learnings",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args(argv)

    if not args.session_id and not args.backfill:
        parser.error("Must specify --session-id or --backfill")

    if args.session_id:
        result = calibrate_session_sync(args.session_id, dry_run=args.dry_run)
    else:
        result = backfill_calibration_sync(dry_run=args.dry_run)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_result(result, args.dry_run))


if __name__ == "__main__":
    # Side effects (dotenv, sys.path) only at CLI entry point
    import sys
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

    main()
