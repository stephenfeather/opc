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
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load env
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
load_dotenv()

project_dir = os.environ.get(
    "CLAUDE_PROJECT_DIR", str(Path(__file__).parent.parent.parent)
)
sys.path.insert(0, project_dir)

# ---------------------------------------------------------------------------
# Scoring patterns
# ---------------------------------------------------------------------------

# Specificity indicators: concrete references in content
_SPECIFICITY_PATTERNS = [
    (r"\b[\w/]+\.\w{1,4}(?::\d+)?\b", 0.15),       # file paths (foo.py:42)
    (r"\b(?:def|class|function|method)\s+\w+", 0.15), # function/class names
    (r"\b[0-9a-f]{7,40}\b", 0.10),                    # git hashes
    (r"`[^`]+`", 0.10),                                # inline code
    (r"\b(?:error|exception|traceback|TypeError|ValueError|KeyError)\b", 0.10),
    (r"\b(?:line \d+|column \d+)\b", 0.05),           # line references
    (r"(?:https?://|localhost:\d+)", 0.05),            # URLs
]

# Actionability indicators: prescriptive language
_ACTIONABILITY_PATTERNS = [
    (r"\b(?:use|always|never|must|should|prefer|avoid|ensure)\b", 0.20),
    (r"\b(?:instead of|rather than|don't|do not)\b", 0.15),
    (r"\b(?:works? (?:by|because|when)|the (?:fix|solution|approach) is)\b", 0.15),
    (r"\b(?:step \d|first|then|next|finally)\b", 0.10),
    (r"\b(?:pattern|technique|approach|strategy|workflow)\b", 0.10),
]

# Evidence indicators: verified/tested claims
_EVIDENCE_PATTERNS = [
    (r"\b(?:commit|merged|PR|pull request)\s*[#:]?\s*\w+", 0.20),
    (r"\b(?:tested|verified|confirmed|validated|proved)\b", 0.20),
    (r"\b(?:tests? pass|build succeed|works correctly)\b", 0.15),
    (r"\b\d+%|\b\d+/\d+\b", 0.10),                     # percentages/ratios
    (r"\b(?:benchmark|measured|profiled|timed)\b", 0.10),
]

# Vagueness indicators (reduce scope score)
_VAGUENESS_PATTERNS = [
    (r"\b(?:sometimes|perhaps|possibly)\b", 0.12),
    (r"\b(?:maybe|might)\b", 0.12),
    (r"\b(?:could|may)\b", 0.08),
    (r"\b(?:generally|usually|often|tends to)\b", 0.10),
    (r"\b(?:seems? to|appears? to|looks? like)\b", 0.10),
    (r"\b(?:in general|overall|broadly)\b", 0.10),
    (r"\b(?:etc|and so on|and more)\b", 0.05),
]


def _score_dimension(
    content: str, patterns: list[tuple[str, float]], cap: float = 1.0
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
    """How focused is the learning scope? (0.0-1.0, higher = more focused)"""
    # Start at 0.7 (decent baseline), reduce for vagueness
    base = 0.7
    vagueness = _score_dimension(content, _VAGUENESS_PATTERNS)
    # Shorter learnings tend to be more focused
    word_count = len(content.split())
    length_bonus = 0.1 if word_count < 50 else (-0.1 if word_count > 200 else 0.0)
    return max(0.0, min(1.0, base - vagueness + length_bonus))


# Dimension weights for final score
WEIGHTS = {
    "specificity": 0.30,
    "actionability": 0.25,
    "evidence": 0.25,
    "scope": 0.20,
}


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

    # Map to categorical confidence
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
# Database operations
# ---------------------------------------------------------------------------


def _pg_connect():
    """Get a PostgreSQL connection."""
    import psycopg2
    db_url = os.environ.get("CONTINUOUS_CLAUDE_DB_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("No DATABASE_URL configured")
    return psycopg2.connect(db_url)


def _ensure_calibration_column():
    """Add confidence_calibrated_at column if missing."""
    conn = _pg_connect()
    cur = conn.cursor()
    cur.execute("""
        ALTER TABLE archival_memory
        ADD COLUMN IF NOT EXISTS confidence_calibrated_at TIMESTAMPTZ
    """)
    conn.commit()
    conn.close()


async def calibrate_session(session_id: str, dry_run: bool = False) -> dict:
    """Calibrate confidence for all learnings in a session.

    Returns:
        dict with counts: total, updated, unchanged, errors
    """
    conn = _pg_connect()
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

    stats = {"total": len(rows), "updated": 0, "unchanged": 0, "errors": 0}
    changes = []

    for row in rows:
        memory_id, content, metadata = row
        if not content:
            stats["errors"] += 1
            continue

        result = calibrate_confidence(content)
        old_confidence = (metadata or {}).get("confidence")
        new_confidence = result["confidence"]

        if old_confidence == new_confidence:
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

        if not dry_run:
            updated_metadata = dict(metadata or {})
            updated_metadata["confidence"] = new_confidence
            updated_metadata["confidence_score"] = result["score"]
            updated_metadata["confidence_dimensions"] = result["dimensions"]
            cur.execute(
                """
                UPDATE archival_memory
                SET metadata = %s,
                    confidence_calibrated_at = NOW()
                WHERE id = %s
                """,
                (json.dumps(updated_metadata), memory_id),
            )

    if not dry_run:
        conn.commit()
    conn.close()

    return {"stats": stats, "changes": changes}


async def backfill_calibration(
    dry_run: bool = False, batch_size: int = 100
) -> dict:
    """Recalibrate all learnings that haven't been calibrated yet.

    Returns:
        dict with total stats across all batches
    """
    _ensure_calibration_column()

    conn = _pg_connect()
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
    conn.close()

    total_stats = {"total": len(rows), "updated": 0, "unchanged": 0, "errors": 0}
    all_changes = []

    # Process in batches
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        conn = _pg_connect()
        cur = conn.cursor()

        for row in batch:
            memory_id, content, metadata = row
            if not content:
                total_stats["errors"] += 1
                continue

            result = calibrate_confidence(content)
            old_confidence = (metadata or {}).get("confidence")
            new_confidence = result["confidence"]

            if old_confidence == new_confidence:
                total_stats["unchanged"] += 1
                if not dry_run:
                    # Still mark as calibrated even if unchanged
                    cur.execute(
                        """
                        UPDATE archival_memory
                        SET confidence_calibrated_at = NOW()
                        WHERE id = %s
                        """,
                        (memory_id,),
                    )
                continue

            total_stats["updated"] += 1
            all_changes.append({
                "id": str(memory_id),
                "old": old_confidence,
                "new": new_confidence,
                "score": result["score"],
            })

            if not dry_run:
                updated_metadata = dict(metadata or {})
                updated_metadata["confidence"] = new_confidence
                updated_metadata["confidence_score"] = result["score"]
                updated_metadata["confidence_dimensions"] = result["dimensions"]
                cur.execute(
                    """
                    UPDATE archival_memory
                    SET metadata = %s,
                        confidence_calibrated_at = NOW()
                    WHERE id = %s
                    """,
                    (json.dumps(updated_metadata), memory_id),
                )

        if not dry_run:
            conn.commit()
        conn.close()

    return {"stats": total_stats, "changes": all_changes}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Calibrate learning confidence scores",
    )
    parser.add_argument(
        "--session-id", help="Calibrate learnings for a specific session",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Recalibrate all uncalibrated learnings",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.session_id and not args.backfill:
        parser.error("Must specify --session-id or --backfill")

    if args.session_id:
        result = await calibrate_session(args.session_id, dry_run=args.dry_run)
    else:
        result = await backfill_calibration(dry_run=args.dry_run)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        stats = result["stats"]
        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"{prefix}Calibration complete:")
        print(f"  Total:     {stats['total']}")
        print(f"  Updated:   {stats['updated']}")
        print(f"  Unchanged: {stats['unchanged']}")
        print(f"  Errors:    {stats['errors']}")

        if result["changes"]:
            print(f"\n{'Proposed' if args.dry_run else 'Applied'} changes:")
            for change in result["changes"][:20]:
                cid = change["id"][:8]
                old, new, sc = change["old"], change["new"], change["score"]
                print(f"  {cid}: {old} -> {new} (score: {sc})")
            if len(result["changes"]) > 20:
                print(f"  ... and {len(result['changes']) - 20} more")


if __name__ == "__main__":
    asyncio.run(main())
