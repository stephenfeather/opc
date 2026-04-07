"""Pure functions for memory metrics computation and formatting.

All functions in this module are side-effect-free: they take data in
and return new data out without I/O, mutation, or global state.

The I/O boundary (database queries, CLI) stays in memory_metrics.py.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def calculate_pct(count: int, total: int, decimals: int = 1) -> float:
    """Calculate a percentage, returning 0.0 when total is zero."""
    if total == 0:
        return 0.0
    return round(100.0 * count / total, decimals)


# ---------------------------------------------------------------------------
# Period parsing
# ---------------------------------------------------------------------------


def parse_period(period_str: str | None) -> tuple[datetime | None, datetime | None]:
    """Parse a period string like '2026-03-01:2026-03-31' into (start, end).

    Returns (None, None) if period_str is None or empty.
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
# Row-to-dict builders (pure transforms of query results)
# ---------------------------------------------------------------------------


def build_confidence_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build confidence distribution from query rows.

    Canonical levels (high, medium, low) appear first in insertion order,
    followed by any extra levels sorted alphabetically.
    Does not mutate the input rows.
    """
    total = sum(r["count"] for r in rows)
    by_level = {r["level"]: r["count"] for r in rows}

    canonical = ["high", "medium", "low"]
    result: dict[str, dict[str, Any]] = {}

    for level in canonical:
        if level in by_level:
            cnt = by_level[level]
            result[level] = {
                "count": cnt,
                "pct": calculate_pct(cnt, total),
            }

    extras = sorted(k for k in by_level if k not in canonical)
    for level in extras:
        cnt = by_level[level]
        result[level] = {
            "count": cnt,
            "pct": calculate_pct(cnt, total),
        }

    return result


def build_classification_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build classification distribution from query rows.

    Preserves input row order. Does not mutate the input.
    """
    total = sum(r["count"] for r in rows)
    return {
        r["learning_type"]: {
            "count": r["count"],
            "pct": calculate_pct(r["count"], total),
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# Report assembly (pure)
# ---------------------------------------------------------------------------


def assemble_report(
    *,
    query_results: dict[str, Any],
    start: datetime | None,
    end: datetime | None,
    all_time_learnings: int,
    version: str,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the final metrics report dict from pre-fetched query results.

    Pure function: does not perform I/O, mutate inputs, or read the clock.
    All nested dicts are deep-copied to prevent aliasing between input and output.

    Args:
        query_results: Dict with keys matching the individual query function names
            (totals, per_session, confidence, classification, dedup,
             embedding_coverage, extraction, stale, tags, superseded,
             temporal, feedback, feedback_velocity, supersession_candidates,
             recall_frequency, type_recall_correlation).
        start: Period start (None if unfiltered).
        end: Period end (None if unfiltered).
        all_time_learnings: Total learning count (all-time) for ratio computation.
        version: Version string to include in the report.
        generated_at: ISO timestamp string. Must be provided by the caller
            from the I/O boundary (e.g. datetime.now(UTC).isoformat()).

    Returns:
        Complete metrics report dict ready for JSON serialization.
    """

    period = None
    if start or end:
        period = {
            "from": start.isoformat() if start else None,
            "to": end.isoformat() if end else None,
        }

    extracted = query_results["extraction"]["extracted"]
    extraction_with_ratio = {
        **copy.deepcopy(query_results["extraction"]),
        "learnings_per_extraction": (
            round(all_time_learnings / extracted, 2) if extracted else 0.0
        ),
    }

    qr = query_results
    return {
        "generated_at": generated_at,
        "period": period,
        "totals": copy.deepcopy(qr["totals"]),
        "per_session": copy.deepcopy(qr["per_session"]),
        "confidence_distribution": copy.deepcopy(qr["confidence"]),
        "classification_distribution": copy.deepcopy(qr["classification"]),
        "dedup_stats_alltime": copy.deepcopy(qr["dedup"]),
        "embedding_coverage_alltime": copy.deepcopy(qr["embedding_coverage"]),
        "extraction_stats_alltime": extraction_with_ratio,
        "stale_learnings": copy.deepcopy(qr["stale"]),
        "top_tags_alltime": copy.deepcopy(qr["tags"]),
        "superseded_alltime": copy.deepcopy(qr["superseded"]),
        "temporal_alltime": copy.deepcopy(qr["temporal"]),
        "feedback_alltime": copy.deepcopy(qr["feedback"]),
        "feedback_velocity": copy.deepcopy(qr["feedback_velocity"]),
        "supersession_candidates": copy.deepcopy(qr["supersession_candidates"]),
        "recall_frequency": copy.deepcopy(qr["recall_frequency"]),
        "type_recall_correlation": copy.deepcopy(qr["type_recall_correlation"]),
        "version": version,
    }


# ---------------------------------------------------------------------------
# Human-readable formatter (pure)
# ---------------------------------------------------------------------------


def format_human(metrics: dict[str, Any]) -> str:
    """Format metrics as a human-readable report.

    Takes a complete metrics report dict (as produced by assemble_report)
    and returns a multi-line string. Pure function — no I/O.
    """
    lines: list[str] = []
    lines.append(f"Memory Metrics Report  (v{metrics['version']})")
    lines.append(f"Generated: {metrics['generated_at']}")
    if metrics["period"]:
        lines.append(f"Period: {metrics['period']['from']} to {metrics['period']['to']}")
    lines.append("")

    t = metrics["totals"]
    lines.append(
        f"Totals:  {t['active_learnings']} active, "
        f"{t['superseded_learnings']} superseded, "
        f"{t['total_learnings']} total"
    )
    lines.append("")

    ps = metrics["per_session"]
    r10 = ps["recent_10_sessions"]
    lines.append(
        f"Per Session (recent 10):  avg {r10['average']}, "
        f"min {r10['min']}, max {r10['max']}"
    )
    ov = ps["overall"]
    lines.append(
        f"Per Session (overall):    avg {ov['average']}, "
        f"{ov['total_sessions_with_learnings']} sessions"
    )
    lines.append("")

    lines.append("Confidence Distribution:")
    for level, data in metrics["confidence_distribution"].items():
        lines.append(f"  {level:8s}  {data['count']:4d}  ({data['pct']:.1f}%)")
    lines.append("")

    lines.append("Classification Distribution:")
    for lt, data in metrics["classification_distribution"].items():
        lines.append(f"  {lt:28s}  {data['count']:4d}  ({data['pct']:.1f}%)")
    lines.append("")

    _format_system_health(metrics, lines)
    _format_recall_quality(metrics, lines)
    _format_storage_extraction(metrics, lines)

    return "\n".join(lines)


def _format_system_health(metrics: dict[str, Any], lines: list[str]) -> None:
    """Append system health section lines."""
    lines.append("\u2550\u2550\u2550 System Health \u2550\u2550\u2550")
    lines.append("")

    emb = metrics.get("embedding_coverage_alltime", {})
    lines.append(
        f"Embedding Coverage:  {emb.get('coverage_pct', 0):.1f}% "
        f"({emb.get('with_embedding', 0)} with, "
        f"{emb.get('without_embedding', 0)} without)"
    )

    d = metrics["dedup_stats_alltime"]
    lines.append(
        f"Dedup Hash Coverage: {d['hash_coverage_pct']:.1f}% "
        f"({d['learnings_with_content_hash']} with, "
        f"{d['learnings_without_content_hash']} without)"
    )

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
    lines.append(
        f"Supersession Candidates: {sc.get('total_candidates', 0)} "
        f"(never recalled, >30d old)"
    )
    by_conf = sc.get("by_confidence", {})
    if any(by_conf.values()):
        lines.append(
            f"  low: {by_conf.get('low', 0)}, "
            f"medium: {by_conf.get('medium', 0)}, "
            f"high: {by_conf.get('high', 0)}"
        )
    lines.append("")


def _format_recall_quality(metrics: dict[str, Any], lines: list[str]) -> None:
    """Append recall quality section lines."""
    lines.append("\u2550\u2550\u2550 Recall Quality \u2550\u2550\u2550")
    lines.append("")

    rf = metrics.get("recall_frequency", {})
    lines.append(
        f"Recall Rate:  {rf.get('recalled_learnings', 0)}/{rf.get('total_active', 0)} "
        f"ever recalled ({rf.get('recall_rate_pct', 0):.1f}%)"
    )
    lines.append(
        f"  Total recall events: {rf.get('total_recall_events', 0)}, "
        f"avg per recalled: {rf.get('avg_recalls_per_recalled_learning', 0):.1f}, "
        f"max: {rf.get('max_recalls_single_learning', 0)}"
    )

    s = metrics["stale_learnings"]
    lines.append(
        f"Stale:  {s['never_recalled']}/{s['total_active']} never recalled "
        f"({s['never_recalled_pct']:.1f}%)"
    )

    trc = metrics.get("type_recall_correlation", {})
    if trc:
        lines.append("")
        lines.append("Type vs Recall:")
        lines.append(
            f"  {'Type':28s}  {'Stored':>6s}  {'Recalled':>8s}  "
            f"{'Rate':>6s}  {'Events':>6s}"
        )
        for lt, data in trc.items():
            lines.append(
                f"  {lt:28s}  {data['stored']:6d}  {data['recalled']:8d}  "
                f"{data['recall_rate_pct']:5.1f}%  {data['total_recall_events']:6d}"
            )
    lines.append("")


def _format_storage_extraction(metrics: dict[str, Any], lines: list[str]) -> None:
    """Append storage & extraction section lines."""
    lines.append("\u2550\u2550\u2550 Storage & Extraction \u2550\u2550\u2550")
    lines.append("")

    e = metrics["extraction_stats_alltime"]
    lines.append(
        f"Extraction (all-time):  {e['extracted']}/{e['total_sessions']} extracted "
        f"({e['extraction_rate_pct']:.1f}%), "
        f"{e['pending']} pending, {e['failed']} failed, {e['retried']} retried"
    )
    lpe = e.get("learnings_per_extraction", 0.0)
    lines.append(f"  Learnings/Extraction:  {lpe:.2f}")
    lines.append("")

    sup = metrics["superseded_alltime"]
    lines.append(
        f"Superseded (all-time):  {sup['superseded_count']}/{sup['total_learnings']} "
        f"({sup['superseded_pct']:.1f}%)"
    )
    lines.append("")

    lines.append("Top Tags (all-time):")
    for tag_data in metrics["top_tags_alltime"]:
        lines.append(f"  {tag_data['tag']:20s}  {tag_data['count']:4d}")
    lines.append("")

    tmp = metrics["temporal_alltime"]
    lines.append(f"Temporal (all-time):  oldest {tmp['oldest_learning']}")
    lines.append(f"           newest {tmp['newest_learning']}")
    lines.append(f"           last 7d: {tmp['last_7_days']},  last 30d: {tmp['last_30_days']}")
