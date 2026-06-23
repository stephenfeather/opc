#!/usr/bin/env python3
"""Memory-review candidate detector (issue #63).

Phase 1 — read-only. Surfaces promotion / near-duplicate / stale candidates from
``archival_memory`` for a single project using hard signals (recall_count, embedding
cosine, age). Emits a grouped report; applies NOTHING. The ``/memory-review`` skill
consumes this output, has the model judge destinations against the live memory layers,
and presents the result for per-item user approval.

Design: thoughts/shared/2026-06-20-issue-63-memory-organization-design.md
SQL prototype: thoughts/shared/2026-06-20-issue-63-candidate-detection.sql
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from asyncpg.exceptions import PostgresError, QueryCanceledError  # noqa: E402

from scripts.core.db.postgres_pool import close_pool, get_pool  # noqa: E402
from scripts.core.project_naming import (  # noqa: E402
    canonicalize_project,
    project_from_path,
)

# --- Constants -------------------------------------------------------------

DEFAULT_MIN_RECALL = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.90
DEFAULT_EF_SEARCH = 40
DEFAULT_MERGE_LIMIT = 200
# Top-k nearest neighbors examined per row before collapsing to canonical pairs.
# k=1 silently drops real near-dup pairs in clustered data; a small k recovers them
# without materially changing scan cost (the HNSW probe dominates, not the LIMIT).
DEFAULT_MERGE_NEIGHBORS = 5
# The merge scan runs one HNSW probe per active learning, so it grows with project
# size and is the only slow detector. Bound it under the pool's 60s command_timeout
# so a large project degrades to "scan skipped" instead of crashing the whole review.
DEFAULT_MERGE_TIMEOUT_S = 50.0
STALE_OPEN_THREAD_DAYS = 30

# Promotion routing: learning_type -> always-loaded destination tier.
# Types absent from this map stay on-demand in archival_memory by design —
# promoting WORKING_SOLUTION/ERROR_FIX/FAILED_APPROACH floods always-loaded
# context with content that recall surfaces only when relevant.
#
# Known Phase-1 boundary: the promotion query filters by learning_type, so a
# high-recall entry MISLABELED as a stay-on-demand (or unknown/future) type is not
# surfaced for promotion. This is a deliberate scope decision — a type-agnostic
# sweep that re-judges every high-recall entry's true type is a Phase-2 concern
# (it ~5x's candidate volume). Cleaning mislabels is the data-quality path, not this
# detector's job in Phase 1.
_ROUTING: dict[str, str] = {
    "USER_PREFERENCE": "rules/",
    "ARCHITECTURAL_DECISION": "CLAUDE.md",
    "CODEBASE_PATTERN": "MEMORY.md",
}
PROMOTABLE_TYPES: tuple[str, ...] = tuple(_ROUTING)


# --- Data structures -------------------------------------------------------


@dataclass(frozen=True)
class PromotionCandidate:
    id: str
    content: str
    recall_count: int
    learning_type: str
    destination: str


@dataclass(frozen=True)
class MergeCandidate:
    id_a: str
    id_b: str
    similarity: float
    preview_a: str
    preview_b: str


@dataclass(frozen=True)
class MergeRow:
    """The keeper-selection signals for one side of a merge pair.

    Distinct from ``MergeCandidate`` (which carries previews for the report): this is the
    minimal row the apply path needs to pick a keeper — recall_count and created_at for the
    tie-break, and ``superseded_by`` so an already-superseded side is refused before any write.
    """

    id: str
    recall_count: int
    created_at: datetime
    superseded_by: str | None


@dataclass(frozen=True)
class StaleBucket:
    label: str
    count: int


@dataclass(frozen=True)
class ReviewReport:
    project: str
    total_active: int
    promotions: list[PromotionCandidate] = field(default_factory=list)
    merges: list[MergeCandidate] = field(default_factory=list)
    stale_buckets: list[StaleBucket] = field(default_factory=list)
    stale_open_threads: int = 0
    merges_timed_out: bool = False
    merge_scanned_model: str | None = None
    merge_skipped_rows: int = 0


# --- Pure functions --------------------------------------------------------


def route_destination(learning_type: str) -> str | None:
    """Map a learning_type to its always-loaded destination, or None to stay on-demand."""
    return _ROUTING.get(learning_type)


def is_promotable(learning_type: str) -> bool:
    """True when the type benefits from promotion to an always-loaded tier."""
    return route_destination(learning_type) is not None


def _truncate(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def format_report(report: ReviewReport) -> str:
    """Render a grouped, approval-oriented text report. Changes nothing."""
    lines: list[str] = []
    lines.append(f"## Memory Review — project: {report.project}")
    lines.append(f"({report.total_active} active learnings)")
    lines.append("")

    # 1. Promotions, grouped by destination
    lines.append(f"### 1. Promotions ({len(report.promotions)} candidates)")
    if report.promotions:
        by_dest: dict[str, list[PromotionCandidate]] = {}
        for c in report.promotions:
            by_dest.setdefault(c.destination, []).append(c)
        for dest in sorted(by_dest):
            group = sorted(by_dest[dest], key=lambda c: c.recall_count, reverse=True)
            lines.append(f"  → {dest}  ({len(group)})")
            for c in group:
                # Full id is shown so an approved promotion can be applied directly:
                # `opc memory-apply <project> --ids <id>` (Phase 2a).
                lines.append(f"      • [{c.learning_type}, recalled {c.recall_count}×] id={c.id}")
                lines.append(f"          {_truncate(c.content)}")
    else:
        lines.append("  (none)")
    lines.append("")

    # 2. Cleanup — near-duplicate merges
    if report.merges_timed_out:
        # Incomplete scan — must NOT read as "zero duplicates found". Cost is one HNSW
        # probe per active learning, so it scales with corpus size, NOT with --threshold
        # (the nearest neighbor is found before the threshold filters output). The real
        # levers are a smaller corpus or skipping merges.
        lines.append("### 2. Cleanup — merges (not scanned: timed out)")
        lines.append(
            f"  ⚠️ merge scan exceeded its time budget on {report.total_active} "
            "learnings (cost scales with corpus size). This is NOT a zero result — the "
            "scan did not complete. Re-run with --promote-only to skip it, or scan a "
            "smaller project."
        )
    else:
        lines.append(f"### 2. Cleanup — merges ({len(report.merges)} near-duplicate pairs)")
        if report.merge_skipped_rows > 0:
            lines.append(
                f"  ⚠️ partial scan: only the '{report.merge_scanned_model}' embedding "
                f"space was scanned; {report.merge_skipped_rows} embedded row(s) in other "
                "spaces were skipped (re-embed in progress?)."
            )
        if report.merges:
            for m in sorted(report.merges, key=lambda m: m.similarity, reverse=True):
                lines.append(
                    f"      • [{m.similarity:.3f}] ({m.id_a[:8]} ⇄ {m.id_b[:8]}) "
                    f"{_truncate(m.preview_a, 55)} ⇄ {_truncate(m.preview_b, 55)}"
                )
        else:
            lines.append("  (none)")
    lines.append("")

    # 3. Cleanup — stale
    lines.append("### 3. Cleanup — stale")
    if report.stale_buckets:
        for b in report.stale_buckets:
            lines.append(f"      • {b.label}: {b.count}")
    lines.append(
        f"      • stale OPEN_THREAD (>{STALE_OPEN_THREAD_DAYS}d, never recalled): "
        f"{report.stale_open_threads}"
    )
    lines.append("")

    lines.append(
        "_Read-only proposal. No changes applied. Approve items individually before any write._"
    )
    return "\n".join(lines)


# --- I/O handlers ----------------------------------------------------------

_PROMOTION_SQL = """
    SELECT id::text AS id,
           content,
           recall_count,
           metadata->>'learning_type' AS learning_type
    FROM archival_memory
    WHERE LOWER(project) = LOWER($1)
      AND superseded_by IS NULL
      AND recall_count >= $2
      AND metadata->>'learning_type' = ANY($3::text[])
    ORDER BY recall_count DESC
"""

# Near-duplicate detection. Guards:
#   1. Single embedding space — cosine across different embedding_models is
#      meaningless (a partial re-embed leaves mixed BGE/Voyage rows). The model is
#      resolved ONCE by fetch_merge_coverage and passed in as $5, so the merge scan
#      and the coverage disclosure can never describe different spaces (no drift), and
#      the dominant-model scan is not duplicated on the hot path.
#   2. Canonical unordered pairs — take the top-k nearest neighbors per row, then
#      collapse to one row per {LEAST,GREATEST} id pair. A naive single-NN +
#      "a.id < nn.id" filter drops real pairs (A's NN is C while B's NN is A is
#      silently lost). Top-k + DISTINCT ON recovers them.
#   3. The lateral reads archival_memory DIRECTLY (not a CTE alias): a multiply-
#      referenced CTE is a materialization boundary, which strips index eligibility.
#      NOTE: per-project near-dup is still inherently O(n^2) — the project filter is
#      selective enough that the planner filter-sorts rather than using the global
#      HNSW index (verified via EXPLAIN), even with iterative scan. The merge_timeout
#      / statement_timeout guard is the real protection on large corpora, not the
#      index. Small/medium projects complete fast; opc-scale degrades gracefully.
_MERGE_SQL = """
    WITH scoped AS (
        SELECT id, content, embedding
        FROM archival_memory
        WHERE LOWER(project) = LOWER($1)
          AND superseded_by IS NULL
          AND embedding IS NOT NULL
          AND embedding_model = $5
    ),
    pairs AS (
        SELECT LEAST(a.id, nn.id) AS lo,
               GREATEST(a.id, nn.id) AS hi,
               (1 - (a.embedding <=> nn.embedding))::float AS similarity,
               -- Tie previews to the canonical ids, not to the directed (a, nn)
               -- roles: lo is the smaller id, so its preview must be that row's
               -- content regardless of which side the lateral emitted it from.
               LEFT(CASE WHEN a.id <= nn.id THEN a.content ELSE nn.content END, 90)
                   AS preview_lo,
               LEFT(CASE WHEN a.id <= nn.id THEN nn.content ELSE a.content END, 90)
                   AS preview_hi
        FROM scoped a
        CROSS JOIN LATERAL (
            SELECT b.id, b.content, b.embedding
            FROM archival_memory b
            WHERE LOWER(b.project) = LOWER($1)
              AND b.superseded_by IS NULL
              AND b.embedding IS NOT NULL
              AND b.embedding_model = $5
              AND b.id <> a.id
            ORDER BY a.embedding <=> b.embedding
            LIMIT $4
        ) nn
        WHERE (1 - (a.embedding <=> nn.embedding)) >= $2
    ),
    canonical AS (
        SELECT DISTINCT ON (lo, hi)
               lo, hi, similarity, preview_lo, preview_hi
        FROM pairs
        ORDER BY lo, hi, similarity DESC
    )
    SELECT lo::text AS id_a,
           hi::text AS id_b,
           similarity,
           preview_lo AS preview_a,
           preview_hi AS preview_b
    FROM canonical
    ORDER BY similarity DESC
    LIMIT $3
"""

# Coverage for the merge scan: which embedding space was scanned and how many
# embedded rows were left out (i.e. live in a different embedding_model). Lets the
# report disclose a partial scan after a re-embed instead of silently under-reporting.
_MERGE_COVERAGE_SQL = """
    WITH embedded AS (
        SELECT embedding_model
        FROM archival_memory
        WHERE LOWER(project) = LOWER($1)
          AND superseded_by IS NULL
          AND embedding IS NOT NULL
    ),
    ranked AS (
        SELECT embedding_model, COUNT(*) AS n
        FROM embedded
        GROUP BY embedding_model
        ORDER BY n DESC, embedding_model ASC
    )
    SELECT
        (SELECT embedding_model FROM ranked LIMIT 1) AS scanned_model,
        COALESCE((SELECT n FROM ranked LIMIT 1), 0) AS scanned_rows,
        (SELECT COUNT(*) FROM embedded) AS total_embedded
"""

_STALE_SQL = """
    WITH scoped AS (
        SELECT recall_count, created_at
        FROM archival_memory
        WHERE LOWER(project) = LOWER($1) AND superseded_by IS NULL
    )
    SELECT bucket AS staleness_bucket, COUNT(*) AS learnings
    FROM (
        SELECT CASE
            WHEN recall_count = 0 AND created_at < NOW() - INTERVAL '60 days'
                THEN 'never recalled, >60d old'
            WHEN recall_count = 0 AND created_at < NOW() - INTERVAL '30 days'
                THEN 'never recalled, 30-60d old'
            WHEN recall_count = 0
                THEN 'never recalled, <30d old'
            ELSE 'recalled at least once'
        END AS bucket
        FROM scoped
    ) t
    GROUP BY bucket
    ORDER BY learnings DESC
"""

_STALE_OPEN_THREAD_SQL = """
    SELECT COUNT(*)
    FROM archival_memory
    WHERE LOWER(project) = LOWER($1)
      AND superseded_by IS NULL
      AND metadata->>'learning_type' = 'OPEN_THREAD'
      AND recall_count = 0
      AND created_at < NOW() - make_interval(days => $2)
"""

_ACTIVE_TOTAL_SQL = """
    SELECT COUNT(*)
    FROM archival_memory
    WHERE LOWER(project) = LOWER($1) AND superseded_by IS NULL
"""

# Resolve both ids of a merge pair in ONE round-trip (plan note N-1: never per-id fetches).
# Carries the keeper-selection signals only (recall_count, created_at) plus superseded_by so
# the apply path can refuse a pair whose side is already superseded. Mirrors
# _CANDIDATES_BY_IDS_SQL (id = ANY, project-scoped) but does NOT filter superseded_by here —
# the apply path needs to SEE an already-superseded side to refuse/skip it, not have it hidden.
_MERGE_PAIR_DETAILS_SQL = """
    SELECT id::text AS id,
           recall_count,
           created_at,
           superseded_by::text AS superseded_by
    FROM archival_memory
    WHERE LOWER(project) = LOWER($1)
      AND id::text = ANY($2::text[])
"""


async def fetch_active_total(pool, project: str) -> int:
    async with pool.acquire() as conn:
        return int(await conn.fetchval(_ACTIVE_TOTAL_SQL, project) or 0)


async def fetch_merge_pair_details(pool, project: str, id_a: str, id_b: str) -> dict[str, MergeRow]:
    """Resolve a merge pair's two ids to ``MergeRow``s in one batched, project-scoped query.

    Returns a dict keyed by id::text so a caller can index either side; an id that does not
    resolve (hard-deleted / wrong project) is simply absent from the dict. Read-only — this
    module never writes. The apply path consumes these rows to pick a keeper and to refuse a
    pair whose side is already superseded.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(_MERGE_PAIR_DETAILS_SQL, project, [id_a, id_b])
    return {
        r["id"]: MergeRow(
            id=r["id"],
            recall_count=int(r["recall_count"]),
            created_at=r["created_at"],
            superseded_by=r["superseded_by"],
        )
        for r in rows
    }


async def fetch_promotion_candidates(
    pool, project: str, min_recall: int = DEFAULT_MIN_RECALL
) -> list[PromotionCandidate]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(_PROMOTION_SQL, project, int(min_recall), list(PROMOTABLE_TYPES))
    out: list[PromotionCandidate] = []
    for r in rows:
        lt = r["learning_type"]
        dest = route_destination(lt)
        if dest is None:  # defensive: query already filters, but never trust the row
            continue
        out.append(
            PromotionCandidate(
                id=r["id"],
                content=r["content"],
                recall_count=int(r["recall_count"]),
                learning_type=lt,
                destination=dest,
            )
        )
    return out


async def fetch_merge_candidates(
    pool,
    project: str,
    model: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ef_search: int = DEFAULT_EF_SEARCH,
    limit: int = DEFAULT_MERGE_LIMIT,
    neighbors: int = DEFAULT_MERGE_NEIGHBORS,
    timeout: float | None = None,
) -> list[MergeCandidate]:
    # ef_search/statement_timeout are session GUCs that cannot be parameterized; coerce
    # to int so a caller value can never carry SQL. threshold/limit/neighbors/model go as
    # bind params. SET LOCAL must run inside a transaction or it is a no-op; the
    # transaction also scopes both GUCs to this query so they never leak to the pool.
    # statement_timeout makes Postgres itself cancel the scan (raising QueryCanceledError)
    # so the backend stops and the connection is freed — the asyncpg client `timeout=` is
    # only a secondary guard that does not stop server-side work on its own.
    # Defense-in-depth clamps (the CLI also validates, but this is also called
    # programmatically): ef_search must be >= 1 or Postgres rejects the GUC; a
    # non-positive timeout must NOT reach the backend — statement_timeout = 0 means
    # "no limit" in Postgres, which would silently disable the degradation guard on a
    # large O(n^2) scan, and a negative asyncpg timeout raises ValueError. Treat any
    # non-positive timeout as "no explicit budget".
    safe_ef = max(1, int(ef_search))
    safe_timeout = timeout if (timeout is not None and timeout > 0) else None
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(f"SET LOCAL hnsw.ef_search = {safe_ef}")
        if safe_timeout is not None:
            await conn.execute(f"SET LOCAL statement_timeout = {int(safe_timeout * 1000)}")
        rows = await conn.fetch(
            _MERGE_SQL,
            project,
            float(threshold),
            int(limit),
            int(neighbors),
            model,
            timeout=safe_timeout,
        )
    return [
        MergeCandidate(
            id_a=r["id_a"],
            id_b=r["id_b"],
            similarity=float(r["similarity"]),
            preview_a=r["preview_a"],
            preview_b=r["preview_b"],
        )
        for r in rows
    ]


async def fetch_merge_coverage(pool, project: str) -> tuple[str | None, int]:
    """Return (scanned_model, skipped_embedded_rows) for the merge scan.

    skipped = embedded rows that live in an embedding_model OTHER than the one the
    merge scan covers, so the report can disclose a partial scan after a re-embed.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(_MERGE_COVERAGE_SQL, project)
    if row is None:
        return None, 0
    scanned_model = row["scanned_model"]
    skipped = int(row["total_embedded"]) - int(row["scanned_rows"])
    return scanned_model, max(0, skipped)


async def fetch_stale_summary(pool, project: str) -> tuple[list[StaleBucket], int]:
    async with pool.acquire() as conn:
        bucket_rows = await conn.fetch(_STALE_SQL, project)
        open_threads = int(
            await conn.fetchval(_STALE_OPEN_THREAD_SQL, project, STALE_OPEN_THREAD_DAYS) or 0
        )
    buckets = [
        StaleBucket(label=r["staleness_bucket"], count=int(r["learnings"])) for r in bucket_rows
    ]
    return buckets, open_threads


# --- Orchestrator ----------------------------------------------------------


async def build_review(
    pool,
    project: str,
    *,
    min_recall: int = DEFAULT_MIN_RECALL,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ef_search: int = DEFAULT_EF_SEARCH,
    merge_timeout: float = DEFAULT_MERGE_TIMEOUT_S,
    promote: bool = True,
    cleanup: bool = True,
) -> ReviewReport:
    total = await fetch_active_total(pool, project)
    promotions: list[PromotionCandidate] = []
    merges: list[MergeCandidate] = []
    stale_buckets: list[StaleBucket] = []
    stale_open_threads = 0
    merges_timed_out = False
    merge_scanned_model: str | None = None
    merge_skipped_rows = 0

    if promote:
        promotions = await fetch_promotion_candidates(pool, project, min_recall)
    if cleanup:
        # Resolve the embedding space ONCE here; the same model string drives both the
        # coverage disclosure and the merge scan, so they can never describe different
        # spaces under a concurrent re-embed.
        merge_scanned_model, merge_skipped_rows = await fetch_merge_coverage(pool, project)
        if merge_scanned_model is not None:
            try:
                merges = await fetch_merge_candidates(
                    pool,
                    project,
                    merge_scanned_model,
                    threshold=threshold,
                    ef_search=ef_search,
                    timeout=merge_timeout,
                )
            except (TimeoutError, QueryCanceledError):
                # The merge scan cost scales with project size; on large corpora it can
                # exceed the timeout (client-side TimeoutError) or be cancelled by the
                # server's statement_timeout (QueryCanceledError). Either way, degrade
                # gracefully — the rest of the review still ships, and the report tells
                # the user how to narrow the scan.
                merges_timed_out = True
        stale_buckets, stale_open_threads = await fetch_stale_summary(pool, project)

    return ReviewReport(
        project=project,
        total_active=total,
        promotions=promotions,
        merges=merges,
        stale_buckets=stale_buckets,
        stale_open_threads=stale_open_threads,
        merges_timed_out=merges_timed_out,
        merge_scanned_model=merge_scanned_model,
        merge_skipped_rows=merge_skipped_rows,
    )


def _default_project() -> str | None:
    """Worktree-aware default project: resolves .claude/worktrees/<branch> to the
    real repo (issue #130), so a worktree session reviews the right corpus rather
    than the branch name. Honors CLAUDE_PROJECT_DIR when set."""
    try:
        cwd = os.getcwd()
    except OSError:
        # cwd can be gone (deleted out from under us) — fall back to env/None rather
        # than crashing default resolution.
        cwd = None
    return project_from_path(os.environ.get("CLAUDE_PROJECT_DIR") or cwd)


def _positive_int(raw: str) -> int:
    """argparse type: an int >= 1 (rejects 0/negative before they reach the GUC)."""
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _positive_float(raw: str) -> float:
    """argparse type: a float > 0 (a non-positive timeout disables the scan guard)."""
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def _non_negative_int(raw: str) -> int:
    """argparse type: an int >= 0 (min-recall of 0 selects everything; negative is invalid)."""
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def _unit_float(raw: str) -> float:
    """argparse type: a cosine threshold in [0, 1]."""
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {value}")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect memory promotion / merge / stale candidates (issue #63). Read-only."
    )
    p.add_argument("project", nargs="?", default=None, help="Project to review (default: cwd name)")
    p.add_argument("--min-recall", type=_non_negative_int, default=DEFAULT_MIN_RECALL)
    p.add_argument("--threshold", type=_unit_float, default=DEFAULT_SIMILARITY_THRESHOLD)
    p.add_argument(
        "--ef-search",
        type=_positive_int,
        default=DEFAULT_EF_SEARCH,
        help="HNSW ef_search for the merge scan (>= 1); lower = faster, less complete",
    )
    p.add_argument(
        "--merge-timeout",
        type=_positive_float,
        default=DEFAULT_MERGE_TIMEOUT_S,
        help="Seconds before the merge scan degrades gracefully (> 0; pool cap is 60s)",
    )
    # Mutually exclusive: passing both would disable promote AND cleanup → empty report.
    only = p.add_mutually_exclusive_group()
    only.add_argument("--promote-only", action="store_true", help="Skip cleanup detectors")
    only.add_argument("--cleanup-only", action="store_true", help="Skip promotion detector")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Explicit arg wins (canonicalized to match stored project values); otherwise
    # derive worktree-aware from the cwd so we never review the branch name.
    project = canonicalize_project(args.project) if args.project else _default_project()
    if not project:
        print(
            "memory-review: could not resolve a project. Pass one explicitly, "
            "e.g. `memory-review opc`.",
            file=sys.stderr,
        )
        return 2
    promote = not args.cleanup_only
    cleanup = not args.promote_only

    try:
        pool = await get_pool()
        report = await build_review(
            pool,
            project,
            min_recall=args.min_recall,
            threshold=args.threshold,
            ef_search=args.ef_search,
            merge_timeout=args.merge_timeout,
            promote=promote,
            cleanup=cleanup,
        )
    except ValueError as exc:
        # Missing/invalid DB config (get_pool raises ValueError when no DATABASE_URL).
        # The message is our own config text, not a DSN, so it is safe to surface.
        print(f"memory-review: configuration error: {exc}", file=sys.stderr)
        return 1
    except (OSError, PostgresError) as exc:
        # Connection/DB failures (DB down, bad DSN, refused) should fail cleanly with
        # a concise message — never dump a traceback that could echo the DSN/host.
        print(f"memory-review: database error ({type(exc).__name__}).", file=sys.stderr)
        return 1
    print(format_report(report))
    if report.total_active == 0:
        # A zero-active corpus usually means the project name is wrong (typo or an
        # unresolved worktree path), not that the project is genuinely empty. Make
        # that visible rather than letting an empty report read as "all clean".
        print(
            f"\nmemory-review: project '{project}' has 0 active learnings — "
            "verify the project name is correct.",
            file=sys.stderr,
        )
    return 0


async def _cli_main(argv: list[str] | None = None) -> int:
    try:
        return await main(argv)
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(_cli_main()))
