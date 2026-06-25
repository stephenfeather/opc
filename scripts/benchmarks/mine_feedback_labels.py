"""Mine golden-set label candidates from the memory-feedback stream.

The recall-tuning loop needs a growing, trustworthy golden set. Users already
emit relevance judgments via ``memory_feedback`` (helpful / not-helpful on a
recalled learning). This tool reconnects those judgments to the query that
surfaced them — using the recall_log query-linkage columns
(add_recall_log_query_link.sql) — and aggregates them into candidate golden
records keyed by content hash (instance-independent, matches the benchmark).

Output is written to ``golden_mined.json`` for HUMAN REVIEW. It is never merged
into the scored ``rerank_queries.json`` automatically — the harness must stay
trustworthy (report §6.5). A reviewer runs ``--merge-into rerank_queries.json``
once the candidates look right.

Requirements:
  * add_recall_log_query_link.sql applied (session_id/query_hash/query_text).
  * recall.log_query_text = true for some history (to get human-readable queries;
    hash-only candidates are still emitted but cannot be merged as queries).

Usage:
    uv run python scripts/benchmarks/mine_feedback_labels.py
    uv run python scripts/benchmarks/mine_feedback_labels.py --min-judgments 5
    uv run python scripts/benchmarks/mine_feedback_labels.py --window-hours 48
    uv run python scripts/benchmarks/mine_feedback_labels.py --merge-into \
        scripts/benchmarks/rerank_queries.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_MIN_JUDGMENTS = 3
DEFAULT_WINDOW_HOURS = 24
DEFAULT_OUTPUT = Path("scripts/benchmarks/golden_mined.json")

# Join feedback back to the query that surfaced the learning. A judgment is
# attributed to a recall only when it happened in the same session, within a
# bounded window AFTER the recall, on a learning that recall actually returned.
#
# DISTINCT ON (mf.id) collapses a feedback row to its single nearest preceding
# recall: a session can recall the same learning many times before feedback is
# given, and memory_feedback stores ONE judgment per row, so without this each
# judgment would fan out to N recall_log rows and be counted N times toward
# min_judgments (and could attach to an older, unrelated query that happened to
# surface the same learning). ORDER BY mf.id, rl.created_at DESC keeps the most
# recent recall at/under the feedback time.
JOIN_SQL = """
SELECT DISTINCT ON (mf.id)
       rl.query_hash       AS query_hash,
       rl.query_text       AS query_text,
       mf.helpful          AS helpful,
       am.content_hash     AS content_hash
FROM memory_feedback mf
JOIN recall_log rl
  ON rl.session_id = mf.session_id
 AND mf.learning_id = ANY(rl.recalled_ids)
 AND rl.created_at <= mf.created_at
 AND rl.created_at > mf.created_at - make_interval(hours => $1)
JOIN archival_memory am ON am.id = mf.learning_id
WHERE rl.session_id IS NOT NULL
  AND rl.query_hash IS NOT NULL
  AND am.content_hash IS NOT NULL
ORDER BY mf.id, rl.created_at DESC
"""


def aggregate_candidates(
    rows: list[dict],
    min_judgments: int = DEFAULT_MIN_JUDGMENTS,
) -> list[dict]:
    """Aggregate joined (query_hash, helpful, content_hash) rows into candidates.

    Pure — no DB. Groups by query_hash; within each query a content hash is a
    positive if helpful votes outnumber unhelpful, a hard negative if the
    reverse, and dropped on a tie (ambiguous signal). A candidate is emitted only
    once it has at least ``min_judgments`` total judgments, so a single stray
    click cannot mint a golden label (report §6.2).
    """
    # query_hash -> {"query_text": str|None, "votes": {content_hash: [pos, neg]}}
    by_query: dict[str, dict] = defaultdict(
        lambda: {"query_text": None, "votes": defaultdict(lambda: [0, 0])}
    )
    for r in rows:
        qh = r["query_hash"]
        bucket = by_query[qh]
        if bucket["query_text"] is None and r.get("query_text"):
            bucket["query_text"] = r["query_text"]
        votes = bucket["votes"][r["content_hash"]]
        if r["helpful"]:
            votes[0] += 1
        else:
            votes[1] += 1

    candidates: list[dict] = []
    for qh, bucket in by_query.items():
        votes = bucket["votes"]
        num_judgments = sum(pos + neg for pos, neg in votes.values())
        if num_judgments < min_judgments:
            continue
        positives = sorted(ch for ch, (p, n) in votes.items() if p > n)
        negatives = sorted(ch for ch, (p, n) in votes.items() if n > p)
        if not positives and not negatives:
            continue
        candidates.append(
            {
                "query_hash": qh,
                "query": bucket["query_text"],  # None if log_query_text was off
                "golden_hashes": positives,
                "golden_negatives": negatives,
                "num_judgments": num_judgments,
                "note": "review before merging into rerank_queries.json",
            }
        )
    # Most-evidenced candidates first.
    candidates.sort(key=lambda c: c["num_judgments"], reverse=True)
    return candidates


async def fetch_join_rows(window_hours: int) -> list[dict]:
    """Run the feedback↔recall join against Postgres. Returns plain dicts."""
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(JOIN_SQL, window_hours)
    return [dict(r) for r in rows]


def merge_into_queries(candidates: list[dict], queries_path: Path) -> int:
    """Append review-approved candidates as new benchmark queries. Returns count.

    Only candidates that carry a human-readable ``query`` (i.e. log_query_text
    was on) can become benchmark queries; hash-only candidates are skipped with a
    warning. New queries default to the ``train`` split so they never silently
    enter the holdout decision set.
    """
    query_data = json.loads(queries_path.read_text(encoding="utf-8"))
    existing_hashes = {
        q.get("query_hash") for q in query_data["queries"] if q.get("query_hash")
    }
    added = 0
    for c in candidates:
        if not c.get("query"):
            print(
                f"  skip {c['query_hash'][:8]}: hash-only (no query text)",
                file=sys.stderr,
            )
            continue
        if c["query_hash"] in existing_hashes:
            continue
        query_data["queries"].append(
            {
                "id": f"mined-{c['query_hash'][:8]}",
                "query": c["query"],
                "query_hash": c["query_hash"],
                "k": 5,
                "category": "mined_feedback",
                "split": "train",
                "golden_hashes": c["golden_hashes"],
                "golden_negatives": c["golden_negatives"],
            }
        )
        added += 1
    if added:
        queries_path.write_text(json.dumps(query_data, indent=2) + "\n", encoding="utf-8")
    return added


async def run(args: argparse.Namespace) -> int:
    rows = await fetch_join_rows(args.window_hours)
    candidates = aggregate_candidates(rows, min_judgments=args.min_judgments)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "min_judgments": args.min_judgments,
        "window_hours": args.window_hours,
        "joined_rows": len(rows),
        "candidates": candidates,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Mined {len(candidates)} candidate(s) from {len(rows)} judgment(s) "
        f"-> {args.output}"
    )
    hash_only = sum(1 for c in candidates if not c.get("query"))
    if hash_only:
        print(
            f"  {hash_only} candidate(s) are hash-only (recall.log_query_text was "
            f"off) and cannot be merged as queries.",
            file=sys.stderr,
        )

    if args.merge_into:
        added = merge_into_queries(candidates, args.merge_into)
        print(f"Merged {added} reviewed candidate(s) into {args.merge_into}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine golden-set label candidates from memory feedback"
    )
    parser.add_argument(
        "--min-judgments", type=int, default=DEFAULT_MIN_JUDGMENTS,
        help=f"Min judgments before a candidate counts (default: {DEFAULT_MIN_JUDGMENTS})",
    )
    parser.add_argument(
        "--window-hours", type=int, default=DEFAULT_WINDOW_HOURS,
        help=(
            "Max hours between a recall and a feedback event for them to be "
            f"linked (default: {DEFAULT_WINDOW_HOURS})"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--merge-into", type=Path, default=None,
        help=(
            "After mining, append review-approved candidates (those with query "
            "text) into this queries file as new train-split benchmark queries."
        ),
    )
    return parser.parse_args()


def main() -> None:
    sys.exit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
