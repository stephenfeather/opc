"""A/B benchmark for --no-rerank flag.

Runs benchmark queries with and without reranking, measures recall quality,
and produces a comparison report.

Usage:
    uv run python scripts/benchmarks/run_rerank_benchmark.py
    uv run python scripts/benchmarks/run_rerank_benchmark.py --queries custom.json
    uv run python scripts/benchmarks/run_rerank_benchmark.py --output results.json
    uv run python scripts/benchmarks/run_rerank_benchmark.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------

@dataclass
class QueryResult:
    """Results from a single query run."""

    query_id: str
    query: str
    mode: str  # "reranked" or "raw"
    result_ids: list[str]
    result_scores: list[float]
    result_contents: list[str]
    rerank_details: list[dict] | None = None
    elapsed_ms: float = 0.0


@dataclass
class ComparisonMetrics:
    """Metrics comparing reranked vs raw for one query."""

    query_id: str
    precision_at_k_reranked: float
    precision_at_k_raw: float
    ndcg_at_k_reranked: float
    ndcg_at_k_raw: float
    mrr_reranked: float
    mrr_raw: float
    mean_rank_displacement: float
    max_rank_displacement: int
    reranked_score_range: list[float]
    raw_score_range: list[float]
    promoted_ids: list[str]
    demoted_ids: list[str]
    dominant_signal: str
    reranked_elapsed_ms: float
    raw_elapsed_ms: float


# -------------------------------------------------------------------
# Metric functions
# -------------------------------------------------------------------

def is_relevant(
    result_id: str,
    content: str,
    golden_ids: list[str],
    golden_keywords: list[str],
) -> bool:
    """Check if a result is relevant based on golden IDs or keywords."""
    if golden_ids and result_id in golden_ids:
        return True
    if golden_keywords:
        content_lower = content.lower()
        return any(kw.lower() in content_lower for kw in golden_keywords)
    return False


def compute_precision_at_k(
    result_ids: list[str],
    result_contents: list[str],
    golden_ids: list[str],
    golden_keywords: list[str],
) -> float:
    """Fraction of top-k results that match the golden set."""
    if not result_ids:
        return 0.0
    relevant = sum(
        1
        for rid, content in zip(result_ids, result_contents)
        if is_relevant(rid, content, golden_ids, golden_keywords)
    )
    return relevant / len(result_ids)


def compute_ndcg(
    result_ids: list[str],
    result_contents: list[str],
    golden_ids: list[str],
    golden_keywords: list[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not result_ids:
        return 0.0

    # Compute DCG
    dcg = 0.0
    for i, (rid, content) in enumerate(
        zip(result_ids[:k], result_contents[:k])
    ):
        rel = 1.0 if is_relevant(
            rid, content, golden_ids, golden_keywords
        ) else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Compute ideal DCG (all relevant items at top)
    num_relevant = sum(
        1
        for rid, content in zip(result_ids, result_contents)
        if is_relevant(rid, content, golden_ids, golden_keywords)
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(num_relevant, k)))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_mrr(
    result_ids: list[str],
    result_contents: list[str],
    golden_ids: list[str],
    golden_keywords: list[str],
) -> float:
    """Mean Reciprocal Rank of first relevant result."""
    for i, (rid, content) in enumerate(
        zip(result_ids, result_contents)
    ):
        if is_relevant(rid, content, golden_ids, golden_keywords):
            return 1.0 / (i + 1)
    return 0.0


def compute_rank_displacement(
    reranked_ids: list[str],
    raw_ids: list[str],
) -> tuple[float, int]:
    """Average and max position change between reranked and raw."""
    if not reranked_ids or not raw_ids:
        return 0.0, 0

    raw_positions = {rid: i for i, rid in enumerate(raw_ids)}
    displacements = []
    for i, rid in enumerate(reranked_ids):
        if rid in raw_positions:
            displacements.append(abs(i - raw_positions[rid]))

    if not displacements:
        return 0.0, 0
    return (
        sum(displacements) / len(displacements),
        max(displacements),
    )


def identify_promoted_demoted(
    reranked_ids: list[str],
    raw_ids: list[str],
    threshold: int = 2,
) -> tuple[list[str], list[str]]:
    """Find IDs that moved up or down by >= threshold positions."""
    raw_positions = {rid: i for i, rid in enumerate(raw_ids)}
    promoted = []
    demoted = []
    for i, rid in enumerate(reranked_ids):
        if rid in raw_positions:
            delta = raw_positions[rid] - i  # positive = promoted
            if delta >= threshold:
                promoted.append(rid)
            elif delta <= -threshold:
                demoted.append(rid)
    return promoted, demoted


def identify_dominant_signal(
    rerank_details: list[dict] | None,
) -> str:
    """Which signal had highest variance across results."""
    if not rerank_details:
        return "none"

    signals = [
        "project_match", "recency", "confidence",
        "recall", "type_match", "tag_overlap", "pattern",
    ]
    max_variance = -1.0
    dominant = "none"
    for signal in signals:
        values = [
            d.get(signal, 0.0)
            for d in rerank_details
            if isinstance(d, dict)
        ]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        if variance > max_variance:
            max_variance = variance
            dominant = signal
    return dominant


def score_range(scores: list[float]) -> list[float]:
    """Return [min, max] of scores."""
    if not scores:
        return [0.0, 0.0]
    return [min(scores), max(scores)]


# -------------------------------------------------------------------
# Query execution
# -------------------------------------------------------------------

SEMAPHORE = asyncio.Semaphore(4)  # limit concurrent DB connections


async def run_query(
    query_id: str,
    query: str,
    k: int,
    rerank: bool,
    project: str | None = None,
    tags: list[str] | None = None,
) -> QueryResult:
    """Run a single recall query via subprocess."""
    cmd = [
        sys.executable,
        "scripts/core/recall_learnings.py",
        "--query", query,
        "--k", str(k),
        "--json",
    ]
    if not rerank:
        cmd.append("--no-rerank")
    if project:
        cmd.extend(["--project", project])
    if tags:
        cmd.extend(["--tags", *tags])

    async with SEMAPHORE:
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        elapsed_ms = (time.monotonic() - start) * 1000

    if proc.returncode != 0:
        err = stderr.decode().strip()
        print(
            f"  WARNING: query {query_id} "
            f"({'reranked' if rerank else 'raw'}) failed: {err}",
            file=sys.stderr,
        )
        return QueryResult(
            query_id=query_id,
            query=query,
            mode="reranked" if rerank else "raw",
            result_ids=[],
            result_scores=[],
            result_contents=[],
            elapsed_ms=elapsed_ms,
        )

    data = json.loads(stdout)
    results = data.get("results", [])
    details = [r.get("rerank_details") for r in results if "rerank_details" in r]

    return QueryResult(
        query_id=query_id,
        query=query,
        mode="reranked" if rerank else "raw",
        result_ids=[r.get("id", "") for r in results],
        result_scores=[r["score"] for r in results],
        result_contents=[r["content"] for r in results],
        rerank_details=details if details else None,
        elapsed_ms=elapsed_ms,
    )


async def run_benchmark(
    queries: list[dict],
) -> list[tuple[QueryResult, QueryResult, ComparisonMetrics]]:
    """Run all queries in both modes and compute metrics."""
    tasks = []
    for q in queries:
        qid = q["id"]
        query = q["query"]
        k = q.get("k", 5)
        project = q.get("project")
        tags = q.get("tags", [])

        tasks.append((
            qid,
            q,
            run_query(qid, query, k, rerank=True,
                      project=project, tags=tags or None),
            run_query(qid, query, k, rerank=False,
                      project=project, tags=tags or None),
        ))

    # Gather all results
    coros = []
    for qid, q, reranked_coro, raw_coro in tasks:
        coros.append(reranked_coro)
        coros.append(raw_coro)
    all_results = await asyncio.gather(*coros)

    # Pair up and compute metrics
    output = []
    for i in range(0, len(all_results), 2):
        reranked = all_results[i]
        raw = all_results[i + 1]
        q = queries[i // 2]
        golden_ids = q.get("golden_ids", [])
        golden_keywords = q.get("golden_keywords", [])
        k = q.get("k", 5)

        p_reranked = compute_precision_at_k(
            reranked.result_ids, reranked.result_contents,
            golden_ids, golden_keywords,
        )
        p_raw = compute_precision_at_k(
            raw.result_ids, raw.result_contents,
            golden_ids, golden_keywords,
        )
        ndcg_reranked = compute_ndcg(
            reranked.result_ids, reranked.result_contents,
            golden_ids, golden_keywords, k,
        )
        ndcg_raw = compute_ndcg(
            raw.result_ids, raw.result_contents,
            golden_ids, golden_keywords, k,
        )
        mrr_reranked = compute_mrr(
            reranked.result_ids, reranked.result_contents,
            golden_ids, golden_keywords,
        )
        mrr_raw = compute_mrr(
            raw.result_ids, raw.result_contents,
            golden_ids, golden_keywords,
        )
        mean_disp, max_disp = compute_rank_displacement(
            reranked.result_ids, raw.result_ids,
        )
        promoted, demoted = identify_promoted_demoted(
            reranked.result_ids, raw.result_ids,
        )
        dominant = identify_dominant_signal(reranked.rerank_details)

        metrics = ComparisonMetrics(
            query_id=reranked.query_id,
            precision_at_k_reranked=p_reranked,
            precision_at_k_raw=p_raw,
            ndcg_at_k_reranked=ndcg_reranked,
            ndcg_at_k_raw=ndcg_raw,
            mrr_reranked=mrr_reranked,
            mrr_raw=mrr_raw,
            mean_rank_displacement=mean_disp,
            max_rank_displacement=max_disp,
            reranked_score_range=score_range(reranked.result_scores),
            raw_score_range=score_range(raw.result_scores),
            promoted_ids=promoted,
            demoted_ids=demoted,
            dominant_signal=dominant,
            reranked_elapsed_ms=reranked.elapsed_ms,
            raw_elapsed_ms=raw.elapsed_ms,
        )
        output.append((reranked, raw, metrics))

    return output


# -------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------

def generate_report(
    results: list[tuple[QueryResult, QueryResult, ComparisonMetrics]],
    queries: list[dict],
) -> dict:
    """Generate JSON report from benchmark results."""
    per_query = []
    wins_rerank = 0
    wins_raw = 0
    ties = 0
    signal_counts: dict[str, int] = {}

    for reranked, raw, m in results:
        entry = {
            "query_id": m.query_id,
            "query": reranked.query,
            **asdict(m),
        }
        per_query.append(entry)

        # Determine winner by precision@k
        if m.precision_at_k_reranked > m.precision_at_k_raw:
            wins_rerank += 1
        elif m.precision_at_k_raw > m.precision_at_k_reranked:
            wins_raw += 1
        else:
            ties += 1

        signal_counts[m.dominant_signal] = (
            signal_counts.get(m.dominant_signal, 0) + 1
        )

    n = len(results)
    def avg(vals: list[float]) -> float:
        return sum(vals) / n if n else 0.0

    p_reranked = avg([m.precision_at_k_reranked for _, _, m in results])
    p_raw = avg([m.precision_at_k_raw for _, _, m in results])
    ndcg_reranked = avg([m.ndcg_at_k_reranked for _, _, m in results])
    ndcg_raw = avg([m.ndcg_at_k_raw for _, _, m in results])
    mrr_reranked = avg([m.mrr_reranked for _, _, m in results])
    mrr_raw = avg([m.mrr_raw for _, _, m in results])
    lat_reranked = avg([m.reranked_elapsed_ms for _, _, m in results])
    lat_raw = avg([m.raw_elapsed_ms for _, _, m in results])

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "num_queries": n,
            "default_k": 5,
        },
        "summary": {
            "precision_at_k": {
                "reranked": round(p_reranked, 4),
                "raw": round(p_raw, 4),
                "delta": round(p_reranked - p_raw, 4),
            },
            "ndcg_at_k": {
                "reranked": round(ndcg_reranked, 4),
                "raw": round(ndcg_raw, 4),
                "delta": round(ndcg_reranked - ndcg_raw, 4),
            },
            "mrr": {
                "reranked": round(mrr_reranked, 4),
                "raw": round(mrr_raw, 4),
                "delta": round(mrr_reranked - mrr_raw, 4),
            },
            "latency_ms": {
                "reranked_avg": round(lat_reranked, 1),
                "raw_avg": round(lat_raw, 1),
                "overhead_ms": round(lat_reranked - lat_raw, 1),
            },
            "rerank_wins": wins_rerank,
            "raw_wins": wins_raw,
            "ties": ties,
            "dominant_signals": dict(
                sorted(
                    signal_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        },
        "per_query": per_query,
    }
    return report


def print_summary(report: dict) -> None:
    """Print human-readable summary to stdout."""
    s = report["summary"]
    n = report["config"]["num_queries"]

    print()
    print("=== Rerank A/B Benchmark Results ===")
    print(f"Queries: {n} | k: {report['config']['default_k']}")
    print()
    print(f"{'':15s} {'Reranked':>10s} {'Raw':>10s} {'Delta':>12s}")
    print("-" * 50)

    for metric_name, key in [
        ("Precision@k", "precision_at_k"),
        ("NDCG@k", "ndcg_at_k"),
        ("MRR", "mrr"),
    ]:
        m = s[key]
        pct = (
            f"({m['delta'] / m['raw'] * 100:+.0f}%)"
            if m["raw"] > 0 else ""
        )
        print(
            f"{metric_name:15s} {m['reranked']:10.4f} "
            f"{m['raw']:10.4f} {m['delta']:+10.4f} {pct}"
        )

    print()
    total = s["rerank_wins"] + s["raw_wins"] + s["ties"]
    print(
        f"Rerank wins: {s['rerank_wins']}/{total} "
        f"({s['rerank_wins'] / total * 100:.0f}%)"
    )
    print(
        f"Raw wins:    {s['raw_wins']}/{total} "
        f"({s['raw_wins'] / total * 100:.0f}%)"
    )
    print(
        f"Ties:        {s['ties']}/{total} "
        f"({s['ties'] / total * 100:.0f}%)"
    )

    lat = s["latency_ms"]
    print(
        f"\nLatency: reranked {lat['reranked_avg']:.0f}ms avg "
        f"vs raw {lat['raw_avg']:.0f}ms avg "
        f"({lat['overhead_ms']:+.0f}ms overhead)"
    )

    print("\nDominant signals driving reordering:")
    for signal, count in s["dominant_signals"].items():
        print(f"  {signal:15s}: {count} queries ({count / n * 100:.0f}%)")

    # Show regressions
    regressions = [
        q for q in report["per_query"]
        if q["precision_at_k_reranked"] < q["precision_at_k_raw"]
    ]
    if regressions:
        print("\nRegressions (rerank hurt):")
        for q in regressions:
            print(
                f"  {q['query_id']} \"{q['query']}\"  "
                f"P@k: {q['precision_at_k_raw']:.2f} -> "
                f"{q['precision_at_k_reranked']:.2f}"
            )

    print()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B benchmark for reranker vs raw retrieval"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("scripts/benchmarks/rerank_queries.json"),
        help="Path to benchmark query JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for JSON report (default: auto-timestamped)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-query detail in stdout",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()

    # Load queries
    query_data = json.loads(args.queries.read_text())
    queries = query_data["queries"]
    print(f"Loaded {len(queries)} benchmark queries from {args.queries}")

    # Run benchmark
    print("Running benchmark (reranked + raw for each query)...")
    start = time.monotonic()
    results = await run_benchmark(queries)
    total_ms = (time.monotonic() - start) * 1000
    print(f"Completed in {total_ms:.0f}ms")

    # Generate report
    report = generate_report(results, queries)
    print_summary(report)

    if args.verbose:
        print("=== Per-Query Detail ===")
        for entry in report["per_query"]:
            print(f"\n{entry['query_id']}: \"{entry['query']}\"")
            print(
                f"  P@k: reranked={entry['precision_at_k_reranked']:.2f} "
                f"raw={entry['precision_at_k_raw']:.2f}"
            )
            print(
                f"  NDCG: reranked={entry['ndcg_at_k_reranked']:.2f} "
                f"raw={entry['ndcg_at_k_raw']:.2f}"
            )
            print(
                f"  MRR: reranked={entry['mrr_reranked']:.2f} "
                f"raw={entry['mrr_raw']:.2f}"
            )
            print(f"  Dominant signal: {entry['dominant_signal']}")
            if entry["promoted_ids"]:
                print(
                    f"  Promoted: {len(entry['promoted_ids'])} results"
                )
            if entry["demoted_ids"]:
                print(
                    f"  Demoted: {len(entry['demoted_ids'])} results"
                )

    # Save JSON report
    output_path = args.output
    if output_path is None:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        output_path = Path(
            f"scripts/benchmarks/results/benchmark-{ts}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {output_path}")

    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
