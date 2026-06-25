"""Mechanical keep/discard tuning loop around the reranker weight sweep.

This is the autoresearch loop applied to OPC recall: take the frozen harness
(golden set + metrics + a pinned candidate pool), perturb ONE surface (the
reranker weights), and for each perturbation apply a mechanical rule —

    holdout NDCG@5 improves AND p95 latency stays within budget  -> keep
    otherwise                                                     -> discard
    tiebreak among keeps: highest holdout NDCG@5, then lower p95 latency, then
    fewer active signals (simpler)

Every run is appended to the journal (journal.py) so dead ends are never
re-tried, and the keep/discard outcome is recorded back into OPC's own memory
(store_learning_v2) as WORKING_SOLUTION / FAILED_APPROACH, making the tuning
history itself recallable.

Decisions are evaluated on the HELD-OUT split only (report §6.1) so the loop
cannot tune to noise on the queries it optimized against. The winning config is
printed as a diff; it is written to opc.toml ONLY with --apply (no silent edits).

Usage:
    uv run python scripts/benchmarks/tune_loop.py
    uv run python scripts/benchmarks/tune_loop.py --split holdout
    uv run python scripts/benchmarks/tune_loop.py --latency-budget-ms 5
    uv run python scripts/benchmarks/tune_loop.py --apply        # write winner
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.benchmarks import journal
from scripts.benchmarks.run_rerank_benchmark import (
    WEIGHT_SWEEPS,
    build_reranker_config,
    evaluate_config,
    fetch_full_results,
    filter_by_split,
)
from scripts.core.config import get_config
from scripts.core.config.models import RerankerConfig

DEFAULT_QUERIES = Path("scripts/benchmarks/rerank_queries.json")
OPC_TOML = Path("opc.toml")
NDCG_EPS = 1e-9  # strict improvement; equal NDCG is not a "win" vs baseline

# Reranker weight keys the sweep perturbs (the only surface --apply may touch).
WEIGHT_KEYS = [
    "project_weight",
    "recency_weight",
    "confidence_weight",
    "recall_weight",
    "type_affinity_weight",
    "tag_overlap_weight",
    "pattern_weight",
]


def active_signal_count(config) -> int:
    """Number of non-zero perturbed weights — the 'simplicity' tiebreak metric."""
    return sum(1 for k in WEIGHT_KEYS if getattr(config, k) > 0)


def decide(candidate: dict, baseline: dict, latency_budget_ms: float) -> str:
    """keep iff holdout NDCG@5 strictly improves AND p95 within budget."""
    improved = candidate["ndcg_at_k"] > baseline["ndcg_at_k"] + NDCG_EPS
    within_budget = candidate["p95_latency_ms"] <= latency_budget_ms
    return "keep" if (improved and within_budget) else "discard"


def latency_budget(baseline_p95: float, override: float | None) -> float:
    """Co-metric ceiling: explicit override, else 25% over baseline (min +1ms)."""
    if override is not None:
        return override
    return max(baseline_p95 * 1.25, baseline_p95 + 1.0)


def apply_weights_to_toml(
    path: Path, weights: dict[str, float], base: RerankerConfig
) -> list[str]:
    """Rewrite the perturbed weight keys inside [reranker]. Returns changed keys.

    Surgical: only lines matching ``<weight_key> = ...`` within the [reranker]
    section are touched; everything else (comments, other sections, other keys)
    is preserved byte-for-byte, and each line keeps its original newline (so a
    CRLF file stays CRLF rather than gaining mixed endings).

    Before writing, the resulting config (the seven new weights layered onto
    ``base`` — the live, file-resolved reranker config) is constructed via
    ``build_reranker_config``, so RerankerConfig.__post_init__ rejects any combo
    that would push ``total_signal_weight`` past 1.0. That guarantees --apply can
    never emit an opc.toml that fails to load (e.g. an operator-raised kg_weight
    plus high swept weights).
    """
    # Validate the would-be config first; raises ValueError on an over-budget mix.
    build_reranker_config({"name": "apply", **weights}, base)

    # newline="" disables universal-newline translation on both read and write,
    # so each line retains its own terminator (\n or \r\n) and a CRLF file is not
    # silently rewritten with mixed/normalized endings.
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = f.readlines()
    in_reranker = False
    changed: list[str] = []
    for i, line in enumerate(lines):
        section = re.match(r"\s*\[([^\]]+)\]", line)
        if section:
            in_reranker = section.group(1) == "reranker"
            continue
        if not in_reranker:
            continue
        m = re.match(r"(\s*)([A-Za-z0-9_]+)(\s*=\s*).*", line)
        if m and m.group(2) in weights:
            key = m.group(2)
            # Preserve this line's original terminator (\r\n / \n / none at EOF).
            ending = (
                "\r\n" if line.endswith("\r\n")
                else "\n" if line.endswith("\n")
                else ""
            )
            newline = f"{m.group(1)}{key}{m.group(3)}{weights[key]}{ending}"
            if newline != line:
                lines[i] = newline
                changed.append(key)
    if changed:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
    return changed


async def record_outcome(winner: dict | None, baseline: dict, n_queries: int) -> None:
    """Best-effort: journal the loop outcome into OPC memory as a learning."""
    try:
        from scripts.core.store_learning import store_learning_v2

        session = f"recall-tune-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        if winner:
            ltype = "WORKING_SOLUTION"
            content = (
                f"Recall reranker tuning: config '{winner['name']}' "
                f"({winner['config_hash']}) improved holdout NDCG@5 from "
                f"{baseline['ndcg_at_k']:.4f} to {winner['ndcg_at_k']:.4f} at "
                f"p95 {winner['p95_latency_ms']:.3f}ms over {n_queries} held-out "
                f"queries. Weights: {json.dumps(winner['weights'])}."
            )
        else:
            ltype = "FAILED_APPROACH"
            content = (
                f"Recall reranker tuning: no swept config beat baseline holdout "
                f"NDCG@5 {baseline['ndcg_at_k']:.4f} over {n_queries} held-out "
                f"queries. The current weights stand; re-running the same sweep is "
                f"a dead end until the golden set or candidate pool changes."
            )
        await store_learning_v2(
            session_id=session,
            content=content,
            learning_type=ltype,
            context="recall reranker weight tuning loop",
            tags=["reranker", "tuning"],
            confidence="medium",
            project="opc",
        )
        print(f"  Recorded {ltype} learning to OPC memory.")
    except Exception as e:  # noqa: BLE001 — telemetry must never break the loop
        print(f"  (could not record learning: {e})", file=sys.stderr)


async def run(args: argparse.Namespace) -> int:
    query_data = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = filter_by_split(query_data["queries"], args.split)
    if not queries:
        print(f"No queries in split={args.split}; nothing to tune.", file=sys.stderr)
        return 1
    print(f"Tuning on {len(queries)} queries (split={args.split}).")

    # Frozen candidate pool: fetch once, rerun every config in-process.
    cache = await fetch_full_results(queries)

    baseline_config = get_config().reranker
    baseline = evaluate_config(cache, queries, baseline_config)
    budget = latency_budget(baseline["p95_latency_ms"], args.latency_budget_ms)
    ts = datetime.now(UTC).isoformat()
    journal.append_entry(
        timestamp=ts,
        cfg_hash=journal.config_hash(baseline_config),
        ndcg=baseline["ndcg_at_k"], p_at_k=baseline["precision_at_k"],
        mrr=baseline["mrr"], p95_latency_ms=baseline["p95_latency_ms"],
        status="keep",  # baseline is the incumbent
        description=f"baseline (live opc.toml) split={args.split}",
        path=args.journal,
    )
    print(
        f"Baseline: NDCG@5={baseline['ndcg_at_k']:.4f} "
        f"p95={baseline['p95_latency_ms']:.3f}ms  (latency budget {budget:.3f}ms)"
    )

    evaluated: list[dict] = []
    for wc in WEIGHT_SWEEPS:
        # Perturb from the live incumbent: non-swept fields (kg_weight, recency
        # half-life, RRF params) inherit baseline_config rather than resetting to
        # dataclass defaults, so a kept winner reproduces after --apply.
        config = build_reranker_config(wc, baseline_config)
        m = evaluate_config(cache, queries, config)
        status = decide(m, baseline, budget)
        cand = {
            "name": wc["name"],
            "config": config,
            "config_hash": journal.config_hash(config),
            "weights": {k: v for k, v in wc.items() if k != "name"},
            "status": status,
            **m,
        }
        evaluated.append(cand)
        journal.append_entry(
            timestamp=ts,
            cfg_hash=cand["config_hash"],
            ndcg=m["ndcg_at_k"], p_at_k=m["precision_at_k"],
            mrr=m["mrr"], p95_latency_ms=m["p95_latency_ms"],
            status=status,
            description=f"{wc['name']} split={args.split} vs baseline "
                        f"{baseline['ndcg_at_k']:.4f}",
            path=args.journal,
        )
        flag = "KEEP " if status == "keep" else "drop "
        print(
            f"  {flag} {wc['name']:16s} NDCG@5={m['ndcg_at_k']:.4f} "
            f"p95={m['p95_latency_ms']:.3f}ms"
        )

    keeps = [c for c in evaluated if c["status"] == "keep"]
    # Tiebreak among keeps: highest NDCG, then lower latency, then fewer signals.
    winner = None
    if keeps:
        winner = max(
            keeps,
            key=lambda c: (
                c["ndcg_at_k"],
                -c["p95_latency_ms"],
                -active_signal_count(c["config"]),
            ),
        )

    print()
    if winner:
        print(
            f"WINNER: {winner['name']} ({winner['config_hash']}) "
            f"NDCG@5 {baseline['ndcg_at_k']:.4f} -> {winner['ndcg_at_k']:.4f}"
        )
        print("Proposed [reranker] weight changes:")
        for k in WEIGHT_KEYS:
            base_v = getattr(baseline_config, k)
            new_v = winner["weights"].get(k, base_v)
            if base_v != new_v:
                print(f"  {k}: {base_v} -> {new_v}")
        if args.apply:
            changed = apply_weights_to_toml(
                OPC_TOML, winner["weights"], baseline_config
            )
            print(f"  Applied to {OPC_TOML}: {', '.join(changed) or '(no lines changed)'}")
        else:
            print("  (re-run with --apply to write these to opc.toml)")
    else:
        print("No swept config beat the baseline on the holdout split. Baseline stands.")

    await record_outcome(winner, baseline, len(queries))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mechanical keep/discard tuning loop over the reranker sweep"
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument(
        "--split", choices=["train", "holdout", "all"], default="holdout",
        help="Split to decide keep/discard on (default: holdout).",
    )
    parser.add_argument(
        "--latency-budget-ms", type=float, default=None,
        help="p95 in-process rerank latency ceiling (default: 25%% over baseline).",
    )
    parser.add_argument("--journal", type=Path, default=journal.DEFAULT_JOURNAL)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the winning weights into opc.toml [reranker] (default: off).",
    )
    return parser.parse_args()


def main() -> None:
    sys.exit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
