#!/usr/bin/env python3
"""Eval harness for learning classification accuracy.

Loads a golden dataset of manually labeled learnings, runs classify_learning()
on each, and reports accuracy metrics including per-class precision/recall
and a confusion matrix.

Usage:
    uv run python tests/eval/learning_classification_eval.py
    uv run python tests/eval/learning_classification_eval.py --output results.json
"""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load BRAINTRUST_API_KEY from .env files
# Precedence: project_root > ~/opc > ~/.claude (first found wins)
# override=True on first hit to replace any stale shell env vars
from dotenv import load_dotenv  # noqa: E402

_env_loaded = False
for env_dir in [project_root, Path.home() / "opc", Path.home() / ".claude"]:
    env_file = env_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=not _env_loaded)
        _env_loaded = True

from scripts.braintrust_analyze import classify_learning  # noqa: E402


GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"

LEARNING_TYPES = [
    "ARCHITECTURAL_DECISION",
    "CODEBASE_PATTERN",
    "ERROR_FIX",
    "FAILED_APPROACH",
    "OPEN_THREAD",
    "USER_PREFERENCE",
    "WORKING_SOLUTION",
]


def load_golden_set(path: Path) -> list[dict]:
    """Load golden set from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


async def run_eval(
    examples: list[dict],
    rate_limit: float = 1.0,
) -> dict:
    """Run classification on all examples and collect results.

    Args:
        examples: Golden set entries with expected_type
        rate_limit: Seconds between API calls

    Returns:
        dict with predictions, metrics, and timing
    """
    predictions = []
    start = time.monotonic()

    for i, ex in enumerate(examples):
        result = await classify_learning(
            content=ex["content"],
            context=ex.get("context"),
        )

        predicted = result.get("learning_type", "UNKNOWN")
        expected = ex["expected_type"]
        correct = predicted == expected

        predictions.append({
            "id": ex["id"],
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "confidence": result.get("confidence", "unknown"),
            "reasoning": result.get("reasoning", ""),
            "error": result.get("error"),
        })

        status = "OK" if correct else "MISS"
        print(
            f"  [{i + 1}/{len(examples)}] {status}: "
            f"expected={expected}, predicted={predicted}"
            f"  ({result.get('confidence', '?')})"
        )

        if i < len(examples) - 1:
            await asyncio.sleep(rate_limit)

    duration = time.monotonic() - start
    return {"predictions": predictions, "duration_seconds": round(duration, 2)}


def compute_metrics(predictions: list[dict]) -> dict:
    """Compute accuracy, per-class precision/recall, and confusion matrix."""
    total = len(predictions)
    correct = sum(1 for p in predictions if p["correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Per-class counts
    tp = Counter()  # true positives
    fp = Counter()  # false positives
    fn = Counter()  # false negatives
    confusion = defaultdict(Counter)  # confusion[expected][predicted]

    for p in predictions:
        expected = p["expected"]
        predicted = p["predicted"]
        confusion[expected][predicted] += 1

        if expected == predicted:
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1

    # Per-class precision and recall
    per_class = {}
    for t in LEARNING_TYPES:
        prec_denom = tp[t] + fp[t]
        rec_denom = tp[t] + fn[t]
        per_class[t] = {
            "precision": round(tp[t] / prec_denom, 3) if prec_denom > 0 else 0.0,
            "recall": round(tp[t] / rec_denom, 3) if rec_denom > 0 else 0.0,
            "support": tp[t] + fn[t],
        }

    # Build confusion matrix as nested dict
    confusion_matrix = {}
    for expected in LEARNING_TYPES:
        confusion_matrix[expected] = {
            predicted: confusion[expected][predicted]
            for predicted in LEARNING_TYPES
            if confusion[expected][predicted] > 0
        }

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def print_report(metrics: dict, duration: float) -> None:
    """Print a human-readable eval report."""
    print("\n" + "=" * 60)
    print("LEARNING CLASSIFICATION EVAL REPORT")
    print("=" * 60)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.1%}"
          f"  ({metrics['correct']}/{metrics['total']})")
    print(f"Duration: {duration:.1f}s")

    print(f"\n{'Type':<25} {'Prec':>6} {'Recall':>6} {'Support':>7}")
    print("-" * 50)
    for t in LEARNING_TYPES:
        c = metrics["per_class"][t]
        print(f"  {t:<23} {c['precision']:>5.1%} {c['recall']:>5.1%} {c['support']:>7}")

    # Confusion matrix
    print("\nConfusion Matrix (rows=expected, cols=predicted):")
    # Header
    abbrevs = {t: t[:4] for t in LEARNING_TYPES}
    header = f"{'':>15} " + " ".join(f"{abbrevs[t]:>5}" for t in LEARNING_TYPES)
    print(header)
    print("-" * len(header))
    for expected in LEARNING_TYPES:
        row = f"  {expected[:13]:>13} "
        for predicted in LEARNING_TYPES:
            count = metrics["confusion_matrix"].get(expected, {}).get(predicted, 0)
            cell = str(count) if count > 0 else "."
            row += f"{cell:>5} "
        print(row)

    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Eval learning classification")
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=GOLDEN_SET_PATH,
        help=f"Path to golden set JSONL (default: {GOLDEN_SET_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write results JSON to file",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.8,
        help="Minimum accuracy to pass (default: 0.8)",
    )
    args = parser.parse_args()

    print(f"Loading golden set from {args.golden_set}")
    examples = load_golden_set(args.golden_set)
    print(f"Loaded {len(examples)} examples\n")

    print("Running classification...")
    eval_result = await run_eval(examples, rate_limit=args.rate_limit)

    metrics = compute_metrics(eval_result["predictions"])
    print_report(metrics, eval_result["duration_seconds"])

    # Build output
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "golden_set": str(args.golden_set.relative_to(project_root)),
        "metrics": metrics,
        "duration_seconds": eval_result["duration_seconds"],
        "predictions": eval_result["predictions"],
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")

    return 0 if metrics["accuracy"] >= args.accuracy_threshold else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
