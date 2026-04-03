#!/usr/bin/env python3
"""Near-duplicate density analysis for the memory system.

Fetches all active learning embeddings from PostgreSQL, computes pairwise
cosine similarities, and produces a histogram showing the distribution of
similarity scores. This helps determine the optimal dedup threshold.

USAGE:
    # Generate histogram PNG (default: output to stdout path)
    uv run python scripts/core/duplicate_density.py

    # Save to specific path
    uv run python scripts/core/duplicate_density.py --output /tmp/dedup_density.png

    # Also dump the pairs above a threshold as JSON
    uv run python scripts/core/duplicate_density.py --dump-pairs 0.85

    # Text-only summary (no graph)
    uv run python scripts/core/duplicate_density.py --text-only

Environment:
    DATABASE_URL or CONTINUOUS_CLAUDE_DB_URL - PostgreSQL connection string
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

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


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

async def fetch_embeddings(conn: Any) -> tuple[list[str], list[str], np.ndarray]:
    """Fetch all active learning IDs, content previews, and embeddings.

    Groups by embedding dimension and uses only the dominant dimension group.
    Mixed-provider corpora (e.g. 1024-dim BGE + 768-dim Ollama) are handled
    by analyzing the largest group and reporting skipped counts.

    Returns:
        (ids, previews, embedding_matrix) where embedding_matrix is (N, D)
    """
    rows = await conn.fetch(
        """
        SELECT id::text, LEFT(content, 120) AS preview, embedding::text
        FROM archival_memory
        WHERE superseded_by IS NULL
          AND embedding IS NOT NULL
        ORDER BY created_at
        """
    )

    # Parse and group by dimension
    by_dim: dict[int, tuple[list[str], list[str], list[list[float]]]] = {}
    for row in rows:
        raw = row["embedding"]
        vec = [float(x) for x in raw.strip("[]").split(",")]
        dim = len(vec)
        if dim not in by_dim:
            by_dim[dim] = ([], [], [])
        ids_list, prev_list, vec_list = by_dim[dim]
        ids_list.append(row["id"])
        prev_list.append(row["preview"])
        vec_list.append(vec)

    if not by_dim:
        return [], [], np.array([], dtype=np.float32)

    # Use the largest dimension group
    dominant_dim = max(by_dim, key=lambda d: len(by_dim[d][0]))
    ids, previews, vectors = by_dim[dominant_dim]

    skipped = sum(len(v[0]) for d, v in by_dim.items() if d != dominant_dim)
    if skipped > 0:
        other_dims = {d: len(v[0]) for d, v in by_dim.items() if d != dominant_dim}
        print(
            f"Using {len(ids)} embeddings (dim={dominant_dim}), "
            f"skipped {skipped} with different dimensions: {other_dims}",
            file=sys.stderr,
        )

    matrix = np.array(vectors, dtype=np.float32)
    return ids, previews, matrix


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def compute_pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for an (N, D) matrix.

    Returns the upper-triangle values as a 1D array (N*(N-1)/2 elements).
    """
    # L2 normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    normed = matrix / norms

    # Cosine similarity matrix
    sim_matrix = normed @ normed.T

    # Extract upper triangle (excluding diagonal)
    upper_indices = np.triu_indices(sim_matrix.shape[0], k=1)
    return sim_matrix[upper_indices], upper_indices, sim_matrix


def compute_threshold_stats(
    similarities: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """Count pairs exceeding each threshold."""
    total_pairs = len(similarities)
    results = []
    for threshold in thresholds:
        count = int(np.sum(similarities >= threshold))
        results.append({
            "threshold": threshold,
            "pair_count": count,
            "pair_pct": round(100.0 * count / total_pairs, 4) if total_pairs else 0.0,
        })
    return results


def find_pairs_above(
    similarities: np.ndarray,
    upper_indices: tuple,
    threshold: float,
    ids: list[str],
    previews: list[str],
    limit: int = 100,
) -> list[dict]:
    """Return the top pairs above a threshold, sorted by similarity descending."""
    mask = similarities >= threshold
    matching_sims = similarities[mask]
    matching_i = upper_indices[0][mask]
    matching_j = upper_indices[1][mask]

    # Sort descending by similarity
    sort_idx = np.argsort(-matching_sims)[:limit]

    pairs = []
    for idx in sort_idx:
        i, j = int(matching_i[idx]), int(matching_j[idx])
        pairs.append({
            "similarity": round(float(matching_sims[idx]), 4),
            "id_a": ids[i],
            "id_b": ids[j],
            "preview_a": previews[i],
            "preview_b": previews[j],
        })
    return pairs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_density(
    similarities: np.ndarray,
    threshold_stats: list[dict],
    output_path: str,
    current_threshold: float | None = None,
) -> None:
    """Generate a histogram of pairwise similarities with threshold annotations."""
    if current_threshold is None:
        from scripts.core.config import get_config
        current_threshold = get_config().dedup.threshold

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, (ax_full, ax_tail) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Near-Duplicate Density Analysis  ({len(similarities):,} pairs)",
        fontsize=14,
        fontweight="bold",
    )

    # --- Left panel: full distribution (0.0 to 1.0) ---
    ax_full.hist(
        similarities,
        bins=200,
        color="#4a90d9",
        alpha=0.8,
        edgecolor="none",
    )
    ax_full.set_xlabel("Cosine Similarity")
    ax_full.set_ylabel("Pair Count")
    ax_full.set_title("Full Distribution")
    ax_full.set_xlim(-0.1, 1.05)
    ax_full.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # --- Right panel: tail distribution (0.5 to 1.0) ---
    tail_mask = similarities >= 0.50
    tail_sims = similarities[tail_mask]

    if len(tail_sims) > 0:
        ax_tail.hist(
            tail_sims,
            bins=100,
            color="#4a90d9",
            alpha=0.8,
            edgecolor="none",
        )

    # Annotate thresholds
    colors = {
        0.80: "#2ecc71",
        0.85: "#f39c12",
        0.90: "#e74c3c",
        0.92: "#9b59b6",
        0.95: "#e91e63",
        0.98: "#1a1a2e",
    }

    for stat in threshold_stats:
        t = stat["threshold"]
        color = colors.get(t, "#666666")
        label = f"{t:.2f}: {stat['pair_count']:,} pairs"
        style = "--" if t != current_threshold else "-"
        width = 1.5 if t != current_threshold else 3.0

        ax_tail.axvline(
            x=t, color=color, linestyle=style, linewidth=width,
            label=label, alpha=0.9,
        )

    ax_tail.set_xlabel("Cosine Similarity")
    ax_tail.set_ylabel("Pair Count")
    ax_tail.set_title("Tail Distribution (≥0.50)")
    ax_tail.set_xlim(0.49, 1.01)
    ax_tail.legend(loc="upper left", fontsize=9)
    ax_tail.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def format_summary(
    n_learnings: int,
    n_pairs: int,
    threshold_stats: list[dict],
    current_threshold: float | None = None,
) -> str:
    """Format a text summary of the density analysis."""
    if current_threshold is None:
        from scripts.core.config import get_config
        current_threshold = get_config().dedup.threshold

    lines = [
        "Near-Duplicate Density Report",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        f"Learnings analyzed: {n_learnings:,}",
        f"Total pairs:        {n_pairs:,}",
        "",
        "Threshold  │  Pairs Above  │  % of Total",
        "───────────┼───────────────┼─────────────",
    ]
    for stat in threshold_stats:
        marker = " ◄── current" if stat["threshold"] == current_threshold else ""
        lines.append(
            f"  {stat['threshold']:.2f}     │  {stat['pair_count']:>11,}  │  {stat['pair_pct']:>8.4f}%{marker}"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Near-duplicate density analysis for memory learnings"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for histogram PNG (default: thoughts/shared/duplicate_density.png)",
    )
    parser.add_argument(
        "--dump-pairs",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Dump pairs above this threshold as JSON to stdout",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print text summary only, no graph",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max pairs to dump (default: 100)",
    )
    return parser


async def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output or str(
        Path(repo_root) / "thoughts" / "shared" / "duplicate_density.png"
    )

    print("Fetching embeddings from PostgreSQL...", file=sys.stderr)
    async with get_connection() as conn:
        ids, previews, matrix = await fetch_embeddings(conn)
    await close_pool()

    n = len(ids)
    if n < 2:
        print(f"Only {n} learnings with embeddings — need at least 2.", file=sys.stderr)
        return 1

    print(f"Computing pairwise cosine similarity for {n:,} learnings...", file=sys.stderr)
    similarities, upper_indices, _ = compute_pairwise_cosine(matrix)
    n_pairs = len(similarities)

    thresholds = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]
    threshold_stats = compute_threshold_stats(similarities, thresholds)

    # Print text summary
    summary = format_summary(n, n_pairs, threshold_stats)
    print(summary)

    # Generate graph unless text-only
    if not args.text_only:
        print(f"Generating histogram → {output_path}", file=sys.stderr)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plot_density(similarities, threshold_stats, output_path)
        print(f"Saved: {output_path}", file=sys.stderr)

    # Dump pairs if requested
    if args.dump_pairs is not None:
        pairs = find_pairs_above(
            similarities, upper_indices, args.dump_pairs,
            ids, previews, args.limit,
        )
        print(json.dumps(pairs, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
