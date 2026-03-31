"""Contextual reranker for memory recall results.

Pure Python post-processing module that re-ranks recall results using
contextual signals (project match, recency, confidence, type affinity,
tag overlap, recall frequency) combined with calibrated retrieval scores.

No I/O, no database calls -- all functions are pure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RecallContext:
    """Context provided by the caller to inform reranking."""

    project: str | None = None
    query_embedding: list[float] | None = None
    type_probabilities: dict[str, float] | None = None
    tags_hint: list[str] | None = None
    retrieval_mode: str | None = None  # "vector", "hybrid_rrf", "text", "sqlite"
    now: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class RerankerConfig:
    """Weights for each contextual signal (should sum to ~0.35)."""

    project_weight: float = 0.15
    recency_weight: float = 0.05
    confidence_weight: float = 0.05
    recall_weight: float = 0.05
    type_affinity_weight: float = 0.05
    tag_overlap_weight: float = 0.05

    @property
    def total_signal_weight(self) -> float:
        return (
            self.project_weight
            + self.recency_weight
            + self.confidence_weight
            + self.recall_weight
            + self.type_affinity_weight
            + self.tag_overlap_weight
        )


# ---------------------------------------------------------------------------
# Score Calibration
# ---------------------------------------------------------------------------

def calibrate_score(
    raw_score: float,
    mode: str | None,
    *,
    rank: int,
    total: int,
) -> float:
    """Normalize a raw retrieval score to [0, 1] based on retrieval mode.

    Parameters
    ----------
    raw_score:  The raw similarity/relevance score from retrieval.
    mode:       Retrieval mode identifier.
    rank:       Zero-based rank of this result in the original list.
    total:      Total number of results.
    """
    if mode == "vector":
        # Cosine similarity [-1, 1] -> [0, 1]
        calibrated = (raw_score + 1) / 2
    elif mode == "hybrid_rrf":
        # RRF scores ~0.01-0.03 -> scale up
        calibrated = raw_score * 60
    elif mode in ("text", "sqlite"):
        # BM25 unbounded -> squash with kappa=1.0
        kappa = 1.0
        calibrated = raw_score / (raw_score + kappa)
    else:
        # Unknown/None: rank-based fallback
        if total <= 0:
            return 1.0
        calibrated = 1 - (rank / total)

    return max(0.0, min(1.0, calibrated))


# ---------------------------------------------------------------------------
# Signal Functions
# ---------------------------------------------------------------------------

def project_match(result: dict, ctx: RecallContext) -> float:
    """Score based on project match: 1.0 exact, 0.5 substring, 0.0 none."""
    if not ctx.project:
        return 0.0

    result_project = result.get("metadata", {}).get("project")
    if not result_project:
        return 0.0

    if result_project == ctx.project:
        return 1.0
    if ctx.project in result_project or result_project in ctx.project:
        return 0.5
    return 0.0


def recency_score(result: dict, ctx: RecallContext) -> float:
    """Exponential decay score: exp(-age_days / 45) over 90-day horizon."""
    created_at = result.get("created_at")
    if created_at is None:
        return 0.5

    # Handle string timestamps
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except (ValueError, TypeError):
            return 0.5

    # Ensure both datetimes are timezone-aware for subtraction
    now = ctx.now
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    age = now - created_at
    age_days = max(0.0, age.total_seconds() / 86400)
    return math.exp(-age_days / 45)


def confidence_score(result: dict) -> float:
    """Map confidence string to numeric score."""
    mapping = {"high": 1.0, "medium": 0.6, "low": 0.2}
    conf = result.get("metadata", {}).get("confidence")
    return mapping.get(conf, 0.5)


def recall_score(result: dict) -> float:
    """Score based on how often a learning has been recalled."""
    count = result.get("recall_count")
    if count is None or count <= 0:
        return 0.0
    return min(1.0, math.log2(1 + count) / 4)


def type_match(result: dict, ctx: RecallContext) -> float:
    """Score based on type affinity from soft distribution."""
    if ctx.type_probabilities is None:
        return 0.5

    learning_type = result.get("metadata", {}).get("learning_type")
    if not learning_type:
        return 0.0

    return ctx.type_probabilities.get(learning_type, 0.0)


def tag_overlap(result: dict, ctx: RecallContext) -> float:
    """Jaccard similarity between result tags and context tags hint."""
    if not ctx.tags_hint:
        return 0.0

    result_tags = set(result.get("metadata", {}).get("tags") or [])
    hint_tags = set(ctx.tags_hint)

    if not result_tags or not hint_tags:
        return 0.0

    intersection = result_tags & hint_tags
    union = result_tags | hint_tags
    if not union:
        return 0.0

    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Embedding Centroid Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0.0 for zero-norm."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_type_centroids(rows: list[dict]) -> dict[str, list[float]]:
    """Compute mean BGE embedding per learning_type.

    Args:
        rows: List of dicts with 'ltype' and 'embedding' keys.
              Typically from: SELECT metadata->>'learning_type' as ltype,
              embedding FROM archival_memory

    Returns:
        Dict mapping learning_type to centroid (mean embedding vector).
    """
    groups: dict[str, list[list[float]]] = {}
    for row in rows:
        ltype = row.get("ltype")
        if ltype is None:
            continue
        embedding = row.get("embedding")
        if embedding is None or (hasattr(embedding, "__len__") and len(embedding) == 0):
            continue
        groups.setdefault(ltype, []).append(embedding)

    centroids: dict[str, list[float]] = {}
    for ltype, vectors in groups.items():
        if not vectors:
            continue
        n = len(vectors)
        centroid = [sum(dims) / n for dims in zip(*vectors)]
        centroids[ltype] = centroid
    return centroids


def infer_query_type(
    query_embedding: list[float],
    centroids: dict[str, list[float]],
) -> dict[str, float]:
    """Infer query type as soft probability distribution over learning types.

    Uses cosine similarity to each type centroid, then softmax.

    Returns:
        Dict mapping learning_type to probability (sums to ~1.0).
    """
    if not centroids:
        return {}

    sims = {lt: _cosine_similarity(query_embedding, c) for lt, c in centroids.items()}

    # Softmax with max-subtraction for numerical stability
    max_sim = max(sims.values())
    exps = {lt: math.exp(s - max_sim) for lt, s in sims.items()}
    total = sum(exps.values())

    if total == 0.0:
        # Uniform fallback (shouldn't happen with valid centroids)
        n = len(centroids)
        return {lt: 1.0 / n for lt in centroids}

    return {lt: e / total for lt, e in exps.items()}


def save_centroids(centroids: dict[str, list[float]], path: str | Path) -> None:
    """Save centroids to a JSON file for caching."""
    with open(path, "w") as f:
        json.dump(centroids, f)


def load_centroids(path: str | Path) -> dict[str, list[float]] | None:
    """Load centroids from JSON file.  Returns None if missing or corrupt."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Main Rerank Function
# ---------------------------------------------------------------------------

def rerank(
    results: list[dict],
    ctx: RecallContext,
    config: RerankerConfig | None = None,
    k: int = 5,
) -> list[dict]:
    """Re-rank recall results using contextual signals.

    Pure function -- no I/O, no database calls, no network.

    Parameters
    ----------
    results:  List of result dicts from recall_learnings.
    ctx:      Contextual information for scoring.
    config:   Signal weights (uses defaults if None).
    k:        Number of top results to return.

    Returns
    -------
    Top-k results sorted by final_score descending, each augmented
    with ``final_score`` and ``rerank_details`` keys.
    """
    if not results:
        return []

    if config is None:
        config = RerankerConfig()

    total_signal_weight = config.total_signal_weight
    if total_signal_weight <= 0:
        return results[:k]

    retrieval_weight = 1.0 - total_signal_weight
    total = len(results)

    scored: list[dict] = []
    for rank, result in enumerate(results):
        # Calibrate raw retrieval score
        raw = result.get("similarity", 0.0)
        cal = calibrate_score(raw, ctx.retrieval_mode, rank=rank, total=total)

        # Compute each signal
        sig_project = project_match(result, ctx)
        sig_recency = recency_score(result, ctx)
        sig_confidence = confidence_score(result)
        sig_recall = recall_score(result)
        sig_type = type_match(result, ctx)
        sig_tags = tag_overlap(result, ctx)

        # Weighted combination
        final = (
            retrieval_weight * cal
            + config.project_weight * sig_project
            + config.recency_weight * sig_recency
            + config.confidence_weight * sig_confidence
            + config.recall_weight * sig_recall
            + config.type_affinity_weight * sig_type
            + config.tag_overlap_weight * sig_tags
        )

        # Augment result (shallow copy to avoid mutating input)
        augmented = {**result}
        augmented["final_score"] = final
        augmented["rerank_details"] = {
            "calibrated_score": cal,
            "project_match": sig_project,
            "recency": sig_recency,
            "confidence": sig_confidence,
            "recall": sig_recall,
            "type_match": sig_type,
            "tag_overlap": sig_tags,
        }
        scored.append(augmented)

    scored.sort(key=lambda r: r["final_score"], reverse=True)
    return scored[:k]
