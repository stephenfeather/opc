"""Contextual reranker for memory recall results.

Re-ranks recall results using contextual signals (project match, recency,
confidence, type affinity, tag overlap, recall frequency) combined with
calibrated retrieval scores.

Signal and scoring logic is pure when an explicit ``RerankerConfig`` is
provided.  If ``config`` is omitted, some helpers lazily resolve defaults via
``_default_config()``, which may consult config-file state on first use.
Explicit I/O helpers (save_centroids, load_centroids) are isolated in a
clearly marked section at the bottom.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    # list[dict] of entities extracted from the query via kg_extractor.
    # Each dict carries at least {"name": str, "type": str}. Used by
    # kg_overlap signal.  None or empty -> signal returns 0.
    query_entities: list[dict] | None = None
    now: datetime = field(default_factory=lambda: datetime.now(UTC))


# Type-salience weights for kg_overlap. Higher weights -> more specific types
# contribute more evidence to the overlap signal. See plan §3.2 for rationale.
KG_TYPE_WEIGHTS: dict[str, float] = {
    "file": 1.0,
    "module": 1.0,
    "error": 0.9,
    "library": 0.8,
    # kg_extractor emits entity_type='config' for env/config variables, so
    # the lookup key must be 'config'. 'config_var' is kept as a harmless
    # alias so older callers that hand-build entity dicts still hit the
    # intended salience. Keep both in sync if the weight changes.
    "config": 0.7,
    "config_var": 0.7,
    "tool": 0.6,
    "concept": 0.5,
    "language": 0.4,
}
_KG_DEFAULT_TYPE_WEIGHT: float = 0.5


# Single source of truth for RerankerConfig lives in config/models.py.
# Re-exported here for backward compatibility with existing callers.
from scripts.core.config.models import RerankerConfig  # noqa: E402


def _default_config() -> RerankerConfig:
    """Lazily load RerankerConfig from opc.toml, falling back to hardcoded defaults.

    This avoids import-time I/O while preserving backward compat with opc.toml overrides.
    """
    try:
        from scripts.core.config import get_config
        return get_config().reranker
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return RerankerConfig()


# ---------------------------------------------------------------------------
# Score Calibration
# ---------------------------------------------------------------------------


def calibrate_score(
    raw_score: float,
    mode: str | None,
    *,
    rank: int,
    total: int,
    config: RerankerConfig | None = None,
) -> float:
    """Normalize a raw retrieval score to [0, 1] based on retrieval mode.

    Parameters
    ----------
    raw_score:  The raw similarity/relevance score from retrieval.
    mode:       Retrieval mode identifier.
    rank:       Zero-based rank of this result in the original list.
    total:      Total number of results.
    config:     Optional config for tuning parameters (uses defaults if None).
    """
    cfg = config if config is not None else _default_config()

    if mode == "vector":
        # Cosine similarity [-1, 1] -> [0, 1]
        calibrated = (raw_score + 1) / 2
    elif mode == "hybrid_rrf":
        # RRF scores ~0.01-0.03 -> scale up
        calibrated = raw_score * cfg.rrf_scale_factor
    elif mode in ("text", "sqlite"):
        # BM25 unbounded -> squash with kappa=1.0; guard denominator
        kappa = 1.0
        denom = raw_score + kappa
        calibrated = raw_score / denom if denom != 0.0 else 0.0
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


def recency_score(
    result: dict,
    ctx: RecallContext,
    *,
    config: RerankerConfig | None = None,
) -> float:
    """Exponential decay score: exp(-age_days / half_life)."""
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

    cfg = config if config is not None else _default_config()
    age = now - created_at
    age_days = max(0.0, age.total_seconds() / 86400)
    half_life = cfg.recency_half_life_days if cfg.recency_half_life_days > 0.0 else 45.0
    return math.exp(-age_days / half_life)


def confidence_score(result: dict) -> float:
    """Map confidence string to numeric score."""
    mapping = {"high": 1.0, "medium": 0.6, "low": 0.2}
    conf = result.get("metadata", {}).get("confidence")
    return mapping.get(conf, 0.5)


def recall_score(
    result: dict,
    *,
    config: RerankerConfig | None = None,
) -> float:
    """Score based on how often a learning has been recalled."""
    count = result.get("recall_count")
    if count is None or count <= 0:
        return 0.0
    cfg = config if config is not None else _default_config()
    normalizer = cfg.recall_log2_normalizer if cfg.recall_log2_normalizer > 0.0 else 4.0
    return min(1.0, math.log2(1 + count) / normalizer)


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


def pattern_score(result: dict, ctx: RecallContext) -> float:
    """Score based on pattern membership, gated by query relevance.

    Only boosts if the query's tags overlap with the pattern's tags,
    preventing generic clustered learnings from drowning novel ones.

    Returns pattern_strength * tag_overlap_ratio, or 0.0 if no overlap.
    """
    strength = result.get("pattern_strength", 0.0)
    if not strength:
        return 0.0

    pattern_tags = result.get("pattern_tags") or []
    query_tags = ctx.tags_hint or []

    if not query_tags or not pattern_tags:
        return 0.0

    unique_query = set(query_tags)
    if not unique_query:
        return 0.0
    overlap = len(unique_query & set(pattern_tags))
    ratio = overlap / len(unique_query)
    return strength * min(1.0, ratio)


def kg_overlap(result: dict, ctx: RecallContext) -> float:
    """Weighted-Jaccard overlap between query entities and result KG entities.

    Each entity is weighted by its type's salience (KG_TYPE_WEIGHTS). Returns
    0.0 when either side lacks entities. Returns score in [0, 1].

    Canonicalization note:
      ``kg_extractor.extract_entities()`` exposes two name fields: ``name``
      (lowercase canonical, used as the KG primary key column) and
      ``display_name`` (original casing). ``_fetch_kg_rows`` stores
      ``display_name`` in ``kg_context.entities[*].name`` for human-readable
      output, while query-side entities carry the canonical lowercase name.
      We lowercase both on compare here so store-side display casing does
      not break overlap matching. If that asymmetry is ever closed (e.g. by
      adding an explicit ``canonical`` field to kg_context entries), the
      ``.lower()`` calls below can be removed.
    """
    if not ctx.query_entities:
        return 0.0

    kg_context = result.get("kg_context")
    if not kg_context:
        return 0.0

    result_entities = kg_context.get("entities") or []
    if not result_entities:
        return 0.0

    def _key(entity: dict) -> tuple[str, str]:
        # Prefer 'canonical' when present (kg_context entities from
        # _fetch_kg_rows carry both display 'name' and canonical 'canonical';
        # query-side entities from extract_entities put the canonical value
        # in 'name'). Fall back to lowercased name so older callers still
        # match. See plan §3.2 / adversarial-review F1 for rationale.
        canonical = entity.get("canonical")
        key_name = canonical if canonical is not None else entity.get("name", "")
        return (str(key_name).lower(), str(entity.get("type", "")))

    def _weight(entity_type: str) -> float:
        return KG_TYPE_WEIGHTS.get(entity_type, _KG_DEFAULT_TYPE_WEIGHT)

    query_map = {_key(e): e for e in ctx.query_entities}
    result_map = {_key(e): e for e in result_entities}

    intersection = set(query_map) & set(result_map)
    union = set(query_map) | set(result_map)
    if not union:
        return 0.0

    num = sum(_weight(query_map[k].get("type", "")) for k in intersection)
    # Use query type for union members on query side, result type on result side.
    den = sum(
        _weight((query_map.get(k) or result_map.get(k)).get("type", ""))
        for k in union
    )
    if den <= 0.0:
        return 0.0
    return max(0.0, min(1.0, num / den))


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
    # Group embeddings by type, filtering out None types and empty embeddings
    groups: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        ltype = row.get("ltype")
        if ltype is None:
            continue
        embedding = row.get("embedding")
        if embedding is None or (hasattr(embedding, "__len__") and len(embedding) == 0):
            continue
        groups[ltype].append(embedding)

    # Compute mean embedding per type
    return {
        ltype: [sum(dims) / len(vectors) for dims in zip(*vectors, strict=True)]
        for ltype, vectors in groups.items()
        if vectors
    }


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

    Pure when an explicit config is provided; otherwise lazily resolves
    defaults via ``_default_config()``.  No database calls, no network.

    Parameters
    ----------
    results:  List of result dicts from recall_learnings.
    ctx:      Contextual information for scoring.
    config:   Signal weights (uses defaults if None).
    k:        Number of top results to return.

    Returns
    -------
    Top-k results sorted by final_score descending, each augmented
    with ``final_score`` and ``rerank_details`` keys.  When
    total_signal_weight is zero, final_score equals the calibrated
    retrieval score and results are sorted accordingly.
    """
    if not results:
        return []

    if config is None:
        config = _default_config()

    total_signal_weight = config.total_signal_weight
    if total_signal_weight <= 0:
        total = len(results)
        scored = [
            {
                **r,
                "final_score": calibrate_score(
                    r.get("similarity", 0.0), ctx.retrieval_mode,
                    rank=i, total=total, config=config,
                ),
                "rerank_details": {},
            }
            for i, r in enumerate(results)
        ]
        scored.sort(key=lambda r: r["final_score"], reverse=True)
        return scored[:k]

    retrieval_weight = 1.0 - total_signal_weight
    total = len(results)

    scored: list[dict] = []
    for rank, result in enumerate(results):
        # Calibrate raw retrieval score
        raw = result.get("similarity", 0.0)
        cal = calibrate_score(raw, ctx.retrieval_mode, rank=rank, total=total, config=config)

        # Compute each signal
        sig_project = project_match(result, ctx)
        sig_recency = recency_score(result, ctx, config=config)
        sig_confidence = confidence_score(result)
        sig_recall = recall_score(result, config=config)
        sig_type = type_match(result, ctx)
        sig_tags = tag_overlap(result, ctx)
        sig_pattern = pattern_score(result, ctx)
        sig_kg = kg_overlap(result, ctx)

        # Weighted combination
        final = (
            retrieval_weight * cal
            + config.project_weight * sig_project
            + config.recency_weight * sig_recency
            + config.confidence_weight * sig_confidence
            + config.recall_weight * sig_recall
            + config.type_affinity_weight * sig_type
            + config.tag_overlap_weight * sig_tags
            + config.pattern_weight * sig_pattern
            + config.kg_weight * sig_kg
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
            "pattern": sig_pattern,
            "kg_overlap": sig_kg,
        }
        scored.append(augmented)

    scored.sort(key=lambda r: r["final_score"], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# I/O Handlers (side effects -- not pure functions)
# ---------------------------------------------------------------------------


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
