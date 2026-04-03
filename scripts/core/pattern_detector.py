"""Cross-session pattern detection engine.

Pure analysis module -- receives data, returns results.
No I/O, no database access, no side effects.

Detects recurring patterns across sessions using:
1. HDBSCAN clustering on L2-normalized BGE embeddings
2. IDF-weighted tag co-occurrence graph
3. Fusion of both signal types

Clustering logic lives in pattern_detector_clustering.py.
This module provides classification, scoring, and the detection pipeline.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from scripts.core.config import get_config as _get_config
from scripts.core.pattern_detector_clustering import (
    cluster_by_embeddings,
    cluster_by_tags,
    compute_tag_idf,
    detect_noise_tags,
    fuse_clusters,
)

_patterns_cfg = _get_config().patterns


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Learning:
    """A single learning from the memory system."""
    id: str
    content: str
    embedding: np.ndarray  # 1024-dim BGE
    learning_type: str
    tags: list[str]
    session_id: str
    context: str
    created_at: datetime
    confidence: str


@dataclass
class DetectedPattern:
    """A cluster of related learnings forming a cross-session pattern."""
    pattern_type: str
    member_ids: list[str]
    representative_id: str
    tags: list[str]
    session_count: int
    confidence: float
    label: str
    metadata: dict = field(default_factory=dict)
    distances: dict[str, float] = field(default_factory=dict)


# Type alias for an async cluster classifier callback.
PatternClassifier = Callable[[list[Learning]], Awaitable[str]]


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------

def classify_pattern_heuristic(
    members: list[Learning],
    reference_time: datetime | None = None,
) -> str:
    """Classify a cluster using heuristics. Priority order:

    1. anti_pattern: >60% FAILED_APPROACH
    2. problem_solution: Both ERROR_FIX and WORKING_SOLUTION, combined >=40%
    3. expertise: 60% recent + >=60% one type + >=4 contexts + >=3 sessions
    4. tool_cluster: >=70% same type + <=3 contexts
    5. cross_project: 4+ contexts OR (>=5 members AND session/member > 0.7)
    """
    if not members:
        return "tool_cluster"

    types = Counter(m.learning_type for m in members)
    sessions = {m.session_id for m in members}
    contexts = {m.context for m in members if m.context}
    total = sum(types.values())
    max_type_count = max(types.values())

    if _is_anti_pattern(types, total):
        return "anti_pattern"
    if _is_problem_solution(types, total):
        return "problem_solution"
    if _is_expertise(members, max_type_count, total, contexts, sessions, reference_time):
        return "expertise"
    if _is_tool_cluster(max_type_count, total, contexts):
        return "tool_cluster"
    if _is_cross_project(members, contexts, sessions):
        return "cross_project"
    return "tool_cluster"


def _is_anti_pattern(types: Counter, total: int) -> bool:
    if len(types) == 1 and "FAILED_APPROACH" in types:
        return True
    return types.get("FAILED_APPROACH", 0) / total > _patterns_cfg.anti_pattern_threshold


def _is_problem_solution(types: Counter, total: int) -> bool:
    error_fix = types.get("ERROR_FIX", 0)
    working_sol = types.get("WORKING_SOLUTION", 0)
    return (
        error_fix > 0
        and working_sol > 0
        and (error_fix + working_sol) / total >= _patterns_cfg.problem_solution_threshold
    )


def _is_expertise(
    members: list[Learning],
    max_type_count: int,
    total: int,
    contexts: set[str],
    sessions: set[str],
    reference_time: datetime | None,
) -> bool:
    now = reference_time or datetime.now(UTC)
    cutoff_seconds = _patterns_cfg.cross_project_days * 24 * 3600
    recent = [
        m for m in members
        if (now - m.created_at).total_seconds() <= cutoff_seconds
    ]
    return (
        len(recent) >= len(members) * _patterns_cfg.expertise_threshold
        and max_type_count / total >= _patterns_cfg.expertise_threshold
        and len(contexts) >= _patterns_cfg.cross_project_contexts
        and len(sessions) >= _patterns_cfg.cross_project_sessions
    )


def _is_tool_cluster(max_type_count: int, total: int, contexts: set[str]) -> bool:
    return max_type_count / total >= _patterns_cfg.tool_cluster_threshold and len(contexts) <= 3


def _is_cross_project(
    members: list[Learning],
    contexts: set[str],
    sessions: set[str],
) -> bool:
    if len(contexts) >= _patterns_cfg.cross_project_contexts:
        return True
    return len(members) >= 5 and len(sessions) / len(members) > 0.7


def _learning_to_dict(m: Learning) -> dict:
    """Convert a Learning to the dict format expected by classify_pattern_llm."""
    return {
        "content": m.content,
        "learning_type": m.learning_type,
        "session_id": m.session_id,
        "context": m.context,
        "tags": m.tags,
    }


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

_TYPE_LABELS = {
    "tool_cluster": "patterns",
    "problem_solution": "problem-solution patterns",
    "cross_project": "cross-project patterns",
    "expertise": "emerging expertise",
    "anti_pattern": "anti-patterns",
}


def generate_label(members: list[Learning], pattern_type: str) -> str:
    """Generate a human-readable label for a pattern."""
    if not members:
        return "Empty pattern"

    all_tags: list[str] = [tag for m in members for tag in set(m.tags)]
    top_tags = [tag for tag, _ in Counter(all_tags).most_common(3)]
    sessions = {m.session_id for m in members}
    tag_str = " + ".join(top_tags) if top_tags else "misc"
    type_label = _TYPE_LABELS.get(pattern_type, "patterns")

    return f"{tag_str} {type_label} across {len(sessions)} sessions"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_cohesion(members: list[Learning], centroid: np.ndarray) -> float:
    """Mean cosine similarity of members to centroid, clamped to [0, 1]."""
    embeddings = np.array([m.embedding for m in members])
    centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    similarities = (embeddings / norms) @ centroid_norm
    return float(max(0.0, min(1.0, np.mean(similarities))))


def compute_diversity(members: list[Learning]) -> float:
    """Distinct session ratio, capped at 1.0 (5 sessions = max)."""
    if not members:
        return 0.0
    sessions = {m.session_id for m in members}
    return min(1.0, len(sessions) / 5)


def compute_temporal_span(members: list[Learning]) -> float:
    """Temporal spread as fraction of 14 days, capped at 1.0."""
    if len(members) < 2:
        return 0.0
    dates = [m.created_at for m in members]
    span_days = (max(dates) - min(dates)).days
    return min(1.0, span_days / 14)


def compute_size_score(count: int) -> float:
    """Logarithmic size score, capped at 1.0 (16 members = max)."""
    if count <= 0:
        return 0.0
    return min(1.0, math.log2(max(1, count)) / 4)


def compute_confidence(
    members: list[Learning],
    centroid: np.ndarray,
) -> float:
    """Score cluster quality on [0, 1].

    confidence = 0.3 * cohesion + 0.3 * diversity + 0.2 * temporal_span + 0.2 * size_score
    """
    if not members:
        return 0.0

    cohesion = compute_cohesion(members, centroid)
    diversity = compute_diversity(members)
    temporal = compute_temporal_span(members)
    size = compute_size_score(len(members))

    return 0.3 * cohesion + 0.3 * diversity + 0.2 * temporal + 0.2 * size


def compute_centroid(members: list[Learning]) -> np.ndarray:
    """Compute mean embedding (centroid) for a set of learnings."""
    return np.mean(np.array([m.embedding for m in members]), axis=0)


def compute_distances(
    members: list[Learning],
    centroid: np.ndarray,
) -> dict[str, float]:
    """Compute cosine distance from each member to the centroid."""
    centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
    return {
        m.id: _cosine_distance(m.embedding, centroid_norm)
        for m in members
    }


def _cosine_distance(embedding: np.ndarray, centroid_norm: np.ndarray) -> float:
    """1 - cosine_similarity, clamped to [0, 1]."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return 1.0
    sim = float((embedding / norm) @ centroid_norm)
    return 1.0 - max(0.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

async def detect_patterns(
    learnings: list[Learning],
    min_cluster_size: int = _patterns_cfg.min_cluster_size,
    min_samples: int = _patterns_cfg.min_samples,
    min_confidence: float = _patterns_cfg.min_confidence,
    tag_noise_percentile: float = _patterns_cfg.tag_noise_percentile,
    classifier: PatternClassifier | None = None,
) -> list[DetectedPattern]:
    """Run the full pattern detection pipeline.

    1. Cluster by embeddings (HDBSCAN)
    2. Compute tag IDF and cluster by tags
    3. Fuse clusters
    4. Classify, label, and score each cluster

    Returns list of DetectedPattern sorted by confidence descending.
    """
    if len(learnings) < min_cluster_size:
        return []

    fused = _cluster_and_fuse(
        learnings, min_cluster_size, min_samples, tag_noise_percentile,
    )

    patterns = []
    for cluster_indices in fused:
        pattern = await _analyze_cluster(
            cluster_indices, learnings, classifier, min_confidence,
        )
        if pattern is not None:
            patterns.append(pattern)

    patterns.sort(key=lambda p: p.confidence, reverse=True)
    return patterns


def _cluster_and_fuse(
    learnings: list[Learning],
    min_cluster_size: int,
    min_samples: int,
    tag_noise_percentile: float,
) -> list[list[int]]:
    """Run embedding + tag clustering and fuse results."""
    emb_clusters = cluster_by_embeddings(learnings, min_cluster_size, min_samples)

    all_tags = {m.id: m.tags for m in learnings}
    tag_idf = compute_tag_idf(all_tags, len(learnings))
    noise_tags = detect_noise_tags(tag_idf, tag_noise_percentile) | {"perception"}
    tag_clusters = cluster_by_tags(
        learnings,
        min_cooccurrence=1.0,
        min_component_size=min_cluster_size,
        exclude_tags=noise_tags,
        tag_idf=tag_idf,
    )

    return fuse_clusters(emb_clusters, tag_clusters)


async def _analyze_cluster(
    cluster_indices: list[int],
    learnings: list[Learning],
    classifier: PatternClassifier | None,
    min_confidence: float,
) -> DetectedPattern | None:
    """Classify, label, and score a single cluster."""
    members = [learnings[i] for i in cluster_indices]
    centroid = compute_centroid(members)
    confidence = compute_confidence(members, centroid)

    if confidence < min_confidence:
        return None

    distances = compute_distances(members, centroid)
    representative_id = min(distances, key=distances.get)
    pattern_type = await _classify_with_fallback(members, classifier)
    label = generate_label(members, pattern_type)

    tag_counter: Counter[str] = Counter()
    for m in members:
        tag_counter.update(set(m.tags))

    sessions = {m.session_id for m in members}

    return DetectedPattern(
        pattern_type=pattern_type,
        member_ids=[m.id for m in members],
        representative_id=representative_id,
        tags=[t for t, _ in tag_counter.most_common(20)],
        session_count=len(sessions),
        confidence=confidence,
        label=label,
        metadata={
            "size": len(members),
            "cohesion": float(np.mean([1.0 - d for d in distances.values()])),
            "temporal_span_days": (
                (max(m.created_at for m in members) - min(m.created_at for m in members)).days
                if len(members) >= 2 else 0
            ),
        },
        distances=distances,
    )


async def _classify_with_fallback(
    members: list[Learning],
    classifier: PatternClassifier | None,
) -> str:
    """Use classifier callback if provided, fall back to heuristic."""
    if classifier is not None:
        try:
            return await classifier(members)
        except Exception:
            pass
    return classify_pattern_heuristic(members)
