"""Cross-session pattern detection engine.

Pure analysis module -- receives data, returns results.
No I/O, no database access, no side effects.

Detects recurring patterns across sessions using:
1. HDBSCAN clustering on L2-normalized BGE embeddings
2. IDF-weighted tag co-occurrence graph
3. Fusion of both signal types
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.cluster import HDBSCAN


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
    pattern_type: str       # 'tool_cluster', 'problem_solution', 'cross_project', 'expertise', 'anti_pattern'
    member_ids: list[str]
    representative_id: str  # closest to centroid
    tags: list[str]         # aggregated, deduplicated
    session_count: int
    confidence: float       # cluster quality metric [0, 1]
    label: str              # human-readable summary
    metadata: dict = field(default_factory=dict)
    distances: dict[str, float] = field(default_factory=dict)  # member_id -> distance to centroid


# ---------------------------------------------------------------------------
# Embedding-based clustering
# ---------------------------------------------------------------------------

def cluster_by_embeddings(
    learnings: list[Learning],
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> list[list[int]]:
    """Cluster learnings by embedding similarity using HDBSCAN.

    L2-normalizes embeddings so Euclidean distance is proportional to
    cosine distance. No dimensionality reduction -- at 2.2K points the
    pairwise matrix is ~19 MB, trivial for numpy.

    Returns list of clusters, each a list of indices into `learnings`.
    Noise points (label -1) are excluded.
    """
    if len(learnings) < min_cluster_size:
        return []

    embeddings = np.array([l.embedding for l in learnings])

    # L2-normalize: Euclidean on unit vectors == cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = embeddings / norms

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(normalized)

    # Group indices by cluster label, skip noise (-1)
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:
            clusters[label].append(idx)

    return list(clusters.values())


# ---------------------------------------------------------------------------
# Tag IDF and co-occurrence clustering
# ---------------------------------------------------------------------------

def compute_tag_idf(
    all_tags: dict[str, list[str]],
    total_docs: int,
) -> dict[str, float]:
    """Compute inverse document frequency for each tag.

    IDF(tag) = log(total_docs / (1 + docs_with_tag))

    High IDF = rare tag = more informative.
    Low IDF = common tag = noise.
    """
    doc_freq: Counter[str] = Counter()
    for tags in all_tags.values():
        for tag in set(tags):  # unique tags per document
            doc_freq[tag] += 1

    return {
        tag: math.log(total_docs / (1 + freq))
        for tag, freq in doc_freq.items()
    }


def detect_noise_tags(
    tag_idf: dict[str, float],
    threshold_percentile: float = 10,
) -> set[str]:
    """Tags with IDF in the bottom percentile are noise.

    Returns set of tag names to exclude from co-occurrence analysis.
    """
    if not tag_idf:
        return set()

    values = sorted(tag_idf.values())
    threshold_idx = max(0, int(len(values) * threshold_percentile / 100))
    threshold = values[threshold_idx]

    return {tag for tag, idf in tag_idf.items() if idf <= threshold}


def cluster_by_tags(
    learnings: list[Learning],
    min_cooccurrence: float = 1.0,
    min_component_size: int = 5,
    exclude_tags: set[str] | None = None,
    tag_idf: dict[str, float] | None = None,
) -> list[list[int]]:
    """Find clusters via IDF-weighted tag co-occurrence graph.

    Builds a weighted graph where nodes are learnings, edges connect
    learnings sharing tags, weights = sum of shared tags' IDF scores.
    Uses connected components with min-weight threshold.

    Large components (>20 members) are split by raising the edge-weight
    threshold iteratively until they break apart.
    """
    if len(learnings) < min_component_size:
        return []

    exclude = exclude_tags or set()

    # Build learning -> filtered tags mapping
    learning_tags: list[set[str]] = []
    for l in learnings:
        filtered = {t for t in l.tags if t not in exclude}
        learning_tags.append(filtered)

    # Build tag -> learning indices inverted index
    tag_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, tags in enumerate(learning_tags):
        for tag in tags:
            tag_to_indices[tag].append(idx)

    # Default IDF: 1.0 for all tags if not provided
    idf = tag_idf or {}

    # Build sparse adjacency matrix with IDF-weighted edges
    n = len(learnings)
    rows, cols, weights = [], [], []

    for tag, indices in tag_to_indices.items():
        if len(indices) < 2:
            continue
        w = idf.get(tag, 1.0)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                rows.append(indices[i])
                cols.append(indices[j])
                weights.append(w)

    if not rows:
        return []

    # Aggregate duplicate edges (same pair may share multiple tags)
    edge_weights: dict[tuple[int, int], float] = defaultdict(float)
    for r, c, w in zip(rows, cols, weights):
        key = (min(r, c), max(r, c))
        edge_weights[key] += w

    # Filter by minimum co-occurrence weight
    filtered_rows, filtered_cols, filtered_weights = [], [], []
    for (r, c), w in edge_weights.items():
        if w >= min_cooccurrence:
            filtered_rows.extend([r, c])
            filtered_cols.extend([c, r])
            filtered_weights.extend([w, w])

    if not filtered_rows:
        return []

    graph = csr_matrix(
        (filtered_weights, (filtered_rows, filtered_cols)),
        shape=(n, n),
    )

    n_components, labels = connected_components(graph, directed=False)

    # Group by component, filter by min size
    components: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        components[label].append(idx)

    clusters = []
    for members in components.values():
        if len(members) < min_component_size:
            continue
        if len(members) <= 20:
            clusters.append(members)
        else:
            # Split large components by raising threshold
            split = _split_large_component(
                members, edge_weights, min_component_size, min_cooccurrence
            )
            clusters.extend(split)

    return clusters


def _split_large_component(
    members: list[int],
    edge_weights: dict[tuple[int, int], float],
    min_size: int,
    base_threshold: float,
) -> list[list[int]]:
    """Split a large component by iteratively raising the edge-weight threshold."""
    member_set = set(members)
    threshold = base_threshold

    for _ in range(10):  # max 10 iterations
        threshold *= 1.5

        rows, cols, weights = [], [], []
        for (r, c), w in edge_weights.items():
            if r in member_set and c in member_set and w >= threshold:
                rows.extend([r, c])
                cols.extend([c, r])
                weights.extend([w, w])

        if not rows:
            # Threshold too high, return the original as-is
            return [members]

        idx_map = {m: i for i, m in enumerate(sorted(member_set))}
        n = len(member_set)
        mapped_rows = [idx_map[r] for r in rows]
        mapped_cols = [idx_map[c] for c in cols]

        graph = csr_matrix((weights, (mapped_rows, mapped_cols)), shape=(n, n))
        _, labels = connected_components(graph, directed=False)

        reverse_map = {i: m for m, i in idx_map.items()}
        sub_components: dict[int, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            sub_components[label].append(reverse_map[i])

        valid = [m for m in sub_components.values() if len(m) >= min_size]
        if len(valid) > 1:
            return valid

    return [members]


# ---------------------------------------------------------------------------
# Cluster fusion
# ---------------------------------------------------------------------------

def fuse_clusters(
    embedding_clusters: list[list[int]],
    tag_clusters: list[list[int]],
    overlap_threshold: float = 0.3,
) -> list[list[int]]:
    """Merge embedding and tag clusters using Jaccard overlap.

    Two clusters from different methods merge if they share > threshold
    members. Tag clusters that don't overlap with any embedding cluster
    are kept as-is.
    """
    if not embedding_clusters and not tag_clusters:
        return []
    if not tag_clusters:
        return [list(c) for c in embedding_clusters]
    if not embedding_clusters:
        return [list(c) for c in tag_clusters]

    # Convert to sets for overlap computation
    emb_sets = [set(c) for c in embedding_clusters]
    tag_sets = [set(c) for c in tag_clusters]

    merged = [set(c) for c in embedding_clusters]
    tag_used = [False] * len(tag_sets)

    for ti, tag_set in enumerate(tag_sets):
        best_overlap = 0.0
        best_ei = -1

        for ei, emb_set in enumerate(emb_sets):
            intersection = len(tag_set & emb_set)
            union = len(tag_set | emb_set)
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard > best_overlap:
                best_overlap = jaccard
                best_ei = ei

        if best_overlap >= overlap_threshold and best_ei >= 0:
            merged[best_ei] = merged[best_ei] | tag_set
            tag_used[ti] = True

    # Keep non-overlapping tag clusters
    for ti, tag_set in enumerate(tag_sets):
        if not tag_used[ti]:
            merged.append(tag_set)

    return [sorted(c) for c in merged]


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------

def classify_pattern_heuristic(members: list[Learning]) -> str:
    """Classify a cluster as one of the pattern types using heuristics.

    Rules (checked in priority order):
    - All FAILED_APPROACH -> 'anti_pattern'
    - Spans 3+ distinct sessions with different contexts -> 'cross_project'
    - Contains both ERROR_FIX and WORKING_SOLUTION -> 'problem_solution'
    - Temporally concentrated (last 2 weeks) and 3+ sessions -> 'expertise'
    - Default: 'tool_cluster'
    """
    if not members:
        return "tool_cluster"

    types = Counter(m.learning_type for m in members)
    sessions = set(m.session_id for m in members)
    contexts = set(m.context for m in members if m.context)

    # All failed approaches -> anti-pattern
    if len(types) == 1 and "FAILED_APPROACH" in types:
        return "anti_pattern"

    # Majority failed approaches -> anti-pattern
    total = sum(types.values())
    if types.get("FAILED_APPROACH", 0) / total > 0.6:
        return "anti_pattern"

    # Cross-project: 3+ sessions with 2+ distinct contexts
    if len(sessions) >= 3 and len(contexts) >= 2:
        return "cross_project"

    # Problem-solution: mix of error fixes and working solutions
    if "ERROR_FIX" in types and "WORKING_SOLUTION" in types:
        return "problem_solution"

    # Expertise: temporally concentrated in last 2 weeks, 3+ sessions
    now = datetime.now(UTC)
    recent = [m for m in members if (now - m.created_at).days <= 14]
    if len(recent) >= len(members) * 0.5 and len(sessions) >= 3:
        return "expertise"

    return "tool_cluster"


def _learning_to_dict(m: Learning) -> dict:
    """Convert a Learning to the dict format expected by classify_pattern_llm."""
    return {
        "content": m.content,
        "learning_type": m.learning_type,
        "session_id": m.session_id,
        "context": m.context,
        "tags": m.tags,
    }


# Type alias for an async cluster classifier callback.
PatternClassifier = Callable[[list[Learning]], Awaitable[str]]


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def generate_label(members: list[Learning], pattern_type: str) -> str:
    """Generate a human-readable label for a pattern.

    Uses the most common tags and learning_type to build a description.
    """
    if not members:
        return "Empty pattern"

    # Collect all tags, find top 3
    all_tags: list[str] = []
    for m in members:
        all_tags.extend(m.tags)
    top_tags = [tag for tag, _ in Counter(all_tags).most_common(3)]

    sessions = set(m.session_id for m in members)
    tag_str = " + ".join(top_tags) if top_tags else "misc"

    type_labels = {
        "tool_cluster": "patterns",
        "problem_solution": "problem-solution patterns",
        "cross_project": "cross-project patterns",
        "expertise": "emerging expertise",
        "anti_pattern": "anti-patterns",
    }
    type_label = type_labels.get(pattern_type, "patterns")

    return f"{tag_str} {type_label} across {len(sessions)} sessions"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(
    members: list[Learning],
    centroid: np.ndarray,
) -> float:
    """Score cluster quality on [0, 1].

    confidence = 0.3 * cohesion + 0.3 * diversity + 0.2 * temporal_span + 0.2 * size_score

    cohesion:      mean cosine similarity to centroid
    diversity:     min(1.0, distinct_sessions / 5)
    temporal_span: min(1.0, (last - first).days / 14)
    size_score:    min(1.0, log2(cluster_size) / 4)
    """
    if not members:
        return 0.0

    # Cohesion: mean cosine similarity to centroid
    embeddings = np.array([m.embedding for m in members])
    centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    similarities = normalized @ centroid_norm
    cohesion = float(np.mean(similarities))
    # Clamp to [0, 1]
    cohesion = max(0.0, min(1.0, cohesion))

    # Diversity: distinct sessions
    sessions = set(m.session_id for m in members)
    diversity = min(1.0, len(sessions) / 5)

    # Temporal span
    dates = [m.created_at for m in members]
    span_days = (max(dates) - min(dates)).days if len(dates) >= 2 else 0
    temporal_span = min(1.0, span_days / 14)

    # Size
    size_score = min(1.0, math.log2(max(1, len(members))) / 4)

    return 0.3 * cohesion + 0.3 * diversity + 0.2 * temporal_span + 0.2 * size_score


def compute_centroid(members: list[Learning]) -> np.ndarray:
    """Compute mean embedding (centroid) for a set of learnings."""
    embeddings = np.array([m.embedding for m in members])
    return np.mean(embeddings, axis=0)


def compute_distances(
    members: list[Learning],
    centroid: np.ndarray,
) -> dict[str, float]:
    """Compute cosine distance from each member to the centroid."""
    centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
    distances = {}
    for m in members:
        norm = np.linalg.norm(m.embedding)
        if norm == 0:
            distances[m.id] = 1.0
        else:
            sim = float((m.embedding / norm) @ centroid_norm)
            distances[m.id] = 1.0 - max(0.0, min(1.0, sim))
    return distances


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

async def detect_patterns(
    learnings: list[Learning],
    min_cluster_size: int = 5,
    min_samples: int = 3,
    min_confidence: float = 0.3,
    tag_noise_percentile: float = 10,
    classifier: PatternClassifier | None = None,
) -> list[DetectedPattern]:
    """Run the full pattern detection pipeline.

    1. Cluster by embeddings (HDBSCAN)
    2. Compute tag IDF and cluster by tags
    3. Fuse clusters
    4. Classify, label, and score each cluster

    Args:
        classifier: Optional async callback for LLM-based classification.
            Falls back to classify_pattern_heuristic() on None or error.

    Returns list of DetectedPattern sorted by confidence descending.
    """
    if len(learnings) < min_cluster_size:
        return []

    # Step 1: Embedding clustering
    emb_clusters = cluster_by_embeddings(learnings, min_cluster_size, min_samples)

    # Step 2: Tag IDF + tag clustering
    all_tags = {l.id: l.tags for l in learnings}
    tag_idf = compute_tag_idf(all_tags, len(learnings))
    noise_tags = detect_noise_tags(tag_idf, tag_noise_percentile)
    tag_clusters = cluster_by_tags(
        learnings,
        min_cooccurrence=1.0,
        min_component_size=min_cluster_size,
        exclude_tags=noise_tags,
        tag_idf=tag_idf,
    )

    # Step 3: Fuse
    fused = fuse_clusters(emb_clusters, tag_clusters)

    # Step 4: Analyze each cluster
    patterns = []
    for cluster_indices in fused:
        members = [learnings[i] for i in cluster_indices]
        centroid = compute_centroid(members)
        confidence = compute_confidence(members, centroid)

        if confidence < min_confidence:
            continue

        # Find representative (closest to centroid)
        distances = compute_distances(members, centroid)
        representative_id = min(distances, key=distances.get)

        if classifier is not None:
            try:
                pattern_type = await classifier(members)
            except Exception:
                pattern_type = classify_pattern_heuristic(members)
        else:
            pattern_type = classify_pattern_heuristic(members)
        label = generate_label(members, pattern_type)

        # Aggregate tags
        tag_counter: Counter[str] = Counter()
        for m in members:
            tag_counter.update(m.tags)
        aggregated_tags = [t for t, _ in tag_counter.most_common(20)]

        sessions = set(m.session_id for m in members)

        patterns.append(DetectedPattern(
            pattern_type=pattern_type,
            member_ids=[m.id for m in members],
            representative_id=representative_id,
            tags=aggregated_tags,
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
        ))

    patterns.sort(key=lambda p: p.confidence, reverse=True)
    return patterns
