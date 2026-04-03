"""Clustering functions for cross-session pattern detection.

Pure functions for grouping learnings by embedding similarity (HDBSCAN)
and tag co-occurrence (IDF-weighted graph components). No I/O.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN

from scripts.core.config import get_config as _get_config

if TYPE_CHECKING:
    from scripts.core.pattern_detector import Learning

_patterns_cfg = _get_config().patterns


# ---------------------------------------------------------------------------
# Embedding-based clustering
# ---------------------------------------------------------------------------

def cluster_by_embeddings(
    learnings: list[Learning],
    min_cluster_size: int = _patterns_cfg.min_cluster_size,
    min_samples: int = _patterns_cfg.min_samples,
) -> list[list[int]]:
    """Cluster learnings by embedding similarity using HDBSCAN.

    L2-normalizes embeddings so Euclidean distance is proportional to
    cosine distance. Returns list of clusters (each a list of indices).
    Noise points (label -1) are excluded.
    """
    if len(learnings) < min_cluster_size:
        return []

    normalized = _l2_normalize(
        np.array([m.embedding for m in learnings]),
    )

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(normalized)

    return _group_by_label(labels)


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero vectors left unchanged."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def _group_by_label(labels: np.ndarray) -> list[list[int]]:
    """Group indices by cluster label, skipping noise (-1)."""
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:
            clusters[label].append(idx)
    return list(clusters.values())


# ---------------------------------------------------------------------------
# Tag IDF
# ---------------------------------------------------------------------------

def compute_tag_idf(
    all_tags: dict[str, list[str]],
    total_docs: int,
) -> dict[str, float]:
    """Compute inverse document frequency for each tag.

    IDF(tag) = log(total_docs / (1 + docs_with_tag))
    High IDF = rare tag = more informative.
    """
    doc_freq: Counter[str] = Counter()
    for tags in all_tags.values():
        for tag in set(tags):
            doc_freq[tag] += 1

    return {
        tag: math.log(total_docs / (1 + freq))
        for tag, freq in doc_freq.items()
    }


def detect_noise_tags(
    tag_idf: dict[str, float],
    threshold_percentile: float = _patterns_cfg.tag_noise_percentile,
) -> set[str]:
    """Tags with IDF in the bottom percentile are noise."""
    if not tag_idf:
        return set()

    values = sorted(tag_idf.values())
    threshold_idx = max(0, int(len(values) * threshold_percentile / 100))
    threshold = values[threshold_idx]

    return {tag for tag, idf in tag_idf.items() if idf <= threshold}


# ---------------------------------------------------------------------------
# Tag co-occurrence clustering
# ---------------------------------------------------------------------------

def build_tag_adjacency(
    learnings: list[Learning],
    exclude_tags: set[str],
    tag_idf: dict[str, float],
) -> dict[tuple[int, int], float]:
    """Build IDF-weighted edge weights between learnings sharing tags.

    Returns dict mapping (i, j) pairs (i < j) to summed IDF weight.
    """
    tag_to_indices = _build_tag_index(learnings, exclude_tags)
    return build_edge_weights(tag_to_indices, tag_idf)


def _build_tag_index(
    learnings: list[Learning],
    exclude: set[str],
) -> dict[str, list[int]]:
    """Inverted index: tag -> list of learning indices.

    Tags are deduplicated per learning to prevent duplicate tags
    from inflating edge weights.
    """
    tag_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, learning in enumerate(learnings):
        for tag in set(learning.tags) - exclude:
            tag_to_indices[tag].append(idx)
    return dict(tag_to_indices)


def build_edge_weights(
    tag_to_indices: dict[str, list[int]],
    tag_idf: dict[str, float],
) -> dict[tuple[int, int], float]:
    """Accumulate IDF-weighted edges for all tag co-occurrences.

    Tags with only one learning produce no edges.
    Default IDF weight is 1.0 for tags not in tag_idf.
    """
    edge_weights: dict[tuple[int, int], float] = defaultdict(float)
    for tag, indices in tag_to_indices.items():
        if len(indices) < 2:
            continue
        weight = tag_idf.get(tag, 1.0)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                key = (min(indices[i], indices[j]), max(indices[i], indices[j]))
                edge_weights[key] += weight
    return dict(edge_weights)


def components_from_edges(
    edge_weights: dict[tuple[int, int], float],
    n: int,
    min_weight: float,
    min_component_size: int,
    max_cluster_size: int = _patterns_cfg.max_cluster_size,
) -> list[list[int]]:
    """Find connected components from weighted edges.

    Filters edges by min_weight, finds connected components, drops
    components smaller than min_component_size, and splits oversized ones.
    """
    if not edge_weights:
        return []

    rows, cols, weights = _filter_edges(edge_weights, min_weight)
    if not rows:
        return []

    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)

    raw_components = _group_components(labels, min_component_size)

    clusters: list[list[int]] = []
    for members in raw_components:
        if len(members) <= max_cluster_size:
            clusters.append(members)
        else:
            clusters.extend(
                _split_large_component(members, edge_weights, min_component_size, min_weight)
            )
    return clusters


def _filter_edges(
    edge_weights: dict[tuple[int, int], float],
    min_weight: float,
) -> tuple[list[int], list[int], list[float]]:
    """Filter edges by minimum weight, return symmetric sparse entries."""
    rows, cols, weights = [], [], []
    for (r, c), w in edge_weights.items():
        if w >= min_weight:
            rows.extend([r, c])
            cols.extend([c, r])
            weights.extend([w, w])
    return rows, cols, weights


def _group_components(
    labels: np.ndarray,
    min_size: int,
) -> list[list[int]]:
    """Group indices by component label, filter by min size."""
    components: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        components[label].append(idx)
    return [m for m in components.values() if len(m) >= min_size]


def _split_large_component(
    members: list[int],
    edge_weights: dict[tuple[int, int], float],
    min_size: int,
    base_threshold: float,
) -> list[list[int]]:
    """Split a large component by iteratively raising the edge-weight threshold."""
    member_set = set(members)
    threshold = base_threshold

    for _ in range(10):
        threshold *= 1.5
        sub_edges = {
            (r, c): w
            for (r, c), w in edge_weights.items()
            if r in member_set and c in member_set and w >= threshold
        }

        if not sub_edges:
            return [members]

        idx_map = {m: i for i, m in enumerate(sorted(member_set))}
        reverse_map = {i: m for m, i in idx_map.items()}
        n = len(member_set)

        rows, cols, weights = [], [], []
        for (r, c), w in sub_edges.items():
            rows.extend([idx_map[r], idx_map[c]])
            cols.extend([idx_map[c], idx_map[r]])
            weights.extend([w, w])

        graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
        _, labels = connected_components(graph, directed=False)

        sub_components: dict[int, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            sub_components[label].append(reverse_map[i])

        valid = [m for m in sub_components.values() if len(m) >= min_size]
        if len(valid) > 1:
            return valid

    return [members]


def cluster_by_tags(
    learnings: list[Learning],
    min_cooccurrence: float = _patterns_cfg.min_cooccurrence,
    min_component_size: int = _patterns_cfg.min_component_size,
    exclude_tags: set[str] | None = None,
    tag_idf: dict[str, float] | None = None,
) -> list[list[int]]:
    """Find clusters via IDF-weighted tag co-occurrence graph.

    Builds a weighted graph where nodes are learnings, edges connect
    learnings sharing tags, weights = sum of shared tags' IDF scores.
    """
    if len(learnings) < min_component_size:
        return []

    edge_weights = build_tag_adjacency(
        learnings,
        exclude_tags=exclude_tags or set(),
        tag_idf=tag_idf or {},
    )

    return components_from_edges(
        edge_weights,
        n=len(learnings),
        min_weight=min_cooccurrence,
        min_component_size=min_component_size,
    )


# ---------------------------------------------------------------------------
# Cluster fusion
# ---------------------------------------------------------------------------

def fuse_clusters(
    embedding_clusters: list[list[int]],
    tag_clusters: list[list[int]],
    overlap_threshold: float = _patterns_cfg.overlap_threshold,
) -> list[list[int]]:
    """Merge embedding and tag clusters using Jaccard overlap.

    Tag clusters overlapping an embedding cluster above threshold are merged.
    Non-overlapping tag clusters are kept as-is.
    """
    if not embedding_clusters and not tag_clusters:
        return []
    if not tag_clusters:
        return [list(c) for c in embedding_clusters]
    if not embedding_clusters:
        return [list(c) for c in tag_clusters]

    emb_sets = [set(c) for c in embedding_clusters]
    tag_sets = [set(c) for c in tag_clusters]

    merged = [set(c) for c in embedding_clusters]
    tag_used = _match_tag_to_embedding(tag_sets, emb_sets, merged, overlap_threshold)

    for ti, tag_set in enumerate(tag_sets):
        if not tag_used[ti]:
            merged.append(tag_set)

    return [sorted(c) for c in merged]


def _match_tag_to_embedding(
    tag_sets: list[set[int]],
    emb_sets: list[set[int]],
    merged: list[set[int]],
    overlap_threshold: float,
) -> list[bool]:
    """Match each tag cluster to its best-overlapping embedding cluster."""
    tag_used = [False] * len(tag_sets)

    for ti, tag_set in enumerate(tag_sets):
        best_overlap = 0.0
        best_ei = -1

        for ei, emb_set in enumerate(emb_sets):
            jaccard = _jaccard(tag_set, emb_set)
            if jaccard > best_overlap:
                best_overlap = jaccard
                best_ei = ei

        if best_overlap >= overlap_threshold and best_ei >= 0:
            merged[best_ei] = merged[best_ei] | tag_set
            tag_used[ti] = True

    return tag_used


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union
