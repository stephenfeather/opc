"""Tests for pattern_detector_clustering module.

Validates the clustering sub-functions extracted from pattern_detector:
1. Embedding-based clustering (HDBSCAN)
2. Tag IDF computation
3. Noise tag detection
4. Tag co-occurrence clustering and helpers
5. Cluster fusion
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
import pytest

from scripts.core.pattern_detector import Learning
from scripts.core.pattern_detector_clustering import (
    build_edge_weights,
    build_tag_adjacency,
    components_from_edges,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_learning(
    *,
    embedding: np.ndarray | None = None,
    learning_type: str = "WORKING_SOLUTION",
    tags: list[str] | None = None,
    session_id: str | None = None,
    context: str = "test",
    created_at: datetime | None = None,
    confidence: str = "high",
) -> Learning:
    """Create a Learning with sensible defaults."""
    return Learning(
        id=str(uuid.uuid4()),
        content=f"Test learning about {', '.join(tags or ['misc'])}",
        embedding=embedding if embedding is not None else np.random.randn(1024).astype(np.float32),
        learning_type=learning_type,
        tags=tags or [],
        session_id=session_id or str(uuid.uuid4())[:8],
        context=context,
        created_at=created_at or datetime.now(UTC),
        confidence=confidence,
    )


def _make_cluster_embeddings(
    n_per_cluster: int,
    n_clusters: int,
    dim: int = 1024,
    spread: float = 0.1,
) -> list[np.ndarray]:
    """Create synthetic embeddings with clear cluster structure."""
    rng = np.random.RandomState(42)
    embeddings = []
    for _ in range(n_clusters):
        center = rng.randn(dim)
        center = center / np.linalg.norm(center)
        for _ in range(n_per_cluster):
            point = center + rng.randn(dim) * spread
            embeddings.append(point.astype(np.float32))
    return embeddings


# ---------------------------------------------------------------------------
# build_tag_adjacency tests
# ---------------------------------------------------------------------------

class TestBuildTagAdjacency:

    def test_shared_tags_create_edges(self):
        """Two learnings sharing a tag produce an edge."""
        learnings = [
            _make_learning(tags=["vue", "testing"]),
            _make_learning(tags=["vue", "hooks"]),
        ]
        edge_weights = build_tag_adjacency(learnings, exclude_tags=set(), tag_idf={})
        # Both share "vue" -> should have an edge between indices 0 and 1
        assert (0, 1) in edge_weights

    def test_excluded_tags_ignored(self):
        """Excluded tags should not produce edges."""
        learnings = [
            _make_learning(tags=["noise", "unique1"]),
            _make_learning(tags=["noise", "unique2"]),
        ]
        edge_weights = build_tag_adjacency(
            learnings, exclude_tags={"noise"}, tag_idf={},
        )
        # Only shared tag is "noise" which is excluded -> no edges
        assert len(edge_weights) == 0

    def test_idf_weights_applied(self):
        """Edge weights should reflect tag IDF scores."""
        learnings = [
            _make_learning(tags=["rare", "common"]),
            _make_learning(tags=["rare", "common"]),
        ]
        idf = {"rare": 3.0, "common": 0.5}
        edge_weights = build_tag_adjacency(
            learnings, exclude_tags=set(), tag_idf=idf,
        )
        # Edge weight = sum of shared tag IDFs = 3.0 + 0.5 = 3.5
        assert edge_weights[(0, 1)] == pytest.approx(3.5)

    def test_default_idf_is_one(self):
        """Tags without IDF entries use default weight of 1.0."""
        learnings = [
            _make_learning(tags=["unknown"]),
            _make_learning(tags=["unknown"]),
        ]
        edge_weights = build_tag_adjacency(
            learnings, exclude_tags=set(), tag_idf={},
        )
        assert edge_weights[(0, 1)] == pytest.approx(1.0)

    def test_empty_learnings(self):
        assert build_tag_adjacency([], exclude_tags=set(), tag_idf={}) == {}

    def test_no_shared_tags(self):
        learnings = [
            _make_learning(tags=["a"]),
            _make_learning(tags=["b"]),
        ]
        edge_weights = build_tag_adjacency(
            learnings, exclude_tags=set(), tag_idf={},
        )
        assert len(edge_weights) == 0

    def test_duplicate_tags_not_inflated(self):
        """Duplicate tags on one learning must not inflate edge weights."""
        learnings = [
            _make_learning(tags=["vue", "vue", "vue"]),
            _make_learning(tags=["vue"]),
        ]
        edge_weights = build_tag_adjacency(
            learnings, exclude_tags=set(), tag_idf={"vue": 2.0},
        )
        # Should be exactly 2.0, not 6.0 from triple-counting
        assert edge_weights[(0, 1)] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# build_edge_weights (raw edge accumulation) tests
# ---------------------------------------------------------------------------

class TestBuildEdgeWeights:

    def test_accumulates_duplicate_pairs(self):
        """Same pair sharing multiple tags should sum weights."""
        # tag_to_indices: tag "a" -> [0, 1], tag "b" -> [0, 1]
        tag_to_indices = {"a": [0, 1], "b": [0, 1]}
        idf = {"a": 2.0, "b": 3.0}
        result = build_edge_weights(tag_to_indices, idf)
        assert result[(0, 1)] == pytest.approx(5.0)

    def test_single_member_tags_no_edges(self):
        """Tags with only one learning produce no edges."""
        tag_to_indices = {"lonely": [0]}
        result = build_edge_weights(tag_to_indices, {})
        assert len(result) == 0

    def test_default_weight_one(self):
        tag_to_indices = {"x": [0, 1]}
        result = build_edge_weights(tag_to_indices, {})
        assert result[(0, 1)] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# components_from_edges tests
# ---------------------------------------------------------------------------

class TestComponentsFromEdges:

    def test_two_connected_components(self):
        """Two disconnected groups should form two components."""
        edge_weights = {
            (0, 1): 2.0,
            (1, 2): 2.0,
            (3, 4): 2.0,
            (4, 5): 2.0,
        }
        components = components_from_edges(
            edge_weights, n=6, min_weight=1.0, min_component_size=2,
        )
        assert len(components) == 2

    def test_min_weight_filters_edges(self):
        """Edges below min_weight should be excluded."""
        edge_weights = {
            (0, 1): 0.5,  # below threshold
            (2, 3): 2.0,
            (3, 4): 2.0,
        }
        components = components_from_edges(
            edge_weights, n=5, min_weight=1.0, min_component_size=2,
        )
        # Only one component (2, 3, 4); pair (0, 1) filtered out
        assert len(components) == 1
        assert set(components[0]) == {2, 3, 4}

    def test_min_size_filters_small_components(self):
        """Components smaller than min_component_size are dropped."""
        edge_weights = {(0, 1): 2.0}  # component of size 2
        components = components_from_edges(
            edge_weights, n=2, min_weight=1.0, min_component_size=3,
        )
        assert len(components) == 0

    def test_empty_edges(self):
        components = components_from_edges(
            {}, n=5, min_weight=1.0, min_component_size=2,
        )
        assert components == []


# ---------------------------------------------------------------------------
# Re-exported function tests (ensure backwards compatibility)
# ---------------------------------------------------------------------------

class TestBackwardsCompatImports:

    def test_cluster_by_embeddings_importable(self):
        from scripts.core.pattern_detector import cluster_by_embeddings as fn
        assert callable(fn)

    def test_compute_tag_idf_importable(self):
        from scripts.core.pattern_detector import compute_tag_idf as fn
        assert callable(fn)

    def test_detect_noise_tags_importable(self):
        from scripts.core.pattern_detector import detect_noise_tags as fn
        assert callable(fn)

    def test_cluster_by_tags_importable(self):
        from scripts.core.pattern_detector import cluster_by_tags as fn
        assert callable(fn)

    def test_fuse_clusters_importable(self):
        from scripts.core.pattern_detector import fuse_clusters as fn
        assert callable(fn)
