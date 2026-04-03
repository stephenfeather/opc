"""Tests for cross-session pattern detection engine.

Validates:
1. HDBSCAN clustering finds groups in synthetic embeddings
2. Tag IDF correctly identifies noise tags
3. Tag co-occurrence clustering groups structurally related learnings
4. Cluster fusion merges overlapping and keeps disjoint clusters
5. Pattern classification assigns correct types
6. Label generation produces readable summaries
7. Confidence scoring differentiates strong from weak patterns
8. Full pipeline detect_patterns() works end-to-end
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio

from scripts.core.pattern_detector import (  # noqa: E402
    Learning,
    classify_pattern_heuristic,
    cluster_by_embeddings,
    cluster_by_tags,
    compute_centroid,
    compute_cohesion,
    compute_confidence,
    compute_distances,
    compute_diversity,
    compute_size_score,
    compute_tag_idf,
    compute_temporal_span,
    detect_noise_tags,
    detect_patterns,
    fuse_clusters,
    generate_label,
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
    """Create synthetic embeddings with clear cluster structure.

    Each cluster is centered around a random unit vector,
    with members scattered by `spread` noise.
    """
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
# Embedding clustering tests
# ---------------------------------------------------------------------------

class TestClusterByEmbeddings:

    def test_finds_known_clusters(self):
        """Synthetic embeddings with 3 clear clusters should be recovered."""
        embeddings = _make_cluster_embeddings(n_per_cluster=10, n_clusters=3, spread=0.05)
        learnings = [
            _make_learning(embedding=e) for e in embeddings
        ]
        clusters = cluster_by_embeddings(learnings, min_cluster_size=5, min_samples=3)
        # Should find at least 2 clusters (HDBSCAN may merge similar ones)
        assert len(clusters) >= 2
        # Each cluster should have at least min_cluster_size members
        for c in clusters:
            assert len(c) >= 5

    def test_noise_returns_no_clusters(self):
        """Random embeddings with high variance should produce few or no clusters."""
        rng = np.random.RandomState(99)
        learnings = [
            _make_learning(embedding=rng.randn(1024).astype(np.float32))
            for _ in range(20)
        ]
        clusters = cluster_by_embeddings(learnings, min_cluster_size=5, min_samples=3)
        # Random data: HDBSCAN should find 0 or maybe 1 spurious cluster
        assert len(clusters) <= 1

    def test_empty_input(self):
        clusters = cluster_by_embeddings([], min_cluster_size=5)
        assert clusters == []

    def test_too_few_learnings(self):
        learnings = [_make_learning() for _ in range(3)]
        clusters = cluster_by_embeddings(learnings, min_cluster_size=5)
        assert clusters == []

    def test_zero_norm_embedding_handled(self):
        """A zero-vector embedding should not crash."""
        learnings = [_make_learning(embedding=np.zeros(1024, dtype=np.float32))]
        learnings.extend([_make_learning() for _ in range(10)])
        # Should not raise
        cluster_by_embeddings(learnings, min_cluster_size=5)


# ---------------------------------------------------------------------------
# Tag IDF tests
# ---------------------------------------------------------------------------

class TestTagIdf:

    def test_rare_tag_high_idf(self):
        all_tags = {
            "a": ["common", "rare_tag"],
            "b": ["common"],
            "c": ["common"],
        }
        idf = compute_tag_idf(all_tags, total_docs=3)
        assert idf["rare_tag"] > idf["common"]

    def test_common_tag_low_idf(self):
        all_tags = {str(i): ["ubiquitous"] for i in range(100)}
        all_tags["special"] = ["ubiquitous", "rare"]
        idf = compute_tag_idf(all_tags, total_docs=101)
        assert idf["rare"] > idf["ubiquitous"]

    def test_empty_tags(self):
        idf = compute_tag_idf({}, total_docs=0)
        assert idf == {}


class TestNoiseDetection:

    def test_identifies_bottom_percentile(self):
        tag_idf = {"common": 0.1, "medium": 1.0, "rare": 3.0}
        noise = detect_noise_tags(tag_idf, threshold_percentile=40)
        assert "common" in noise
        assert "rare" not in noise

    def test_empty_idf(self):
        assert detect_noise_tags({}) == set()

    def test_single_tag(self):
        noise = detect_noise_tags({"only": 1.0}, threshold_percentile=10)
        # Single tag at percentile 0 -- always at or below threshold
        assert "only" in noise


# ---------------------------------------------------------------------------
# Tag co-occurrence clustering tests
# ---------------------------------------------------------------------------

class TestClusterByTags:

    def test_shared_tags_cluster_together(self):
        """Learnings sharing tags should be grouped."""
        session = "s1"
        learnings = []
        # Group A: 6 learnings sharing "vue", "testing"
        for i in range(6):
            learnings.append(_make_learning(
                tags=["vue", "testing"], session_id=session,
            ))
        # Group B: 6 learnings sharing "hooks", "daemon"
        for i in range(6):
            learnings.append(_make_learning(
                tags=["hooks", "daemon"], session_id=session,
            ))
        # 2 loners with unique tags
        learnings.append(_make_learning(tags=["unique1"], session_id=session))
        learnings.append(_make_learning(tags=["unique2"], session_id=session))

        clusters = cluster_by_tags(
            learnings, min_cooccurrence=0.5, min_component_size=5,
        )
        assert len(clusters) >= 2

    def test_excludes_noise_tags(self):
        """Excluded tags should not create spurious connections."""
        learnings = []
        for i in range(10):
            learnings.append(_make_learning(
                tags=["high-frequency-noise", f"unique_{i}"],
            ))
        clusters = cluster_by_tags(
            learnings, exclude_tags={"high-frequency-noise"}, min_component_size=5,
        )
        # Without noise tag, each learning has unique tags only -> no clusters
        assert len(clusters) == 0

    def test_empty_input(self):
        assert cluster_by_tags([], min_component_size=5) == []

    def test_too_few_learnings(self):
        learnings = [_make_learning(tags=["a"]) for _ in range(3)]
        assert cluster_by_tags(learnings, min_component_size=5) == []


# ---------------------------------------------------------------------------
# Fusion tests
# ---------------------------------------------------------------------------

class TestFuseClusters:

    def test_overlapping_clusters_merge(self):
        emb = [[0, 1, 2, 3, 4]]
        tag = [[2, 3, 4, 5, 6]]
        fused = fuse_clusters(emb, tag, overlap_threshold=0.2)
        # Should merge into one cluster containing 0-6
        assert len(fused) == 1
        assert set(fused[0]) == {0, 1, 2, 3, 4, 5, 6}

    def test_disjoint_clusters_kept_separate(self):
        emb = [[0, 1, 2]]
        tag = [[3, 4, 5]]
        fused = fuse_clusters(emb, tag, overlap_threshold=0.3)
        assert len(fused) == 2

    def test_empty_embedding_clusters(self):
        tag = [[0, 1, 2]]
        fused = fuse_clusters([], tag)
        assert len(fused) == 1

    def test_empty_tag_clusters(self):
        emb = [[0, 1, 2]]
        fused = fuse_clusters(emb, [])
        assert len(fused) == 1

    def test_both_empty(self):
        assert fuse_clusters([], []) == []


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassifyPattern:

    # -- Rule 1: anti_pattern --

    def test_anti_pattern_all_failed(self):
        # All FAILED_APPROACH -> anti_pattern (rule 1, pure case)
        members = [
            _make_learning(learning_type="FAILED_APPROACH", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC))
            for _ in range(5)
        ]
        assert classify_pattern_heuristic(members) == "anti_pattern"

    def test_anti_pattern_majority_failed(self):
        # >60% FAILED_APPROACH -> anti_pattern (rule 1, majority case)
        # 4/5 = 80% FAILED_APPROACH
        members = [
            _make_learning(learning_type="FAILED_APPROACH", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC))
            for _ in range(4)
        ]
        members.append(
            _make_learning(learning_type="WORKING_SOLUTION", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC))
        )
        assert classify_pattern_heuristic(members) == "anti_pattern"

    # -- Rule 2: problem_solution --

    def test_problem_solution(self):
        # Both ERROR_FIX and WORKING_SOLUTION present, combined >= 40%
        # 2 ERROR_FIX + 1 WORKING_SOLUTION out of 3 = 100% combined
        # Does NOT match anti_pattern (no FAILED_APPROACH)
        members = [
            _make_learning(learning_type="ERROR_FIX", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="ERROR_FIX", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="WORKING_SOLUTION", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC)),
        ]
        assert classify_pattern_heuristic(members) == "problem_solution"

    def test_problem_solution_at_40_percent_boundary(self):
        # Exactly 40% combined ERROR_FIX + WORKING_SOLUTION -> problem_solution
        # 2 ERROR_FIX + 2 WORKING_SOLUTION + 6 CODEBASE_PATTERN = 4/10 = 40%
        # Does NOT match anti_pattern (no FAILED_APPROACH)
        members = [
            _make_learning(learning_type="ERROR_FIX", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="ERROR_FIX", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="WORKING_SOLUTION", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="WORKING_SOLUTION", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC)),
        ] + [
            _make_learning(learning_type="CODEBASE_PATTERN", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC))
            for _ in range(6)
        ]
        assert classify_pattern_heuristic(members) == "problem_solution"

    def test_problem_solution_requires_both_types(self):
        # Pure ERROR_FIX (no WORKING_SOLUTION) should NOT match problem_solution
        # 3 ERROR_FIX + 0 WORKING_SOLUTION -> rule 2 requires both present
        # Falls through: 1 session, 1 context, 100% one type -> tool_cluster (rule 4)
        members = [
            _make_learning(learning_type="ERROR_FIX", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC))
            for _ in range(3)
        ]
        assert classify_pattern_heuristic(members) == "tool_cluster"

    # -- Rule 3: expertise --

    def test_expertise(self):
        # 60% recent (last 30d) + >=60% one type + >=4 contexts + >=3 sessions
        # 5 members, all recent, all WORKING_SOLUTION, 5 contexts, 5 sessions
        # Does NOT match anti_pattern (no FAILED_APPROACH)
        # Does NOT match problem_solution (no ERROR_FIX+WORKING_SOLUTION pair)
        now = datetime(2026, 3, 31, tzinfo=UTC)
        members = [
            _make_learning(
                learning_type="WORKING_SOLUTION",
                session_id=f"s{i}",
                context=f"ctx{i}",
                created_at=now - timedelta(days=i),
            )
            for i in range(5)
        ]
        assert classify_pattern_heuristic(members, reference_time=now) == "expertise"

    def test_expertise_needs_3_sessions(self):
        # Only 2 sessions -> should NOT match expertise (needs >=3)
        # 5 members, all recent, all same type, 4 contexts, but only 2 sessions
        # Falls through to tool_cluster (<=3 contexts? no, 4 contexts)
        # Then cross_project (4 contexts >= 4)
        now = datetime(2026, 3, 31, tzinfo=UTC)
        members = [
            _make_learning(learning_type="WORKING_SOLUTION", session_id="s1",
                           context=f"ctx{i}", created_at=now - timedelta(days=1))
            for i in range(4)
        ] + [
            _make_learning(learning_type="WORKING_SOLUTION", session_id="s2",
                           context="ctx4", created_at=now - timedelta(days=1))
        ]
        result = classify_pattern_heuristic(members, reference_time=now)
        assert result == "cross_project"

    def test_expertise_needs_4_contexts(self):
        # Only 3 contexts -> should NOT match expertise (needs >=4)
        # 6 members, all recent, all same type, 3 contexts, 3 sessions
        # Falls through to tool_cluster (100% one type, <=3 contexts -> rule 4)
        now = datetime(2026, 3, 31, tzinfo=UTC)
        members = [
            _make_learning(learning_type="WORKING_SOLUTION", session_id=f"s{i}",
                           context=f"ctx{i}", created_at=now - timedelta(days=1))
            for i in range(3)
        ] + [
            _make_learning(learning_type="WORKING_SOLUTION", session_id=f"s{i}",
                           context=f"ctx{i}", created_at=now - timedelta(days=1))
            for i in range(3)
        ]
        result = classify_pattern_heuristic(members, reference_time=now)
        assert result != "expertise"
        assert result == "tool_cluster"

    # -- Rule 4: tool_cluster --

    def test_tool_cluster_homogeneous_narrow(self):
        # >=70% same type + <=3 contexts -> tool_cluster (rule 4)
        # 5/5 = 100% CODEBASE_PATTERN, 1 context
        # Does NOT match anti_pattern (not FAILED_APPROACH)
        # Does NOT match problem_solution (no ERROR_FIX/WORKING_SOLUTION pair)
        # Does NOT match expertise (only 1 context < 4)
        members = [
            _make_learning(learning_type="CODEBASE_PATTERN", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC))
            for _ in range(5)
        ]
        assert classify_pattern_heuristic(members) == "tool_cluster"

    def test_tool_cluster_boundary_at_70_percent(self):
        # Exactly 70% same type + <=3 contexts -> tool_cluster
        # 7/10 = 70% CODEBASE_PATTERN, 2 contexts
        # Does NOT match problem_solution (no ERROR_FIX+WORKING_SOLUTION pair)
        members = [
            _make_learning(learning_type="CODEBASE_PATTERN", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC))
            for _ in range(7)
        ] + [
            _make_learning(learning_type="ARCHITECTURAL_DECISION", session_id="s1",
                           context="ctx2", created_at=datetime(2026, 1, 1, tzinfo=UTC))
            for _ in range(3)
        ]
        assert classify_pattern_heuristic(members) == "tool_cluster"

    # -- Rule 5: cross_project --

    def test_cross_project_4_contexts(self):
        # 4+ contexts -> cross_project (rule 5, context path)
        # Does NOT match anti_pattern (no FAILED_APPROACH)
        # Does NOT match problem_solution (no ERROR_FIX+WORKING_SOLUTION pair)
        # Does NOT match expertise (not enough recent, single session)
        # Does NOT match tool_cluster (4 contexts > 3)
        members = [
            _make_learning(learning_type="CODEBASE_PATTERN", session_id="s1",
                           context=f"project{i}", created_at=datetime(2025, 1, 1, tzinfo=UTC))
            for i in range(4)
        ]
        assert classify_pattern_heuristic(
            members, reference_time=datetime(2026, 3, 31, tzinfo=UTC)
        ) == "cross_project"

    def test_cross_project_ratio_path(self):
        # >=5 members AND session/member > 0.7 -> cross_project (rule 5, ratio path)
        # 5 members, 5 sessions, ratio = 1.0, but only 3 contexts (<=3)
        # Does NOT match tool_cluster because types are mixed (<70% any single type)
        # 2 CODEBASE_PATTERN + 2 ARCHITECTURAL_DECISION + 1 USER_PREFERENCE = no type >=70%
        members = [
            _make_learning(learning_type="CODEBASE_PATTERN", session_id="s0",
                           context="ctx1", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="CODEBASE_PATTERN", session_id="s1",
                           context="ctx2", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="ARCHITECTURAL_DECISION", session_id="s2",
                           context="ctx3", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="ARCHITECTURAL_DECISION", session_id="s3",
                           context="ctx3", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="USER_PREFERENCE", session_id="s4",
                           context="ctx3", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
        ]
        assert classify_pattern_heuristic(
            members, reference_time=datetime(2026, 3, 31, tzinfo=UTC)
        ) == "cross_project"

    def test_cross_project_small_cluster_no_ratio(self):
        # 3 members, 3 sessions, ratio = 1.0 but <5 members -> ratio path blocked
        # 3 contexts <= 3 -> context path blocked (needs >=4)
        # Mixed types (no type >=70%) -> tool_cluster blocked too
        # Falls to default tool_cluster
        members = [
            _make_learning(learning_type="CODEBASE_PATTERN", session_id="s0",
                           context="ctx1", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="ARCHITECTURAL_DECISION", session_id="s1",
                           context="ctx2", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="USER_PREFERENCE", session_id="s2",
                           context="ctx3", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
        ]
        result = classify_pattern_heuristic(
            members, reference_time=datetime(2026, 3, 31, tzinfo=UTC)
        )
        assert result == "tool_cluster"

    def test_4_context_gap_closed(self):
        # A cluster with exactly 4 contexts should be classified (not fall through)
        # 4 contexts, heterogeneous types -> not tool_cluster (4 > 3 contexts)
        # 4 contexts >= 4 -> cross_project
        members = [
            _make_learning(learning_type="CODEBASE_PATTERN", session_id="s1",
                           context="ctx1", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="ARCHITECTURAL_DECISION", session_id="s1",
                           context="ctx2", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="USER_PREFERENCE", session_id="s1",
                           context="ctx3", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _make_learning(learning_type="CODEBASE_PATTERN", session_id="s1",
                           context="ctx4", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
        ]
        result = classify_pattern_heuristic(
            members, reference_time=datetime(2026, 3, 31, tzinfo=UTC)
        )
        # Must not be default tool_cluster; should be cross_project
        assert result == "cross_project"

    # -- Default / empty --

    def test_default_tool_cluster(self):
        # Same session, same context, same type -> falls through to default
        # Does NOT match any rule above: no FAILED_APPROACH, no ERROR_FIX+WORKING_SOLUTION,
        # not enough contexts for expertise or cross_project
        # 100% one type + 1 context -> matches tool_cluster (rule 4)
        members = [
            _make_learning(learning_type="WORKING_SOLUTION", session_id="s1", context="ctx1",
                           created_at=datetime(2026, 1, 1, tzinfo=UTC))
            for _ in range(5)
        ]
        assert classify_pattern_heuristic(members) == "tool_cluster"

    def test_empty_members(self):
        assert classify_pattern_heuristic([]) == "tool_cluster"


# ---------------------------------------------------------------------------
# Label generation tests
# ---------------------------------------------------------------------------

class TestGenerateLabel:

    def test_includes_top_tags(self):
        members = [
            _make_learning(tags=["vue", "testing"]),
            _make_learning(tags=["vue", "vitest"]),
        ]
        label = generate_label(members, "tool_cluster")
        assert "vue" in label

    def test_includes_session_count(self):
        members = [
            _make_learning(tags=["a"], session_id="s1"),
            _make_learning(tags=["a"], session_id="s2"),
        ]
        label = generate_label(members, "tool_cluster")
        assert "2 sessions" in label

    def test_empty_members(self):
        label = generate_label([], "tool_cluster")
        assert "Empty" in label

    def test_duplicate_tags_not_inflated(self):
        """Duplicate tags on one learning should not skew tag ranking."""
        members = [
            _make_learning(tags=["rare", "rare", "rare", "rare", "rare"]),
            _make_learning(tags=["common", "rare"]),
        ]
        label_with_dupes = generate_label(members, "tool_cluster")
        # "rare" appears in both learnings (count=2), "common" in one (count=1)
        # Without dedup, "rare" would get count=6 - but ranking should be the same
        members_deduped = [
            _make_learning(tags=["rare"]),
            _make_learning(tags=["common", "rare"]),
        ]
        label_without_dupes = generate_label(members_deduped, "tool_cluster")
        assert label_with_dupes == label_without_dupes


# ---------------------------------------------------------------------------
# Confidence scoring tests
# ---------------------------------------------------------------------------

class TestComputeConfidence:

    def test_tight_cluster_high_cohesion(self):
        """Members near the centroid should score high cohesion."""
        center = np.random.randn(1024).astype(np.float32)
        center = center / np.linalg.norm(center)
        members = [
            _make_learning(
                embedding=(center + np.random.randn(1024) * 0.01).astype(np.float32),
                session_id=f"s{i}",
                created_at=datetime.now(UTC) - timedelta(days=i * 5),
            )
            for i in range(10)
        ]
        centroid = compute_centroid(members)
        score = compute_confidence(members, centroid)
        assert score > 0.5

    def test_diverse_sessions_boost(self):
        """More distinct sessions should increase confidence."""
        center = np.ones(1024, dtype=np.float32)
        members_few = [
            _make_learning(embedding=center.copy(), session_id="s1")
            for _ in range(5)
        ]
        members_many = [
            _make_learning(embedding=center.copy(), session_id=f"s{i}")
            for i in range(5)
        ]
        centroid = compute_centroid(members_few)
        score_few = compute_confidence(members_few, centroid)
        score_many = compute_confidence(members_many, centroid)
        assert score_many > score_few

    def test_single_member(self):
        m = _make_learning()
        centroid = compute_centroid([m])
        score = compute_confidence([m], centroid)
        # Single member: low diversity, low temporal span, low size
        assert 0.0 <= score <= 1.0

    def test_empty_members(self):
        assert compute_confidence([], np.zeros(1024)) == 0.0


# ---------------------------------------------------------------------------
# Confidence sub-component tests
# ---------------------------------------------------------------------------

class TestComputeCohesion:

    def test_identical_embeddings_perfect_cohesion(self):
        """All members identical to centroid -> cohesion = 1.0."""
        emb = np.ones(1024, dtype=np.float32)
        members = [_make_learning(embedding=emb.copy()) for _ in range(5)]
        centroid = np.ones(1024, dtype=np.float32)
        assert compute_cohesion(members, centroid) == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_embeddings_low_cohesion(self):
        """Members orthogonal to centroid -> cohesion near 0."""
        centroid = np.zeros(1024, dtype=np.float32)
        centroid[0] = 1.0
        emb = np.zeros(1024, dtype=np.float32)
        emb[1] = 1.0
        members = [_make_learning(embedding=emb.copy())]
        result = compute_cohesion(members, centroid)
        assert result < 0.1

    def test_zero_norm_centroid_safe(self):
        """Zero centroid should not crash."""
        members = [_make_learning()]
        result = compute_cohesion(members, np.zeros(1024, dtype=np.float32))
        assert 0.0 <= result <= 1.0

    def test_empty_members_returns_zero(self):
        """Empty members list should return 0.0, not crash."""
        assert compute_cohesion([], np.zeros(1024, dtype=np.float32)) == 0.0


class TestComputeDiversity:

    def test_many_sessions_high_diversity(self):
        members = [_make_learning(session_id=f"s{i}") for i in range(5)]
        assert compute_diversity(members) == pytest.approx(1.0)

    def test_single_session_low_diversity(self):
        members = [_make_learning(session_id="s1") for _ in range(5)]
        assert compute_diversity(members) == pytest.approx(0.2)

    def test_empty_returns_zero(self):
        assert compute_diversity([]) == 0.0


class TestComputeTemporalSpan:

    def test_two_week_span(self):
        members = [
            _make_learning(created_at=datetime(2026, 1, 1, tzinfo=UTC)),
            _make_learning(created_at=datetime(2026, 1, 15, tzinfo=UTC)),
        ]
        assert compute_temporal_span(members) == pytest.approx(1.0)

    def test_same_day_zero_span(self):
        members = [
            _make_learning(created_at=datetime(2026, 1, 1, tzinfo=UTC)),
            _make_learning(created_at=datetime(2026, 1, 1, tzinfo=UTC)),
        ]
        assert compute_temporal_span(members) == pytest.approx(0.0)

    def test_single_member(self):
        members = [_make_learning()]
        assert compute_temporal_span(members) == 0.0

    def test_empty_returns_zero(self):
        assert compute_temporal_span([]) == 0.0


class TestComputeSizeScore:

    def test_16_members_max_score(self):
        # log2(16) / 4 = 4/4 = 1.0
        assert compute_size_score(16) == pytest.approx(1.0)

    def test_single_member(self):
        # log2(1) / 4 = 0/4 = 0.0
        assert compute_size_score(1) == pytest.approx(0.0)

    def test_zero_returns_zero(self):
        assert compute_size_score(0) == 0.0

    def test_large_clamped_to_one(self):
        assert compute_size_score(1000) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Distance computation tests
# ---------------------------------------------------------------------------

class TestComputeDistances:

    def test_identical_embedding_zero_distance(self):
        emb = np.random.randn(1024).astype(np.float32)
        m = _make_learning(embedding=emb.copy())
        distances = compute_distances([m], emb)
        assert distances[m.id] < 0.01

    def test_orthogonal_embedding_high_distance(self):
        e1 = np.zeros(1024, dtype=np.float32)
        e1[0] = 1.0
        e2 = np.zeros(1024, dtype=np.float32)
        e2[1] = 1.0
        m = _make_learning(embedding=e1)
        distances = compute_distances([m], e2)
        assert distances[m.id] > 0.9


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestDetectPatterns:

    def test_end_to_end_with_clear_clusters(self):
        """Synthetic data with clear clusters should produce patterns."""
        embeddings = _make_cluster_embeddings(n_per_cluster=8, n_clusters=3, spread=0.05)
        learnings = []
        cluster_tags = [["vue", "testing"], ["hooks", "daemon"], ["mcp", "api"]]
        for i, emb in enumerate(embeddings):
            cluster_idx = i // 8
            learnings.append(_make_learning(
                embedding=emb,
                tags=cluster_tags[cluster_idx],
                session_id=f"s{i % 5}",
                context=f"project{cluster_idx}",
                created_at=datetime.now(UTC) - timedelta(days=i),
            ))

        patterns = asyncio.run(detect_patterns(
            learnings, min_cluster_size=5, min_samples=3, min_confidence=0.1,
        ))
        assert len(patterns) >= 1
        for p in patterns:
            assert p.confidence > 0
            assert len(p.member_ids) >= 5
            assert p.representative_id in p.member_ids
            assert p.label
            assert p.pattern_type in (
                "tool_cluster", "problem_solution", "cross_project",
                "expertise", "anti_pattern",
            )

    def test_too_few_learnings_returns_empty(self):
        learnings = [_make_learning() for _ in range(3)]
        assert asyncio.run(detect_patterns(learnings)) == []

    def test_patterns_sorted_by_confidence(self):
        """Output should be sorted by confidence descending."""
        embeddings = _make_cluster_embeddings(n_per_cluster=10, n_clusters=3, spread=0.05)
        learnings = [
            _make_learning(
                embedding=e,
                session_id=f"s{i % 6}",
                created_at=datetime.now(UTC) - timedelta(days=i),
            )
            for i, e in enumerate(embeddings)
        ]
        patterns = asyncio.run(detect_patterns(learnings, min_cluster_size=5, min_confidence=0.0))
        if len(patterns) >= 2:
            for i in range(len(patterns) - 1):
                assert patterns[i].confidence >= patterns[i + 1].confidence
