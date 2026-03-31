"""Tests for contextual reranker.

Validates that:
1. Signal functions compute correct scores for various inputs
2. Per-mode score calibration normalizes scores to [0,1]
3. rerank() combines signals correctly and reorders results
4. All functions handle missing data gracefully
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.reranker import (  # noqa: E402
    RecallContext,
    RerankerConfig,
    _cosine_similarity,
    calibrate_score,
    compute_type_centroids,
    confidence_score,
    infer_query_type,
    load_centroids,
    pattern_score,
    project_match,
    recall_score,
    recency_score,
    rerank,
    save_centroids,
    tag_overlap,
    type_match,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    *,
    similarity: float = 0.5,
    project: str | None = None,
    learning_type: str | None = None,
    confidence: str | None = None,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
    recall_count: int | None = None,
) -> dict:
    """Build a minimal result dict for testing."""
    metadata: dict = {"type": "session_learning"}
    if learning_type is not None:
        metadata["learning_type"] = learning_type
    if confidence is not None:
        metadata["confidence"] = confidence
    if tags is not None:
        metadata["tags"] = tags
    if project is not None:
        metadata["project"] = project

    result: dict = {
        "id": "test-id",
        "session_id": "test-session",
        "content": "test content",
        "metadata": metadata,
        "similarity": similarity,
    }
    if created_at is not None:
        result["created_at"] = created_at
    if recall_count is not None:
        result["recall_count"] = recall_count
    return result


# ---------------------------------------------------------------------------
# Signal Function Tests: project_match
# ---------------------------------------------------------------------------

class TestProjectMatch:
    def test_project_match_exact(self):
        result = _make_result(project="opc")
        ctx = RecallContext(project="opc")
        assert project_match(result, ctx) == 1.0

    def test_project_match_partial(self):
        result = _make_result(project="opc-memory")
        ctx = RecallContext(project="opc")
        assert project_match(result, ctx) == 0.5

    def test_project_match_none(self):
        result = _make_result(project="other-project")
        ctx = RecallContext(project="opc")
        assert project_match(result, ctx) == 0.0

    def test_project_match_missing_metadata(self):
        result = _make_result()  # no project in metadata
        ctx = RecallContext(project="opc")
        assert project_match(result, ctx) == 0.0

    def test_project_match_no_ctx_project(self):
        result = _make_result(project="opc")
        ctx = RecallContext()  # no project in context
        assert project_match(result, ctx) == 0.0


# ---------------------------------------------------------------------------
# Signal Function Tests: recency_score
# ---------------------------------------------------------------------------

class TestRecencyScore:
    def test_recency_fresh(self):
        now = datetime.now(UTC)
        result = _make_result(created_at=now - timedelta(days=1))
        ctx = RecallContext(now=now)
        score = recency_score(result, ctx)
        assert abs(score - math.exp(-1 / 45)) < 0.01

    def test_recency_old(self):
        now = datetime.now(UTC)
        result = _make_result(created_at=now - timedelta(days=90))
        ctx = RecallContext(now=now)
        score = recency_score(result, ctx)
        assert abs(score - math.exp(-90 / 45)) < 0.01

    def test_recency_missing_timestamp(self):
        result = _make_result()  # no created_at
        ctx = RecallContext()
        assert recency_score(result, ctx) == 0.5


# ---------------------------------------------------------------------------
# Signal Function Tests: confidence_score
# ---------------------------------------------------------------------------

class TestConfidenceScore:
    def test_confidence_high(self):
        result = _make_result(confidence="high")
        assert confidence_score(result) == 1.0

    def test_confidence_medium(self):
        result = _make_result(confidence="medium")
        assert confidence_score(result) == 0.6

    def test_confidence_low(self):
        result = _make_result(confidence="low")
        assert confidence_score(result) == 0.2

    def test_confidence_none(self):
        result = _make_result()  # no confidence
        assert confidence_score(result) == 0.5


# ---------------------------------------------------------------------------
# Signal Function Tests: recall_score
# ---------------------------------------------------------------------------

class TestRecallScore:
    def test_recall_score_zero(self):
        result = _make_result(recall_count=0)
        assert recall_score(result) == 0.0

    def test_recall_score_moderate(self):
        result = _make_result(recall_count=3)
        expected = min(1.0, math.log2(1 + 3) / 4)
        assert abs(recall_score(result) - expected) < 0.01

    def test_recall_score_missing(self):
        result = _make_result()  # no recall_count
        assert recall_score(result) == 0.0


# ---------------------------------------------------------------------------
# Signal Function Tests: type_match
# ---------------------------------------------------------------------------

class TestTypeMatch:
    def test_type_match_with_probabilities(self):
        result = _make_result(learning_type="WORKING_SOLUTION")
        ctx = RecallContext(
            type_probabilities={
                "WORKING_SOLUTION": 0.7,
                "ERROR_FIX": 0.2,
                "CODEBASE_PATTERN": 0.1,
            }
        )
        assert type_match(result, ctx) == 0.7

    def test_type_match_no_probabilities(self):
        result = _make_result(learning_type="WORKING_SOLUTION")
        ctx = RecallContext()
        assert type_match(result, ctx) == 0.5

    def test_type_match_missing_type_in_result(self):
        result = _make_result()  # no learning_type
        ctx = RecallContext(
            type_probabilities={"WORKING_SOLUTION": 0.7}
        )
        assert type_match(result, ctx) == 0.0


# ---------------------------------------------------------------------------
# Signal Function Tests: tag_overlap
# ---------------------------------------------------------------------------

class TestTagOverlap:
    def test_tag_overlap_full(self):
        result = _make_result(tags=["hooks", "typescript"])
        ctx = RecallContext(tags_hint=["hooks", "typescript"])
        assert tag_overlap(result, ctx) == 1.0

    def test_tag_overlap_partial(self):
        result = _make_result(tags=["hooks", "typescript", "build"])
        ctx = RecallContext(tags_hint=["hooks", "python"])
        score = tag_overlap(result, ctx)
        # intersection={"hooks"}, union={"hooks","typescript","build","python"}
        assert abs(score - 1 / 4) < 0.01

    def test_tag_overlap_disjoint(self):
        result = _make_result(tags=["hooks", "typescript"])
        ctx = RecallContext(tags_hint=["python", "django"])
        assert tag_overlap(result, ctx) == 0.0

    def test_tag_overlap_empty(self):
        result = _make_result(tags=[])
        ctx = RecallContext(tags_hint=["hooks"])
        assert tag_overlap(result, ctx) == 0.0

    def test_tag_overlap_no_hint(self):
        result = _make_result(tags=["hooks"])
        ctx = RecallContext()  # no tags_hint
        assert tag_overlap(result, ctx) == 0.0


# ---------------------------------------------------------------------------
# Calibration Tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibrate_vector(self):
        score = calibrate_score(0.5, "vector", rank=0, total=5)
        assert abs(score - 0.75) < 0.001

    def test_calibrate_rrf(self):
        # 0.017 * 60 = 1.02, clamped to 1.0
        score = calibrate_score(0.017, "hybrid_rrf", rank=0, total=5)
        assert score == 1.0

    def test_calibrate_rrf_normal(self):
        # 0.01 * 60 = 0.6
        score = calibrate_score(0.01, "hybrid_rrf", rank=0, total=5)
        assert abs(score - 0.6) < 0.001

    def test_calibrate_bm25_squash(self):
        # score / (score + 1.0) with score=1.0 => 0.5
        score = calibrate_score(1.0, "text", rank=0, total=5)
        assert abs(score - 0.5) < 0.001

    def test_calibrate_sqlite_squash(self):
        score = calibrate_score(2.0, "sqlite", rank=0, total=5)
        # 2.0 / (2.0 + 1.0) = 0.667
        assert abs(score - 2 / 3) < 0.001

    def test_calibrate_unknown_rank_fallback(self):
        # rank=2, total=5 => 1 - (2/5) = 0.6
        score = calibrate_score(999.0, None, rank=2, total=5)
        assert abs(score - 0.6) < 0.001

    def test_calibrate_single_result(self):
        # rank=0, total=1 => 1 - (0/1) = 1.0
        score = calibrate_score(0.01, None, rank=0, total=1)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Integration Tests: rerank
# ---------------------------------------------------------------------------

class TestRerank:
    def test_rerank_reorders_by_project(self):
        """A result matching the project should rise to the top."""
        results = [
            _make_result(similarity=0.8, project="other"),
            _make_result(similarity=0.7, project="opc"),
        ]
        # Give the second result a distinct id so we can find it
        results[0]["id"] = "no-match"
        results[1]["id"] = "match"

        ctx = RecallContext(
            project="opc",
            retrieval_mode="vector",
        )
        config = RerankerConfig(project_weight=0.15)
        ranked = rerank(results, ctx, config=config, k=5)
        assert ranked[0]["id"] == "match"

    def test_rerank_preserves_order_no_context(self):
        """With empty context, order is mostly by raw score."""
        results = [
            _make_result(similarity=0.9),
            _make_result(similarity=0.5),
            _make_result(similarity=0.3),
        ]
        results[0]["id"] = "a"
        results[1]["id"] = "b"
        results[2]["id"] = "c"

        ctx = RecallContext(retrieval_mode="vector")
        ranked = rerank(results, ctx, k=5)
        assert ranked[0]["id"] == "a"
        assert ranked[1]["id"] == "b"
        assert ranked[2]["id"] == "c"

    def test_rerank_trims_to_k(self):
        results = [_make_result(similarity=0.5 + i * 0.01) for i in range(10)]
        ctx = RecallContext(retrieval_mode="vector")
        ranked = rerank(results, ctx, k=3)
        assert len(ranked) == 3

    def test_rerank_adds_details(self):
        results = [_make_result(similarity=0.5)]
        ctx = RecallContext(retrieval_mode="vector")
        ranked = rerank(results, ctx, k=5)
        assert "final_score" in ranked[0]
        assert "rerank_details" in ranked[0]
        assert isinstance(ranked[0]["rerank_details"], dict)

    def test_rerank_empty_results(self):
        ctx = RecallContext()
        assert rerank([], ctx, k=5) == []

    def test_rerank_none_config(self):
        """rerank with config=None should use defaults."""
        results = [_make_result(similarity=0.5)]
        ctx = RecallContext(retrieval_mode="vector")
        ranked = rerank(results, ctx, config=None, k=5)
        assert len(ranked) == 1
        assert "final_score" in ranked[0]


# ---------------------------------------------------------------------------
# Phase 4: Cosine Similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert abs(_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_opposite_vectors(self):
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-9

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_both_zero_vectors(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# Phase 4: Compute Type Centroids
# ---------------------------------------------------------------------------

class TestComputeCentroids:
    def test_single_type(self):
        rows = [
            {"ltype": "WORKING_SOLUTION", "embedding": [1.0, 3.0]},
            {"ltype": "WORKING_SOLUTION", "embedding": [3.0, 1.0]},
        ]
        centroids = compute_type_centroids(rows)
        assert "WORKING_SOLUTION" in centroids
        assert abs(centroids["WORKING_SOLUTION"][0] - 2.0) < 1e-9
        assert abs(centroids["WORKING_SOLUTION"][1] - 2.0) < 1e-9

    def test_multiple_types(self):
        rows = [
            {"ltype": "A", "embedding": [1.0, 0.0]},
            {"ltype": "A", "embedding": [3.0, 0.0]},
            {"ltype": "B", "embedding": [0.0, 5.0]},
        ]
        centroids = compute_type_centroids(rows)
        assert len(centroids) == 2
        assert abs(centroids["A"][0] - 2.0) < 1e-9
        assert abs(centroids["B"][1] - 5.0) < 1e-9

    def test_empty_rows(self):
        assert compute_type_centroids([]) == {}

    def test_skips_none_ltype(self):
        rows = [
            {"ltype": None, "embedding": [1.0, 0.0]},
            {"ltype": "A", "embedding": [2.0, 3.0]},
        ]
        centroids = compute_type_centroids(rows)
        assert len(centroids) == 1
        assert "A" in centroids


# ---------------------------------------------------------------------------
# Phase 4: Infer Query Type
# ---------------------------------------------------------------------------

class TestInferQueryType:
    def test_returns_distribution(self):
        centroids = {
            "A": [1.0, 0.0],
            "B": [0.0, 1.0],
        }
        probs = infer_query_type([0.5, 0.5], centroids)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_closest_type_highest(self):
        centroids = {
            "A": [1.0, 0.0],
            "B": [0.0, 1.0],
        }
        # Query is much closer to A
        probs = infer_query_type([1.0, 0.0], centroids)
        assert probs["A"] > probs["B"]

    def test_single_centroid(self):
        centroids = {"ONLY": [1.0, 0.0]}
        probs = infer_query_type([0.5, 0.5], centroids)
        assert abs(probs["ONLY"] - 1.0) < 1e-6

    def test_empty_centroids(self):
        probs = infer_query_type([1.0, 0.0], {})
        assert probs == {}


# ---------------------------------------------------------------------------
# Phase 4: Centroid Cache (save/load)
# ---------------------------------------------------------------------------

class TestCentroidCache:
    def test_save_and_load(self, tmp_path):
        centroids = {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}
        path = tmp_path / "centroids.json"
        save_centroids(centroids, path)
        loaded = load_centroids(path)
        assert loaded == centroids

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_centroids(path) is None

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        assert load_centroids(path) is None


# ---------------------------------------------------------------------------
# Pattern score tests
# ---------------------------------------------------------------------------

class TestPatternScore:

    def test_no_pattern_strength_returns_zero(self):
        result = _make_result()  # no pattern_strength key
        ctx = RecallContext(tags_hint=["vue"])
        assert pattern_score(result, ctx) == 0.0

    def test_zero_strength_returns_zero(self):
        result = _make_result()
        result["pattern_strength"] = 0.0
        result["pattern_tags"] = ["vue"]
        ctx = RecallContext(tags_hint=["vue"])
        assert pattern_score(result, ctx) == 0.0

    def test_no_query_tags_returns_zero(self):
        result = _make_result()
        result["pattern_strength"] = 0.8
        result["pattern_tags"] = ["vue", "testing"]
        ctx = RecallContext(tags_hint=None)
        assert pattern_score(result, ctx) == 0.0

    def test_no_pattern_tags_returns_zero(self):
        result = _make_result()
        result["pattern_strength"] = 0.8
        result["pattern_tags"] = []
        ctx = RecallContext(tags_hint=["vue"])
        assert pattern_score(result, ctx) == 0.0

    def test_full_overlap_returns_strength(self):
        result = _make_result()
        result["pattern_strength"] = 0.8
        result["pattern_tags"] = ["vue", "testing"]
        ctx = RecallContext(tags_hint=["vue", "testing"])
        score = pattern_score(result, ctx)
        assert score == 0.8  # 2/2 overlap * 0.8

    def test_partial_overlap_scales(self):
        result = _make_result()
        result["pattern_strength"] = 0.8
        result["pattern_tags"] = ["vue"]
        ctx = RecallContext(tags_hint=["vue", "testing"])
        score = pattern_score(result, ctx)
        assert score == 0.4  # 1/2 overlap * 0.8

    def test_no_overlap_returns_zero(self):
        result = _make_result()
        result["pattern_strength"] = 0.8
        result["pattern_tags"] = ["hooks", "daemon"]
        ctx = RecallContext(tags_hint=["vue", "testing"])
        assert pattern_score(result, ctx) == 0.0


class TestRerankWithPattern:

    def test_pattern_signal_boosts_ranking(self):
        r1 = _make_result(similarity=0.5)
        r1["id"] = "id1"
        r2 = _make_result(similarity=0.4)
        r2["id"] = "id2"
        r2["pattern_strength"] = 0.9
        r2["pattern_tags"] = ["test"]
        ctx = RecallContext(
            tags_hint=["test"],
            retrieval_mode="hybrid_rrf",
        )
        ranked = rerank([r1, r2], ctx, k=2)
        # Pattern-boosted result should rank higher
        assert ranked[0]["id"] == "id2"
        assert ranked[0]["rerank_details"]["pattern"] > 0

    def test_pattern_weight_in_config(self):
        config = RerankerConfig()
        assert config.pattern_weight == 0.05
        assert config.total_signal_weight > 0.35
