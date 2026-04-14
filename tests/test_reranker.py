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
    KG_TYPE_WEIGHTS,
    RecallContext,
    RerankerConfig,
    _cosine_similarity,
    calibrate_score,
    compute_type_centroids,
    confidence_score,
    infer_query_type,
    kg_overlap,
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

    def test_recency_string_timestamp(self):
        """ISO format string timestamps should be parsed correctly."""
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
        result = _make_result()
        result["created_at"] = "2026-03-31T12:00:00+00:00"
        ctx = RecallContext(now=now)
        score = recency_score(result, ctx)
        # 1 day old => exp(-1/45)
        assert abs(score - math.exp(-1 / 45)) < 0.01

    def test_recency_invalid_string(self):
        """Unparseable string timestamps should return 0.5 fallback."""
        result = _make_result()
        result["created_at"] = "not-a-date"
        ctx = RecallContext()
        assert recency_score(result, ctx) == 0.5

    def test_recency_naive_datetime(self):
        """Timezone-naive datetimes should get UTC assumed."""
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
        naive_dt = datetime(2026, 3, 31, 12, 0, 0)  # no tzinfo
        result = _make_result()
        result["created_at"] = naive_dt
        ctx = RecallContext(now=now)
        score = recency_score(result, ctx)
        # 1 day old => exp(-1/45)
        assert abs(score - math.exp(-1 / 45)) < 0.01


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

    def test_calibrate_negative_score(self):
        """Negative scores should be clamped to 0.0."""
        # (-0.5 + 1) / 2 = 0.25 -- positive, not clamped
        assert calibrate_score(-0.5, "vector", rank=0, total=5) == 0.25
        # (-2.0 + 1) / 2 = -0.5 => clamped to 0.0
        assert calibrate_score(-2.0, "vector", rank=0, total=5) == 0.0

    def test_calibrate_text_negative_one_no_zerodiv(self):
        """raw_score=-1.0 with text mode: denominator (-1+1)=0, should return 0.0."""
        score = calibrate_score(-1.0, "text", rank=0, total=5)
        assert score == 0.0

    def test_calibrate_total_zero(self):
        """total=0 with unknown mode should return 1.0 (edge case in rank fallback)."""
        score = calibrate_score(0.5, None, rank=0, total=0)
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

    def test_rerank_does_not_mutate_input(self):
        """The original results list should be unchanged after rerank."""
        results = [
            _make_result(similarity=0.9),
            _make_result(similarity=0.5),
        ]
        results[0]["id"] = "first"
        results[1]["id"] = "second"
        original_ids = [r["id"] for r in results]
        original_keys = [set(r.keys()) for r in results]

        ctx = RecallContext(retrieval_mode="vector")
        rerank(results, ctx, k=5)

        # Order unchanged
        assert [r["id"] for r in results] == original_ids
        # No extra keys added to originals
        assert [set(r.keys()) for r in results] == original_keys

    def test_rerank_zero_total_signal_weight(self):
        """With zero total_signal_weight, returns results with consistent output shape."""
        results = [
            _make_result(similarity=0.5),
            _make_result(similarity=0.9),
        ]
        results[0]["id"] = "first"
        results[1]["id"] = "second"
        config = RerankerConfig(
            project_weight=0.0,
            recency_weight=0.0,
            confidence_weight=0.0,
            recall_weight=0.0,
            type_affinity_weight=0.0,
            tag_overlap_weight=0.0,
            pattern_weight=0.0,
        )
        ctx = RecallContext(retrieval_mode="vector")
        ranked = rerank(results, ctx, config=config, k=5)
        # Should sort by calibrated score (0.9 > 0.5 in vector mode)
        assert ranked[0]["id"] == "second"
        assert ranked[1]["id"] == "first"
        # Output shape should still include augmented keys
        assert "final_score" in ranked[0]
        assert "rerank_details" in ranked[0]


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

    def test_skips_empty_embeddings(self):
        """Rows with empty embedding lists should be skipped."""
        rows = [
            {"ltype": "A", "embedding": []},
            {"ltype": "A", "embedding": [1.0, 2.0]},
            {"ltype": "B", "embedding": []},
        ]
        centroids = compute_type_centroids(rows)
        # A should only use the non-empty embedding
        assert "A" in centroids
        assert abs(centroids["A"][0] - 1.0) < 1e-9
        assert abs(centroids["A"][1] - 2.0) < 1e-9
        # B should be absent (only had empty embedding)
        assert "B" not in centroids


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


# ---------------------------------------------------------------------------
# FP Compliance: Config defaults are hardcoded (no import-time I/O)
# ---------------------------------------------------------------------------

class TestRerankerConfigDefaults:
    """RerankerConfig defaults should match opc.toml values without I/O at import."""

    def test_default_project_weight(self):
        config = RerankerConfig()
        assert config.project_weight == 0.15

    def test_default_recency_weight(self):
        config = RerankerConfig()
        assert config.recency_weight == 0.05

    def test_config_is_frozen(self):
        config = RerankerConfig()
        with pytest.raises(AttributeError):
            config.project_weight = 0.99

    def test_default_total_signal_weight(self):
        config = RerankerConfig()
        # project + 7 secondary signals (recency, confidence, recall,
        # type_affinity, tag_overlap, pattern, kg) -- all at default 0.05.
        expected = 0.15 + 0.05 * 7
        assert abs(config.total_signal_weight - expected) < 1e-9


# ---------------------------------------------------------------------------
# FP Compliance: calibrate_score accepts config parameter (no module globals)
# ---------------------------------------------------------------------------

class TestCalibrateScoreWithConfig:
    """calibrate_score should accept an optional config parameter for RRF scale."""

    def test_calibrate_rrf_with_custom_config(self):
        """RRF calibration should use config's rrf_scale_factor, not a global."""
        config = RerankerConfig(rrf_scale_factor=100.0)
        score = calibrate_score(0.01, "hybrid_rrf", rank=0, total=5, config=config)
        # 0.01 * 100 = 1.0 (clamped)
        assert score == 1.0

    def test_calibrate_rrf_default_without_config(self):
        """calibrate_score without config should still work with default scale."""
        score = calibrate_score(0.01, "hybrid_rrf", rank=0, total=5)
        assert abs(score - 0.6) < 0.001


class TestRecencyScoreWithConfig:
    """recency_score should accept config for half-life parameter."""

    def test_recency_with_custom_half_life(self):
        """recency_score should use config's half-life, not a module global."""
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
        result = _make_result(created_at=now - timedelta(days=10))
        ctx = RecallContext(now=now)
        config = RerankerConfig(recency_half_life_days=10.0)
        score = recency_score(result, ctx, config=config)
        # exp(-10 / 10) = exp(-1) ≈ 0.368
        assert abs(score - math.exp(-1)) < 0.01

    def test_recency_zero_half_life_falls_back(self):
        """Zero half-life should fall back to default 45.0, not ZeroDivisionError."""
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
        result = _make_result(created_at=now - timedelta(days=10))
        ctx = RecallContext(now=now)
        config = RerankerConfig(recency_half_life_days=0.0)
        score = recency_score(result, ctx, config=config)
        assert abs(score - math.exp(-10 / 45)) < 0.01


class TestRecallScoreWithConfig:
    """recall_score should accept config for log2 normalizer."""

    def test_recall_with_custom_normalizer(self):
        """recall_score should use config's normalizer, not a module global."""
        result = _make_result(recall_count=3)
        config = RerankerConfig(recall_log2_normalizer=2.0)
        # log2(4) / 2.0 = 2.0 / 2.0 = 1.0
        score = recall_score(result, config=config)
        assert abs(score - 1.0) < 0.01

    def test_recall_zero_normalizer_falls_back(self):
        """Zero normalizer should fall back to default 4.0, not ZeroDivisionError."""
        result = _make_result(recall_count=3)
        config = RerankerConfig(recall_log2_normalizer=0.0)
        expected = min(1.0, math.log2(1 + 3) / 4.0)
        score = recall_score(result, config=config)
        assert abs(score - expected) < 0.01


# ---------------------------------------------------------------------------
# Signal Function Tests: kg_overlap (Phase 3)
# ---------------------------------------------------------------------------


def _result_with_kg(entities: list[dict]) -> dict:
    """Build a result dict carrying a kg_context entities list."""
    result = _make_result()
    result["kg_context"] = {"entities": entities, "edges": []}
    return result


class TestKGOverlap:
    def test_zero_when_query_entities_none(self):
        result = _result_with_kg([{"name": "pytest", "type": "tool"}])
        ctx = RecallContext(query_entities=None)
        assert kg_overlap(result, ctx) == 0.0

    def test_zero_when_query_entities_empty(self):
        result = _result_with_kg([{"name": "pytest", "type": "tool"}])
        ctx = RecallContext(query_entities=[])
        assert kg_overlap(result, ctx) == 0.0

    def test_zero_when_result_has_no_kg_context(self):
        result = _make_result()  # no kg_context key
        ctx = RecallContext(query_entities=[{"name": "pytest", "type": "tool"}])
        assert kg_overlap(result, ctx) == 0.0

    def test_one_point_zero_on_identical_sets(self):
        entities = [
            {"name": "pytest", "type": "tool"},
            {"name": "store_learning.py", "type": "file"},
        ]
        result = _result_with_kg(entities)
        ctx = RecallContext(query_entities=list(entities))
        assert kg_overlap(result, ctx) == 1.0

    def test_partial_overlap_between_zero_and_one(self):
        result = _result_with_kg([
            {"name": "pytest", "type": "tool"},
            {"name": "asyncpg", "type": "library"},
        ])
        ctx = RecallContext(query_entities=[{"name": "pytest", "type": "tool"}])
        score = kg_overlap(result, ctx)
        assert 0.0 < score < 1.0

    def test_case_insensitive_name_match(self):
        result = _result_with_kg([{"name": "Pytest", "type": "tool"}])
        ctx = RecallContext(query_entities=[{"name": "PYTEST", "type": "tool"}])
        assert kg_overlap(result, ctx) == 1.0

    def test_file_type_overlap_scores_higher_than_language(self):
        """Type weights: file (1.0) > language (0.4) for equal-size overlap."""
        file_result = _result_with_kg([{"name": "a.py", "type": "file"}])
        file_ctx = RecallContext(query_entities=[{"name": "a.py", "type": "file"}])
        lang_result = _result_with_kg([{"name": "python", "type": "language"}])
        lang_ctx = RecallContext(query_entities=[{"name": "python", "type": "language"}])
        # Both are full matches in their own set, so weighted-Jaccard = 1.0 for
        # each individually. The differentiation appears when mixed with
        # non-matching entities of the same type. Test a mixed case instead.

        mixed_result = _result_with_kg([
            {"name": "a.py", "type": "file"},
            {"name": "b.py", "type": "file"},        # not in query
            {"name": "python", "type": "language"},
            {"name": "ruby", "type": "language"},    # not in query
        ])
        file_match_ctx = RecallContext(
            query_entities=[{"name": "a.py", "type": "file"}]
        )
        lang_match_ctx = RecallContext(
            query_entities=[{"name": "python", "type": "language"}]
        )
        file_score = kg_overlap(mixed_result, file_match_ctx)
        lang_score = kg_overlap(mixed_result, lang_match_ctx)
        # file weight (1.0) >> language weight (0.4), so overlap contributes
        # relatively more when the matching entity is typed 'file'.
        assert file_score > lang_score
        # Sanity: individual full matches are still 1.0
        assert kg_overlap(file_result, file_ctx) == 1.0
        assert kg_overlap(lang_result, lang_ctx) == 1.0

    def test_type_weights_table_has_expected_entries(self):
        """KG_TYPE_WEIGHTS covers the entity types emitted by kg_extractor."""
        required = {"file", "module", "library", "tool", "language", "concept", "error"}
        assert required.issubset(set(KG_TYPE_WEIGHTS.keys()))
        assert KG_TYPE_WEIGHTS["file"] > KG_TYPE_WEIGHTS["language"]

    def test_canonical_field_matches_across_case_and_path_norm(self):
        """Finding F1 fix: when kg_context entities carry a 'canonical' field
        (added by _fetch_kg_rows), kg_overlap uses it for matching so display
        casing and un-normalized paths don't break overlap."""
        # Result has display 'name' preserving the stored form, and the
        # canonical form the extractor would produce.
        result = _result_with_kg([
            {"name": "./scripts/core/reranker.py",
             "canonical": "scripts/core/reranker.py",
             "type": "file"},
        ])
        # Query-side entity uses the canonical value directly (extract_entities
        # returns canonical in .name).
        ctx = RecallContext(
            query_entities=[{"name": "scripts/core/reranker.py", "type": "file"}]
        )
        assert kg_overlap(result, ctx) == 1.0

    def test_config_type_weight_matches_extractor_type_name(self):
        """Finding F3 fix: kg_extractor emits entity_type='config' for env
        variables; KG_TYPE_WEIGHTS must key on 'config' (not 'config_var')
        or the intended salience is silently lost to the default weight."""
        assert "config" in KG_TYPE_WEIGHTS
        assert KG_TYPE_WEIGHTS["config"] != 0.5  # not the default fallback


class TestRerankKGWeight:
    def test_rerank_boosts_kg_matches_when_weight_positive(self):
        """With kg_weight > 0, a matching result ranks above a non-matching one
        that would otherwise be tied on retrieval."""
        matching = _result_with_kg([{"name": "pytest", "type": "tool"}])
        matching["id"] = "matching"
        non_matching = _result_with_kg([{"name": "other", "type": "tool"}])
        non_matching["id"] = "non_matching"

        ctx = RecallContext(
            query_entities=[{"name": "pytest", "type": "tool"}],
            retrieval_mode="vector",
        )
        # Kill all other signal weights to isolate kg_weight effect.
        config = RerankerConfig(
            project_weight=0.0,
            recency_weight=0.0,
            confidence_weight=0.0,
            recall_weight=0.0,
            type_affinity_weight=0.0,
            tag_overlap_weight=0.0,
            pattern_weight=0.0,
            kg_weight=0.2,
        )

        ranked = rerank([non_matching, matching], ctx, config=config, k=2)
        assert ranked[0]["id"] == "matching"
        assert "kg_overlap" in ranked[0]["rerank_details"]

    def test_kg_inactive_scores_match_pre_phase3_exactly(self):
        """Finding D1 fix: when KG is inactive (no query entities OR no
        result carries kg_context), final_score must be byte-identical to
        the pre-Phase-3 reranker with kg_weight=0. Proves kg_weight is
        redirected to retrieval, not deducted. Load-bearing invariant."""
        # Two results, no kg_context attached, no query entities.
        results = [
            {"id": "a", "session_id": "s", "content": "x",
             "metadata": {"learning_type": "WORKING_SOLUTION"}, "similarity": 0.5},
            {"id": "b", "session_id": "s", "content": "y",
             "metadata": {"learning_type": "ERROR_FIX"}, "similarity": 0.3},
        ]
        ctx_active_but_empty = RecallContext(
            query_entities=None, retrieval_mode="vector",
        )
        # Reference: pre-Phase-3 behavior simulated by kg_weight=0.
        ref_config = RerankerConfig(kg_weight=0.0)
        # Current config: kg_weight=0.05 default.
        current_config = RerankerConfig()

        ref_ranked = rerank(
            [dict(r) for r in results], ctx_active_but_empty,
            config=ref_config, k=2,
        )
        cur_ranked = rerank(
            [dict(r) for r in results], ctx_active_but_empty,
            config=current_config, k=2,
        )
        # Byte-identical final_score, byte-identical order.
        assert [r["id"] for r in ref_ranked] == [r["id"] for r in cur_ranked]
        for ref_r, cur_r in zip(ref_ranked, cur_ranked):
            assert ref_r["final_score"] == cur_r["final_score"], (
                f"Score drift on {ref_r['id']}: "
                f"ref={ref_r['final_score']} vs cur={cur_r['final_score']}"
            )

    def test_kg_active_flag_reported_in_rerank_details(self):
        """Operators need to tell which mode was used. kg_active lands in
        rerank_details."""
        active_result = _result_with_kg([{"name": "pytest", "type": "tool"}])
        active_result["id"] = "active"
        ctx_active = RecallContext(
            query_entities=[{"name": "pytest", "type": "tool"}],
            retrieval_mode="vector",
        )
        ranked_active = rerank([active_result], ctx_active, k=1)
        assert ranked_active[0]["rerank_details"]["kg_active"] is True

        inactive_result = dict(active_result)
        inactive_result.pop("kg_context")
        ctx_inactive = RecallContext(retrieval_mode="vector")
        ranked_inactive = rerank([inactive_result], ctx_inactive, k=1)
        assert ranked_inactive[0]["rerank_details"]["kg_active"] is False

    def test_rerank_zero_kg_weight_is_noop(self):
        """With kg_weight=0.0, ranking is unchanged from non-KG scoring."""
        matching = _result_with_kg([{"name": "pytest", "type": "tool"}])
        matching["id"] = "matching"
        matching["similarity"] = 0.3  # lower retrieval
        non_matching = _result_with_kg([{"name": "other", "type": "tool"}])
        non_matching["id"] = "non_matching"
        non_matching["similarity"] = 0.9  # higher retrieval

        ctx = RecallContext(
            query_entities=[{"name": "pytest", "type": "tool"}],
            retrieval_mode="vector",
        )
        config = RerankerConfig(
            project_weight=0.0,
            recency_weight=0.0,
            confidence_weight=0.0,
            recall_weight=0.0,
            type_affinity_weight=0.0,
            tag_overlap_weight=0.0,
            pattern_weight=0.0,
            kg_weight=0.0,
        )

        ranked = rerank([matching, non_matching], ctx, config=config, k=2)
        # Retrieval dominates: non_matching wins.
        assert ranked[0]["id"] == "non_matching"
