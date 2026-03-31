"""Tests for confidence calibration.

Validates that:
1. Individual dimension scorers detect the right patterns
2. calibrate_confidence() produces correct mappings
3. Edge cases (empty content, missing fields) are handled
4. Calibration thresholds produce reasonable distributions
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.confidence_calibrator import (  # noqa: E402
    calibrate_confidence,
    score_actionability,
    score_evidence,
    score_scope,
    score_specificity,
)


# ---------------------------------------------------------------------------
# Specificity scoring
# ---------------------------------------------------------------------------

class TestSpecificity:
    def test_file_path_reference(self):
        score = score_specificity("The fix is in scripts/core/store_learning.py:42")
        assert score >= 0.15

    def test_function_name(self):
        score = score_specificity("def calibrate_confidence handles the mapping")
        assert score >= 0.15

    def test_git_hash(self):
        score = score_specificity("Fixed in commit abc1234")
        assert score >= 0.10

    def test_inline_code(self):
        score = score_specificity("Use `store_learning_v2` for new code")
        assert score >= 0.10

    def test_error_type(self):
        score = score_specificity("Got a TypeError when passing None")
        assert score >= 0.10

    def test_vague_content(self):
        score = score_specificity("Things work better now")
        assert score < 0.15

    def test_multiple_indicators(self):
        content = "Fixed TypeError in `store_learning.py:86` by checking for None"
        score = score_specificity(content)
        assert score >= 0.35


class TestActionability:
    def test_prescriptive_language(self):
        score = score_actionability("Always use store_learning_v2 instead of the legacy function")
        assert score >= 0.35

    def test_pattern_description(self):
        score = score_actionability("The pattern for hooks is to check availability first")
        assert score >= 0.10

    def test_step_instructions(self):
        score = score_actionability("First check the config, then validate, then store")
        assert score >= 0.10

    def test_descriptive_only(self):
        score = score_actionability("The system processes data in batches")
        assert score < 0.20


class TestEvidence:
    def test_commit_reference(self):
        score = score_evidence("Tested and merged in PR #5, commit abc1234")
        assert score >= 0.20

    def test_verification_language(self):
        score = score_evidence("Verified that the fix works with all test cases")
        assert score >= 0.20

    def test_test_results(self):
        score = score_evidence("Tests pass with 95% coverage after the change")
        assert score >= 0.25

    def test_no_evidence(self):
        score = score_evidence("This approach should work well")
        assert score < 0.10


class TestScope:
    def test_focused_short(self):
        score = score_scope("Use psycopg2.connect() not psycopg2.pool for single queries")
        assert score >= 0.6

    def test_vague_hedging(self):
        score = score_scope("Sometimes this might work, maybe it could help generally")
        assert score < 0.5

    def test_long_content(self):
        content = "word " * 250
        score = score_scope(content)
        assert score < score_scope("Short focused learning")


# ---------------------------------------------------------------------------
# End-to-end calibration
# ---------------------------------------------------------------------------

class TestCalibrateConfidence:
    def test_high_confidence_learning(self):
        content = (
            "Fixed TypeError in `store_learning.py:86` by checking for None. "
            "Always validate content before calling embed(). "
            "Verified with tests, merged in commit abc1234."
        )
        result = calibrate_confidence(content)
        assert result["confidence"] == "high"
        assert result["score"] >= 0.30
        assert "dimensions" in result
        assert set(result["dimensions"].keys()) == {
            "specificity", "actionability", "evidence", "scope"
        }

    def test_medium_confidence_learning(self):
        content = (
            "The hook system uses a pattern where you check availability first. "
            "Use `checkService()` before proceeding with the workflow."
        )
        result = calibrate_confidence(content)
        assert result["confidence"] in ("medium", "high")
        assert result["score"] >= 0.20

    def test_low_confidence_learning(self):
        content = "Things seem to maybe work sometimes."
        result = calibrate_confidence(content)
        assert result["confidence"] == "low"
        assert result["score"] < 0.15

    def test_empty_content(self):
        result = calibrate_confidence("")
        assert result["confidence"] == "low"

    def test_result_structure(self):
        result = calibrate_confidence("test content")
        assert "confidence" in result
        assert "score" in result
        assert "dimensions" in result
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0

    def test_score_is_weighted_average(self):
        """Verify score equals weighted sum of dimensions."""
        from scripts.core.confidence_calibrator import WEIGHTS
        result = calibrate_confidence("Use `foo.py` always, verified with tests")
        expected = sum(
            WEIGHTS[k] * result["dimensions"][k] for k in WEIGHTS
        )
        assert abs(result["score"] - round(expected, 3)) < 0.01


# ---------------------------------------------------------------------------
# Distribution sanity check
# ---------------------------------------------------------------------------

class TestDistribution:
    """Verify calibration produces reasonable distribution across sample learnings."""

    SAMPLE_LEARNINGS = [
        # Should be HIGH
        "Fixed TypeError in `store_learning.py:86` by adding None check. Always validate before embed(). Commit abc1234.",
        "Use `psycopg2.connect()` not pool for single queries. Tested: 3x faster for one-shot operations.",
        # Should be MEDIUM
        "The hook system works by checking availability first, then proceeding.",
        "Parallel agent spawning works well for independent tasks.",
        # Should be LOW
        "Things might work better sometimes.",
        "Generally the system seems fine overall.",
    ]

    def test_not_all_same_confidence(self):
        results = [calibrate_confidence(s) for s in self.SAMPLE_LEARNINGS]
        confidences = {r["confidence"] for r in results}
        assert len(confidences) >= 2, (
            f"All learnings got same confidence: {confidences}"
        )

    def test_high_samples_score_higher(self):
        high_scores = [
            calibrate_confidence(s)["score"]
            for s in self.SAMPLE_LEARNINGS[:2]
        ]
        low_scores = [
            calibrate_confidence(s)["score"]
            for s in self.SAMPLE_LEARNINGS[-2:]
        ]
        assert min(high_scores) > max(low_scores)
