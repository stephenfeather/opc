"""Tests for confidence calibration.

Validates that:
1. Individual dimension scorers detect the right patterns
2. calibrate_confidence() produces correct mappings
3. Edge cases (empty content, missing fields) are handled
4. Calibration thresholds produce reasonable distribution
5. Pure row-calibration logic (calibrate_rows) works without DB
6. Handler functions delegate correctly with mocked DB
7. CLI dispatches to the right handler
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.confidence_calibrator import (  # noqa: E402
    WEIGHTS,
    backfill_calibration,
    backfill_calibration_sync,
    calibrate_confidence,
    calibrate_rows,
    calibrate_session,
    calibrate_session_sync,
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
            "specificity", "actionability", "evidence", "scope",
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
        result = calibrate_confidence("Use `foo.py` always, verified with tests")
        expected = sum(WEIGHTS[k] * result["dimensions"][k] for k in WEIGHTS)
        assert abs(result["score"] - round(expected, 3)) < 0.01


# ---------------------------------------------------------------------------
# Distribution sanity check
# ---------------------------------------------------------------------------


class TestDistribution:
    """Verify calibration produces reasonable distribution across sample learnings."""

    SAMPLE_LEARNINGS = [
        # Should be HIGH
        "Fixed TypeError in `store_learning.py:86` by adding None check. "
        "Always validate before embed(). Commit abc1234.",
        "Use `psycopg2.connect()` not pool for single queries. "
        "Tested: 3x faster for one-shot operations.",
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
            calibrate_confidence(s)["score"] for s in self.SAMPLE_LEARNINGS[:2]
        ]
        low_scores = [
            calibrate_confidence(s)["score"] for s in self.SAMPLE_LEARNINGS[-2:]
        ]
        assert min(high_scores) > max(low_scores)


# ---------------------------------------------------------------------------
# Pure row calibration (calibrate_rows)
# ---------------------------------------------------------------------------


class TestCalibrateRows:
    """Test the pure function that processes DB rows without touching DB."""

    def test_basic_row_processing(self):
        rows = [
            ("id-1", "Fixed TypeError in `foo.py:10`. Commit abc1234.", {"confidence": "low"}),
        ]
        result = calibrate_rows(rows)
        assert result["stats"]["total"] == 1
        assert result["stats"]["updated"] == 1
        assert len(result["changes"]) == 1
        assert result["changes"][0]["old"] == "low"
        assert result["changes"][0]["new"] in ("medium", "high")

    def test_unchanged_row_with_full_calibration(self):
        """Row is unchanged only if label + score + dimensions all present."""
        content = "Fixed TypeError in `foo.py:10`. Always check None. Commit abc1234."
        cal = calibrate_confidence(content)
        rows = [
            ("id-1", content, {
                "confidence": cal["confidence"],
                "confidence_score": cal["score"],
                "confidence_dimensions": cal["dimensions"],
            }),
        ]
        result = calibrate_rows(rows)
        assert result["stats"]["unchanged"] == 1
        assert len(result["changes"]) == 0

    def test_row_missing_score_is_updated(self):
        """Row with matching label but missing score/dimensions is updated."""
        content = "Fixed TypeError in `foo.py:10`. Always check None. Commit abc1234."
        cal = calibrate_confidence(content)
        rows = [
            ("id-1", content, {"confidence": cal["confidence"]}),
        ]
        result = calibrate_rows(rows)
        assert result["stats"]["updated"] == 1

    def test_empty_content_row(self):
        rows = [("id-1", "", {"confidence": "high"})]
        result = calibrate_rows(rows)
        assert result["stats"]["errors"] == 1

    def test_none_content_row(self):
        rows = [("id-1", None, {"confidence": "high"})]
        result = calibrate_rows(rows)
        assert result["stats"]["errors"] == 1

    def test_none_metadata(self):
        rows = [("id-1", "Always use `foo()` instead. Verified.", None)]
        result = calibrate_rows(rows)
        assert result["stats"]["total"] == 1
        # old confidence should be None since metadata was None
        if result["changes"]:
            assert result["changes"][0]["old"] is None

    def test_multiple_rows(self):
        rows = [
            ("id-1", "Fixed TypeError in `foo.py`. Commit abc1234.", {"confidence": "low"}),
            ("id-2", "Things seem ok.", {"confidence": "high"}),
            ("id-3", "", None),
        ]
        result = calibrate_rows(rows)
        assert result["stats"]["total"] == 3
        assert result["stats"]["errors"] == 1
        assert result["stats"]["updated"] + result["stats"]["unchanged"] == 2

    def test_changes_include_dimensions(self):
        rows = [
            ("id-1", "Fixed TypeError in `foo.py:10`. Commit abc1234.", {"confidence": "low"}),
        ]
        result = calibrate_rows(rows)
        change = result["changes"][0]
        assert "dimensions" in change
        assert set(change["dimensions"].keys()) == {
            "specificity", "actionability", "evidence", "scope",
        }

    def test_changes_do_not_include_metadata(self):
        """Public changes should not leak internal metadata."""
        rows = [
            ("id-1", "Always use `foo()`. Verified with tests.",
             {"confidence": "low", "tags": ["a"], "session_id": "s1"}),
        ]
        result = calibrate_rows(rows)
        change = result["changes"][0]
        assert "updated_metadata" not in change
        assert "session_id" not in change
        assert "tags" not in change
        # Only public fields
        assert set(change.keys()) == {
            "id", "old", "new", "score", "dimensions",
        }

    def test_returns_immutable_stats(self):
        """Verify calling calibrate_rows twice doesn't accumulate state."""
        rows = [("id-1", "Always use `foo()`. Verified.", {"confidence": "low"})]
        r1 = calibrate_rows(rows)
        r2 = calibrate_rows(rows)
        assert r1["stats"] == r2["stats"]


# ---------------------------------------------------------------------------
# Handler: calibrate_session (mocked DB)
# ---------------------------------------------------------------------------


class TestCalibrateSessionHandler:
    """Test calibrate_session_sync with mocked database."""

    def _make_mock_conn(self, rows):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value = cur
        cur.fetchall.return_value = rows
        return conn, cur

    @patch("scripts.core.confidence_calibrator._pg_connect")
    def test_session_dry_run_does_not_write(self, mock_connect):
        rows = [
            ("id-1", "Fixed TypeError in `foo.py`. Commit abc.", {"confidence": "low"}),
        ]
        conn, cur = self._make_mock_conn(rows)
        mock_connect.return_value = conn

        result = calibrate_session_sync("sess-1", dry_run=True)

        assert result["stats"]["updated"] >= 1
        # No UPDATE executed (DDL ALTER is allowed, not UPDATE)
        update_calls = [
            c for c in cur.execute.call_args_list
            if c[0][0].strip().startswith("UPDATE")
        ]
        assert len(update_calls) == 0

    @patch("scripts.core.confidence_calibrator._pg_connect")
    def test_session_writes_when_not_dry_run(self, mock_connect):
        rows = [
            ("id-1", "Fixed TypeError in `foo.py`. Commit abc.", {"confidence": "low"}),
        ]
        conn, cur = self._make_mock_conn(rows)
        mock_connect.return_value = conn

        result = calibrate_session_sync("sess-1", dry_run=False)

        assert result["stats"]["updated"] >= 1
        # UPDATE was executed using JSONB merge (not full overwrite)
        update_calls = [
            c for c in cur.execute.call_args_list
            if c[0][0].strip().startswith("UPDATE")
        ]
        assert len(update_calls) >= 1
        # Verify JSONB merge pattern (|| operator)
        update_sql = update_calls[0][0][0]
        assert "||" in update_sql, "Should use JSONB merge, not overwrite"
        # Commit called for DDL + data
        assert conn.commit.call_count >= 2

    @patch("scripts.core.confidence_calibrator._pg_connect")
    def test_session_passes_session_id_to_query(self, mock_connect):
        conn, cur = self._make_mock_conn([])
        mock_connect.return_value = conn

        calibrate_session_sync("my-session-42")

        # Verify the SELECT used the session_id (index 1, after DDL)
        select_calls = [
            c for c in cur.execute.call_args_list
            if "SELECT" in c[0][0]
        ]
        assert len(select_calls) == 1
        assert "my-session-42" in select_calls[0][0][1]


# ---------------------------------------------------------------------------
# Handler: backfill_calibration (mocked DB)
# ---------------------------------------------------------------------------


class TestBackfillHandler:
    def _make_mock_conn(self, rows):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value = cur
        cur.fetchall.return_value = rows
        return conn, cur

    @patch("scripts.core.confidence_calibrator._pg_connect")
    def test_backfill_dry_run_no_data_updates(self, mock_connect):
        rows = [
            ("id-1", "Always use `foo()`. Verified.", {"confidence": "low"}),
            ("id-2", "Maybe things work.", {"confidence": "high"}),
        ]
        conn, cur = self._make_mock_conn(rows)
        mock_connect.return_value = conn

        result = backfill_calibration_sync(dry_run=True, batch_size=10)

        assert result["stats"]["total"] == 2
        # DDL commit (_ensure_calibration_column) is allowed;
        # no UPDATE statements should be executed
        update_calls = [
            c for c in cur.execute.call_args_list
            if c[0][0].strip().startswith("UPDATE")
        ]
        assert len(update_calls) == 0

    @patch("scripts.core.confidence_calibrator._pg_connect")
    def test_backfill_writes_updates(self, mock_connect):
        rows = [
            ("id-1", "Always use `foo()`. Verified.", {"confidence": "low"}),
        ]
        conn, cur = self._make_mock_conn(rows)
        mock_connect.return_value = conn

        result = backfill_calibration_sync(dry_run=False, batch_size=10)

        assert result["stats"]["total"] == 1
        conn.commit.assert_called()

    @patch("scripts.core.confidence_calibrator._pg_connect")
    def test_backfill_marks_unchanged_as_calibrated(self, mock_connect):
        """Even unchanged rows should get confidence_calibrated_at set."""
        content = "Always use `foo()`. Verified with tests, commit abc."
        cal = calibrate_confidence(content)
        rows = [
            ("id-1", content, {
                "confidence": cal["confidence"],
                "confidence_score": cal["score"],
                "confidence_dimensions": cal["dimensions"],
            }),
        ]
        conn, cur = self._make_mock_conn(rows)
        mock_connect.return_value = conn

        result = backfill_calibration_sync(dry_run=False, batch_size=10)

        assert result["stats"]["unchanged"] == 1
        # Still should have executed an UPDATE for calibrated_at
        update_calls = [
            c for c in cur.execute.call_args_list
            if c[0][0].strip().startswith("UPDATE")
        ]
        assert len(update_calls) >= 1


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------


class TestCLI:
    @patch("scripts.core.confidence_calibrator.backfill_calibration_sync")
    def test_main_backfill_flag(self, mock_backfill):
        from scripts.core.confidence_calibrator import main

        mock_backfill.return_value = {
            "stats": {"total": 0, "updated": 0, "unchanged": 0, "errors": 0},
            "changes": [],
        }
        main(["--backfill", "--dry-run"])
        mock_backfill.assert_called_once_with(dry_run=True)

    @patch("scripts.core.confidence_calibrator.calibrate_session_sync")
    def test_main_session_flag(self, mock_session):
        from scripts.core.confidence_calibrator import main

        mock_session.return_value = {
            "stats": {"total": 0, "updated": 0, "unchanged": 0, "errors": 0},
            "changes": [],
        }
        main(["--session-id", "abc123"])
        mock_session.assert_called_once_with("abc123", dry_run=False)

    def test_main_no_args_exits(self):
        from scripts.core.confidence_calibrator import main

        with pytest.raises(SystemExit):
            main([])

    @patch("scripts.core.confidence_calibrator.calibrate_session_sync")
    def test_json_output_does_not_leak_metadata(self, mock_session, capsys):
        """CLI --json must not expose internal metadata fields."""
        from scripts.core.confidence_calibrator import main

        mock_session.return_value = {
            "stats": {"total": 1, "updated": 1, "unchanged": 0, "errors": 0},
            "changes": [{
                "id": "abc12345",
                "old": "low",
                "new": "high",
                "score": 0.45,
                "dimensions": {
                    "specificity": 0.5,
                    "actionability": 0.4,
                    "evidence": 0.5,
                    "scope": 0.7,
                },
            }],
        }
        main(["--session-id", "s1", "--json"])
        output = capsys.readouterr().out
        import json
        data = json.loads(output)
        for change in data["changes"]:
            forbidden = {
                "updated_metadata", "session_id", "host_id",
                "context", "project", "classification_reasoning",
            }
            assert not forbidden & set(change.keys()), (
                f"Leaked internal fields: {forbidden & set(change.keys())}"
            )


# ---------------------------------------------------------------------------
# Async wrappers (backward compat with memory_daemon)
# ---------------------------------------------------------------------------


class TestAsyncWrappers:
    """Verify async wrappers delegate to sync implementations."""

    @patch("scripts.core.confidence_calibrator._pg_connect")
    async def test_calibrate_session_async(self, mock_connect):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value = cur
        cur.fetchall.return_value = []
        mock_connect.return_value = conn

        result = await calibrate_session("sess-1", dry_run=True)
        assert result["stats"]["total"] == 0

    @patch("scripts.core.confidence_calibrator._pg_connect")
    async def test_backfill_calibration_async(self, mock_connect):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value = cur
        cur.fetchall.return_value = []
        mock_connect.return_value = conn

        result = await backfill_calibration(dry_run=True)
        assert result["stats"]["total"] == 0
