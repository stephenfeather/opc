"""Tests for re_embed_voyage.py — TDD+FP refactor."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from scripts.core.re_embed_voyage import (
    EXCLUDED_STATES,
    FAILURE_STATES,
    BatchResult,
    build_batch_texts,
    build_excluded_states,
    classify_pending,
    format_progress_line,
    format_summary,
    mark_failed_rows,
    process_single_batch,
)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestClassifyPending:
    """classify_pending(total, done) -> (pending, already_done)"""

    def test_all_pending(self):
        assert classify_pending(total=100, done=0) == (100, 0)

    def test_all_done(self):
        assert classify_pending(total=50, done=50) == (0, 50)

    def test_partial(self):
        assert classify_pending(total=200, done=75) == (125, 75)

    def test_zero_total(self):
        assert classify_pending(total=0, done=0) == (0, 0)


class TestBuildBatchTexts:
    """build_batch_texts extracts content strings from row dicts."""

    def test_extracts_content(self):
        rows = [
            {"id": uuid4(), "content": "hello"},
            {"id": uuid4(), "content": "world"},
        ]
        assert build_batch_texts(rows) == ["hello", "world"]

    def test_empty_rows(self):
        assert build_batch_texts([]) == []


class TestFormatProgressLine:
    """format_progress_line produces a human-readable progress string."""

    def test_basic_format(self):
        line = format_progress_line(
            batch_num=3, batch_len=16, converted=32, pending=100, elapsed=45.0
        )
        assert "Batch 3" in line
        assert "16 rows" in line
        assert "32/100" in line
        assert "32%" in line
        assert "45s" in line

    def test_zero_pending(self):
        line = format_progress_line(batch_num=1, batch_len=0, converted=0, pending=0, elapsed=1.0)
        assert "100%" in line


class TestFormatSummary:
    """format_summary produces a multi-line summary string."""

    def test_no_failures(self):
        summary = format_summary(converted=50, failed_ids=[], elapsed=12.5)
        assert "50" in summary
        assert "12.5s" in summary
        assert "Failed:    0" in summary
        assert "retry" not in summary.lower()

    def test_with_failures(self):
        summary = format_summary(converted=40, failed_ids=["a", "b"], elapsed=30.0)
        assert "Failed:    2" in summary
        assert "--retry-failed" in summary


class TestBatchResult:
    """BatchResult data class."""

    def test_success(self):
        r = BatchResult(converted=10, failed_ids=[])
        assert r.converted == 10
        assert r.failed_ids == []

    def test_failure(self):
        r = BatchResult(converted=0, failed_ids=["id1", "id2"])
        assert r.converted == 0
        assert len(r.failed_ids) == 2

    def test_frozen(self):
        r = BatchResult(converted=5)
        with pytest.raises(AttributeError):
            r.converted = 10


# ---------------------------------------------------------------------------
# I/O boundary tests (mocked)
# ---------------------------------------------------------------------------


class TestProcessSingleBatch:
    """process_single_batch embeds and updates a batch, returning BatchResult."""

    @pytest.mark.asyncio
    async def test_success(self):
        rows = [{"id": uuid4(), "content": "text1"}, {"id": uuid4(), "content": "text2"}]
        mock_provider = AsyncMock()
        mock_provider.embed_batch.return_value = [[0.1] * 1024, [0.2] * 1024]
        mock_update = AsyncMock()

        result = await process_single_batch(
            rows=rows,
            provider=mock_provider,
            target_model="voyage-code-3",
            update_fn=mock_update,
        )

        assert result.converted == 2
        assert result.failed_ids == []
        mock_provider.embed_batch.assert_awaited_once_with(["text1", "text2"])
        mock_update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embedding_error_marks_bge_failed(self):
        from scripts.core.db.embedding_service import EmbeddingError

        rows = [{"id": uuid4(), "content": "text1"}]
        mock_provider = AsyncMock()
        mock_provider.embed_batch.side_effect = EmbeddingError("API down")
        mock_mark_failed = AsyncMock()

        result = await process_single_batch(
            rows=rows,
            provider=mock_provider,
            target_model="voyage-code-3",
            update_fn=AsyncMock(),
            mark_failed_fn=mock_mark_failed,
        )

        assert result.converted == 0
        assert len(result.failed_ids) == 1
        mock_mark_failed.assert_awaited_once()
        call_kwargs = mock_mark_failed.await_args
        assert call_kwargs[1]["status"] == "bge-failed"

    @pytest.mark.asyncio
    async def test_db_update_failure_marks_embed_failed_db(self):
        """When embed succeeds but DB update fails, rows get 'embed-failed-db'."""
        rows = [{"id": uuid4(), "content": "text1"}]
        mock_provider = AsyncMock()
        mock_provider.embed_batch.return_value = [[0.1] * 1024]
        mock_update = AsyncMock(side_effect=Exception("DB connection lost"))
        mock_mark_failed = AsyncMock()

        result = await process_single_batch(
            rows=rows,
            provider=mock_provider,
            target_model="voyage-code-3",
            update_fn=mock_update,
            mark_failed_fn=mock_mark_failed,
        )

        assert result.converted == 0
        assert len(result.failed_ids) == 1
        mock_mark_failed.assert_awaited_once()
        call_kwargs = mock_mark_failed.await_args
        assert call_kwargs[1]["status"] == "embed-failed-db"

    @pytest.mark.asyncio
    async def test_embedding_count_mismatch_fails_batch(self):
        """If provider returns wrong number of embeddings, batch fails."""
        rows = [{"id": uuid4(), "content": "text1"}, {"id": uuid4(), "content": "text2"}]
        mock_provider = AsyncMock()
        mock_provider.embed_batch.return_value = [[0.1] * 1024]  # only 1 for 2 rows
        mock_mark_failed = AsyncMock()

        result = await process_single_batch(
            rows=rows,
            provider=mock_provider,
            target_model="voyage-code-3",
            update_fn=AsyncMock(),
            mark_failed_fn=mock_mark_failed,
        )

        assert result.converted == 0
        assert len(result.failed_ids) == 2
        mock_mark_failed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_mark_failed_fn_still_returns_failure(self):
        """Without mark_failed_fn, failures still return correct BatchResult."""
        from scripts.core.db.embedding_service import EmbeddingError

        rows = [{"id": uuid4(), "content": "text1"}]
        mock_provider = AsyncMock()
        mock_provider.embed_batch.side_effect = EmbeddingError("API down")

        result = await process_single_batch(
            rows=rows,
            provider=mock_provider,
            target_model="voyage-code-3",
            update_fn=AsyncMock(),
        )

        assert result.converted == 0
        assert len(result.failed_ids) == 1


class TestMarkFailedRows:
    """mark_failed_rows updates DB rows with specified failure status."""

    @pytest.mark.asyncio
    async def test_marks_rows_default_status(self):
        row_ids = [str(uuid4()), str(uuid4())]
        mock_conn = AsyncMock()

        with patch("scripts.core.re_embed_voyage.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)
            await mark_failed_rows(row_ids)

        mock_conn.execute.assert_awaited_once()
        call_args = mock_conn.execute.await_args[0]
        assert "bge-failed" in call_args[1]

    @pytest.mark.asyncio
    async def test_marks_rows_custom_status(self):
        row_ids = [str(uuid4())]
        mock_conn = AsyncMock()

        with patch("scripts.core.re_embed_voyage.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)
            await mark_failed_rows(row_ids, status="embed-failed-db")

        call_args = mock_conn.execute.await_args[0]
        assert "embed-failed-db" in call_args[1]

    @pytest.mark.asyncio
    async def test_rejects_invalid_status(self):
        with pytest.raises(ValueError, match="Invalid failure status"):
            await mark_failed_rows([str(uuid4())], status="bogus")


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------


class TestStateMachine:
    """Verify the embedding_model state machine is correctly defined."""

    def test_failure_states_are_subset_of_excluded(self):
        """All failure states must be in the excluded set."""
        for state in FAILURE_STATES:
            assert state in EXCLUDED_STATES

    def test_excluded_states_include_in_progress(self):
        """in-progress is excluded from claiming."""
        assert "in-progress" in EXCLUDED_STATES

    def test_excluded_states_include_both_failure_types(self):
        assert "bge-failed" in EXCLUDED_STATES
        assert "embed-failed-db" in EXCLUDED_STATES

    def test_build_excluded_states_includes_target(self):
        """Target model is excluded so already-converted rows aren't reclaimed."""
        excluded = build_excluded_states("voyage-code-3")
        assert "voyage-code-3" in excluded
        for state in EXCLUDED_STATES:
            assert state in excluded

    def test_build_excluded_states_supports_model_migration(self):
        """Switching target model only excludes the new target, not the old one."""
        excluded = build_excluded_states("voyage-3")
        assert "voyage-3" in excluded
        # A row with voyage-code-3 is NOT excluded — it can be re-embedded to voyage-3
        assert "voyage-code-3" not in excluded
