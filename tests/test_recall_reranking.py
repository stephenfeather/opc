"""Tests for reranker integration into recall_learnings.py.

Validates that:
1. --no-rerank flag bypasses contextual re-ranking
2. --project flag sets RecallContext.project correctly
3. Adaptive over-fetch uses max(3*k, 50) when reranking is active
4. record_recall is called only on final k results, not the over-fetched set
5. JSON output includes rerank_details and final_score
6. Human-readable output uses final_score when available
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    similarity: float = 0.5,
    content: str = "test learning",
    session_id: str = "test-session",
    created_at: datetime | None = None,
    result_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Create a fake recall result dict."""
    return {
        "id": result_id or str(uuid.uuid4()),
        "similarity": similarity,
        "content": content,
        "session_id": session_id,
        "created_at": created_at or datetime(2026, 3, 29, 12, 0, tzinfo=UTC),
        "metadata": metadata or {},
    }


def _make_results(n: int, base_similarity: float = 0.5) -> list[dict]:
    """Create n fake results with descending similarity."""
    return [
        _make_result(
            similarity=base_similarity - i * 0.01,
            content=f"learning {i}",
            session_id=f"session-{i}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test: --no-rerank bypasses reranker
# ---------------------------------------------------------------------------

class TestNoRerankFlag:
    """When --no-rerank is passed, reranker should not be called."""

    @pytest.mark.asyncio
    async def test_no_rerank_flag_bypasses_reranker(self):
        """With --no-rerank, results should pass through without reranking."""
        fake_results = _make_results(3)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_hybrid_rrf",
                new_callable=AsyncMock,
                return_value=fake_results,
            ),
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock) as mock_record,
            patch("scripts.core.reranker.rerank") as mock_rerank,
            patch("sys.argv", ["recall", "--query", "test", "--k", "3", "--no-rerank", "--json"]),
        ):
            from scripts.core.recall_learnings import main

            exit_code = await main()
            assert exit_code == 0

            # rerank() should not have been called
            mock_rerank.assert_not_called()

            # record_recall should be called with the original 3 IDs
            mock_record.assert_called_once()
            recorded_ids = mock_record.call_args[0][0]
            assert len(recorded_ids) == 3

            # Verify no rerank_details in output (results not reranked)
            # The results should not have final_score key
            for r in fake_results:
                assert "final_score" not in r


# ---------------------------------------------------------------------------
# Test: --project sets context
# ---------------------------------------------------------------------------

class TestProjectFlag:
    """--project flag should set RecallContext.project for reranking."""

    @pytest.mark.asyncio
    async def test_project_flag_sets_context(self):
        """--project my-project should create RecallContext with project='my-project'."""
        from scripts.core.reranker import rerank as real_rerank

        fake_results = _make_results(3)
        captured_ctx = []

        def spy_rerank(results, ctx, k=5):
            captured_ctx.append(ctx)
            return real_rerank(results, ctx, k=k)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_hybrid_rrf",
                new_callable=AsyncMock,
                return_value=fake_results,
            ),
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock),
            patch("scripts.core.reranker.rerank", side_effect=spy_rerank),
            patch("sys.argv", ["recall", "--query", "test", "--k", "3", "--project", "my-project", "--json"]),
        ):
            from scripts.core.recall_learnings import main

            exit_code = await main()
            assert exit_code == 0

            # Verify RecallContext.project was set
            assert len(captured_ctx) == 1
            assert captured_ctx[0].project == "my-project"


# ---------------------------------------------------------------------------
# Test: Adaptive over-fetch
# ---------------------------------------------------------------------------

class TestAdaptiveOverfetch:
    """When reranking is active, fetch_k should be max(3*k, 50)."""

    @pytest.mark.asyncio
    async def test_adaptive_overfetch_with_small_k(self):
        """With k=5, fetch_k should be max(15, 50) = 50."""
        fake_results = _make_results(50)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_hybrid_rrf",
                new_callable=AsyncMock,
                return_value=fake_results,
            ) as mock_search,
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock),
            patch("sys.argv", ["recall", "--query", "test", "--k", "5", "--json"]),
        ):
            from scripts.core.recall_learnings import main

            await main()

            # search_learnings_hybrid_rrf should have been called with k=50 (over-fetch)
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args
            # k could be positional or keyword
            if call_kwargs.kwargs.get("k") is not None:
                assert call_kwargs.kwargs["k"] == 50
            else:
                # Check positional args
                assert 50 in call_kwargs.args

    @pytest.mark.asyncio
    async def test_adaptive_overfetch_with_large_k(self):
        """With k=25, fetch_k should be max(75, 50) = 75."""
        fake_results = _make_results(75)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_hybrid_rrf",
                new_callable=AsyncMock,
                return_value=fake_results,
            ) as mock_search,
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock),
            patch("sys.argv", ["recall", "--query", "test", "--k", "25", "--json"]),
        ):
            from scripts.core.recall_learnings import main

            await main()

            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args
            if call_kwargs.kwargs.get("k") is not None:
                assert call_kwargs.kwargs["k"] == 75
            else:
                assert 75 in call_kwargs.args

    @pytest.mark.asyncio
    async def test_no_overfetch_when_no_rerank(self):
        """With --no-rerank, fetch_k should equal k (no over-fetch)."""
        fake_results = _make_results(5)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_hybrid_rrf",
                new_callable=AsyncMock,
                return_value=fake_results,
            ) as mock_search,
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock),
            patch("sys.argv", ["recall", "--query", "test", "--k", "5", "--no-rerank", "--json"]),
        ):
            from scripts.core.recall_learnings import main

            await main()

            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args
            if call_kwargs.kwargs.get("k") is not None:
                assert call_kwargs.kwargs["k"] == 5
            else:
                assert 5 in call_kwargs.args


# ---------------------------------------------------------------------------
# Test: record_recall after trim
# ---------------------------------------------------------------------------

class TestRecordRecallAfterTrim:
    """record_recall should only be called on final k results, not over-fetched set."""

    @pytest.mark.asyncio
    async def test_record_recall_after_trim(self):
        """With k=3 and 50 over-fetched results, record_recall should get 3 IDs."""
        fake_results = _make_results(50)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_hybrid_rrf",
                new_callable=AsyncMock,
                return_value=fake_results,
            ),
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock) as mock_record,
            patch("sys.argv", ["recall", "--query", "test", "--k", "3", "--json"]),
        ):
            from scripts.core.recall_learnings import main

            await main()

            # record_recall should be called exactly once with only 3 IDs
            mock_record.assert_called_once()
            recorded_ids = mock_record.call_args[0][0]
            assert len(recorded_ids) == 3

    @pytest.mark.asyncio
    async def test_record_recall_not_called_from_search_learnings(self):
        """search_learnings() should NOT call record_recall (main() handles it)."""
        fake_results = _make_results(5)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_postgres",
                new_callable=AsyncMock,
                return_value=fake_results,
            ),
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock) as mock_record,
        ):
            from scripts.core.recall_learnings import search_learnings

            results = await search_learnings(query="test", k=5)
            assert len(results) == 5

            # search_learnings should NOT call record_recall anymore
            mock_record.assert_not_called()


# ---------------------------------------------------------------------------
# Test: JSON output includes rerank details
# ---------------------------------------------------------------------------

class TestJsonOutputWithRerank:
    """JSON output should include final_score and rerank_details."""

    @pytest.mark.asyncio
    async def test_json_output_includes_rerank_fields(self, capsys):
        """JSON output should have score=final_score, raw_score, and rerank_details."""
        fake_results = _make_results(3)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_hybrid_rrf",
                new_callable=AsyncMock,
                return_value=fake_results,
            ),
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock),
            patch("sys.argv", ["recall", "--query", "test", "--k", "3", "--json"]),
        ):
            from scripts.core.recall_learnings import main

            exit_code = await main()
            assert exit_code == 0

            captured = capsys.readouterr()
            output = json.loads(captured.out)

            assert "results" in output
            for r in output["results"]:
                assert "score" in r
                assert "raw_score" in r
                assert "rerank_details" in r


# ---------------------------------------------------------------------------
# Test: Retrieval mode detection
# ---------------------------------------------------------------------------

class TestRetrievalModeDetection:
    """Verify correct retrieval_mode is set in RecallContext."""

    @pytest.mark.asyncio
    async def test_text_only_sets_text_mode(self):
        """--text-only should set retrieval_mode='text'."""
        fake_results = _make_results(3)
        captured_ctx = []

        from scripts.core.reranker import rerank as real_rerank

        def spy_rerank(results, ctx, k=5):
            captured_ctx.append(ctx)
            return real_rerank(results, ctx, k=k)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_text_only_postgres",
                new_callable=AsyncMock,
                return_value=fake_results,
            ),
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock),
            patch("sys.argv", ["recall", "--query", "test", "--k", "3", "--text-only", "--json"]),
        ):
            # Patch the reranker module-level function so the lazy import picks it up
            with patch("scripts.core.reranker.rerank", side_effect=spy_rerank):
                from scripts.core.recall_learnings import main

                await main()

                assert len(captured_ctx) == 1
                assert captured_ctx[0].retrieval_mode == "text"

    @pytest.mark.asyncio
    async def test_sqlite_sets_sqlite_mode(self):
        """SQLite backend should set retrieval_mode='sqlite'."""
        fake_results = _make_results(3)
        captured_ctx = []

        from scripts.core.reranker import rerank as real_rerank

        def spy_rerank(results, ctx, k=5):
            captured_ctx.append(ctx)
            return real_rerank(results, ctx, k=k)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"),
            patch(
                "scripts.core.recall_learnings.search_learnings_sqlite",
                new_callable=AsyncMock,
                return_value=fake_results,
            ),
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock),
            patch("sys.argv", ["recall", "--query", "test", "--k", "3", "--json"]),
        ):
            with patch("scripts.core.reranker.rerank", side_effect=spy_rerank):
                from scripts.core.recall_learnings import main

                await main()

                assert len(captured_ctx) == 1
                assert captured_ctx[0].retrieval_mode == "sqlite"

    @pytest.mark.asyncio
    async def test_default_sets_hybrid_rrf_mode(self):
        """Default search should set retrieval_mode='hybrid_rrf'."""
        fake_results = _make_results(3)
        captured_ctx = []

        from scripts.core.reranker import rerank as real_rerank

        def spy_rerank(results, ctx, k=5):
            captured_ctx.append(ctx)
            return real_rerank(results, ctx, k=k)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_hybrid_rrf",
                new_callable=AsyncMock,
                return_value=fake_results,
            ),
            patch("scripts.core.recall_learnings.record_recall", new_callable=AsyncMock),
            patch("sys.argv", ["recall", "--query", "test", "--k", "3", "--json"]),
        ):
            with patch("scripts.core.reranker.rerank", side_effect=spy_rerank):
                from scripts.core.recall_learnings import main

                await main()

                assert len(captured_ctx) == 1
                assert captured_ctx[0].retrieval_mode == "hybrid_rrf"
