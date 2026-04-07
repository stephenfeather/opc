"""Tests for recall_learnings.py — TDD+FP refactored pure functions and I/O wrappers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.core.recall_learnings import (
    apply_pattern_enrichment,
    build_pattern_lookup,
    compute_fetch_k,
    determine_retrieval_mode,
    filter_by_tags,
    get_backend,
    make_recall_context,
    resolve_search_params,
    select_output,
)


def _make_pool_mock(mock_conn):
    """Build a mock pool whose acquire() returns an async context manager yielding mock_conn."""
    mock_pool = MagicMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = mock_conn
    cm.__aexit__.return_value = False
    mock_pool.acquire.return_value = cm
    return mock_pool


def _patch_pool(mock_pool):
    """Return a patch context for get_pool that returns mock_pool."""
    async def fake_get_pool():
        return mock_pool
    return patch("scripts.core.db.postgres_pool.get_pool", side_effect=fake_get_pool)


# ---------------------------------------------------------------------------
# compute_fetch_k: pure
# ---------------------------------------------------------------------------
class TestComputeFetchK:
    def test_no_rerank_returns_k(self):
        assert compute_fetch_k(5, no_rerank=True) == 5

    def test_rerank_returns_3x_k_when_large(self):
        assert compute_fetch_k(20, no_rerank=False) == 60

    def test_rerank_returns_min_50(self):
        assert compute_fetch_k(5, no_rerank=False) == 50

    def test_rerank_3x_k_above_50(self):
        assert compute_fetch_k(25, no_rerank=False) == 75

    def test_zero_k_no_rerank(self):
        assert compute_fetch_k(0, no_rerank=True) == 0

    def test_zero_k_rerank(self):
        assert compute_fetch_k(0, no_rerank=False) == 50


# ---------------------------------------------------------------------------
# determine_retrieval_mode: pure
# ---------------------------------------------------------------------------
class TestDetermineRetrievalMode:
    def test_sqlite_backend(self):
        assert determine_retrieval_mode("sqlite", text_only=False, vector_only=False) == "sqlite"

    def test_text_only(self):
        assert determine_retrieval_mode("postgres", text_only=True, vector_only=False) == "text"

    def test_vector_only(self):
        assert determine_retrieval_mode("postgres", text_only=False, vector_only=True) == "vector"

    def test_hybrid_rrf_default(self):
        assert (
            determine_retrieval_mode("postgres", text_only=False, vector_only=False) == "hybrid_rrf"
        )

    def test_sqlite_ignores_flags(self):
        assert determine_retrieval_mode("sqlite", text_only=True, vector_only=True) == "sqlite"


# ---------------------------------------------------------------------------
# filter_by_tags: pure
# ---------------------------------------------------------------------------
class TestFilterByTags:
    def test_no_tags_returns_all(self):
        results = [{"id": "1", "metadata": {"tags": ["a"]}}]
        assert filter_by_tags(results, tags=None, strict=False) == results

    def test_not_strict_returns_all(self):
        results = [{"id": "1", "metadata": {"tags": ["a"]}}]
        assert filter_by_tags(results, tags=["b"], strict=False) == results

    def test_strict_filters(self):
        results = [
            {"id": "1", "metadata": {"tags": ["a", "b"]}},
            {"id": "2", "metadata": {"tags": ["c"]}},
        ]
        filtered = filter_by_tags(results, tags=["a"], strict=True)
        assert len(filtered) == 1
        assert filtered[0]["id"] == "1"

    def test_strict_empty_tags_returns_all(self):
        results = [{"id": "1", "metadata": {"tags": ["a"]}}]
        assert filter_by_tags(results, tags=[], strict=True) == results

    def test_strict_no_metadata(self):
        results = [{"id": "1"}]
        filtered = filter_by_tags(results, tags=["a"], strict=True)
        assert filtered == []

    def test_does_not_mutate_input(self):
        results = [
            {"id": "1", "metadata": {"tags": ["a"]}},
            {"id": "2", "metadata": {"tags": ["b"]}},
        ]
        original_len = len(results)
        filter_by_tags(results, tags=["a"], strict=True)
        assert len(results) == original_len


# ---------------------------------------------------------------------------
# build_pattern_lookup: pure
# ---------------------------------------------------------------------------
class TestBuildPatternLookup:
    def test_empty_rows(self):
        assert build_pattern_lookup([]) == {}

    def test_single_row(self):
        rows = [
            {
                "memory_id": "abc-123",
                "pattern_strength": 0.85,
                "pattern_tags": ["hook", "error"],
            }
        ]
        lookup = build_pattern_lookup(rows)
        assert "abc-123" == list(lookup.keys())[0]
        assert lookup["abc-123"]["pattern_strength"] == 0.85
        assert lookup["abc-123"]["pattern_tags"] == ["hook", "error"]

    def test_none_strength_defaults_zero(self):
        rows = [{"memory_id": "x", "pattern_strength": None, "pattern_tags": None}]
        lookup = build_pattern_lookup(rows)
        assert lookup["x"]["pattern_strength"] == 0.0
        assert lookup["x"]["pattern_tags"] == []


# ---------------------------------------------------------------------------
# apply_pattern_enrichment: pure
# ---------------------------------------------------------------------------
class TestApplyPatternEnrichment:
    def test_empty_results(self):
        assert apply_pattern_enrichment([], {}) == []

    def test_enriches_matching_results(self):
        results = [
            {"id": "abc", "content": "test"},
            {"id": "xyz", "content": "other"},
        ]
        lookup = {
            "abc": {"pattern_strength": 0.9, "pattern_tags": ["hook"]},
        }
        enriched = apply_pattern_enrichment(results, lookup)
        assert enriched[0]["pattern_strength"] == 0.9
        assert enriched[0]["pattern_tags"] == ["hook"]
        assert "pattern_strength" not in enriched[1]

    def test_does_not_mutate_input(self):
        results = [{"id": "abc", "content": "test"}]
        lookup = {"abc": {"pattern_strength": 0.9, "pattern_tags": ["hook"]}}
        enriched = apply_pattern_enrichment(results, lookup)
        assert "pattern_strength" not in results[0]
        assert "pattern_strength" in enriched[0]


# ---------------------------------------------------------------------------
# make_recall_context: pure
# ---------------------------------------------------------------------------
class TestMakeRecallContext:
    def test_basic(self):
        ctx = make_recall_context(
            project="opc", tags=["hook"], retrieval_mode="hybrid_rrf"
        )
        assert ctx.project == "opc"
        assert ctx.tags_hint == ["hook"]
        assert ctx.retrieval_mode == "hybrid_rrf"

    def test_none_project(self):
        ctx = make_recall_context(project=None, tags=None, retrieval_mode="text")
        assert ctx.project is None
        assert ctx.tags_hint is None


# ---------------------------------------------------------------------------
# get_backend: env-dependent (test with patches)
# ---------------------------------------------------------------------------
class TestGetBackend:
    def test_explicit_sqlite(self):
        with patch.dict("os.environ", {"AGENTICA_MEMORY_BACKEND": "sqlite"}, clear=False):
            assert get_backend() == "sqlite"

    def test_explicit_postgres(self):
        with patch.dict("os.environ", {"AGENTICA_MEMORY_BACKEND": "postgres"}, clear=False):
            assert get_backend() == "postgres"

    def test_database_url_means_postgres(self):
        env = {"DATABASE_URL": "postgresql://localhost/test"}
        with patch.dict(
            "os.environ", env, clear=False
        ), patch.dict("os.environ", {"AGENTICA_MEMORY_BACKEND": ""}, clear=False):
            assert get_backend() == "postgres"

    def test_continuous_claude_db_url_means_postgres(self):
        env = {"CONTINUOUS_CLAUDE_DB_URL": "postgresql://localhost/test"}
        with patch.dict(
            "os.environ", env, clear=False
        ), patch.dict("os.environ", {"AGENTICA_MEMORY_BACKEND": ""}, clear=False):
            assert get_backend() == "postgres"

    def test_no_env_defaults_sqlite(self):
        env_clear = {
            "AGENTICA_MEMORY_BACKEND": "",
            "CONTINUOUS_CLAUDE_DB_URL": "",
            "DATABASE_URL": "",
        }
        with patch.dict("os.environ", env_clear, clear=False):
            assert get_backend() == "sqlite"


# ---------------------------------------------------------------------------
# resolve_search_params: pure
# ---------------------------------------------------------------------------
class TestResolveSearchParams:
    def test_sqlite_mode(self):
        params = resolve_search_params(
            backend="sqlite",
            text_only=False,
            vector_only=False,
            query="test",
            fetch_k=50,
            provider="local",
            threshold=0.2,
            recency=0.1,
            no_rerank=False,
            no_expand=False,
            expand_terms=5,
            rebuild_idf=False,
        )
        assert params["mode"] == "sqlite"

    def test_text_only_mode(self):
        params = resolve_search_params(
            backend="postgres",
            text_only=True,
            vector_only=False,
            query="test",
            fetch_k=50,
            provider="local",
            threshold=0.2,
            recency=0.1,
            no_rerank=False,
            no_expand=False,
            expand_terms=5,
            rebuild_idf=False,
        )
        assert params["mode"] == "text_only"

    def test_vector_only_with_rerank(self):
        params = resolve_search_params(
            backend="postgres",
            text_only=False,
            vector_only=True,
            query="test",
            fetch_k=50,
            provider="local",
            threshold=0.2,
            recency=0.1,
            no_rerank=False,
            no_expand=False,
            expand_terms=5,
            rebuild_idf=False,
        )
        assert params["mode"] == "vector"
        assert params["recency_weight"] == 0.0  # suppressed for reranking
        assert params["text_fallback"] is True

    def test_vector_only_no_rerank_keeps_recency(self):
        params = resolve_search_params(
            backend="postgres",
            text_only=False,
            vector_only=True,
            query="test",
            fetch_k=50,
            provider="local",
            threshold=0.2,
            recency=0.3,
            no_rerank=True,
            no_expand=False,
            expand_terms=5,
            rebuild_idf=False,
        )
        assert params["mode"] == "vector"
        assert params["recency_weight"] == 0.3

    def test_hybrid_rrf_default(self):
        params = resolve_search_params(
            backend="postgres",
            text_only=False,
            vector_only=False,
            query="test",
            fetch_k=50,
            provider="local",
            threshold=0.2,
            recency=0.1,
            no_rerank=False,
            no_expand=False,
            expand_terms=5,
            rebuild_idf=False,
        )
        assert params["mode"] == "hybrid_rrf"
        assert params["similarity_threshold"] == pytest.approx(0.002)  # 0.2 * 0.01

    def test_hybrid_rrf_excludes_recency_weight(self):
        params = resolve_search_params(
            backend="postgres",
            text_only=False,
            vector_only=False,
            query="test",
            fetch_k=50,
            provider="local",
            threshold=0.2,
            recency=0.5,
            no_rerank=False,
            no_expand=False,
            expand_terms=5,
            rebuild_idf=False,
        )
        assert params["mode"] == "hybrid_rrf"
        assert "recency_weight" not in params

    def test_hybrid_no_expand(self):
        params = resolve_search_params(
            backend="postgres",
            text_only=False,
            vector_only=False,
            query="test",
            fetch_k=50,
            provider="local",
            threshold=0.2,
            recency=0.1,
            no_rerank=False,
            no_expand=True,
            expand_terms=5,
            rebuild_idf=False,
        )
        assert params["expand"] is False


# ---------------------------------------------------------------------------
# select_output: pure
# ---------------------------------------------------------------------------
class TestSelectOutput:
    def test_json_full(self):
        assert select_output(json_flag=False, json_full=True) == "json_full"

    def test_json(self):
        assert select_output(json_flag=True, json_full=False) == "json"

    def test_human(self):
        assert select_output(json_flag=False, json_full=False) == "human"

    def test_json_full_takes_priority(self):
        assert select_output(json_flag=True, json_full=True) == "json_full"


# ---------------------------------------------------------------------------
# _dispatch_search: I/O dispatch
# ---------------------------------------------------------------------------
class TestDispatchSearch:
    @pytest.mark.asyncio
    async def test_sqlite_mode(self):
        from scripts.core.recall_learnings import _dispatch_search

        with patch(
            "scripts.core.recall_learnings.search_learnings_sqlite",
            new_callable=AsyncMock,
            return_value=[{"id": "1"}],
        ) as mock:
            result = await _dispatch_search({"mode": "sqlite", "query": "test", "k": 5})
            mock.assert_called_once_with("test", 5)
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_text_only_mode(self):
        from scripts.core.recall_learnings import _dispatch_search

        with patch(
            "scripts.core.recall_learnings.search_learnings_text_only_postgres",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock:
            await _dispatch_search({"mode": "text_only", "query": "x", "k": 5})
            mock.assert_called_once_with("x", 5)

    @pytest.mark.asyncio
    async def test_vector_mode_passes_text_fallback(self):
        from scripts.core.recall_learnings import _dispatch_search

        with patch(
            "scripts.core.recall_learnings.search_learnings_postgres",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock:
            params = {
                "mode": "vector",
                "query": "x",
                "k": 5,
                "provider": "local",
                "similarity_threshold": 0.2,
                "recency_weight": 0.0,
                "text_fallback": True,
            }
            await _dispatch_search(params)
            mock.assert_called_once()
            _, kwargs = mock.call_args
            assert kwargs["text_fallback"] is True

    @pytest.mark.asyncio
    async def test_hybrid_rrf_mode(self):
        from scripts.core.recall_learnings import _dispatch_search

        with patch(
            "scripts.core.recall_learnings.search_learnings_hybrid_rrf",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock:
            params = {
                "mode": "hybrid_rrf",
                "query": "x",
                "k": 10,
                "provider": "local",
                "similarity_threshold": 0.002,
                "expand": True,
                "max_expansion_terms": 5,
                "rebuild_idf": False,
            }
            await _dispatch_search(params)
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# record_recall: I/O wrapper
# ---------------------------------------------------------------------------
class TestRecordRecall:
    @pytest.mark.asyncio
    async def test_empty_ids_noop(self):
        from scripts.core.recall_learnings import record_recall

        # Should return without touching DB
        await record_recall([])

    @pytest.mark.asyncio
    async def test_sqlite_backend_noop(self):
        from scripts.core.recall_learnings import record_recall

        with patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"):
            await record_recall(["abc"])  # should not raise

    @pytest.mark.asyncio
    async def test_postgres_calls_execute(self):
        from scripts.core.recall_learnings import record_recall

        mock_conn = AsyncMock()
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            _patch_pool(mock_pool),
        ):
            await record_recall(["id-1", "id-2"])
            mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_error_graceful(self):
        from scripts.core.recall_learnings import record_recall

        async def failing_pool():
            raise Exception("DB down")

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch("scripts.core.db.postgres_pool.get_pool", side_effect=failing_pool),
        ):
            await record_recall(["id-1"])  # should not raise


# ---------------------------------------------------------------------------
# enrich_with_pattern_strength: I/O wrapper
# ---------------------------------------------------------------------------
class TestEnrichWithPatternStrength:
    @pytest.mark.asyncio
    async def test_empty_results_noop(self):
        from scripts.core.recall_learnings import enrich_with_pattern_strength

        result = await enrich_with_pattern_strength([])
        assert result == []

    @pytest.mark.asyncio
    async def test_sqlite_backend_noop(self):
        from scripts.core.recall_learnings import enrich_with_pattern_strength

        results = [{"id": "abc"}]
        with patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"):
            out = await enrich_with_pattern_strength(results)
            assert out == results

    @pytest.mark.asyncio
    async def test_enriches_from_db(self):
        from scripts.core.recall_learnings import enrich_with_pattern_strength

        test_uuid = "12345678-1234-5678-1234-567812345678"
        results = [{"id": test_uuid, "content": "test"}]
        mock_rows = [
            {
                "memory_id": test_uuid,
                "pattern_strength": 0.85,
                "pattern_tags": ["hook"],
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            _patch_pool(mock_pool),
        ):
            out = await enrich_with_pattern_strength(results)
            assert out[0]["pattern_strength"] == 0.85

    @pytest.mark.asyncio
    async def test_db_error_returns_unchanged(self):
        from scripts.core.recall_learnings import enrich_with_pattern_strength

        results = [{"id": "abc", "content": "test"}]

        async def failing_pool():
            raise OSError("unreachable")

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch("scripts.core.db.postgres_pool.get_pool", side_effect=failing_pool),
        ):
            out = await enrich_with_pattern_strength(results)
            assert out == results
            assert "pattern_strength" not in out[0]


# ---------------------------------------------------------------------------
# search_learnings: dispatcher
# ---------------------------------------------------------------------------
class TestSearchLearnings:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        from scripts.core.recall_learnings import search_learnings

        assert await search_learnings("") == []
        assert await search_learnings("   ") == []

    @pytest.mark.asyncio
    async def test_dispatches_to_sqlite(self):
        from scripts.core.recall_learnings import search_learnings

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"),
            patch(
                "scripts.core.recall_learnings.search_learnings_sqlite",
                new_callable=AsyncMock,
                return_value=[{"id": "1"}],
            ) as mock_sqlite,
        ):
            results = await search_learnings("test", k=5)
            mock_sqlite.assert_called_once_with("test", 5)
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatches_to_postgres(self):
        from scripts.core.recall_learnings import search_learnings

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
            patch(
                "scripts.core.recall_learnings.search_learnings_postgres",
                new_callable=AsyncMock,
                return_value=[{"id": "1"}],
            ) as mock_pg,
        ):
            results = await search_learnings("test", k=5)
            mock_pg.assert_called_once()
            assert len(results) == 1


# ---------------------------------------------------------------------------
# Regression: SQLite path must not require Postgres imports
# ---------------------------------------------------------------------------
class TestSqlitePathIndependence:
    """Ensure SQLite code path works even when Postgres pool is unavailable."""

    @pytest.mark.asyncio
    async def test_search_learnings_sqlite_no_postgres(self):
        from scripts.core.recall_learnings import search_learnings

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"),
            patch(
                "scripts.core.recall_learnings.search_learnings_sqlite",
                new_callable=AsyncMock,
                return_value=[{"id": "1", "content": "test"}],
            ),
            patch(
                "scripts.core.db.postgres_pool.get_pool",
                side_effect=ImportError("asyncpg not installed"),
            ),
        ):
            results = await search_learnings("test query", k=3)
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_record_recall_sqlite_skips_postgres(self):
        from scripts.core.recall_learnings import record_recall

        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"),
            patch(
                "scripts.core.db.postgres_pool.get_pool",
                side_effect=ImportError("asyncpg not installed"),
            ),
        ):
            await record_recall(["some-id"])  # should not raise

    @pytest.mark.asyncio
    async def test_enrich_sqlite_skips_postgres(self):
        from scripts.core.recall_learnings import enrich_with_pattern_strength

        results = [{"id": "abc", "content": "test"}]
        with (
            patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"),
            patch(
                "scripts.core.db.postgres_pool.get_pool",
                side_effect=ImportError("asyncpg not installed"),
            ),
        ):
            out = await enrich_with_pattern_strength(results)
            assert out == results
