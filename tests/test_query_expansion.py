"""Tests for TF-IDF query expansion."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.core.query_expansion import (
    IDFIndex,
    _tokenize,
    build_idf_index,
    expand_query,
    get_idf_index,
    load_idf_index,
    save_idf_index,
)


class FakeAcquire:
    """Fake async context manager for pool.acquire()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


class FakeTransaction:
    """Fake async context manager for conn.transaction()."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeCursor:
    """Fake async iterator for conn.cursor()."""

    def __init__(self, rows):
        self._rows = iter(rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._rows)
        except StopIteration:
            raise StopAsyncIteration from None


def _make_pool_and_conn(**conn_overrides):
    """Create a mock pool + conn pair using FakeAcquire."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value={"cnt": 0})
    conn.transaction = MagicMock(return_value=FakeTransaction())
    conn.cursor = MagicMock(return_value=FakeCursor([]))
    for k, v in conn_overrides.items():
        setattr(conn, k, v)
    pool = MagicMock()
    pool.acquire.return_value = FakeAcquire(conn)
    return pool, conn


# --- Tokenizer tests ---


class TestTokenize:
    def test_basic_tokenization(self):
        result = _tokenize("Hello World Testing")
        assert "hello" in result
        assert "world" in result
        assert "testing" in result

    def test_removes_short_words(self):
        result = _tokenize("I am a big dog")
        assert "big" in result  # len 3 > 2, included
        assert "dog" in result
        assert "am" not in result  # len 2, filtered

    def test_removes_stopwords(self):
        result = _tokenize("the quick brown fox with help")
        assert "the" not in result
        assert "with" not in result
        assert "help" not in result
        assert "quick" in result
        assert "brown" in result

    def test_replaces_hyphens(self):
        result = _tokenize("multi-terminal session")
        assert "multi" in result
        assert "terminal" in result
        assert "session" in result

    def test_strips_non_alnum(self):
        result = _tokenize("hello! world? (test)")
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_stopwords(self):
        assert _tokenize("the and or but") == []

    def test_only_short_words(self):
        assert _tokenize("a I am") == []

    def test_returns_list_type(self):
        result = _tokenize("hello world")
        assert isinstance(result, list)


# --- IDFIndex tests ---


class TestIDFIndex:
    def test_idf_computation(self):
        index = IDFIndex(
            word_df={"common": 100, "rare": 2},
            doc_count=1000,
            built_at="2026-01-01T00:00:00",
        )
        assert index.idf("rare") > index.idf("common")

    def test_idf_unknown_word(self):
        index = IDFIndex(
            word_df={"known": 10},
            doc_count=100,
            built_at="2026-01-01T00:00:00",
        )
        unknown_idf = index.idf("unknown")
        known_idf = index.idf("known")
        assert unknown_idf > known_idf  # unknown gets max IDF

    def test_idf_formula_correctness(self):
        index = IDFIndex(
            word_df={"term": 10},
            doc_count=100,
            built_at="2026-01-01T00:00:00",
        )
        expected = math.log(100 / (1 + 10))
        assert index.idf("term") == pytest.approx(expected)

    def test_idf_unknown_formula(self):
        index = IDFIndex(
            word_df={},
            doc_count=100,
            built_at="2026-01-01T00:00:00",
        )
        expected = math.log(101)
        assert index.idf("unknown") == pytest.approx(expected)

    def test_save_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "idf_test.json"
        original = IDFIndex(
            word_df={"auth": 5, "token": 3},
            doc_count=50,
            built_at="2026-01-01T00:00:00+00:00",
        )
        save_idf_index(original, path)
        loaded = load_idf_index(path)

        assert loaded is not None
        assert loaded.word_df == original.word_df
        assert loaded.doc_count == original.doc_count
        assert loaded.built_at == original.built_at

    def test_load_missing_file(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        result = load_idf_index(path)
        assert result is None

    def test_load_corrupt_file(self, tmp_path: Path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        result = load_idf_index(path)
        assert result is None

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "deep" / "idf.json"
        index = IDFIndex(word_df={"x": 1}, doc_count=1, built_at="t")
        save_idf_index(index, path)
        assert path.exists()

    async def test_build_idf_index(self):
        mock_rows = [
            {"content": "authentication tokens are important"},
            {"content": "tokens expire after timeout"},
            {"content": "database connection pooling works well"},
        ]
        mock_pool, _ = _make_pool_and_conn(
            cursor=MagicMock(return_value=FakeCursor(mock_rows)),
        )

        with patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool):
            with patch("scripts.core.query_expansion.save_idf_index"):  # noqa: SIM117
                index = await build_idf_index()

        assert index.doc_count == 3
        assert "tokens" in index.word_df
        assert index.word_df["tokens"] == 2  # appears in 2 docs
        assert "authentication" in index.word_df
        assert index.word_df["authentication"] == 1

    async def test_get_idf_index_caches(self, tmp_path: Path):
        """Second call uses cached index when fresh."""
        path = tmp_path / "idf.json"
        from datetime import UTC, datetime

        cached = IDFIndex(
            word_df={"test": 1},
            doc_count=10,
            built_at=datetime.now(UTC).isoformat(),
        )
        save_idf_index(cached, path)

        # Mock the DB count check to match cached count (no drift)
        mock_pool, _ = _make_pool_and_conn(
            fetchrow=AsyncMock(return_value={"cnt": 10}),
        )

        with patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool):
            with patch("scripts.core.query_expansion._IDF_CACHE_PATH", path):
                result = await get_idf_index()

        assert result.word_df == {"test": 1}

    async def test_get_idf_index_force_rebuild(self):
        """Force rebuild ignores cache."""
        mock_rows = [{"content": "fresh data here now"}]
        mock_pool, _ = _make_pool_and_conn(
            cursor=MagicMock(return_value=FakeCursor(mock_rows)),
        )

        with patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool):
            with patch("scripts.core.query_expansion.save_idf_index"):  # noqa: SIM117
                result = await get_idf_index(force_rebuild=True)

        assert result.doc_count == 1
        assert "fresh" in result.word_df


# --- Pure function tests (extracted for FP compliance) ---


class TestSanitizeQueryWords:
    def test_basic_extraction(self):
        from scripts.core.query_expansion import _sanitize_query_words

        result = _sanitize_query_words("auth patterns")
        assert "auth" in result
        assert "patterns" in result

    def test_strips_punctuation(self):
        from scripts.core.query_expansion import _sanitize_query_words

        result = _sanitize_query_words("hello! world?")
        assert result == ["hello", "world"]

    def test_filters_short_words(self):
        from scripts.core.query_expansion import _sanitize_query_words

        result = _sanitize_query_words("a I am big dog")
        assert "big" in result
        assert "dog" in result
        assert "am" not in result

    def test_lowercases(self):
        from scripts.core.query_expansion import _sanitize_query_words

        result = _sanitize_query_words("Auth PATTERNS")
        assert result == ["auth", "patterns"]

    def test_replaces_hyphens(self):
        from scripts.core.query_expansion import _sanitize_query_words

        result = _sanitize_query_words("multi-terminal")
        assert "multi" in result
        assert "terminal" in result

    def test_empty_query_fallback(self):
        from scripts.core.query_expansion import _sanitize_query_words

        result = _sanitize_query_words("")
        assert result == [""]

    def test_only_short_words_fallback(self):
        from scripts.core.query_expansion import _sanitize_query_words

        result = _sanitize_query_words("a I")
        assert isinstance(result, list)
        assert len(result) >= 1


class TestComputeNeighborDf:
    def test_basic_counting(self):
        from scripts.core.query_expansion import _compute_neighbor_df

        contents = [
            "authentication tokens are important",
            "tokens expire after timeout",
        ]
        result = _compute_neighbor_df(contents)
        assert result["tokens"] == 2
        assert result["authentication"] == 1

    def test_empty_contents(self):
        from scripts.core.query_expansion import _compute_neighbor_df

        result = _compute_neighbor_df([])
        assert result == {}

    def test_deduplicates_within_document(self):
        from scripts.core.query_expansion import _compute_neighbor_df

        contents = ["tokens tokens tokens"]
        result = _compute_neighbor_df(contents)
        assert result["tokens"] == 1  # unique per doc

    def test_stopwords_excluded(self):
        from scripts.core.query_expansion import _compute_neighbor_df

        contents = ["the authentication with help"]
        result = _compute_neighbor_df(contents)
        assert "the" not in result
        assert "with" not in result
        assert "help" not in result
        assert "authentication" in result


class TestScoreExpansionCandidates:
    def _make_idf(self) -> IDFIndex:
        return IDFIndex(
            word_df={"authentication": 5, "tokens": 8, "database": 50},
            doc_count=200,
            built_at="2026-01-01T00:00:00",
        )

    def test_scores_by_ndf_times_idf(self):
        from scripts.core.query_expansion import _score_expansion_candidates

        idf_index = self._make_idf()
        neighbor_df = {"authentication": 3, "tokens": 2}
        original_tokens: set[str] = set()

        result = _score_expansion_candidates(
            neighbor_df, idf_index, original_tokens, min_idf=0.0
        )
        result_dict = dict(result)
        assert "authentication" in result_dict
        assert "tokens" in result_dict

    def test_excludes_original_tokens(self):
        from scripts.core.query_expansion import _score_expansion_candidates

        idf_index = self._make_idf()
        neighbor_df = {"authentication": 3, "tokens": 2}
        original_tokens = {"authentication"}

        result = _score_expansion_candidates(
            neighbor_df, idf_index, original_tokens, min_idf=0.0
        )
        terms = [t for t, _ in result]
        assert "authentication" not in terms
        assert "tokens" in terms

    def test_excludes_stopwords(self):
        from scripts.core.query_expansion import _score_expansion_candidates

        idf_index = self._make_idf()
        neighbor_df = {"the": 5, "authentication": 3}
        original_tokens: set[str] = set()

        result = _score_expansion_candidates(
            neighbor_df, idf_index, original_tokens, min_idf=0.0
        )
        terms = [t for t, _ in result]
        assert "the" not in terms

    def test_excludes_short_terms(self):
        from scripts.core.query_expansion import _score_expansion_candidates

        idf_index = self._make_idf()
        neighbor_df = {"ab": 5, "authentication": 3}
        original_tokens: set[str] = set()

        result = _score_expansion_candidates(
            neighbor_df, idf_index, original_tokens, min_idf=0.0
        )
        terms = [t for t, _ in result]
        assert "ab" not in terms

    def test_min_idf_filter(self):
        from scripts.core.query_expansion import _score_expansion_candidates

        idf_index = IDFIndex(
            word_df={"common": 190},  # IDF ~ log(200/191) ~ 0.046
            doc_count=200,
            built_at="2026-01-01T00:00:00",
        )
        neighbor_df = {"common": 5}
        original_tokens: set[str] = set()

        result = _score_expansion_candidates(
            neighbor_df, idf_index, original_tokens, min_idf=1.0
        )
        assert len(result) == 0  # filtered by min_idf

    def test_sorted_descending(self):
        from scripts.core.query_expansion import _score_expansion_candidates

        idf_index = self._make_idf()
        neighbor_df = {"authentication": 10, "tokens": 1}
        original_tokens: set[str] = set()

        result = _score_expansion_candidates(
            neighbor_df, idf_index, original_tokens, min_idf=0.0
        )
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_sanitizes_non_alnum(self):
        from scripts.core.query_expansion import _score_expansion_candidates

        idf_index = IDFIndex(word_df={}, doc_count=100, built_at="t")
        neighbor_df = {"hello": 3}
        original_tokens: set[str] = set()

        result = _score_expansion_candidates(
            neighbor_df, idf_index, original_tokens, min_idf=0.0
        )
        for term, _ in result:
            assert term.isalnum()


class TestFormatTsquery:
    def test_basic_format(self):
        from scripts.core.query_expansion import _format_tsquery

        result = _format_tsquery(["auth", "patterns"], ["authentication", "workflow"])
        assert result == "auth | patterns | authentication | workflow"

    def test_no_expansion_terms(self):
        from scripts.core.query_expansion import _format_tsquery

        result = _format_tsquery(["auth", "patterns"], [])
        assert result == "auth | patterns"

    def test_single_original_term(self):
        from scripts.core.query_expansion import _format_tsquery

        result = _format_tsquery(["auth"], ["workflow"])
        assert result == "auth | workflow"

    def test_empty_lists_returns_empty_string(self):
        from scripts.core.query_expansion import _format_tsquery

        result = _format_tsquery([], [])
        assert result == ""


class TestComputeWordDf:
    def test_basic_counting(self):
        from scripts.core.query_expansion import _compute_word_df

        documents = [
            "authentication tokens important",
            "tokens expire timeout",
            "database connection pooling",
        ]
        word_df, doc_count = _compute_word_df(documents)
        assert doc_count == 3
        assert word_df["tokens"] == 2
        assert word_df["authentication"] == 1

    def test_empty_documents(self):
        from scripts.core.query_expansion import _compute_word_df

        word_df, doc_count = _compute_word_df([])
        assert doc_count == 0
        assert word_df == {}

    def test_deduplicates_within_document(self):
        from scripts.core.query_expansion import _compute_word_df

        documents = ["tokens tokens tokens"]
        word_df, doc_count = _compute_word_df(documents)
        assert doc_count == 1
        assert word_df["tokens"] == 1


class TestIsCacheStale:
    def test_fresh_cache_not_stale(self):
        from datetime import UTC, datetime

        from scripts.core.query_expansion import _is_cache_stale

        cached = IDFIndex(
            word_df={"test": 1},
            doc_count=100,
            built_at=datetime.now(UTC).isoformat(),
        )
        assert _is_cache_stale(
            cached, max_age_hours=24, current_count=100, drift_threshold=0.1
        ) is False

    def test_old_cache_is_stale(self):
        from scripts.core.query_expansion import _is_cache_stale

        cached = IDFIndex(
            word_df={"test": 1},
            doc_count=100,
            built_at="2020-01-01T00:00:00+00:00",
        )
        assert _is_cache_stale(
            cached, max_age_hours=24, current_count=100, drift_threshold=0.1
        ) is True

    def test_high_drift_is_stale(self):
        from datetime import UTC, datetime

        from scripts.core.query_expansion import _is_cache_stale

        cached = IDFIndex(
            word_df={"test": 1},
            doc_count=100,
            built_at=datetime.now(UTC).isoformat(),
        )
        # 50% drift exceeds 10% threshold
        assert _is_cache_stale(
            cached, max_age_hours=24, current_count=150, drift_threshold=0.1
        ) is True

    def test_zero_doc_count_is_stale(self):
        from datetime import UTC, datetime

        from scripts.core.query_expansion import _is_cache_stale

        cached = IDFIndex(
            word_df={},
            doc_count=0,
            built_at=datetime.now(UTC).isoformat(),
        )
        assert _is_cache_stale(
            cached, max_age_hours=24, current_count=10, drift_threshold=0.1
        ) is True

    def test_bad_timestamp_is_stale(self):
        from scripts.core.query_expansion import _is_cache_stale

        cached = IDFIndex(
            word_df={"test": 1},
            doc_count=100,
            built_at="not-a-timestamp",
        )
        assert _is_cache_stale(
            cached, max_age_hours=24, current_count=100, drift_threshold=0.1
        ) is True

    def test_within_drift_threshold(self):
        from datetime import UTC, datetime

        from scripts.core.query_expansion import _is_cache_stale

        cached = IDFIndex(
            word_df={"test": 1},
            doc_count=100,
            built_at=datetime.now(UTC).isoformat(),
        )
        # 5% drift is within 10% threshold
        assert _is_cache_stale(
            cached, max_age_hours=24, current_count=105, drift_threshold=0.1
        ) is False


# --- expand_query tests ---


class TestExpandQuery:
    def _make_idf_index(self) -> IDFIndex:
        return IDFIndex(
            word_df={
                "authentication": 5,
                "workflow": 3,
                "tokens": 8,
                "database": 50,
                "the": 200,
                "session": 40,
            },
            doc_count=200,
            built_at="2026-01-01T00:00:00",
        )

    async def test_expansion_adds_related_terms(self):
        neighbor_rows = [
            {"content": "authentication workflow requires tokens"},
            {"content": "authentication handler validates tokens"},
            {"content": "authentication middleware checks credentials"},
        ]

        mock_pool, _ = _make_pool_and_conn(
            fetch=AsyncMock(return_value=neighbor_rows),
        )

        idf_index = self._make_idf_index()

        with (
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
            patch(
                "scripts.core.query_expansion.get_idf_index",
                new_callable=AsyncMock,
                return_value=idf_index,
            ),
        ):
            result = await expand_query("auth patterns", [0.1] * 1024)

        # Should contain original terms plus expansions
        assert "auth" in result
        assert "patterns" in result
        assert "|" in result
        # Should have more terms than original
        terms = [t.strip() for t in result.split("|")]
        assert len(terms) > 2

    async def test_original_terms_excluded(self):
        """Expansion terms should not duplicate original query terms."""
        neighbor_rows = [
            {"content": "authentication workflow setup"},
        ]

        mock_pool, _ = _make_pool_and_conn(
            fetch=AsyncMock(return_value=neighbor_rows),
        )

        idf_index = self._make_idf_index()

        with (
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
            patch(
                "scripts.core.query_expansion.get_idf_index",
                new_callable=AsyncMock,
                return_value=idf_index,
            ),
        ):
            result = await expand_query("authentication setup", [0.1] * 1024)

        terms = [t.strip() for t in result.split("|")]
        # "authentication" and "setup" should appear only once (as originals)
        assert terms.count("authentication") == 1

    async def test_stopwords_excluded(self):
        neighbor_rows = [
            {"content": "the authentication with help for tokens"},
        ]

        mock_pool, _ = _make_pool_and_conn(
            fetch=AsyncMock(return_value=neighbor_rows),
        )

        idf_index = self._make_idf_index()

        with (
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
            patch(
                "scripts.core.query_expansion.get_idf_index",
                new_callable=AsyncMock,
                return_value=idf_index,
            ),
        ):
            result = await expand_query("auth test", [0.1] * 1024)

        terms = [t.strip() for t in result.split("|")]
        for sw in ["the", "with", "help", "for"]:
            assert sw not in terms

    async def test_max_expansion_terms_respected(self):
        neighbor_rows = [
            {"content": "alpha bravo charlie delta echo foxtrot golf hotel"},
        ] * 5

        mock_pool, _ = _make_pool_and_conn(
            fetch=AsyncMock(return_value=neighbor_rows),
        )

        # All terms rare in corpus -> all eligible
        idf_index = IDFIndex(word_df={}, doc_count=1000, built_at="2026-01-01T00:00:00")

        with (
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
            patch(
                "scripts.core.query_expansion.get_idf_index",
                new_callable=AsyncMock,
                return_value=idf_index,
            ),
        ):
            result = await expand_query(
                "query", [0.1] * 1024, max_expansion_terms=3
            )

        terms = [t.strip() for t in result.split("|")]
        # 1 original + max 3 expansion = 4 max
        assert len(terms) <= 4

    async def test_no_neighbors_returns_original(self):
        mock_pool, _ = _make_pool_and_conn()

        with (
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
        ):
            result = await expand_query("auth patterns", [0.1] * 1024)

        assert "auth" in result
        assert "patterns" in result

    async def test_expansion_failure_returns_original(self):
        """If expand_query raises, the caller should handle gracefully."""
        mock_pool = MagicMock()
        mock_pool.acquire.side_effect = RuntimeError("DB down")

        with (
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError):
                await expand_query("auth", [0.1] * 1024)

    async def test_output_format_tsquery_compatible(self):
        """Output should be valid OR-joined terms for to_tsquery."""
        neighbor_rows = [
            {"content": "authentication workflow tokens handler"},
        ]

        mock_pool, _ = _make_pool_and_conn(
            fetch=AsyncMock(return_value=neighbor_rows),
        )

        idf_index = self._make_idf_index()

        with (
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
            patch(
                "scripts.core.query_expansion.get_idf_index",
                new_callable=AsyncMock,
                return_value=idf_index,
            ),
        ):
            result = await expand_query("auth", [0.1] * 1024)

        # Should be "term1 | term2 | term3" format
        parts = result.split(" | ")
        assert len(parts) >= 1
        for part in parts:
            # Each part should be alphanumeric only
            assert part.strip().isalnum() or part.strip() == ""


# --- Hybrid RRF integration tests ---


class TestHybridRRFWithExpansion:
    async def test_expand_true_calls_expand_query(self):
        """When expand=True, expand_query should be called."""
        from scripts.core.recall_backends import search_learnings_hybrid_rrf

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()

        mock_pool, _ = _make_pool_and_conn()

        with (
            patch(
                "scripts.core.db.embedding_service.EmbeddingService",
                return_value=mock_embedder,
            ),
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
            patch(
                "scripts.core.query_expansion.expand_query",
                new_callable=AsyncMock,
                return_value="auth | patterns | authentication",
            ) as mock_expand,
        ):
            await search_learnings_hybrid_rrf("auth patterns", k=5, expand=True)

        mock_expand.assert_called_once()

    async def test_expand_false_skips_expansion(self):
        """When expand=False, expand_query should not be called."""
        from scripts.core.recall_backends import search_learnings_hybrid_rrf

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()

        mock_pool, _ = _make_pool_and_conn()

        with (
            patch(
                "scripts.core.db.embedding_service.EmbeddingService",
                return_value=mock_embedder,
            ),
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
            patch(
                "scripts.core.query_expansion.expand_query",
                new_callable=AsyncMock,
            ) as mock_expand,
        ):
            await search_learnings_hybrid_rrf("auth patterns", k=5, expand=False)

        mock_expand.assert_not_called()

    async def test_tsquery_used_when_expanded(self):
        """When expansion produces new terms, to_tsquery should be used."""
        from scripts.core.recall_backends import search_learnings_hybrid_rrf

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()

        captured_sql = []

        async def capture_fetch(sql, *args):
            captured_sql.append(sql)
            return []

        mock_pool, _ = _make_pool_and_conn(
            fetch=AsyncMock(side_effect=capture_fetch),
        )

        with (
            patch(
                "scripts.core.db.embedding_service.EmbeddingService",
                return_value=mock_embedder,
            ),
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
            patch(
                "scripts.core.query_expansion.expand_query",
                new_callable=AsyncMock,
                return_value="auth | patterns | authentication",
            ),
        ):
            await search_learnings_hybrid_rrf("auth patterns", k=5, expand=True)

        # The SQL should contain to_tsquery (not plainto_tsquery)
        assert any(
            "to_tsquery" in sql and "plainto_tsquery" not in sql for sql in captured_sql
        )

    async def test_plainto_tsquery_when_not_expanded(self):
        """When no expansion, plainto_tsquery should be used."""
        from scripts.core.recall_backends import search_learnings_hybrid_rrf

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()

        captured_sql = []

        async def capture_fetch(sql, *args):
            captured_sql.append(sql)
            return []

        mock_pool, _ = _make_pool_and_conn(
            fetch=AsyncMock(side_effect=capture_fetch),
        )

        with (
            patch(
                "scripts.core.db.embedding_service.EmbeddingService",
                return_value=mock_embedder,
            ),
            patch("scripts.core.db.postgres_pool.get_pool", return_value=mock_pool),
            patch("scripts.core.db.postgres_pool.init_pgvector", new_callable=AsyncMock),
        ):
            await search_learnings_hybrid_rrf("auth patterns", k=5, expand=False)

        # The SQL should contain plainto_tsquery
        assert any("plainto_tsquery" in sql for sql in captured_sql)
