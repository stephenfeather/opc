"""Tests for recall_backends — tsquery sanitization, query building, and result formatting."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID


# ==================== tsquery Sanitization ====================


class TestSanitizeTsqueryWords:
    """Ensure tsquery metacharacters are stripped before building OR queries."""

    def test_plain_words_unchanged(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["session", "affinity"]) == ["session", "affinity"]

    def test_strips_exclamation(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["hello!", "world"]) == ["hello", "world"]

    def test_strips_ampersand(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["test&exploit"]) == ["testexploit"]

    def test_strips_parentheses(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["(inject)", "normal"]) == ["inject", "normal"]

    def test_strips_pipe_operator(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["a|b"]) == []  # "ab" is len 2, filtered

    def test_strips_angle_brackets(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["<->proximity"]) == ["proximity"]

    def test_strips_colon(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["weight:A"]) == ["weightA"]

    def test_filters_short_words_after_strip(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        # "!!" becomes "" -> filtered out
        assert sanitize_tsquery_words(["!!", "valid"]) == ["valid"]

    def test_empty_input(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words([]) == []

    def test_all_metacharacters(self):
        """All tsquery operators: ! & | ( ) < > : * are stripped."""
        from scripts.core.recall_backends import sanitize_tsquery_words

        result = sanitize_tsquery_words(
            ["!not", "&and", "|or", "(group)", "<prox>", ":weight", "*prefix"]
        )
        # Each should be stripped to just alphanumeric
        for word in result:
            assert re.match(r"^[a-zA-Z0-9]+$", word), f"Unclean word: {word!r}"

    def test_preserves_digits(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["error404", "http500"]) == ["error404", "http500"]


class TestBuildOrQuery:
    """Test the full or_query building pipeline including sanitization."""

    def test_injection_via_not_operator(self):
        """Query with ! should not produce tsquery NOT operator."""
        from scripts.core.recall_backends import sanitize_tsquery_words

        words = ["!secret", "data"]
        sanitized = sanitize_tsquery_words(words)
        or_query = " | ".join(sanitized)
        assert "!" not in or_query

    def test_injection_via_followed_by(self):
        """Query with <-> should not produce tsquery FOLLOWED BY operator."""
        from scripts.core.recall_backends import sanitize_tsquery_words

        words = ["a<->b", "test"]
        sanitized = sanitize_tsquery_words(words)
        or_query = " | ".join(sanitized)
        assert "<" not in or_query
        assert ">" not in or_query


# ==================== clean_query_text ====================


class TestCleanQueryText:
    """Pure function: normalize query string and remove stopwords."""

    def test_removes_stopwords(self):
        from scripts.core.recall_backends import clean_query_text

        stopwords = {"the", "is", "a"}
        result = clean_query_text("the quick brown fox", stopwords)
        assert result == "quick brown fox"

    def test_normalizes_hyphens_to_spaces(self):
        from scripts.core.recall_backends import clean_query_text

        result = clean_query_text("multi-terminal session", set())
        assert result == "multi terminal session"

    def test_lowercases_input(self):
        from scripts.core.recall_backends import clean_query_text

        result = clean_query_text("HELLO World", set())
        assert result == "hello world"

    def test_falls_back_to_original_when_all_stopwords(self):
        from scripts.core.recall_backends import clean_query_text

        stopwords = {"the", "is"}
        result = clean_query_text("the is", stopwords)
        assert result == "the is"

    def test_empty_query(self):
        from scripts.core.recall_backends import clean_query_text

        result = clean_query_text("", set())
        assert result == ""

    def test_preserves_non_stopwords(self):
        from scripts.core.recall_backends import clean_query_text

        stopwords = {"a", "an"}
        result = clean_query_text("a database an error", stopwords)
        assert result == "database error"


# ==================== build_fallback_words ====================


class TestBuildFallbackWords:
    """Pure function: produce fallback word list when sanitization yields nothing."""

    def test_extracts_first_word_alphanumeric(self):
        from scripts.core.recall_backends import build_fallback_words

        result = build_fallback_words("hello world")
        assert result == ["hello"]

    def test_strips_non_alnum_from_first_word(self):
        from scripts.core.recall_backends import build_fallback_words

        result = build_fallback_words("!test something")
        assert result == ["test"]

    def test_short_first_word_returns_as_is(self):
        from scripts.core.recall_backends import build_fallback_words

        # "a" is short but we still need something
        result = build_fallback_words("a")
        assert result == ["a"]

    def test_empty_query_returns_fallback(self):
        from scripts.core.recall_backends import build_fallback_words

        result = build_fallback_words("")
        assert result == ["a"]


# ==================== build_or_query ====================


class TestBuildOrQueryFunction:
    """Pure function: full pipeline from raw query to OR-joined tsquery string."""

    def test_simple_query(self):
        from scripts.core.recall_backends import build_or_query

        result = build_or_query("session affinity terminal", set())
        assert result == "session | affinity | terminal"

    def test_with_stopwords(self):
        from scripts.core.recall_backends import build_or_query

        result = build_or_query("the session is good", {"the", "is"})
        assert result == "session | good"

    def test_metacharacters_stripped(self):
        from scripts.core.recall_backends import build_or_query

        result = build_or_query("!inject & data", set())
        assert "!" not in result
        assert "&" not in result

    def test_fallback_when_all_filtered(self):
        from scripts.core.recall_backends import build_or_query

        # All words become too short after sanitization
        result = build_or_query("!!", set())
        assert len(result) > 0  # should produce a fallback


# ==================== normalize_bm25_score ====================


class TestNormalizeBm25Score:
    """Pure function: convert raw BM25 rank to 0.0-1.0 range."""

    def test_negative_rank_normalizes(self):
        from scripts.core.recall_backends import normalize_bm25_score

        result = normalize_bm25_score(-5.0, 10.0)
        assert result == 0.5

    def test_zero_rank(self):
        from scripts.core.recall_backends import normalize_bm25_score

        assert normalize_bm25_score(0.0, 10.0) == 0.0

    def test_clamps_to_max_one(self):
        from scripts.core.recall_backends import normalize_bm25_score

        result = normalize_bm25_score(-100.0, 10.0)
        assert result == 1.0

    def test_clamps_to_min_zero(self):
        from scripts.core.recall_backends import normalize_bm25_score

        result = normalize_bm25_score(5.0, 10.0)
        assert result == 0.0

    def test_none_rank_treated_as_zero(self):
        from scripts.core.recall_backends import normalize_bm25_score

        assert normalize_bm25_score(None, 10.0) == 0.0


# ==================== format_row_metadata ====================


class TestFormatRowMetadata:
    """Pure function: parse metadata from row, handling str or dict."""

    def test_dict_passthrough(self):
        from scripts.core.recall_backends import format_row_metadata

        meta = {"type": "session_learning", "tags": ["a"]}
        assert format_row_metadata(meta) == meta

    def test_json_string_parsed(self):
        from scripts.core.recall_backends import format_row_metadata

        result = format_row_metadata('{"type": "session_learning"}')
        assert result == {"type": "session_learning"}

    def test_invalid_json_returns_empty_dict(self):
        from scripts.core.recall_backends import format_row_metadata

        result = format_row_metadata("not json")
        assert result == {}

    def test_none_returns_empty_dict(self):
        from scripts.core.recall_backends import format_row_metadata

        result = format_row_metadata(None)
        assert result == {}


# ==================== format_text_result ====================


class TestFormatTextResult:
    """Pure function: convert a text search DB row to result dict."""

    def test_formats_basic_row(self):
        from scripts.core.recall_backends import format_text_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-abc123",
            "content": "test content",
            "metadata": {"type": "session_learning"},
            "created_at": datetime(2026, 1, 1),
            "similarity": 0.85,
        }
        result = format_text_result(row)
        assert result["id"] == "12345678-1234-1234-1234-123456789abc"
        assert result["session_id"] == "s-abc123"
        assert result["content"] == "test content"
        assert result["similarity"] == 0.85
        assert isinstance(result["metadata"], dict)

    def test_parses_string_metadata(self):
        from scripts.core.recall_backends import format_text_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-abc",
            "content": "x",
            "metadata": '{"type": "session_learning"}',
            "created_at": datetime(2026, 1, 1),
            "similarity": 0.5,
        }
        result = format_text_result(row)
        assert result["metadata"] == {"type": "session_learning"}


# ==================== format_sqlite_result ====================


class TestFormatSqliteResult:
    """Pure function: convert SQLite FTS5 row to result dict with normalized BM25."""

    def test_normalizes_bm25_score(self):
        from scripts.core.recall_backends import format_sqlite_result

        row = MagicMock()
        row.__getitem__ = lambda self, k: {
            "id": "abc",
            "session_id": "s-1",
            "content": "test",
            "metadata_json": '{"type": "session_learning"}',
            "created_at": 1704067200,  # 2024-01-01
            "rank": -5.0,
        }[k]
        result = format_sqlite_result(row, divisor=10.0)
        assert result["similarity"] == 0.5

    def test_missing_fields_use_defaults(self):
        from scripts.core.recall_backends import format_sqlite_result

        row = MagicMock()
        row.__getitem__ = lambda self, k: {
            "id": None,
            "session_id": None,
            "content": None,
            "metadata_json": None,
            "created_at": None,
            "rank": None,
        }[k]
        result = format_sqlite_result(row, divisor=10.0)
        assert result["id"] == ""
        assert result["session_id"] == "unknown"
        assert result["content"] == ""
        assert result["metadata"] == {}
        assert result["created_at"] is None
        assert result["similarity"] == 0.0


# ==================== format_rrf_result ====================


class TestFormatRrfResult:
    """Pure function: convert RRF row to result dict."""

    def test_with_decay_columns(self):
        from scripts.core.recall_backends import format_rrf_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-1",
            "content": "test",
            "metadata": {"type": "session_learning"},
            "created_at": datetime(2026, 1, 1),
            "boosted_score": 0.045,
            "raw_rrf_score": 0.032,
            "recall_count": 3,
            "last_recalled": datetime(2026, 3, 1),
            "fts_rank": 1,
            "vec_rank": 2,
        }
        result = format_rrf_result(row, has_decay=True)
        assert result["similarity"] == 0.045
        assert result["raw_rrf_score"] == 0.032
        assert result["recall_count"] == 3
        assert result["fts_rank"] == 1
        assert result["vec_rank"] == 2

    def test_without_decay_columns(self):
        from scripts.core.recall_backends import format_rrf_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-1",
            "content": "test",
            "metadata": '{"type": "session_learning"}',
            "created_at": datetime(2026, 1, 1),
            "rrf_score": 0.028,
            "fts_rank": 3,
            "vec_rank": 5,
        }
        result = format_rrf_result(row, has_decay=False)
        assert result["similarity"] == 0.028
        assert "raw_rrf_score" not in result
        assert "recall_count" not in result


# ==================== format_vector_result ====================


class TestFormatVectorResult:
    """Pure function: convert vector search row to result dict."""

    def test_basic_vector_result(self):
        from scripts.core.recall_backends import format_vector_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-1",
            "content": "test",
            "metadata": {"type": "session_learning"},
            "created_at": datetime(2026, 1, 1),
            "similarity": 0.72,
        }
        result = format_vector_result(row)
        assert result["similarity"] == 0.72
        assert result["id"] == "12345678-1234-1234-1234-123456789abc"

    def test_with_recency_boost(self):
        from scripts.core.recall_backends import format_vector_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-1",
            "content": "test",
            "metadata": {"type": "session_learning"},
            "created_at": datetime(2026, 1, 1),
            "similarity": 0.72,
            "combined_score": 0.81,
            "recency": 0.9,
        }
        result = format_vector_result(row)
        assert result["similarity"] == 0.81  # uses combined_score
        assert result["raw_similarity"] == 0.72
        assert result["recency"] == 0.9

    def test_below_threshold_returns_none(self):
        from scripts.core.recall_backends import format_vector_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-1",
            "content": "test",
            "metadata": {"type": "session_learning"},
            "created_at": datetime(2026, 1, 1),
            "similarity": 0.1,
        }
        result = format_vector_result(row, similarity_threshold=0.5)
        assert result is None


# ==================== build_rrf_cte ====================


class TestBuildRrfCte:
    """Pure function: build SQL CTE string for RRF queries."""

    def test_chain_filter_included(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=True, use_tsquery=False)
        assert "superseded_by IS NULL" in sql

    def test_chain_filter_excluded(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=False, use_tsquery=False)
        assert "superseded_by" not in sql

    def test_uses_to_tsquery_when_flagged(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=True, use_tsquery=True)
        assert "to_tsquery" in sql
        assert "plainto_tsquery" not in sql

    def test_uses_plainto_tsquery_by_default(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=True, use_tsquery=False)
        assert "plainto_tsquery" in sql

    def test_returns_string(self):
        from scripts.core.recall_backends import build_rrf_cte

        result = build_rrf_cte(chain_filter=False, use_tsquery=False)
        assert isinstance(result, str)
        assert "WITH fts_ranked AS" in result
