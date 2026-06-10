"""Tests for recall_backends — tsquery sanitization, query building, and result formatting."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
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

        row: dict[str, Any] = {
            "id": "abc",
            "session_id": "s-1",
            "content": "test",
            "metadata_json": '{"type": "session_learning"}',
            "created_at": 1704067200,  # 2024-01-01
            "rank": -5.0,
        }
        result = format_sqlite_result(row, divisor=10.0)
        assert result["similarity"] == 0.5

    def test_missing_fields_use_defaults(self):
        from scripts.core.recall_backends import format_sqlite_result

        row: dict[str, Any] = {
            "id": None,
            "session_id": None,
            "content": None,
            "metadata_json": None,
            "created_at": None,
            "rank": None,
        }
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


# ==================== project column merge (issue #130) ====================


class TestMergeProjectIntoMetadata:
    """Pure function: overlay archival_memory.project column onto metadata."""

    def test_column_set_and_metadata_missing(self):
        from scripts.core.recall_backends import merge_project_into_metadata

        merged = merge_project_into_metadata({"type": "x"}, {"project": "binbrain"})
        assert merged["project"] == "binbrain"

    def test_column_overrides_stale_metadata(self):
        from scripts.core.recall_backends import merge_project_into_metadata

        merged = merge_project_into_metadata(
            {"project": "stale-value"}, {"project": "opc"}
        )
        assert merged["project"] == "opc"

    def test_null_column_preserves_existing_metadata(self):
        from scripts.core.recall_backends import merge_project_into_metadata

        merged = merge_project_into_metadata({"project": "kept"}, {"project": None})
        assert merged["project"] == "kept"

    def test_missing_column_passes_through(self):
        from scripts.core.recall_backends import merge_project_into_metadata

        metadata = {"type": "session_learning"}
        merged = merge_project_into_metadata(metadata, {"id": "abc"})
        assert merged == {"type": "session_learning"}

    def test_does_not_mutate_input_metadata(self):
        from scripts.core.recall_backends import merge_project_into_metadata

        metadata: dict[str, Any] = {"type": "x"}
        merge_project_into_metadata(metadata, {"project": "opc"})
        assert "project" not in metadata


class TestFormattersCarryProject:
    """Each row formatter must surface the project column for the reranker."""

    def test_format_text_result_merges_project(self):
        from scripts.core.recall_backends import format_text_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-abc123",
            "content": "test content",
            "metadata": {"type": "session_learning"},
            "created_at": datetime(2026, 1, 1),
            "similarity": 0.85,
            "project": "binbrain",
        }
        result = format_text_result(row)
        assert result["metadata"]["project"] == "binbrain"

    def test_format_rrf_result_merges_project(self):
        from scripts.core.recall_backends import format_rrf_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-abc",
            "content": "x",
            "metadata": {"project": "stale"},
            "created_at": datetime(2026, 1, 1),
            "rrf_score": 0.03,
            "fts_rank": 1,
            "vec_rank": 2,
            "project": "opc",
        }
        result = format_rrf_result(row, has_decay=False)
        assert result["metadata"]["project"] == "opc"

    def test_format_vector_result_merges_project(self):
        from scripts.core.recall_backends import format_vector_result

        row: dict[str, Any] = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "session_id": "s-abc",
            "content": "x",
            "metadata": {},
            "created_at": datetime(2026, 1, 1),
            "similarity": 0.9,
            "project": "agentic-work",
        }
        result = format_vector_result(row)
        assert result is not None
        assert result["metadata"]["project"] == "agentic-work"

    def test_format_sqlite_result_without_project_column(self):
        from scripts.core.recall_backends import format_sqlite_result

        row: dict[str, Any] = {
            "id": "abc",
            "session_id": "s-1",
            "content": "x",
            "metadata_json": '{"type": "session_learning"}',
            "created_at": 1700000000,
            "rank": -1.0,
        }
        result = format_sqlite_result(row, divisor=10.0)
        assert "project" not in result["metadata"]


class TestRecallSqlSelectsProject:
    """Every postgres recall SQL must SELECT the project column when the
    database has it, and omit it on pre-migration databases (issue #130)."""

    def _simple_templates(self):
        from scripts.core import recall_backends as rb

        return [
            rb._TEXT_ONLY_FTS_SQL,
            rb._TEXT_ONLY_FTS_NO_CHAIN_SQL,
            rb._TEXT_ONLY_ILIKE_SQL,
            rb._TEXT_ONLY_ILIKE_NO_CHAIN_SQL,
        ]

    def _chain_templates(self):
        from scripts.core import recall_backends as rb

        return [rb._PG_RECENCY_SQL, rb._PG_VECTOR_SQL, rb._PG_TEXT_FALLBACK_SQL]

    def test_postgres_recall_sql_selects_project_when_available(self):
        from scripts.core.recall_backends import render_recall_sql

        for tmpl in self._simple_templates():
            sql = render_recall_sql(tmpl, include_project=True)
            assert "project" in sql.split("FROM")[0], f"missing: {sql[:120]}"
        for tmpl in self._chain_templates():
            sql = render_recall_sql(tmpl, include_project=True, chain_filter="")
            assert "project" in sql.split("FROM")[0], f"missing: {sql[:120]}"

    def test_postgres_recall_sql_omits_project_when_unavailable(self):
        from scripts.core.recall_backends import render_recall_sql

        for tmpl in self._simple_templates():
            sql = render_recall_sql(tmpl, include_project=False)
            assert "project" not in sql.split("FROM")[0], f"present: {sql[:120]}"
        for tmpl in self._chain_templates():
            sql = render_recall_sql(tmpl, include_project=False, chain_filter="")
            assert "project" not in sql.split("FROM")[0], f"present: {sql[:120]}"

    def test_rrf_tails_select_project_when_available(self):
        from scripts.core import recall_backends as rb

        for tmpl in (rb._RRF_BOOSTED_TAIL_SQL, rb._RRF_PLAIN_TAIL_SQL):
            sql = rb.render_recall_sql(
                tmpl, include_project=True, project_expr=", a.project",
            )
            assert "a.project" in sql.split("FROM")[0]
            sql_without = rb.render_recall_sql(
                tmpl, include_project=False, project_expr=", a.project",
            )
            assert "a.project" not in sql_without.split("FROM")[0]

    def test_rendered_sql_has_no_unfilled_placeholders(self):
        from scripts.core import recall_backends as rb

        for tmpl in self._simple_templates():
            for include in (True, False):
                sql = rb.render_recall_sql(tmpl, include_project=include)
                assert "{" not in sql and "}" not in sql
        for tmpl in self._chain_templates():
            sql = rb.render_recall_sql(
                tmpl, include_project=True,
                chain_filter="AND superseded_by IS NULL",
            )
            assert "{" not in sql and "}" not in sql


def _undefined_column_error() -> Exception:
    from asyncpg.exceptions import UndefinedColumnError

    return UndefinedColumnError("column \"project\" does not exist")


class TestProjectColumnProbe:
    """Capability probe: recall degrades gracefully on pre-migration DBs.

    Cache semantics (review round 2): only definitive answers are cached.
    Transient probe failures must NOT permanently disable project scoping.
    """

    def _make_conn(self, outcome: Exception | None):
        class FakeConn:
            def __init__(self) -> None:
                self.calls = 0
                self.outcome: Exception | None = outcome

            async def fetch(self, _sql: str, *args: Any) -> list[Any]:
                self.calls += 1
                if self.outcome is not None:
                    raise self.outcome
                return []

        return FakeConn()

    async def test_probe_true_when_column_exists(self):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = self._make_conn(None)
        assert await rb.project_column_available(conn) is True

    async def test_probe_false_when_column_missing(self):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = self._make_conn(_undefined_column_error())
        assert await rb.project_column_available(conn) is False

    async def test_probe_targets_actual_relation(self):
        """Probe must touch archival_memory.project itself, not
        information_schema by bare table name (schema/search_path skew)."""
        from scripts.core import recall_backends as rb

        sql = rb._PROJECT_COLUMN_PROBE_SQL
        assert "archival_memory" in sql
        assert "project" in sql
        assert "information_schema" not in sql

    async def test_definitive_results_are_cached(self):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = self._make_conn(None)
        await rb.project_column_available(conn)
        await rb.project_column_available(conn)
        assert conn.calls == 1

        rb.reset_project_column_cache()
        conn = self._make_conn(_undefined_column_error())
        await rb.project_column_available(conn)
        await rb.project_column_available(conn)
        assert conn.calls == 1

    async def test_transient_failure_not_cached(self):
        """A timeout/permission hiccup must not disable scoping for the
        process lifetime — the next call retries the probe."""
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = self._make_conn(RuntimeError("connection reset"))
        assert await rb.project_column_available(conn) is False
        conn.outcome = None  # transient issue clears
        assert await rb.project_column_available(conn) is True
        assert conn.calls == 2

    async def test_mark_project_column_missing_downgrades_cache(self):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = self._make_conn(None)
        assert await rb.project_column_available(conn) is True
        rb.mark_project_column_missing()
        assert await rb.project_column_available(conn) is False
        assert conn.calls == 1  # downgrade did not trigger a re-probe


class _FakeRecallDb:
    """Fake pool/conn pair simulating a DB with or without the project column."""

    def __init__(self, has_project_column: bool) -> None:
        self.has_project_column = has_project_column
        self.executed: list[str] = []

    def _project_referenced(self, sql: str) -> bool:
        return "project" in sql.split("FROM")[0]

    def make_pool(self):
        db = self

        class FakeConn:
            async def fetch(self, sql: str, *args: Any) -> list[Any]:
                if not db.has_project_column and db._project_referenced(sql):
                    from asyncpg.exceptions import UndefinedColumnError

                    raise UndefinedColumnError('column "project" does not exist')
                db.executed.append(sql)
                return []

        class FakeAcquire:
            async def __aenter__(self) -> FakeConn:
                return FakeConn()

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        class FakePool:
            def acquire(self) -> FakeAcquire:
                return FakeAcquire()

        return FakePool()


class TestOldDatabaseDegradation:
    """End-to-end: pre-migration DBs get project-free SQL, not errors."""

    async def test_text_only_runs_without_project_on_old_db(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        db = _FakeRecallDb(has_project_column=False)

        async def fake_get_pool():
            return db.make_pool()

        import scripts.core.db.postgres_pool as pool_mod

        monkeypatch.setattr(pool_mod, "get_pool", fake_get_pool)

        results = await rb.search_learnings_text_only_postgres("query terms", k=3)
        assert results == []
        assert db.executed, "expected SQL to be executed"
        for sql in db.executed:
            assert "project" not in sql.split("FROM")[0], sql[:120]

    async def test_stale_true_cache_recovers_dynamically(self, monkeypatch):
        """Review round 2: a wrong/stale has_project=True (schema drift after
        probe) must fall back to project-free SQL, not crash recall."""
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        db = _FakeRecallDb(has_project_column=False)

        async def fake_get_pool():
            return db.make_pool()

        import scripts.core.db.postgres_pool as pool_mod

        monkeypatch.setattr(pool_mod, "get_pool", fake_get_pool)

        # Poison the cache as if the column existed at probe time.
        rb._set_project_column_cache_for_tests(True)

        results = await rb.search_learnings_text_only_postgres("query terms", k=3)
        assert results == []
        assert db.executed, "expected fallback SQL to be executed"
        for sql in db.executed:
            assert "project" not in sql.split("FROM")[0], sql[:120]
