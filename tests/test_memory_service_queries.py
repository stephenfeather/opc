"""Tests for memory_service_queries — pure functions extracted from MemoryServicePG.

Tests the SQL building, embedding padding, result formatting, and ID generation
functions that contain no I/O and can be tested without mocking.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from scripts.core.db.memory_service_queries import (
    build_date_conditions,
    build_hybrid_search_sql,
    build_text_search_sql,
    build_vector_search_sql,
    filter_core_by_query,
    format_archival_row,
    format_context_string,
    format_recall_text,
    generate_memory_id,
    pad_embedding,
)

# ==================== generate_memory_id ====================


class TestGenerateMemoryId:
    """Tests for UUID generation."""

    def test_returns_valid_uuid_string(self):
        result = generate_memory_id()
        # Should be a valid UUID (parseable)
        parsed = uuid.UUID(result)
        assert str(parsed) == result

    def test_returns_unique_ids(self):
        ids = {generate_memory_id() for _ in range(100)}
        assert len(ids) == 100


# ==================== pad_embedding ====================


class TestPadEmbedding:
    """Tests for embedding normalization to target dimension."""

    def test_pads_short_embedding(self):
        short = [1.0, 2.0, 3.0]
        result = pad_embedding(short, target_dim=5)
        assert len(result) == 5
        assert result[:3] == [1.0, 2.0, 3.0]
        assert result[3:] == [0.0, 0.0]

    def test_truncates_long_embedding(self):
        long = list(range(2048))
        result = pad_embedding(long, target_dim=1024)
        assert len(result) == 1024
        assert result == list(range(1024))

    def test_exact_dimension_unchanged(self):
        exact = [0.5] * 1024
        result = pad_embedding(exact, target_dim=1024)
        assert len(result) == 1024
        assert result == exact

    def test_default_target_dim_is_1024(self):
        short = [1.0, 2.0]
        result = pad_embedding(short)
        assert len(result) == 1024

    def test_returns_list_not_ndarray(self):
        result = pad_embedding([1.0, 2.0], target_dim=4)
        assert isinstance(result, list)

    def test_empty_embedding_returns_zeros(self):
        result = pad_embedding([], target_dim=3)
        assert result == [0.0, 0.0, 0.0]


# ==================== format_archival_row ====================


class TestFormatArchivalRow:
    """Tests for converting database rows to result dicts."""

    def test_formats_basic_row(self):
        row = {
            "id": "abc-123",
            "content": "test fact",
            "metadata": json.dumps({"type": "test"}),
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        }
        result = format_archival_row(row)
        assert result == {
            "id": "abc-123",
            "content": "test fact",
            "metadata": {"type": "test"},
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        }

    def test_null_metadata_returns_empty_dict(self):
        row = {
            "id": "abc-123",
            "content": "test",
            "metadata": None,
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        }
        result = format_archival_row(row)
        assert result["metadata"] == {}

    def test_includes_extra_fields(self):
        row = {
            "id": "abc-123",
            "content": "test",
            "metadata": "{}",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            "similarity": 0.95,
        }
        result = format_archival_row(row, extra_fields=["similarity"])
        assert result["similarity"] == 0.95

    def test_extra_field_rrf_score(self):
        row = {
            "id": "abc-123",
            "content": "test",
            "metadata": "{}",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            "rrf_score": 0.033,
        }
        result = format_archival_row(row, extra_fields=["rrf_score"])
        assert result["rrf_score"] == 0.033

    def test_extra_field_with_float_conversion(self):
        """RRF scores from DB may be Decimal — ensure float conversion."""
        row = {
            "id": "abc-123",
            "content": "test",
            "metadata": "{}",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            "rrf_score": Decimal("0.033"),
        }
        result = format_archival_row(
            row, extra_fields=["rrf_score"], float_fields=["rrf_score"]
        )
        assert isinstance(result["rrf_score"], float)


# ==================== build_date_conditions ====================


class TestBuildDateConditions:
    """Tests for dynamic date filter SQL building."""

    def test_no_dates_returns_empty(self):
        conditions, params, next_idx = build_date_conditions(
            start_date=None, end_date=None, param_start_idx=4
        )
        assert conditions == []
        assert params == []
        assert next_idx == 4

    def test_start_date_only(self):
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        conditions, params, next_idx = build_date_conditions(
            start_date=dt, end_date=None, param_start_idx=4
        )
        assert conditions == ["created_at >= $4"]
        assert params == [dt]
        assert next_idx == 5

    def test_end_date_only(self):
        dt = datetime(2024, 12, 31, tzinfo=UTC)
        conditions, params, next_idx = build_date_conditions(
            start_date=None, end_date=dt, param_start_idx=4
        )
        assert conditions == ["created_at <= $4"]
        assert params == [dt]
        assert next_idx == 5

    def test_both_dates(self):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        conditions, params, next_idx = build_date_conditions(
            start_date=start, end_date=end, param_start_idx=4
        )
        assert conditions == ["created_at >= $4", "created_at <= $5"]
        assert params == [start, end]
        assert next_idx == 6

    def test_custom_start_idx(self):
        dt = datetime(2024, 6, 15, tzinfo=UTC)
        conditions, _params, next_idx = build_date_conditions(
            start_date=dt, end_date=None, param_start_idx=7
        )
        assert conditions == ["created_at >= $7"]
        assert next_idx == 8


# ==================== build_text_search_sql ====================


class TestBuildTextSearchSql:
    """Tests for text search SQL generation."""

    def test_basic_text_search(self):
        sql, params = build_text_search_sql(
            session_id="s1",
            agent_id=None,
            query="test query",
            limit=10,
        )
        assert "plainto_tsquery" in sql
        assert "ORDER BY rank DESC" in sql
        assert "s1" in params
        assert "test query" in params
        assert 10 in params

    def test_with_date_range(self):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        sql, params = build_text_search_sql(
            session_id="s1",
            agent_id=None,
            query="test",
            limit=10,
            start_date=start,
        )
        assert "created_at >= " in sql
        assert start in params


# ==================== build_vector_search_sql ====================


class TestBuildVectorSearchSql:
    """Tests for vector search SQL generation."""

    def test_basic_vector_search(self):
        embedding = [0.1] * 1024
        sql, params = build_vector_search_sql(
            session_id="s1",
            agent_id=None,
            query_embedding=embedding,
            limit=10,
        )
        assert "embedding" in sql
        assert "similarity" in sql
        assert "s1" in params

    def test_with_date_range(self):
        embedding = [0.1] * 1024
        end = datetime(2024, 12, 31, tzinfo=UTC)
        sql, params = build_vector_search_sql(
            session_id="s1",
            agent_id=None,
            query_embedding=embedding,
            limit=5,
            end_date=end,
        )
        assert "created_at <= " in sql
        assert end in params


# ==================== build_hybrid_search_sql ====================


class TestBuildHybridSearchSql:
    """Tests for hybrid (text + vector) search SQL generation."""

    def test_basic_hybrid_search(self):
        embedding = [0.1] * 1024
        sql, params = build_hybrid_search_sql(
            session_id="s1",
            agent_id=None,
            text_query="test",
            query_embedding=embedding,
            limit=10,
        )
        assert "combined_score" in sql
        assert "plainto_tsquery" in sql
        assert "embedding" in sql

    def test_custom_weights(self):
        embedding = [0.1] * 1024
        sql, params = build_hybrid_search_sql(
            session_id="s1",
            agent_id=None,
            text_query="test",
            query_embedding=embedding,
            limit=10,
            text_weight=0.5,
            vector_weight=0.5,
        )
        assert 0.5 in params


# ==================== format_recall_text ====================


class TestFormatRecallText:
    """Tests for recall output formatting."""

    def test_empty_returns_no_memories(self):
        result = format_recall_text(core_matches={}, archival_results=[])
        assert result == "No relevant memories found."

    def test_core_matches_formatted(self):
        result = format_recall_text(
            core_matches={"persona": "helpful assistant"},
            archival_results=[],
        )
        assert "[Core/persona]: helpful assistant" in result

    def test_archival_results_formatted(self):
        result = format_recall_text(
            core_matches={},
            archival_results=[{"content": "Python is preferred"}],
        )
        assert "[Archival]: Python is preferred" in result

    def test_both_combined(self):
        result = format_recall_text(
            core_matches={"lang": "Python"},
            archival_results=[{"content": "Uses pytest"}],
        )
        assert "[Core/lang]" in result
        assert "[Archival]" in result


# ==================== format_context_string ====================


class TestFormatContextString:
    """Tests for context generation formatting."""

    def test_empty_core_and_archival(self):
        result = format_context_string(core={}, archival_contents=[])
        assert "## Core Memory" in result
        assert "(empty)" in result

    def test_with_core_data(self):
        result = format_context_string(
            core={"persona": "helper", "task": "coding"},
            archival_contents=[],
        )
        assert "**persona:** helper" in result
        assert "**task:** coding" in result

    def test_with_archival_data(self):
        result = format_context_string(
            core={},
            archival_contents=["Fact 1", "Fact 2"],
        )
        assert "- Fact 1" in result
        assert "- Fact 2" in result


# ==================== filter_core_by_query ====================


class TestFilterCoreByQuery:
    """Tests for core memory key matching during recall."""

    def test_exact_key_match(self):
        core = {"persona": "assistant", "task": "coding"}
        result = filter_core_by_query(core, "persona")
        assert result == {"persona": "assistant"}

    def test_partial_match(self):
        core = {"persona": "assistant", "task": "coding"}
        result = filter_core_by_query(core, "What is the persona?")
        assert "persona" in result

    def test_no_match(self):
        core = {"persona": "assistant"}
        result = filter_core_by_query(core, "weather")
        assert result == {}

    def test_case_insensitive(self):
        core = {"Persona": "assistant"}
        result = filter_core_by_query(core, "persona")
        assert "Persona" in result


# ==================== Superseded row filtering ====================


class TestSupersededFiltering:
    """SQL builders must exclude superseded rows (superseded_by IS NULL)."""

    def test_text_search_excludes_superseded(self):
        sql, _ = build_text_search_sql(
            session_id="s1", agent_id=None, query="test", limit=10,
        )
        assert "superseded_by IS NULL" in sql

    def test_vector_search_excludes_superseded(self):
        sql, _ = build_vector_search_sql(
            session_id="s1", agent_id=None,
            query_embedding=[0.1] * 1024, limit=10,
        )
        assert "superseded_by IS NULL" in sql

    def test_hybrid_search_excludes_superseded(self):
        sql, _ = build_hybrid_search_sql(
            session_id="s1", agent_id=None,
            text_query="test", query_embedding=[0.1] * 1024, limit=10,
        )
        assert "superseded_by IS NULL" in sql

    def test_text_search_omits_filter_when_disabled(self):
        sql, _ = build_text_search_sql(
            session_id="s1", agent_id=None, query="test", limit=10,
            include_active_filter=False,
        )
        assert "superseded_by" not in sql

    def test_vector_search_omits_filter_when_disabled(self):
        sql, _ = build_vector_search_sql(
            session_id="s1", agent_id=None,
            query_embedding=[0.1] * 1024, limit=10,
            include_active_filter=False,
        )
        assert "superseded_by" not in sql

    def test_hybrid_search_omits_filter_when_disabled(self):
        sql, _ = build_hybrid_search_sql(
            session_id="s1", agent_id=None,
            text_query="test", query_embedding=[0.1] * 1024, limit=10,
            include_active_filter=False,
        )
        assert "superseded_by" not in sql


# ==================== format_archival_row metadata handling ====================


class TestFormatArchivalRowMetadataTypes:
    """format_archival_row must handle both string and already-decoded metadata."""

    def test_string_metadata_decoded(self):
        row = {
            "id": "abc", "content": "t", "metadata": '{"k": "v"}',
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        }
        assert format_archival_row(row)["metadata"] == {"k": "v"}

    def test_dict_metadata_passed_through(self):
        row = {
            "id": "abc", "content": "t", "metadata": {"k": "v"},
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        }
        assert format_archival_row(row)["metadata"] == {"k": "v"}

    def test_none_metadata_returns_empty_dict(self):
        row = {
            "id": "abc", "content": "t", "metadata": None,
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        }
        assert format_archival_row(row)["metadata"] == {}

    def test_id_coerced_to_string(self):
        """UUIDs from asyncpg should be stringified."""
        uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        row = {
            "id": uid, "content": "t", "metadata": "{}",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        }
        result = format_archival_row(row)
        assert isinstance(result["id"], str)
        assert result["id"] == "12345678-1234-5678-1234-567812345678"
