"""Tests for store_learning.py TDD+FP refactor (S23).

Validates:
1. Pure functions: backend detection, metadata building, content building,
   dedup result checking, tag parsing, output formatting
2. I/O handlers: store_learning_v2, store_learning (legacy), rejection recording
3. CLI: argument parsing, main orchestration
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.store_learning import (  # noqa: E402
    CONFIDENCE_LEVELS,
    LEARNING_TYPES,
    build_learning_content,
    build_metadata,
    check_dedup_result,
    detect_backend,
    format_output,
    get_rejection_count,
    parse_cli_args,
    parse_tags,
    store_learning,
    store_learning_v2,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TYPES = [
    "FAILED_APPROACH",
    "WORKING_SOLUTION",
    "USER_PREFERENCE",
    "CODEBASE_PATTERN",
    "ARCHITECTURAL_DECISION",
    "ERROR_FIX",
    "OPEN_THREAD",
]


# ===========================================================================
# Pure Function Tests
# ===========================================================================


class TestDetectBackend:
    """Tests for detect_backend() — pure env-based backend selection."""

    def test_returns_postgres_when_database_url_set(self) -> None:
        env = {"DATABASE_URL": "postgresql://localhost/test"}
        assert detect_backend(env) == "postgres"

    def test_returns_postgres_when_continuous_claude_db_url_set(self) -> None:
        env = {"CONTINUOUS_CLAUDE_DB_URL": "postgresql://localhost/test"}
        assert detect_backend(env) == "postgres"

    def test_prefers_postgres_over_fallback(self) -> None:
        env = {"DATABASE_URL": "postgresql://localhost/test"}
        assert detect_backend(env, fallback="sqlite") == "postgres"

    def test_returns_fallback_when_no_env_vars(self) -> None:
        assert detect_backend({}, fallback="sqlite") == "sqlite"

    def test_returns_fallback_default_when_no_env_vars(self) -> None:
        # When no fallback and no env vars, delegates to get_default_backend
        with patch("scripts.core.store_learning.get_default_backend", return_value="sqlite"):
            result = detect_backend({})
        assert result == "sqlite"

    def test_ignores_empty_string_env_vars(self) -> None:
        env = {"DATABASE_URL": "", "CONTINUOUS_CLAUDE_DB_URL": ""}
        assert detect_backend(env, fallback="sqlite") == "sqlite"


class TestBuildMetadata:
    """Tests for build_metadata() — pure metadata dict construction."""

    def test_minimal_metadata(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        result = build_metadata(session_id="s1", timestamp=ts)
        assert result["type"] == "session_learning"
        assert result["session_id"] == "s1"
        assert result["timestamp"] == ts.isoformat()

    def test_includes_learning_type(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        result = build_metadata(session_id="s1", timestamp=ts, learning_type="ERROR_FIX")
        assert result["learning_type"] == "ERROR_FIX"

    def test_includes_all_optional_fields(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        result = build_metadata(
            session_id="s1",
            timestamp=ts,
            learning_type="WORKING_SOLUTION",
            context="hook dev",
            tags=["hooks", "patterns"],
            confidence="high",
            confidence_score=0.92,
            confidence_dimensions={"specificity": 0.9},
            host_id="mac-1",
            embedding_model="bge",
            project="opc",
            classification_reasoning="matched pattern",
        )
        assert result["context"] == "hook dev"
        assert result["tags"] == ["hooks", "patterns"]
        assert result["confidence"] == "high"
        assert result["confidence_score"] == 0.92
        assert result["confidence_dimensions"] == {"specificity": 0.9}
        assert result["host_id"] == "mac-1"
        assert result["embedding_model"] == "bge"
        assert result["project"] == "opc"
        assert result["classification_reasoning"] == "matched pattern"
        assert result["classified_by"] == "llm_judge"

    def test_omits_none_optional_fields(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        result = build_metadata(session_id="s1", timestamp=ts)
        assert "learning_type" not in result
        assert "context" not in result
        assert "tags" not in result
        assert "host_id" not in result

    def test_returns_new_dict_each_call(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        r1 = build_metadata(session_id="s1", timestamp=ts)
        r2 = build_metadata(session_id="s1", timestamp=ts)
        assert r1 is not r2


class TestBuildLearningContent:
    """Tests for build_learning_content() — legacy content assembly."""

    def test_assembles_all_parts(self) -> None:
        result = build_learning_content(
            worked="X worked", failed="Y failed",
            decisions="chose Z", patterns="pattern A",
        )
        assert "What worked: X worked" in result
        assert "What failed: Y failed" in result
        assert "Decisions: chose Z" in result
        assert "Patterns: pattern A" in result

    def test_skips_none_parts(self) -> None:
        result = build_learning_content(
            worked="X worked", failed="None",
            decisions="none", patterns="None",
        )
        assert "What worked: X worked" in result
        assert "failed" not in result
        assert "Decisions" not in result

    def test_returns_none_when_all_empty(self) -> None:
        result = build_learning_content(
            worked="None", failed="none",
            decisions="NONE", patterns="None",
        )
        assert result is None

    def test_returns_none_for_empty_strings(self) -> None:
        result = build_learning_content(
            worked="", failed="", decisions="", patterns="",
        )
        assert result is None

    def test_joins_with_newlines(self) -> None:
        result = build_learning_content(
            worked="A", failed="B", decisions="None", patterns="None",
        )
        assert result is not None
        lines = result.split("\n")
        assert len(lines) == 2


class TestCheckDedupResult:
    """Tests for check_dedup_result() — pure dedup decision logic."""

    def test_no_matches_returns_none(self) -> None:
        assert check_dedup_result(existing=[], threshold=0.85) is None

    def test_empty_list_returns_none(self) -> None:
        assert check_dedup_result(existing=None, threshold=0.85) is None

    def test_below_threshold_returns_none(self) -> None:
        matches = [{"similarity": 0.80, "session_id": "s1", "id": "abc"}]
        assert check_dedup_result(existing=matches, threshold=0.85) is None

    def test_above_threshold_returns_dedup_info(self) -> None:
        matches = [{"similarity": 0.90, "session_id": "s1", "id": "abc"}]
        result = check_dedup_result(existing=matches, threshold=0.85)
        assert result is not None
        assert result["similarity"] == 0.90
        assert result["existing_session"] == "s1"
        assert result["existing_id"] == "abc"

    def test_exact_threshold_returns_dedup_info(self) -> None:
        matches = [{"similarity": 0.85, "session_id": "s1", "id": "abc"}]
        result = check_dedup_result(existing=matches, threshold=0.85)
        assert result is not None

    def test_default_session_used_when_missing(self) -> None:
        """When match lacks session_id, default_session is used."""
        matches = [{"similarity": 0.90, "id": "abc"}]
        result = check_dedup_result(
            existing=matches, threshold=0.85, default_session="fallback-sess"
        )
        assert result is not None
        assert result["existing_session"] == "fallback-sess"

    def test_match_session_preferred_over_default(self) -> None:
        """When match has session_id, it takes priority over default."""
        matches = [{"similarity": 0.90, "session_id": "from-match", "id": "abc"}]
        result = check_dedup_result(
            existing=matches, threshold=0.85, default_session="fallback"
        )
        assert result is not None
        assert result["existing_session"] == "from-match"

    def test_uses_first_match_only(self) -> None:
        matches = [
            {"similarity": 0.90, "session_id": "s1", "id": "first"},
            {"similarity": 0.95, "session_id": "s2", "id": "second"},
        ]
        result = check_dedup_result(existing=matches, threshold=0.85)
        assert result is not None
        assert result["existing_id"] == "first"


class TestParseTags:
    """Tests for parse_tags() — pure comma-separated tag parsing."""

    def test_parses_comma_separated(self) -> None:
        assert parse_tags("hooks,patterns,test") == ["hooks", "patterns", "test"]

    def test_strips_whitespace(self) -> None:
        assert parse_tags(" hooks , patterns ") == ["hooks", "patterns"]

    def test_filters_empty_strings(self) -> None:
        assert parse_tags("hooks,,patterns,") == ["hooks", "patterns"]

    def test_returns_none_for_none_input(self) -> None:
        assert parse_tags(None) is None

    def test_returns_none_for_empty_string(self) -> None:
        assert parse_tags("") is None

    def test_returns_none_for_whitespace_only(self) -> None:
        assert parse_tags("  , , ") is None


class TestFormatOutput:
    """Tests for format_output() — pure CLI output formatting."""

    def test_skipped_result(self) -> None:
        result = {"success": True, "skipped": True, "reason": "duplicate"}
        output = format_output(result, json_mode=False)
        assert "skipped" in output.lower()
        assert "duplicate" in output

    def test_success_result(self) -> None:
        result = {
            "success": True,
            "memory_id": "abc-123",
            "backend": "postgres",
            "content_length": 42,
        }
        output = format_output(result, json_mode=False)
        assert "abc-123" in output
        assert "postgres" in output

    def test_failure_result(self) -> None:
        result = {"success": False, "error": "Connection refused"}
        output = format_output(result, json_mode=False)
        assert "Connection refused" in output

    def test_json_mode(self) -> None:
        result = {"success": True, "memory_id": "abc"}
        output = format_output(result, json_mode=True)
        parsed = json.loads(output)
        assert parsed["success"] is True
        assert "version" in parsed

    def test_json_mode_includes_version(self) -> None:
        result = {"success": True}
        output = format_output(result, json_mode=True)
        parsed = json.loads(output)
        assert "version" in parsed


class TestParseCliArgs:
    """Tests for parse_cli_args() — pure arg parsing."""

    def test_v2_mode_with_content(self) -> None:
        args = parse_cli_args([
            "--session-id", "s1",
            "--content", "something learned",
            "--type", "ERROR_FIX",
        ])
        assert args.session_id == "s1"
        assert args.content == "something learned"
        assert args.type == "ERROR_FIX"

    def test_legacy_mode_with_worked(self) -> None:
        args = parse_cli_args([
            "--session-id", "s1",
            "--worked", "X worked",
        ])
        assert args.session_id == "s1"
        assert args.worked == "X worked"

    def test_tags_as_string(self) -> None:
        args = parse_cli_args([
            "--session-id", "s1",
            "--content", "test",
            "--tags", "a,b,c",
        ])
        assert args.tags == "a,b,c"

    def test_auto_classify_flag(self) -> None:
        args = parse_cli_args([
            "--session-id", "s1",
            "--content", "test",
            "--auto-classify",
        ])
        assert args.auto_classify is True

    def test_supersedes_arg(self) -> None:
        args = parse_cli_args([
            "--session-id", "s1",
            "--content", "test",
            "--supersedes", "some-uuid",
        ])
        assert args.supersedes == "some-uuid"


# ===========================================================================
# Constants Validation
# ===========================================================================


class TestConstants:
    """Validate module-level constants."""

    def test_learning_types_complete(self) -> None:
        assert LEARNING_TYPES == VALID_TYPES

    def test_confidence_levels(self) -> None:
        assert CONFIDENCE_LEVELS == ["high", "medium", "low"]


# ===========================================================================
# I/O Handler Tests (mocked)
# ===========================================================================


class TestStoreLearningV2:
    """Tests for store_learning_v2 — I/O handler with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_empty_content_returns_error(self) -> None:
        result = await store_learning_v2(session_id="s1", content="")
        assert result["success"] is False
        assert "content" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_whitespace_content_returns_error(self) -> None:
        result = await store_learning_v2(session_id="s1", content="   ")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_successful_store(self) -> None:
        mock_memory = AsyncMock()
        mock_memory.search_vector_global = AsyncMock(return_value=[])
        mock_memory.store = AsyncMock(return_value="new-uuid")
        mock_memory.close = AsyncMock()

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder._provider = MagicMock(model="bge")

        with (
            patch(
                "scripts.core.store_learning.create_memory_service",
                new_callable=AsyncMock,
                return_value=mock_memory,
            ),
            patch(
                "scripts.core.store_learning.EmbeddingService",
                return_value=mock_embedder,
            ),
        ):
            result = await store_learning_v2(session_id="s1", content="Test learning")

        assert result["success"] is True
        assert result["memory_id"] == "new-uuid"
        mock_memory.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dedup_skips_duplicate(self) -> None:
        mock_memory = AsyncMock()
        mock_memory.search_vector_global = AsyncMock(
            return_value=[{"similarity": 0.95, "session_id": "s0", "id": "existing-id"}]
        )
        mock_memory.close = AsyncMock()

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder._provider = MagicMock(model="bge")

        with (
            patch(
                "scripts.core.store_learning.create_memory_service",
                new_callable=AsyncMock,
                return_value=mock_memory,
            ),
            patch(
                "scripts.core.store_learning.EmbeddingService",
                return_value=mock_embedder,
            ),
            patch("scripts.core.store_learning._record_rejection"),
        ):
            result = await store_learning_v2(session_id="s1", content="Dup content")

        assert result["success"] is True
        assert result["skipped"] is True
        assert "duplicate" in result["reason"]

    @pytest.mark.asyncio
    async def test_content_hash_dedup(self) -> None:
        """When store returns empty string, it's a content_hash duplicate."""
        mock_memory = AsyncMock()
        mock_memory.search_vector_global = AsyncMock(return_value=[])
        mock_memory.store = AsyncMock(return_value="")  # content_hash match
        mock_memory.close = AsyncMock()

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder._provider = MagicMock(model="bge")

        with (
            patch(
                "scripts.core.store_learning.create_memory_service",
                new_callable=AsyncMock,
                return_value=mock_memory,
            ),
            patch(
                "scripts.core.store_learning.EmbeddingService",
                return_value=mock_embedder,
            ),
            patch("scripts.core.store_learning._record_rejection"),
        ):
            result = await store_learning_v2(session_id="s1", content="Dup hash")

        assert result["skipped"] is True
        assert "content_hash" in result["reason"]

    @pytest.mark.asyncio
    async def test_supersedes_included_in_result(self) -> None:
        mock_memory = AsyncMock()
        mock_memory.search_vector_global = AsyncMock(return_value=[])
        mock_memory.store = AsyncMock(return_value="new-uuid")
        mock_memory.close = AsyncMock()

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder._provider = MagicMock(model="bge")

        with (
            patch(
                "scripts.core.store_learning.create_memory_service",
                new_callable=AsyncMock,
                return_value=mock_memory,
            ),
            patch(
                "scripts.core.store_learning.EmbeddingService",
                return_value=mock_embedder,
            ),
        ):
            result = await store_learning_v2(
                session_id="s1", content="New version",
                supersedes="old-uuid",
            )

        assert result["superseded"] == "old-uuid"


class TestStoreLearningLegacy:
    """Tests for store_learning() — legacy I/O handler."""

    @pytest.mark.asyncio
    async def test_all_none_returns_error(self) -> None:
        result = await store_learning(
            session_id="s1", worked="None", failed="None",
            decisions="None", patterns="None",
        )
        assert result["success"] is False
        assert "content" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_legacy_store(self) -> None:
        mock_memory = AsyncMock()
        mock_memory.store = AsyncMock(return_value="legacy-uuid")
        mock_memory.close = AsyncMock()

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)

        with (
            patch(
                "scripts.core.store_learning.create_memory_service",
                new_callable=AsyncMock,
                return_value=mock_memory,
            ),
            patch(
                "scripts.core.store_learning.EmbeddingService",
                return_value=mock_embedder,
            ),
        ):
            result = await store_learning(
                session_id="s1", worked="X worked",
                failed="None", decisions="None", patterns="None",
            )

        assert result["success"] is True
        assert result["memory_id"] == "legacy-uuid"


class TestGetRejectionCount:
    """Tests for get_rejection_count() — I/O handler."""

    def test_returns_zero_when_no_pg_url(self) -> None:
        with patch("scripts.core.store_learning._pg_url", return_value=None):
            assert get_rejection_count("s1") == 0

    def test_returns_count_from_db(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (5,)
        mock_conn.cursor.return_value = mock_cur

        with (
            patch("scripts.core.store_learning._pg_url", return_value="postgresql://test"),
            patch("psycopg2.connect", return_value=mock_conn),
        ):
            assert get_rejection_count("s1") == 5

    def test_returns_zero_on_exception(self) -> None:
        with (
            patch("scripts.core.store_learning._pg_url", return_value="postgresql://test"),
            patch("psycopg2.connect", side_effect=Exception("conn failed")),
        ):
            assert get_rejection_count("s1") == 0
