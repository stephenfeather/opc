"""Tests for scripts/core/db/postgres_pool.py — TDD + FP compliance refactor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.core.db.postgres_pool import (
    _decode_vector,
    _encode_vector,
    _sanitize_log_message,
    build_pool_config,
    close_pool,
    get_connection,
    get_connection_string,
    get_pool,
    get_pool_with_retry,
    get_transaction,
    health_check,
    init_pgvector,
    reset_pool,
    resolve_connection_url,
)

# ---------------------------------------------------------------------------
# Pure functions: _sanitize_log_message
# ---------------------------------------------------------------------------


class TestSanitizeLogMessage:
    def test_redacts_credentials(self):
        msg = "connection to postgresql://claude:claude_dev@localhost:5432/db failed"
        result = _sanitize_log_message(msg)
        assert "claude_dev" not in result
        assert "***@localhost" in result

    def test_no_credentials_unchanged(self):
        msg = "connection timed out"
        assert _sanitize_log_message(msg) == msg

    def test_multiple_urls_redacted(self):
        msg = "primary=postgresql://u:p@h1 fallback=postgresql://u2:p2@h2"
        result = _sanitize_log_message(msg)
        assert "p@" not in result
        assert "p2@" not in result


# ---------------------------------------------------------------------------
# Pure functions: _encode_vector
# ---------------------------------------------------------------------------


class TestEncodeVector:
    def test_list_of_floats(self):
        result = _encode_vector([1.0, 2.0, 3.0])
        assert result == "[1.0,2.0,3.0]"

    def test_single_element(self):
        result = _encode_vector([0.5])
        assert result == "[0.5]"

    def test_empty_list(self):
        result = _encode_vector([])
        assert result == "[]"

    def test_string_passthrough(self):
        already_formatted = "[1.0,2.0,3.0]"
        result = _encode_vector(already_formatted)
        assert result == already_formatted

    def test_integers_converted(self):
        result = _encode_vector([1, 2, 3])
        assert result == "[1,2,3]"


# ---------------------------------------------------------------------------
# Pure functions: _decode_vector
# ---------------------------------------------------------------------------


class TestDecodeVector:
    def test_standard_vector_string(self):
        result = _decode_vector("[1.0,2.0,3.0]")
        assert result == [1.0, 2.0, 3.0]

    def test_single_element(self):
        result = _decode_vector("[0.5]")
        assert result == [0.5]

    def test_integer_strings(self):
        result = _decode_vector("[1,2,3]")
        assert result == [1.0, 2.0, 3.0]

    def test_whitespace_handling(self):
        result = _decode_vector("  [1.0, 2.0, 3.0]  ")
        assert result == [1.0, 2.0, 3.0]

    def test_empty_vector(self):
        assert _decode_vector("[]") == []

    def test_empty_vector_with_whitespace(self):
        assert _decode_vector("  []  ") == []


# ---------------------------------------------------------------------------
# Pure functions: build_pool_config
# ---------------------------------------------------------------------------


class TestBuildPoolConfig:
    def test_defaults(self):
        result = build_pool_config("10")
        assert result == {"min_size": 2, "max_size": 10, "command_timeout": 60}

    def test_custom_max_size(self):
        result = build_pool_config("20")
        assert result["max_size"] == 20
        assert result["min_size"] == 4  # 20 // 5

    def test_min_size_floor_is_two(self):
        result = build_pool_config("5")
        assert result["min_size"] == 2  # max(2, 5 // 5) = max(2, 1) = 2

    def test_invalid_max_size_uses_default(self):
        result = build_pool_config("invalid")
        assert result["max_size"] == 10

    def test_zero_max_size_uses_default(self):
        result = build_pool_config("0")
        assert result["max_size"] == 10

    def test_negative_max_size_uses_default(self):
        result = build_pool_config("-5")
        assert result["max_size"] == 10

    def test_command_timeout_always_60(self):
        result = build_pool_config("10")
        assert result["command_timeout"] == 60

    def test_max_size_one_clamps_min_size(self):
        result = build_pool_config("1")
        assert result["max_size"] == 1
        assert result["min_size"] <= result["max_size"]


# ---------------------------------------------------------------------------
# Pure functions: resolve_connection_url
# ---------------------------------------------------------------------------


class TestResolveConnectionUrl:
    def test_canonical_url_wins(self):
        result = resolve_connection_url(
            continuous_claude_db_url="postgresql://canonical",
            database_url="postgresql://fallback",
            opc_postgres_url="postgresql://legacy",
            environment="",
        )
        assert result == "postgresql://canonical"

    def test_database_url_fallback(self):
        result = resolve_connection_url(
            continuous_claude_db_url=None,
            database_url="postgresql://fallback",
            opc_postgres_url=None,
            environment="",
        )
        assert result == "postgresql://fallback"

    def test_opc_postgres_url_fallback(self):
        result = resolve_connection_url(
            continuous_claude_db_url=None,
            database_url=None,
            opc_postgres_url="postgresql://legacy",
            environment="",
        )
        assert result == "postgresql://legacy"

    def test_dev_default_when_no_url_and_empty_env(self):
        result = resolve_connection_url(
            continuous_claude_db_url=None,
            database_url=None,
            opc_postgres_url=None,
            environment="",
        )
        assert result == "postgresql://claude:claude_dev@localhost:5432/continuous_claude"

    def test_dev_default_when_explicit_development(self):
        result = resolve_connection_url(
            continuous_claude_db_url=None,
            database_url=None,
            opc_postgres_url=None,
            environment="development",
        )
        assert result == "postgresql://claude:claude_dev@localhost:5432/continuous_claude"

    def test_test_env_raises_without_url(self):
        with pytest.raises(ValueError, match="Database URL must be set"):
            resolve_connection_url(
                continuous_claude_db_url=None,
                database_url=None,
                opc_postgres_url=None,
                environment="test",
            )

    def test_production_raises_without_url(self):
        with pytest.raises(ValueError, match="Database URL must be set"):
            resolve_connection_url(
                continuous_claude_db_url=None,
                database_url=None,
                opc_postgres_url=None,
                environment="production",
            )

    def test_staging_raises_without_url(self):
        with pytest.raises(ValueError, match="Database URL must be set"):
            resolve_connection_url(
                continuous_claude_db_url=None,
                database_url=None,
                opc_postgres_url=None,
                environment="staging",
            )

    def test_unknown_env_raises_without_url(self):
        with pytest.raises(ValueError, match="Database URL must be set"):
            resolve_connection_url(
                continuous_claude_db_url=None,
                database_url=None,
                opc_postgres_url=None,
                environment="prod",
            )

    def test_production_with_url_succeeds(self):
        result = resolve_connection_url(
            continuous_claude_db_url="postgresql://prod",
            database_url=None,
            opc_postgres_url=None,
            environment="production",
        )
        assert result == "postgresql://prod"

    def test_empty_string_urls_treated_as_none(self):
        result = resolve_connection_url(
            continuous_claude_db_url="",
            database_url="",
            opc_postgres_url="",
            environment="",
        )
        assert result == "postgresql://claude:claude_dev@localhost:5432/continuous_claude"


# ---------------------------------------------------------------------------
# I/O wrapper: get_connection_string (reads env, delegates to pure)
# ---------------------------------------------------------------------------


class TestGetConnectionString:
    def test_reads_env_and_delegates(self):
        with patch.dict(
            "os.environ",
            {"CONTINUOUS_CLAUDE_DB_URL": "postgresql://from-env"},
            clear=False,
        ):
            result = get_connection_string()
            assert result == "postgresql://from-env"

    def test_falls_back_to_dev_default(self):
        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            result = get_connection_string()
            assert result == "postgresql://claude:claude_dev@localhost:5432/continuous_claude"


# ---------------------------------------------------------------------------
# I/O handlers: get_pool, close_pool, reset_pool
# ---------------------------------------------------------------------------


class TestGetPool:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_pool()
        yield
        reset_pool()

    async def test_creates_pool_on_first_call(self):
        mock_pool = MagicMock()
        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            pool = await get_pool()
            assert pool is mock_pool

    async def test_returns_same_pool_on_second_call(self):
        mock_pool = MagicMock()
        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ) as mock_create:
            pool1 = await get_pool()
            pool2 = await get_pool()
            assert pool1 is pool2
            assert mock_create.call_count == 1


class TestClosePool:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_pool()
        yield
        reset_pool()

    async def test_closes_existing_pool(self):
        mock_pool = AsyncMock()
        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            await get_pool()
            await close_pool()
            mock_pool.close.assert_called_once()

    async def test_noop_when_no_pool(self):
        # Should not raise
        await close_pool()


class TestResetPool:
    def test_clears_pool_reference(self):
        # reset_pool is synchronous and should not raise
        reset_pool()


# ---------------------------------------------------------------------------
# I/O handlers: get_connection, get_transaction
# ---------------------------------------------------------------------------


class TestGetConnection:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_pool()
        yield
        reset_pool()

    async def test_yields_connection(self):
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            async with get_connection() as conn:
                assert conn is mock_conn


class TestGetTransaction:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_pool()
        yield
        reset_pool()

    async def test_yields_connection_in_transaction(self):
        mock_conn = MagicMock()
        mock_txn = MagicMock()
        mock_txn.__aenter__ = AsyncMock(return_value=mock_txn)
        mock_txn.__aexit__ = AsyncMock(return_value=False)
        mock_conn.transaction.return_value = mock_txn

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            async with get_transaction() as conn:
                assert conn is mock_conn


# ---------------------------------------------------------------------------
# I/O handler: get_pool_with_retry
# ---------------------------------------------------------------------------


class TestGetPoolWithRetry:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_pool()
        yield
        reset_pool()

    async def test_succeeds_on_first_try(self):
        mock_pool = MagicMock()
        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            pool = await get_pool_with_retry(max_retries=3, initial_delay=0.01)
            assert pool is mock_pool

    async def test_retries_on_failure_then_succeeds(self):
        mock_pool = MagicMock()
        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            side_effect=[ConnectionError("fail"), mock_pool],
        ):
            pool = await get_pool_with_retry(max_retries=3, initial_delay=0.01)
            assert pool is mock_pool

    async def test_raises_after_all_retries_exhausted(self):
        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            side_effect=ConnectionError("fail"),
        ):
            with pytest.raises(ConnectionError, match="fail"):
                await get_pool_with_retry(max_retries=2, initial_delay=0.01)

    async def test_returns_existing_pool_without_retry(self):
        mock_pool = MagicMock()
        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ) as mock_create:
            await get_pool_with_retry(max_retries=1, initial_delay=0.01)
            pool2 = await get_pool_with_retry(max_retries=1, initial_delay=0.01)
            assert pool2 is mock_pool
            assert mock_create.call_count == 1


# ---------------------------------------------------------------------------
# I/O handler: health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_pool()
        yield
        reset_pool()

    async def test_healthy_returns_true(self):
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            is_healthy, error = await health_check()
            assert is_healthy is True
            assert error is None

    async def test_failure_returns_false_with_error_type(self):
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = OSError("conn failed")

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            is_healthy, error = await health_check()
            assert is_healthy is False
            assert error == "OSError"

    async def test_log_errors_sanitizes_credentials(self, caplog):
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = OSError(
            "connection to postgresql://user:secret@host:5432/db refused"
        )

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "scripts.core.db.postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            import logging

            with caplog.at_level(logging.WARNING):
                is_healthy, error = await health_check(log_errors=True)

            assert is_healthy is False
            assert "secret" not in caplog.text
            assert "***@host" in caplog.text


# ---------------------------------------------------------------------------
# I/O handler: init_pgvector
# ---------------------------------------------------------------------------


class TestInitPgvector:
    async def test_registers_vector_codec(self):
        mock_conn = AsyncMock()
        await init_pgvector(mock_conn)
        mock_conn.execute.assert_called_once_with(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )
        mock_conn.set_type_codec.assert_called_once()
        call_kwargs = mock_conn.set_type_codec.call_args
        assert call_kwargs[0][0] == "vector"  # first positional arg
        assert call_kwargs[1]["schema"] == "public"
        assert call_kwargs[1]["format"] == "text"
