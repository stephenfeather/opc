"""Tests for memory_factory.py — TDD+FP refactor (S6).

Tests pure validation functions and I/O factory functions with mocked backends.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.core.db.memory_factory import (
    PROTOCOL_METHODS,
    validate_backend,
    validate_backend_type,
)


# ---------------------------------------------------------------------------
# Pure function: validate_backend_type
# ---------------------------------------------------------------------------


class TestValidateBackendType:
    """Tests for validate_backend_type() — pure validation."""

    def test_sqlite_is_valid(self) -> None:
        is_valid, error = validate_backend_type("sqlite")
        assert is_valid is True
        assert error == ""

    def test_postgres_is_valid(self) -> None:
        is_valid, error = validate_backend_type("postgres")
        assert is_valid is True
        assert error == ""

    def test_unknown_backend_is_invalid(self) -> None:
        is_valid, error = validate_backend_type("mysql")
        assert is_valid is False
        assert "mysql" in error

    def test_empty_string_is_invalid(self) -> None:
        is_valid, error = validate_backend_type("")
        assert is_valid is False

    def test_none_is_invalid(self) -> None:
        is_valid, error = validate_backend_type(None)  # type: ignore[arg-type]
        assert is_valid is False


# ---------------------------------------------------------------------------
# Pure function: validate_backend
# ---------------------------------------------------------------------------


def _make_backend_stub(methods: tuple[str, ...] | None = None) -> MagicMock:
    """Create a stub object with the given method names."""
    stub = MagicMock()
    # Remove all attributes except the ones we want
    all_methods = methods if methods is not None else PROTOCOL_METHODS
    for method in PROTOCOL_METHODS:
        if method not in all_methods:
            delattr(stub, method)
    return stub


class TestValidateBackend:
    """Tests for validate_backend() — protocol conformance check."""

    def test_valid_backend_returns_empty_list(self) -> None:
        stub = _make_backend_stub(PROTOCOL_METHODS)
        missing = validate_backend(stub)
        assert missing == []

    def test_missing_one_method(self) -> None:
        stub = _make_backend_stub(PROTOCOL_METHODS)
        delattr(stub, "connect")
        missing = validate_backend(stub)
        assert "connect" in missing

    def test_missing_multiple_methods(self) -> None:
        stub = _make_backend_stub(PROTOCOL_METHODS)
        delattr(stub, "store")
        delattr(stub, "search")
        missing = validate_backend(stub)
        assert "store" in missing
        assert "search" in missing

    def test_empty_object_returns_all_methods(self) -> None:
        stub = object()
        missing = validate_backend(stub)
        assert set(missing) == set(PROTOCOL_METHODS)

    def test_none_returns_all_methods(self) -> None:
        missing = validate_backend(None)  # type: ignore[arg-type]
        assert len(missing) == len(PROTOCOL_METHODS)


# ---------------------------------------------------------------------------
# PROTOCOL_METHODS constant
# ---------------------------------------------------------------------------


class TestProtocolMethods:
    """Ensure PROTOCOL_METHODS matches the MemoryBackend protocol."""

    def test_contains_all_expected_methods(self) -> None:
        expected = {
            "set_core", "get_core", "list_core_keys", "delete_core", "get_all_core",
            "store", "search", "delete_archival", "recall", "to_context",
            "connect", "close",
        }
        assert set(PROTOCOL_METHODS) == expected

    def test_is_tuple(self) -> None:
        assert isinstance(PROTOCOL_METHODS, tuple)


# ---------------------------------------------------------------------------
# I/O: create_memory_service
# ---------------------------------------------------------------------------


class TestCreateMemoryService:
    """Tests for create_memory_service() — I/O factory function."""

    async def test_unknown_backend_raises_value_error(self) -> None:
        from scripts.core.db.memory_factory import create_memory_service

        with pytest.raises(ValueError, match="Unknown backend.*mysql"):
            await create_memory_service("mysql")  # type: ignore[arg-type]

    async def test_sqlite_backend_creates_and_connects(self) -> None:
        from scripts.core.db.memory_factory import create_memory_service

        mock_service = MagicMock()
        # Add all protocol methods so validation passes
        for method in PROTOCOL_METHODS:
            setattr(mock_service, method, AsyncMock())
        mock_service.connect = AsyncMock()

        mock_cls = MagicMock(return_value=mock_service)

        with patch(
            "scripts.core.db.memory_factory._import_sqlite_backend",
            return_value=mock_cls,
        ):
            result = await create_memory_service("sqlite", session_id="test-1")

        mock_cls.assert_called_once_with(session_id="test-1", db_path=None)
        mock_service.connect.assert_awaited_once()
        assert result is mock_service

    async def test_postgres_backend_creates_and_connects(self) -> None:
        from scripts.core.db.memory_factory import create_memory_service

        mock_service = MagicMock()
        for method in PROTOCOL_METHODS:
            setattr(mock_service, method, AsyncMock())
        mock_service.connect = AsyncMock()

        mock_cls = MagicMock(return_value=mock_service)

        with patch(
            "scripts.core.db.memory_factory._import_postgres_backend",
            return_value=mock_cls,
        ):
            result = await create_memory_service(
                "postgres", session_id="test-2", agent_id="agent-a"
            )

        mock_cls.assert_called_once_with(session_id="test-2", agent_id="agent-a")
        mock_service.connect.assert_awaited_once()
        assert result is mock_service

    async def test_validation_called_before_return(self) -> None:
        """Factory must validate the backend satisfies the protocol."""
        from scripts.core.db.memory_factory import create_memory_service

        # Create a service missing a protocol method
        mock_service = MagicMock()
        for method in PROTOCOL_METHODS:
            setattr(mock_service, method, AsyncMock())
        # Remove one method to trigger validation failure
        delattr(mock_service, "store")
        mock_service.connect = AsyncMock()

        mock_cls = MagicMock(return_value=mock_service)

        with (
            patch(
                "scripts.core.db.memory_factory._import_sqlite_backend",
                return_value=mock_cls,
            ),
            pytest.raises(TypeError, match="store"),
        ):
            await create_memory_service("sqlite", session_id="test-3")

    async def test_sqlite_passes_db_path_kwarg(self) -> None:
        from scripts.core.db.memory_factory import create_memory_service

        mock_service = MagicMock()
        for method in PROTOCOL_METHODS:
            setattr(mock_service, method, AsyncMock())
        mock_service.connect = AsyncMock()

        mock_cls = MagicMock(return_value=mock_service)

        with patch(
            "scripts.core.db.memory_factory._import_sqlite_backend",
            return_value=mock_cls,
        ):
            await create_memory_service("sqlite", session_id="s1", db_path="/tmp/test.db")

        mock_cls.assert_called_once_with(session_id="s1", db_path="/tmp/test.db")


# ---------------------------------------------------------------------------
# I/O: get_default_backend
# ---------------------------------------------------------------------------


class TestGetDefaultBackend:
    """Tests for get_default_backend() — reads env vars."""

    def test_defaults_to_sqlite(self) -> None:
        from scripts.core.db.memory_factory import get_default_backend

        with patch.dict("os.environ", {}, clear=True):
            result = get_default_backend()
        assert result == "sqlite"

    def test_reads_postgres_from_env(self) -> None:
        from scripts.core.db.memory_factory import get_default_backend

        with patch.dict("os.environ", {"AGENTICA_MEMORY_BACKEND": "postgres"}):
            result = get_default_backend()
        assert result == "postgres"

    def test_invalid_env_raises_value_error(self) -> None:
        from scripts.core.db.memory_factory import get_default_backend

        with (
            patch.dict("os.environ", {"AGENTICA_MEMORY_BACKEND": "redis"}),
            pytest.raises(ValueError, match="redis"),
        ):
            get_default_backend()


# ---------------------------------------------------------------------------
# I/O: create_default_memory_service
# ---------------------------------------------------------------------------


class TestCreateDefaultMemoryService:
    """Tests for create_default_memory_service() — delegates to factory."""

    async def test_delegates_to_create_memory_service(self) -> None:
        from scripts.core.db.memory_factory import create_default_memory_service

        mock_backend = MagicMock()
        for method in PROTOCOL_METHODS:
            setattr(mock_backend, method, AsyncMock())

        with (
            patch(
                "scripts.core.db.memory_factory.get_default_backend",
                return_value="sqlite",
            ),
            patch(
                "scripts.core.db.memory_factory.create_memory_service",
                new_callable=AsyncMock,
                return_value=mock_backend,
            ) as mock_create,
        ):
            result = await create_default_memory_service(
                session_id="sess-1", agent_id="ag-1"
            )

        mock_create.assert_awaited_once_with("sqlite", "sess-1", "ag-1")
        assert result is mock_backend


# ---------------------------------------------------------------------------
# Module-level: no faulthandler side-effect
# ---------------------------------------------------------------------------


class TestModuleLevelCleanliness:
    """Ensure the refactored module has no side effects at import time."""

    def test_no_faulthandler_at_module_level(self) -> None:
        """The module should not call faulthandler.enable() at import."""
        import inspect

        import scripts.core.db.memory_factory as mod

        source = inspect.getsource(mod)
        # faulthandler should not appear outside function bodies
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "faulthandler" in stripped and not stripped.startswith("#"):
                # It should not be at module level (non-indented)
                assert line.startswith(" ") or line.startswith("\t"), (
                    f"faulthandler reference at module-level line {i + 1}: {line}"
                )
