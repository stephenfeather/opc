"""Tests for scripts.core.db.memory_protocol.

Verifies the MemoryBackend Protocol contract:
- Structural typing works (classes satisfying the protocol are recognized)
- Classes missing methods are not recognized
- Method signatures match expected types
"""

from __future__ import annotations

from typing import Any

from scripts.core.db.memory_protocol import MemoryBackend, validate_backend

# ---------------------------------------------------------------------------
# Test helpers: concrete implementations for structural typing checks
# ---------------------------------------------------------------------------


class CompleteBackend:
    """A class that satisfies the full MemoryBackend protocol."""

    async def set_core(self, key: str, value: str) -> None:
        pass

    async def get_core(self, key: str) -> str | None:
        return None

    async def list_core_keys(self) -> list[str]:
        return []

    async def delete_core(self, key: str) -> None:
        pass

    async def get_all_core(self) -> dict[str, str]:
        return {}

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        tags: list[str] | None = None,
        content_hash: str | None = None,
        host_id: str | None = None,
        supersedes: str | None = None,
    ) -> str:
        return "test-id"

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    async def delete_archival(self, memory_id: str) -> None:
        pass

    async def recall(
        self,
        query: str,
        include_core: bool = True,
        limit: int = 5,
    ) -> str:
        return ""

    async def to_context(self, max_archival: int = 10) -> str:
        return ""

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass


class IncompleteBackend:
    """A class missing required methods — should NOT satisfy the protocol."""

    async def set_core(self, key: str, value: str) -> None:
        pass

    # Missing all other methods


class SyncBackend:
    """A class with all method names but synchronous — NOT a valid backend.

    Demonstrates the limitation of runtime_checkable (name-only checks).
    The validate_backend function catches this case.
    """

    def set_core(self, key: str, value: str) -> None:
        pass

    def get_core(self, key: str) -> str | None:
        return None

    def list_core_keys(self) -> list[str]:
        return []

    def delete_core(self, key: str) -> None:
        pass

    def get_all_core(self) -> dict[str, str]:
        return {}

    def store(self, content: str, **kwargs: Any) -> str:
        return "test-id"

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def delete_archival(self, memory_id: str) -> None:
        pass

    def recall(self, query: str, include_core: bool = True, limit: int = 5) -> str:
        return ""

    def to_context(self, max_archival: int = 10) -> str:
        return ""

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass


class WrongArityBackend:
    """Async methods with wrong parameter counts — should fail validation."""

    async def set_core(self) -> None:  # missing key, value
        pass

    async def get_core(self) -> str | None:  # missing key
        return None

    async def list_core_keys(self) -> list[str]:
        return []

    async def delete_core(self) -> None:  # missing key
        pass

    async def get_all_core(self) -> dict[str, str]:
        return {}

    async def store(self) -> str:  # missing content
        return "test-id"

    async def search(self) -> list[dict[str, Any]]:  # missing query
        return []

    async def delete_archival(self) -> None:  # missing memory_id
        pass

    async def recall(self) -> str:  # missing query
        return ""

    async def to_context(self) -> str:
        return ""

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass


class KeywordOnlyBackend:
    """Async methods with keyword-only params — should fail validation.

    Protocol expects positional params, but this uses keyword-only syntax.
    """

    async def set_core(self, *, key: str, value: str) -> None:
        pass

    async def get_core(self, *, key: str) -> str | None:
        return None

    async def list_core_keys(self) -> list[str]:
        return []

    async def delete_core(self, *, key: str) -> None:
        pass

    async def get_all_core(self) -> dict[str, str]:
        return {}

    async def store(self, *, content: str) -> str:
        return "test-id"

    async def search(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    async def delete_archival(self, *, memory_id: str) -> None:
        pass

    async def recall(
        self, *, query: str, include_core: bool = True, limit: int = 5
    ) -> str:
        return ""

    async def to_context(self, *, max_archival: int = 10) -> str:
        return ""

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests: Protocol is importable and well-formed
# ---------------------------------------------------------------------------


class TestMemoryBackendProtocol:
    """Verify the MemoryBackend Protocol definition."""

    def test_protocol_is_importable(self) -> None:
        """MemoryBackend should be importable from the db package."""
        assert MemoryBackend is not None

    def test_protocol_is_a_protocol(self) -> None:
        """MemoryBackend should be a typing.Protocol subclass."""
        from typing import is_protocol

        assert is_protocol(MemoryBackend)

    def test_protocol_defines_expected_methods(self) -> None:
        """Protocol should define all expected async method stubs."""
        expected_methods = [
            "set_core",
            "get_core",
            "list_core_keys",
            "delete_core",
            "get_all_core",
            "store",
            "search",
            "delete_archival",
            "recall",
            "to_context",
            "connect",
            "close",
        ]
        for method_name in expected_methods:
            assert hasattr(MemoryBackend, method_name), (
                f"MemoryBackend missing method: {method_name}"
            )


# ---------------------------------------------------------------------------
# Tests: Structural typing (runtime_checkable)
# ---------------------------------------------------------------------------


class TestStructuralTyping:
    """Verify structural typing works with runtime_checkable."""

    def test_complete_backend_is_recognized(self) -> None:
        """A class implementing all methods should satisfy the protocol."""
        # Protocol must be runtime_checkable for isinstance checks
        assert isinstance(CompleteBackend(), MemoryBackend)

    def test_incomplete_backend_is_not_recognized(self) -> None:
        """A class missing methods should NOT satisfy the protocol."""
        assert not isinstance(IncompleteBackend(), MemoryBackend)


# ---------------------------------------------------------------------------
# Tests: No import side effects
# ---------------------------------------------------------------------------


class TestNoSideEffects:
    """Protocol module should be free of import-time side effects."""

    def test_no_faulthandler_in_source(self) -> None:
        """Module source should not contain faulthandler.enable calls."""
        import inspect as _inspect
        import sys

        module_name = "scripts.core.db.memory_protocol"
        if module_name in sys.modules:
            source = _inspect.getsource(sys.modules[module_name])
            assert "faulthandler.enable" not in source, (
                "Protocol module should not have faulthandler side effects"
            )

    def test_import_does_not_enable_faulthandler(self) -> None:
        """Re-importing the module should not toggle faulthandler state."""
        import faulthandler
        import importlib
        import sys

        module_name = "scripts.core.db.memory_protocol"
        # Remove cached module so import runs init code again
        saved = sys.modules.pop(module_name, None)
        try:
            before = faulthandler.is_enabled()
            importlib.import_module(module_name)
            after = faulthandler.is_enabled()
            assert before == after, (
                f"faulthandler state changed on import: {before} -> {after}"
            )
        finally:
            # Restore original module to avoid side effects on other tests
            if saved is not None:
                sys.modules[module_name] = saved


# ---------------------------------------------------------------------------
# Tests: validate_backend (deep protocol validation)
# ---------------------------------------------------------------------------


class TestValidateBackend:
    """Verify validate_backend catches issues isinstance misses."""

    def test_complete_async_backend_passes(self) -> None:
        """A fully async backend should pass validation."""
        is_valid, errors = validate_backend(CompleteBackend())
        assert is_valid
        assert errors == []

    def test_sync_backend_fails_validation(self) -> None:
        """A backend with sync methods should fail validation.

        This is the case that @runtime_checkable misses -- isinstance
        returns True for SyncBackend, but validate_backend catches it.
        """
        is_valid, errors = validate_backend(SyncBackend())
        assert not is_valid
        assert len(errors) == 12  # all 12 methods are sync, not async
        assert all("not a coroutine function" in e for e in errors)

    def test_incomplete_backend_fails_validation(self) -> None:
        """A backend missing methods should fail validation."""
        is_valid, errors = validate_backend(IncompleteBackend())
        assert not is_valid
        assert any("missing method" in e for e in errors)

    def test_empty_object_fails_validation(self) -> None:
        """An object with no methods should fail validation."""
        is_valid, errors = validate_backend(object())
        assert not is_valid
        assert len(errors) == 12

    def test_sync_backend_passes_isinstance_but_fails_validate(self) -> None:
        """Demonstrates runtime_checkable limitation and validate_backend fix."""
        backend = SyncBackend()
        # isinstance is name-only -- passes incorrectly
        assert isinstance(backend, MemoryBackend)
        # validate_backend catches the async requirement
        is_valid, _ = validate_backend(backend)
        assert not is_valid

    def test_wrong_arity_backend_fails_validation(self) -> None:
        """Async methods with wrong parameter counts should fail validation."""
        is_valid, errors = validate_backend(WrongArityBackend())
        assert not is_valid
        # Methods with wrong arity: set_core(needs 2), get_core(1),
        # delete_core(1), store(1), search(1), delete_archival(1), recall(1)
        arity_errors = [e for e in errors if "required params" in e]
        assert len(arity_errors) >= 6, f"Expected >=6 arity errors, got: {arity_errors}"

    def test_wrong_arity_backend_passes_isinstance(self) -> None:
        """Wrong-arity backend still passes isinstance (name-only check)."""
        assert isinstance(WrongArityBackend(), MemoryBackend)

    def test_keyword_only_backend_fails_validation(self) -> None:
        """Backend with keyword-only params should fail validation.

        Protocol methods use positional params (e.g., set_core(key, value)).
        A backend using keyword-only (*, key, value) can't be called positionally.
        """
        is_valid, errors = validate_backend(KeywordOnlyBackend())
        assert not is_valid
        kw_errors = [e for e in errors if "keyword-only" in e]
        assert len(kw_errors) >= 5, f"Expected >=5 kw-only errors, got: {kw_errors}"

    def test_kwargs_backend_passes_validation(self) -> None:
        """Backend using **kwargs for forward-compatibility should pass.

        Implementations may use **kwargs to accept future protocol params
        without breaking. The validator should treat this as compatible.
        """

        class KwargsBackend:
            async def set_core(self, key: str, value: str, **kw: Any) -> None:
                pass

            async def get_core(self, key: str, **kw: Any) -> str | None:
                return None

            async def list_core_keys(self, **kw: Any) -> list[str]:
                return []

            async def delete_core(self, key: str, **kw: Any) -> None:
                pass

            async def get_all_core(self, **kw: Any) -> dict[str, str]:
                return {}

            async def store(self, content: str, **kw: Any) -> str:
                return "id"

            async def search(self, query: str, limit: int = 10, **kw: Any) -> list[dict[str, Any]]:
                return []

            async def delete_archival(self, memory_id: str, **kw: Any) -> None:
                pass

            async def recall(self, query: str, **kw: Any) -> str:
                return ""

            async def to_context(self, **kw: Any) -> str:
                return ""

            async def connect(self, **kw: Any) -> None:
                pass

            async def close(self, **kw: Any) -> None:
                pass

        is_valid, errors = validate_backend(KwargsBackend())
        assert is_valid, f"KwargsBackend should pass validation, got errors: {errors}"
