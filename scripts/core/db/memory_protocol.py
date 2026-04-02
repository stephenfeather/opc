"""Memory backend Protocol definition.

Defines the contract that all memory service backends must implement.
Uses Protocol for structural typing (duck typing with type checking).

Both SQLite and PostgreSQL implementations satisfy this protocol.
"""

import inspect
from typing import Any, Protocol, runtime_checkable

_VARIADIC_KINDS = frozenset({
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
})


def _get_protocol_methods() -> dict[str, inspect.Signature]:
    """Extract method names and signatures from MemoryBackend Protocol.

    Uses __protocol_attrs__ to enumerate only the declared protocol members,
    avoiding inherited metaclass/ABC callables like register().

    Returns:
        Dict mapping method name to its inspect.Signature.
    """
    attrs = getattr(MemoryBackend, "__protocol_attrs__", set())
    methods: dict[str, inspect.Signature] = {}
    for name in attrs:
        attr = getattr(MemoryBackend, name, None)
        if callable(attr):
            methods[name] = inspect.signature(attr)
    return methods


def _check_signature_compatible(
    name: str,
    expected_sig: inspect.Signature,
    actual_sig: inspect.Signature,
) -> list[str]:
    """Check that actual_sig can accept calls shaped like expected_sig.

    Verifies:
    - Required parameter count matches (excluding *args/**kwargs).
    - Positional params in the protocol are positional in the implementation.
    - Implementations using **kwargs satisfy any missing required params.

    Args:
        name: Method name (for error messages).
        expected_sig: Signature from the Protocol (excludes self).
        actual_sig: Signature from the instance method (self already bound).

    Returns:
        List of error strings (empty if compatible).
    """
    errors: list[str] = []
    expected_params = [
        p for p in expected_sig.parameters.values()
        if p.name != "self" and p.kind not in _VARIADIC_KINDS
    ]
    actual_params = list(actual_sig.parameters.values())
    has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in actual_params
    )
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in actual_params
    )
    actual_non_variadic = [
        p for p in actual_params if p.kind not in _VARIADIC_KINDS
    ]

    expected_required = [
        p for p in expected_params
        if p.default is inspect.Parameter.empty
    ]
    actual_required = [
        p for p in actual_non_variadic
        if p.default is inspect.Parameter.empty
    ]

    # If impl has *args or **kwargs, it can absorb extra params
    if not (has_var_positional or has_var_keyword):
        if len(actual_required) != len(expected_required):
            errors.append(
                f"method {name} expects {len(expected_required)} required params, "
                f"got {len(actual_required)}"
            )
            return errors

    # Check that positional params in protocol are positional in impl
    for ep in expected_params:
        if ep.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            matching = [ap for ap in actual_non_variadic if ap.name == ep.name]
            if matching and matching[0].kind == inspect.Parameter.KEYWORD_ONLY:
                errors.append(
                    f"method {name} param '{ep.name}' is keyword-only "
                    f"but protocol expects positional"
                )
    return errors


def validate_backend(instance: object) -> tuple[bool, list[str]]:
    """Validate that an object fully satisfies the MemoryBackend protocol.

    Goes beyond isinstance (which only checks attribute names) by verifying:
    1. Every required method exists on the instance.
    2. Every method is a coroutine function (async def).
    3. Every method's signature is call-compatible with the protocol.

    Args:
        instance: Object to validate.

    Returns:
        Tuple of (is_valid, list_of_errors). Empty error list means valid.
    """
    errors: list[str] = []
    protocol_methods = _get_protocol_methods()

    for name, expected_sig in protocol_methods.items():
        method = getattr(instance, name, None)
        if method is None:
            errors.append(f"missing method: {name}")
            continue
        if not inspect.iscoroutinefunction(method):
            errors.append(f"method {name} is not a coroutine function")
            continue
        actual_sig = inspect.signature(method)
        errors.extend(_check_signature_compatible(name, expected_sig, actual_sig))
    return (len(errors) == 0, errors)


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for memory service backends.

    Defines the async interface that all backends must implement.
    Uses structural typing - no need to inherit from this class.

    Backends implementing this protocol:
    - SQLite (scripts.agentica.memory_service)
    - PostgreSQL (scripts.agentica.memory_service_pg)
    """

    # Core Memory Operations
    async def set_core(self, key: str, value: str) -> None:
        """Set a core memory block.

        Args:
            key: Memory block identifier
            value: Memory block content

        Raises:
            ConnectionError: If database unavailable
        """
        ...

    async def get_core(self, key: str) -> str | None:
        """Get a core memory block.

        Args:
            key: Memory block identifier

        Returns:
            Memory block content if exists, None otherwise

        Raises:
            ConnectionError: If database unavailable
        """
        ...

    async def list_core_keys(self) -> list[str]:
        """List all core memory block keys.

        Returns:
            List of core memory keys

        Raises:
            ConnectionError: If database unavailable
        """
        ...

    async def delete_core(self, key: str) -> None:
        """Delete a core memory block.

        Args:
            key: Memory block identifier

        Raises:
            ConnectionError: If database unavailable
        """
        ...

    async def get_all_core(self) -> dict[str, str]:
        """Get all core memory blocks.

        Returns:
            Dictionary mapping keys to values

        Raises:
            ConnectionError: If database unavailable
        """
        ...

    # Archival Memory Operations
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
        """Store content in archival memory.

        Args:
            content: Content to store
            metadata: Optional metadata dictionary
            embedding: Optional embedding vector
            tags: Optional list of tags for categorization
            content_hash: SHA-256 hash for deduplication
            host_id: Machine identifier for multi-system support
            supersedes: UUID of an older learning this one replaces

        Returns:
            Memory ID (UUID)

        Raises:
            ConnectionError: If database unavailable
            ValueError: If content is empty
        """
        ...

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search archival memory (full-text search).

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching memories with metadata

        Raises:
            ConnectionError: If database unavailable
        """
        ...

    async def delete_archival(self, memory_id: str) -> None:
        """Delete an archival memory entry.

        Args:
            memory_id: Memory ID to delete

        Raises:
            ConnectionError: If database unavailable
        """
        ...

    # Recall Operations
    async def recall(
        self,
        query: str,
        include_core: bool = True,
        limit: int = 5,
    ) -> str:
        """Recall relevant memories for a query.

        Combines archival search with optional core memory inclusion.

        Args:
            query: Query string
            include_core: Whether to include core memory in results
            limit: Max archival memories to retrieve

        Returns:
            Formatted context string

        Raises:
            ConnectionError: If database unavailable
        """
        ...

    async def to_context(self, max_archival: int = 10) -> str:
        """Convert memory state to context string.

        Args:
            max_archival: Maximum archival memories to include

        Returns:
            Formatted context string with core and archival memory

        Raises:
            ConnectionError: If database unavailable
        """
        ...

    # Lifecycle Management
    async def connect(self) -> None:
        """Initialize backend connection.

        Should be idempotent - safe to call multiple times.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    async def close(self) -> None:
        """Close backend connection and cleanup resources.

        Should be idempotent - safe to call multiple times.
        """
        ...
