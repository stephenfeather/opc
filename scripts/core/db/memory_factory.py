"""Memory backend factory for creating SQLite or PostgreSQL backends.

Provides factory functions for backend instantiation with:
- Explicit backend selection via validate_backend_type()
- Protocol conformance checking via validate_backend()
- Environment-based configuration via get_default_backend()

Structure (FP):
- Pure: validate_backend_type, validate_backend, PROTOCOL_METHODS
- I/O:  create_memory_service, get_default_backend, create_default_memory_service
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .memory_protocol import MemoryBackend

BackendType = Literal["sqlite", "postgres"]

_VALID_BACKENDS: tuple[str, ...] = ("sqlite", "postgres")

PROTOCOL_METHODS: tuple[str, ...] = (
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
)


# ---------------------------------------------------------------------------
# Pure validation functions
# ---------------------------------------------------------------------------


def validate_backend_type(backend: str) -> tuple[bool, str]:
    """Check whether *backend* is a recognised backend name.

    Returns:
        (True, "") on success, (False, error_message) on failure.
    """
    if backend not in _VALID_BACKENDS:
        return False, f"Unknown backend: {backend!r}. Must be one of {_VALID_BACKENDS}"
    return True, ""


def validate_backend(backend: object) -> list[str]:
    """Return protocol methods missing from *backend*.

    An empty list means *backend* satisfies MemoryBackend.
    """
    return [m for m in PROTOCOL_METHODS if not hasattr(backend, m)]


# ---------------------------------------------------------------------------
# I/O helper: lazy imports (isolated for testability)
# ---------------------------------------------------------------------------


def _import_sqlite_backend() -> type:
    """Import and return the SQLite MemoryService class."""
    try:
        from .memory_service import MemoryService
    except ImportError as e:
        raise ImportError(
            f"SQLite backend requires: uv pip install aiosqlite\nOriginal error: {e}"
        ) from e
    return MemoryService


def _import_postgres_backend() -> type:
    """Import and return the PostgreSQL MemoryServicePG class."""
    try:
        from .memory_service_pg import MemoryServicePG
    except ImportError as e:
        raise ImportError(
            f"Postgres backend requires: uv pip install asyncpg\nOriginal error: {e}"
        ) from e
    return MemoryServicePG


# ---------------------------------------------------------------------------
# I/O factory functions
# ---------------------------------------------------------------------------


async def create_memory_service(
    backend: BackendType = "sqlite",
    session_id: str = "default",
    agent_id: str | None = None,
    **kwargs: Any,
) -> MemoryBackend:
    """Create a memory service with the specified backend.

    Args:
        backend: "sqlite" or "postgres"
        session_id: Session identifier for isolation
        agent_id: Optional agent identifier within session
        **kwargs: Backend-specific options (e.g. db_path for SQLite)

    Returns:
        Connected MemoryBackend implementation.

    Raises:
        ValueError: If backend type is unknown.
        ImportError: If required backend dependencies not installed.
        TypeError: If created service does not satisfy MemoryBackend protocol.
    """
    is_valid, error = validate_backend_type(backend)
    if not is_valid:
        raise ValueError(error)

    if backend == "sqlite":
        cls = _import_sqlite_backend()
        service = cls(session_id=session_id, db_path=kwargs.get("db_path"))
    else:  # postgres
        cls = _import_postgres_backend()
        service = cls(session_id=session_id, agent_id=agent_id)

    missing = validate_backend(service)
    if missing:
        raise TypeError(
            f"Backend {type(service).__name__} missing protocol methods: {', '.join(missing)}"
        )

    await service.connect()
    return service


def check_backend_available(backend: str) -> tuple[bool, str]:
    """Check whether *backend*'s dependencies can be imported.

    Returns:
        (True, "") on success, (False, error_message) on failure.
    """
    if backend == "postgres":
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            return False, "postgres backend requires: uv pip install asyncpg"
    elif backend == "sqlite":
        try:
            from .memory_service import MemoryService  # noqa: F401
        except ImportError:
            return False, "sqlite backend requires memory_service module and aiosqlite"
    return True, ""


def get_default_backend() -> BackendType:
    """Read backend preference from AGENTICA_MEMORY_BACKEND env var.

    Validates the backend type and checks that its dependencies are available.

    Returns:
        "sqlite" (default) or "postgres".

    Raises:
        ValueError: If env var contains an unrecognised value.
        ImportError: If the selected backend's dependencies are unavailable.
    """
    backend = os.environ.get("AGENTICA_MEMORY_BACKEND", "sqlite")
    is_valid, error = validate_backend_type(backend)
    if not is_valid:
        raise ValueError(error)
    available, dep_error = check_backend_available(backend)
    if not available:
        raise ImportError(dep_error)
    return backend  # type: ignore[return-value]


async def create_default_memory_service(
    session_id: str = "default",
    agent_id: str | None = None,
) -> MemoryBackend:
    """Create memory service using environment config.

    Delegates to get_default_backend() then create_memory_service().

    Args:
        session_id: Session identifier for isolation
        agent_id: Optional agent identifier within session

    Returns:
        Connected MemoryBackend implementation.
    """
    backend = get_default_backend()
    return await create_memory_service(backend, session_id, agent_id)
