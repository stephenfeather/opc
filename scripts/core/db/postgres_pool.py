"""PostgreSQL connection pool for async memory operations.

Provides connection pooling with proper lifecycle management.
Uses asyncpg for true async/await without blocking.

Usage:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.fetch("SELECT * FROM core_memory WHERE session_id = $1", session_id)

    # Or use the convenience context manager:
    async with get_connection() as conn:
        await conn.fetch(...)

    # For transactions:
    async with get_transaction() as conn:
        await conn.execute("INSERT ...")
        await conn.execute("UPDATE ...")
        # Auto-commits on success, rolls back on exception

    # For connection with retry:
    pool = await get_pool_with_retry(max_retries=5, initial_delay=0.5)

    # Health check:
    is_healthy, error = await health_check()

    # Clean up when done:
    await close_pool()
"""

import asyncio
import faulthandler
import logging
import os
import random
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from asyncpg import Connection, Pool

from scripts.core.db.backend_resolution import resolve_url

# Issue #62: no hardcoded credentialed fallback. The connection URL must
# always come from the environment (CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL,
# or OPC_POSTGRES_URL). For local Docker development, export these from
# docker/.env after `docker compose up`.


def _enable_faulthandler() -> None:
    """Enable faulthandler for crash diagnostics.

    Writes to ~/.claude/logs/opc_crash.log, creating the directory if needed.
    Falls back to stderr if file logging cannot be set up.
    """
    log_dir = Path.home() / ".claude" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        faulthandler.enable(
            file=open(log_dir / "opc_crash.log", "a"),  # noqa: SIM115
            all_threads=True,
        )
    except OSError:
        faulthandler.enable(all_threads=True)


_enable_faulthandler()

# Global pool instance
_pool: Pool | None = None
_pool_lock = asyncio.Lock()

_logger = logging.getLogger(__name__)

_LOG_REDACTION_MARKER = "***"
_SENSITIVE_QUERY_KEYS = (
    "api[_-]?key",
    "access[_-]?token",
    "auth[_-]?token",
    "client[_-]?secret",
    "password",
    "secret",
    "token",
)
_LOG_REDACTION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Connection-string userinfo: scheme://user:password@host or scheme://token@host.
    (re.compile(r"://[^@\s]+@"), f"://{_LOG_REDACTION_MARKER}@"),
    # Quoted header dumps from HTTP clients commonly use Python dict/repr syntax.
    (
        re.compile(
            r"(\b(?:authorization|x-api-key|api-key)\b['\"]?\s*[:=]\s*)(['\"])[^'\"]+(['\"])",
            re.IGNORECASE,
        ),
        rf"\1\2{_LOG_REDACTION_MARKER}\3",
    ),
    # Preserve the auth scheme for unquoted Authorization headers.
    (
        re.compile(
            r"\b(Authorization\s*[:=]\s*(?:[A-Za-z]+\s+)?)[^'\"\s,;}]+",
            re.IGNORECASE,
        ),
        rf"\1{_LOG_REDACTION_MARKER}",
    ),
    # Authorization bearer values may also appear outside a structured headers dict.
    (
        re.compile(r"\b(Bearer\s+)[^'\"\s,;}]{4,}", re.IGNORECASE),
        rf"\1{_LOG_REDACTION_MARKER}",
    ),
    # Unquoted API-key header dumps using either ":" or "=" delimiters.
    (
        re.compile(
            r"(\b(?:x-api-key|api-key)\b\s*[:=]\s*)[^'\"\s,;}]+",
            re.IGNORECASE,
        ),
        rf"\1{_LOG_REDACTION_MARKER}",
    ),
    # URL query params with sensitive key names; preserve delimiters and unrelated params.
    (
        re.compile(
            rf"([?&;](?:{'|'.join(_SENSITIVE_QUERY_KEYS)})=)[^&#\s'\",;:)}}\]]+",
            re.IGNORECASE,
        ),
        rf"\1{_LOG_REDACTION_MARKER}",
    ),
    # Provider-style keys in free text.
    (re.compile(r"\b(?:sk|pa)-[A-Za-z0-9_-]{8,}\b"), _LOG_REDACTION_MARKER),
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _sanitize_log_message(msg: str) -> str:
    """Redact credentials and API secrets from log messages."""
    for pattern, replacement in _LOG_REDACTION_RULES:
        msg = pattern.sub(replacement, msg)
    return msg


# Public alias so other modules (e.g. recall_learnings degrade warnings, which
# reach hook-captured stderr) can redact exception text without importing a
# private name. Same behavior; do not diverge (aegis MEDIUM-2).
def sanitize_log_message(msg: str) -> str:
    """Public redactor: strip credentials and API secrets from a message."""
    return _sanitize_log_message(msg)


def _encode_vector(v: list[float] | str) -> str:
    """Encode a vector for pgvector.

    Handles both:
    - list[float]: Converts to pgvector string format
    - str: Already formatted, pass through (handles ::vector cast)
    """
    if isinstance(v, str):
        return v
    return f"[{','.join(map(str, v))}]"


def _decode_vector(v: str) -> list[float]:
    """Decode a pgvector string to a list of floats."""
    inner = v.strip().strip("[]")
    if not inner:
        return []
    return [float(x) for x in inner.split(",")]


def build_pool_config(max_size_str: str) -> dict[str, int]:
    """Build pool configuration from a max-size string.

    Args:
        max_size_str: Maximum connections as a string (default-safe: "10").

    Returns:
        Dict with min_size, max_size, and command_timeout keys.

    Note:
        Invalid values (non-numeric, zero, or negative) fall back to 10.
    """
    try:
        max_size = int(max_size_str)
        if max_size <= 0:
            max_size = 10
    except ValueError:
        max_size = 10

    min_size = min(max(2, max_size // 5), max_size)

    return {
        "min_size": min_size,
        "max_size": max_size,
        "command_timeout": 60,
    }


def resolve_connection_url(
    *,
    continuous_claude_db_url: str | None,
    database_url: str | None,
    opc_postgres_url: str | None,
    environment: str,
) -> str:
    """Resolve the PostgreSQL connection URL from candidates.

    Checks candidates in priority order:
    1. continuous_claude_db_url — canonical name (``CONTINUOUS_CLAUDE_DB_URL``)
    2. database_url — backwards compatible (``DATABASE_URL``)
    3. opc_postgres_url — legacy (``OPC_POSTGRES_URL``, hooks)

    Issue #62: no hardcoded development fallback. All callers — including
    local dev — must supply one of the three env vars. For Docker local
    dev, export the credentials from ``docker/.env`` after
    ``docker compose -f docker/docker-compose.yml up -d``.

    Args:
        continuous_claude_db_url: Primary URL (``CONTINUOUS_CLAUDE_DB_URL``).
        database_url: Fallback URL (``DATABASE_URL``).
        opc_postgres_url: Legacy URL (``OPC_POSTGRES_URL``).
        environment: Value of ``AGENTICA_ENV`` (lowercased). Retained
            only so that the error message can reflect the active
            environment name; it no longer gates any code path.

    Returns:
        The resolved connection URL.

    Raises:
        ValueError: If none of the three env-var inputs are set.
    """
    url = resolve_url(
        {
            "CONTINUOUS_CLAUDE_DB_URL": continuous_claude_db_url or "",
            "DATABASE_URL": database_url or "",
            "OPC_POSTGRES_URL": opc_postgres_url or "",
        }
    )
    if url:
        return url

    env_label = environment or "<unset>"
    raise ValueError(
        f"Database URL not set (environment={env_label!r}). "
        "Set CONTINUOUS_CLAUDE_DB_URL (preferred), DATABASE_URL, or "
        "OPC_POSTGRES_URL in your shell / launcher. "
        "For local Docker dev, run `docker compose -f docker/docker-compose.yml up -d` "
        "and export the credentials from docker/.env before invoking this script."
    )


# ---------------------------------------------------------------------------
# I/O wrappers (thin delegates to pure functions)
# ---------------------------------------------------------------------------


def _get_pool_config() -> dict[str, int]:
    """Read pool config from environment and delegate to build_pool_config."""
    from scripts.core.config import get_config

    return build_pool_config(
        max_size_str=os.environ.get(
            "AGENTICA_MAX_POOL_SIZE", str(get_config().database.max_pool_size)
        ),
    )


def get_connection_string() -> str:
    """Get PostgreSQL connection string from environment or defaults.

    Delegates to resolve_connection_url with values read from env vars.
    """
    return resolve_connection_url(
        continuous_claude_db_url=os.environ.get("CONTINUOUS_CLAUDE_DB_URL"),
        database_url=os.environ.get("DATABASE_URL"),
        opc_postgres_url=os.environ.get("OPC_POSTGRES_URL"),
        environment=os.environ.get("AGENTICA_ENV", "").lower(),
    )


# ---------------------------------------------------------------------------
# I/O handlers: pool lifecycle
# ---------------------------------------------------------------------------


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Initialize a new connection with pgvector codec.

    Called automatically by asyncpg for each new connection in the pool.
    """
    await init_pgvector(conn)


async def get_pool() -> Pool:
    """Get or create the global connection pool.

    Thread-safe via asyncio.Lock.
    """
    global _pool

    async with _pool_lock:
        if _pool is None:
            config = _get_pool_config()
            _pool = await asyncpg.create_pool(
                get_connection_string(),
                init=_init_connection,
                **config,
            )

    return _pool


def _reset_schema_capability_caches() -> None:
    """Clear MemoryServicePG's process-wide schema probe caches (issue #63
    Phase 2b round-2 finding 4).

    The capability caches (`_has_superseded_column`, `_has_archived_at_column`)
    are class-level, so they outlive any single pool. Closing/resetting the pool
    may point the process at a DB with a different migration state, so the cached
    probe results must be invalidated. Imported lazily here to avoid a circular
    import (memory_service_pg imports from this module at module load).
    """
    try:
        from scripts.core.db.memory_service_pg import MemoryServicePG
    except ImportError:  # pragma: no cover - defensive; module should import
        return
    MemoryServicePG.reset_capability_caches()


async def close_pool() -> None:
    """Close the connection pool gracefully."""
    global _pool

    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None
        _reset_schema_capability_caches()


def reset_pool() -> None:
    """Reset the pool reference and lock without closing.

    Used for testing when the event loop changes.
    """
    global _pool, _pool_lock
    _pool = None
    _pool_lock = asyncio.Lock()
    _reset_schema_capability_caches()


@asynccontextmanager
async def get_connection() -> AsyncGenerator[Connection]:
    """Acquire a connection from the pool."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


async def init_pgvector(conn: asyncpg.Connection) -> None:
    """Initialize pgvector extension for a connection.

    Registers the vector type codec using module-level pure encode/decode functions.
    """
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.set_type_codec(
        "vector",
        encoder=_encode_vector,
        decoder=_decode_vector,
        schema="public",
        format="text",
    )


async def get_pool_with_retry(
    max_retries: int = 5,
    initial_delay: float = 0.5,
) -> Pool:
    """Get or create the global connection pool with retry on transient failures.

    Uses exponential backoff with full jitter between retries.

    Args:
        max_retries: Maximum number of connection attempts.
        initial_delay: Initial delay in seconds between retries (doubles each time).

    Returns:
        The connection pool.

    Raises:
        The last connection error after all retries are exhausted.
    """
    global _pool

    async with _pool_lock:
        if _pool is not None:
            return _pool

        last_error: Exception | None = None
        config = _get_pool_config()

        for attempt in range(max_retries):
            try:
                pool = await asyncpg.create_pool(
                    get_connection_string(),
                    init=_init_connection,
                    **config,
                )
                _pool = pool
                return pool
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    max_delay = initial_delay * (2**attempt)
                    jittered_delay = random.uniform(0, max_delay)
                    await asyncio.sleep(jittered_delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to create pool after retries")


async def health_check(log_errors: bool = False) -> tuple[bool, str | None]:
    """Check if the connection pool is healthy.

    Args:
        log_errors: If True, log error details at WARNING level.

    Returns:
        A tuple of (is_healthy, error_type).
    """
    try:
        async with get_connection() as conn:
            await conn.fetchval("SELECT 1")
        return (True, None)
    except asyncpg.PostgresConnectionError as e:
        error_type = "connection_error"
        if log_errors:
            _logger.warning("Health check failed: %s", _sanitize_log_message(str(e)))
        return (False, error_type)
    except asyncpg.PostgresError as e:
        error_type = "postgres_error"
        if log_errors:
            _logger.warning("Health check failed: %s", _sanitize_log_message(str(e)))
        return (False, error_type)
    except (OSError, TimeoutError, RuntimeError) as e:
        error_type = type(e).__name__
        if log_errors:
            sanitized = _sanitize_log_message(str(e))
            _logger.warning("Health check failed (%s): %s", error_type, sanitized)
        return (False, error_type)


@asynccontextmanager
async def get_transaction() -> AsyncGenerator[Connection]:
    """Acquire a connection from the pool with a transaction.

    Auto-commits on success, rolls back on exception.
    """
    async with get_connection() as conn:
        async with conn.transaction():
            yield conn
