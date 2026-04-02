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

_DEV_DEFAULT_URL = "postgresql://claude:claude_dev@localhost:5432/continuous_claude"


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


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _sanitize_log_message(msg: str) -> str:
    """Redact credentials from connection strings in log messages."""
    return re.sub(r"://[^@]+@", "://***@", msg)


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
    return [float(x) for x in v.strip().strip("[]").split(",")]


def build_pool_config(max_size_str: str) -> dict:
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


_LOCAL_DEV_ENVS = frozenset({"development", ""})


def resolve_connection_url(
    *,
    continuous_claude_db_url: str | None,
    database_url: str | None,
    opc_postgres_url: str | None,
    environment: str,
) -> str:
    """Resolve the PostgreSQL connection URL from candidates.

    Checks candidates in priority order:
    1. continuous_claude_db_url — canonical name
    2. database_url — backwards compatible
    3. opc_postgres_url — legacy (hooks)
    4. Development default only if environment is explicitly local

    The dev default is only used when environment is one of: "development",
    "test", or "" (unset). Any other environment (production, staging, etc.)
    requires an explicit URL.

    Args:
        continuous_claude_db_url: Primary URL (CONTINUOUS_CLAUDE_DB_URL).
        database_url: Fallback URL (DATABASE_URL).
        opc_postgres_url: Legacy URL (OPC_POSTGRES_URL).
        environment: Value of AGENTICA_ENV (lowercased by caller).

    Returns:
        The resolved connection URL.

    Raises:
        ValueError: If no URL is provided and environment is not local dev.
    """
    url = continuous_claude_db_url or database_url or opc_postgres_url
    if url:
        return url

    if environment not in _LOCAL_DEV_ENVS:
        raise ValueError(
            f"Database URL must be set for environment '{environment}'. "
            "Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL. "
            "Set AGENTICA_ENV=development for local defaults."
        )

    return _DEV_DEFAULT_URL


# ---------------------------------------------------------------------------
# I/O wrappers (thin delegates to pure functions)
# ---------------------------------------------------------------------------


def _get_pool_config() -> dict:
    """Read pool config from environment and delegate to build_pool_config."""
    return build_pool_config(
        max_size_str=os.environ.get("AGENTICA_MAX_POOL_SIZE", "10"),
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


async def close_pool() -> None:
    """Close the connection pool gracefully."""
    global _pool

    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None


def reset_pool() -> None:
    """Reset the pool reference and lock without closing.

    Used for testing when the event loop changes.
    """
    global _pool, _pool_lock
    _pool = None
    _pool_lock = asyncio.Lock()


@asynccontextmanager
async def get_connection() -> AsyncGenerator[Connection, None]:
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
    except Exception as e:
        error_type = type(e).__name__
        if log_errors:
            sanitized = _sanitize_log_message(str(e))
            _logger.warning("Health check failed (%s): %s", error_type, sanitized)
        return (False, error_type)


@asynccontextmanager
async def get_transaction() -> AsyncGenerator[Connection, None]:
    """Acquire a connection from the pool with a transaction.

    Auto-commits on success, rolls back on exception.
    """
    async with get_connection() as conn:
        async with conn.transaction():
            yield conn
