"""Memory Service with PostgreSQL + pgvector backend.

Async I/O layer that delegates query building and formatting to pure
functions in memory_service_queries.py.

Architecture:
- Core Memory: Key-value blocks per session/agent
- Archival Memory: Long-term storage with FTS + vector search
- Recall Memory: Cross-source query combining all sources

Scoping model (R-Flow):
- session_id: Claude Code session
- agent_id: Optional agent identifier within session

Usage:
    memory = MemoryServicePG(session_id="abc123")
    await memory.connect()

    await memory.set_core("persona", "You are a helpful assistant")
    await memory.store("User prefers Python")
    result = await memory.recall("What language?")

    await memory.close()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import asyncpg

if TYPE_CHECKING:
    from .memory_block import Block

from .memory_service_queries import (
    ACTIVE_ROW_FILTER,
    build_hybrid_search_sql,
    build_text_search_sql,
    build_vector_search_sql,
    filter_core_by_query,
    format_context_string,
    format_recall_text,
    format_rows,
    generate_memory_id,
    pad_embedding,
)
from .postgres_pool import get_connection, get_pool, get_transaction, init_pgvector

# faulthandler is enabled by postgres_pool._enable_faulthandler() on import.

logger = logging.getLogger(__name__)

# Load config defaults lazily for search methods.
from scripts.core.config import get_config as _get_config

# Capability probe for the embedding_model column (issue #151, round 1 FIX 1).
# Deferred import (matches the config import above) and circular-safe:
# recall_backends does not import this module.
from scripts.core.recall_backends import embedding_model_column_available


def _recall_cfg() -> Any:
    """Return current recall config.

    Keep this at call time so tests and long-lived processes can reset or
    override config after this module has already been imported.
    """
    return _get_config().recall


def _db_cfg() -> Any:
    """Return current database config at call time."""
    return _get_config().database


def _search_limit(limit: int | None) -> int:
    return _recall_cfg().default_search_limit if limit is None else limit


def _rrf_k(k: int | None) -> int:
    return _recall_cfg().rrf_k if k is None else k


def _max_archival(max_archival: int | None) -> int:
    return _db_cfg().max_archival_context if max_archival is None else max_archival


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


@dataclass
class ArchivalFact:
    """A fact stored in archival memory."""

    id: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime
    similarity: float | None = None


class MemoryServicePG:
    """Async memory service with PostgreSQL + pgvector backend.

    Thin I/O wrapper — pure logic lives in memory_service_queries.py.
    """

    # Module-level cache: None = not checked, True/False = result
    _has_superseded_column: bool | None = None

    def __init__(
        self,
        session_id: str = "default",
        agent_id: str | None = None,
    ):
        self.session_id = session_id
        self.agent_id = agent_id
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        self._pool = await get_pool()

    async def _check_superseded_column(self) -> bool:
        """Check if the superseded_by column exists (schema migration compat).

        Caches the result at the class level so the check runs at most once
        per process lifetime.
        """
        if MemoryServicePG._has_superseded_column is not None:
            return MemoryServicePG._has_superseded_column
        try:
            async with get_connection() as conn:
                await conn.fetchval(
                    "SELECT 1 FROM archival_memory WHERE superseded_by IS NULL LIMIT 1"
                )
            MemoryServicePG._has_superseded_column = True
        except (asyncpg.UndefinedColumnError, asyncpg.UndefinedTableError):
            logger.debug("superseded_by column not found, disabling active-row filter")
            MemoryServicePG._has_superseded_column = False
        return MemoryServicePG._has_superseded_column

    async def close(self) -> None:
        """Release connection (pool stays open for other services)."""
        pass

    # ==================== Core Memory ====================

    async def set_core(self, key: str, value: str) -> None:
        """Set a core memory block (DELETE + INSERT for NULL agent_id compat)."""
        async with get_transaction() as conn:
            await conn.execute(
                """
                DELETE FROM core_memory
                WHERE session_id = $1 AND agent_id IS NOT DISTINCT FROM $2 AND key = $3
                """,
                self.session_id, self.agent_id, key,
            )
            await conn.execute(
                """
                INSERT INTO core_memory (session_id, agent_id, key, value, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                """,
                self.session_id, self.agent_id, key, value,
            )

    async def get_core(self, key: str) -> str | None:
        """Get a core memory block value."""
        async with get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT value FROM core_memory
                WHERE session_id = $1 AND agent_id IS NOT DISTINCT FROM $2 AND key = $3
                """,
                self.session_id, self.agent_id, key,
            )
            return row["value"] if row else None

    async def list_core_keys(self) -> list[str]:
        """List all core memory block keys."""
        async with get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT key FROM core_memory
                WHERE session_id = $1 AND agent_id IS NOT DISTINCT FROM $2
                ORDER BY key
                """,
                self.session_id, self.agent_id,
            )
            return [row["key"] for row in rows]

    async def delete_core(self, key: str) -> None:
        """Delete a core memory block."""
        async with get_connection() as conn:
            await conn.execute(
                """
                DELETE FROM core_memory
                WHERE session_id = $1 AND agent_id IS NOT DISTINCT FROM $2 AND key = $3
                """,
                self.session_id, self.agent_id, key,
            )

    async def get_all_core(self) -> dict[str, str]:
        """Get all core memory blocks as key→value dict."""
        async with get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT key, value FROM core_memory
                WHERE session_id = $1 AND agent_id IS NOT DISTINCT FROM $2
                ORDER BY key
                """,
                self.session_id, self.agent_id,
            )
            return {row["key"]: row["value"] for row in rows}

    # ==================== Block-Based Core Memory ====================

    async def set_block(self, block: Block) -> None:
        """Set a Block in core memory (Letta-compatible)."""
        block_data = json.dumps({
            "value": block.value,
            "limit": block.limit,
            "metadata": block.metadata,
        })
        await self.set_core(block.label, block_data)

    async def get_block(self, label: str) -> Block | None:
        """Get a Block from core memory."""
        from .memory_block import Block

        raw = await self.get_core(label)
        if raw is None:
            return None

        try:
            data = json.loads(raw)
            return Block(
                label=label,
                value=data.get("value", ""),
                limit=data.get("limit", 5000),
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, TypeError):
            return Block(label=label, value=raw)

    async def get_all_blocks(self) -> dict[str, Block]:
        """Get all Blocks from core memory."""
        from .memory_block import Block

        all_core = await self.get_all_core()
        blocks: dict[str, Block] = {}

        for label, raw in all_core.items():
            try:
                data = json.loads(raw)
                if isinstance(data, dict) and "value" in data:
                    blocks[label] = Block(
                        label=label,
                        value=data.get("value", ""),
                        limit=data.get("limit", 5000),
                        metadata=data.get("metadata", {}),
                    )
                else:
                    blocks[label] = Block(label=label, value=raw)
            except (json.JSONDecodeError, TypeError):
                blocks[label] = Block(label=label, value=raw)

        return blocks

    # ==================== Archival Memory ====================

    @staticmethod
    def _build_embedding_insert(
        *,
        memory_id: str,
        session_id: str,
        agent_id: str | None,
        content: str,
        metadata: dict[str, Any] | None,
        padded_embedding: Any,
        content_hash: str | None,
        host_id: str | None,
        project: str | None,
        source_time: datetime | None,
        embedding_model: str | None,
    ) -> tuple[str, list[Any]]:
        """Build the (sql, vals) for an embedding INSERT.

        When ``embedding_model`` is set the labeled column is included;
        otherwise the legacy (pre-#151) shape is produced. Pure — no I/O.
        """
        base_cols = (
            "id, session_id, agent_id, content, "
            "metadata, embedding, content_hash, host_id, project"
        )
        base_vals: list[Any] = [
            memory_id,
            session_id,
            agent_id,
            content,
            json.dumps(metadata or {}),
            padded_embedding,
            content_hash,
            host_id,
            project,
        ]
        model_col = ", embedding_model" if embedding_model else ""
        model_val = [embedding_model] if embedding_model else []
        if source_time is not None:
            cols = f"{base_cols}{model_col}, created_at"
            vals = [*base_vals, *model_val, source_time]
        else:
            cols = f"{base_cols}{model_col}"
            vals = [*base_vals, *model_val]
        placeholders = ", ".join(f"${i + 1}" for i in range(len(vals)))
        sql = f"""
                    INSERT INTO archival_memory
                        ({cols})
                    VALUES ({placeholders})
                    ON CONFLICT (content_hash)
                        WHERE content_hash IS NOT NULL
                        DO NOTHING
                    """
        return sql, vals

    async def _insert_with_embedding(
        self,
        conn: Any,
        *,
        memory_id: str,
        content: str,
        metadata: dict[str, Any] | None,
        padded_embedding: Any,
        content_hash: str | None,
        host_id: str | None,
        project: str | None,
        source_time: datetime | None,
        embedding_model: str | None,
    ) -> str:
        """Run the embedding INSERT, labeling the row evidence-based.

        FIX 3 (round 2, issue #151): the WRITE path must NOT be governed by
        the read-degradation capability probe — a probe cached False (process
        lifetime) or a transient probe error would make stores fall back to
        the unlabeled INSERT and let the 'bge' column default permanently
        mislabel voyage/openai rows. Instead, when a label is present we ALWAYS
        attempt the labeled INSERT first (inside a savepoint so the outer
        transaction survives) and retry the legacy INSERT exactly once on a
        real UndefinedColumnError. This self-heals after a mid-process
        migration with no cache to invalidate.
        """
        labeled_sql, labeled_vals = self._build_embedding_insert(
            memory_id=memory_id,
            session_id=self.session_id,
            agent_id=self.agent_id,
            content=content,
            metadata=metadata,
            padded_embedding=padded_embedding,
            content_hash=content_hash,
            host_id=host_id,
            project=project,
            source_time=source_time,
            embedding_model=embedding_model,
        )
        if not embedding_model:
            # No label -> nothing to degrade; one plain INSERT.
            return await conn.execute(labeled_sql, *labeled_vals)
        try:
            # Savepoint: a labeled-INSERT failure rolls back to here, leaving
            # the outer transaction usable for the legacy retry.
            async with conn.transaction():
                return await conn.execute(labeled_sql, *labeled_vals)
        except asyncpg.exceptions.UndefinedColumnError:
            logger.debug(
                "embedding_model column absent; retrying legacy INSERT "
                "(row written with the 'bge' column default until migrated)",
                exc_info=True,
            )
            legacy_sql, legacy_vals = self._build_embedding_insert(
                memory_id=memory_id,
                session_id=self.session_id,
                agent_id=self.agent_id,
                content=content,
                metadata=metadata,
                padded_embedding=padded_embedding,
                content_hash=content_hash,
                host_id=host_id,
                project=project,
                source_time=source_time,
                embedding_model=None,
            )
            return await conn.execute(legacy_sql, *legacy_vals)

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        tags: list[str] | None = None,
        content_hash: str | None = None,
        host_id: str | None = None,
        supersedes: str | None = None,
        project: str | None = None,
        source_time: datetime | None = None,
        embedding_model: str | None = None,
    ) -> str:
        """Store a fact in archival memory.

        ``embedding_model`` (issue #151) is the embedding-space label written
        to the ``embedding_model`` column so recall can pin a single space.
        When ``None`` the column is omitted and the DB default ('bge')
        applies, keeping legacy callers byte-identical.

        Returns memory ID (or empty string if deduplicated).
        """
        memory_id = generate_memory_id()

        padded_embedding = None
        if embedding is not None and len(embedding) > 0:
            padded_embedding = pad_embedding(embedding)

        async with get_transaction() as conn:
            if padded_embedding is not None:
                await init_pgvector(conn)
                result = await self._insert_with_embedding(
                    conn,
                    memory_id=memory_id,
                    content=content,
                    metadata=metadata,
                    padded_embedding=padded_embedding,
                    content_hash=content_hash,
                    host_id=host_id,
                    project=project,
                    source_time=source_time,
                    embedding_model=embedding_model,
                )
            else:
                if source_time is not None:
                    result = await conn.execute(
                        """
                        INSERT INTO archival_memory
                            (id, session_id, agent_id, content,
                             metadata, content_hash, host_id, project,
                             created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (content_hash)
                            WHERE content_hash IS NOT NULL
                            DO NOTHING
                        """,
                        memory_id,
                        self.session_id,
                        self.agent_id,
                        content,
                        json.dumps(metadata or {}),
                        content_hash,
                        host_id,
                        project,
                        source_time,
                    )
                else:
                    result = await conn.execute(
                        """
                        INSERT INTO archival_memory
                            (id, session_id, agent_id, content,
                             metadata, content_hash, host_id, project)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (content_hash)
                            WHERE content_hash IS NOT NULL
                            DO NOTHING
                        """,
                        memory_id,
                        self.session_id,
                        self.agent_id,
                        content,
                        json.dumps(metadata or {}),
                        content_hash,
                        host_id,
                        project,
                    )

            if result == "INSERT 0 0":
                return ""

            if tags:
                for tag in set(tags):
                    await conn.execute(
                        """
                        INSERT INTO memory_tags (memory_id, tag, session_id)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (memory_id, tag) DO NOTHING
                        """,
                        memory_id, tag, self.session_id,
                    )

            if supersedes:
                try:
                    await conn.execute(
                        """
                        UPDATE archival_memory
                        SET superseded_by = $1::uuid, superseded_at = NOW()
                        WHERE id = $2::uuid AND superseded_by IS NULL
                        """,
                        memory_id, supersedes,
                    )
                except asyncpg.UndefinedColumnError:
                    logger.debug(
                        "Supersede UPDATE failed (column missing) for %s -> %s",
                        supersedes, memory_id, exc_info=True,
                    )

        return memory_id

    async def search_text(
        self,
        query: str,
        limit: int | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Search archival memory with full-text search."""
        resolved_limit = _search_limit(limit)
        has_col = await self._check_superseded_column()
        sql, params = build_text_search_sql(
            self.session_id, self.agent_id, query, resolved_limit,
            start_date=start_date, end_date=end_date,
            include_active_filter=has_col,
        )
        async with get_connection() as conn:
            rows = await conn.fetch(sql, *params)
            return format_rows(rows, extra_fields=["rank"])

    async def search_vector(
        self,
        query_embedding: list[float],
        limit: int | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        embedding_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search archival memory with vector similarity.

        ``embedding_model`` (issue #151, round 1 FIX 2): the session-scoped
        dedup fallback pins to one space, mirroring search_vector_global. The
        capability probe drops the filter on a pre-migration DB.
        """
        resolved_limit = _search_limit(limit)
        has_col = await self._check_superseded_column()
        padded_query = pad_embedding(query_embedding)
        async with get_connection() as conn:
            await init_pgvector(conn)
            model_label = embedding_model
            if model_label and not await embedding_model_column_available(conn):
                model_label = None
            sql, params = build_vector_search_sql(
                self.session_id, self.agent_id, padded_query, resolved_limit,
                start_date=start_date, end_date=end_date,
                include_active_filter=has_col,
                embedding_model=model_label,
            )
            rows = await conn.fetch(sql, *params)
            return format_rows(rows, extra_fields=["similarity"])

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Search archival memory with FTS (backward compatible alias)."""
        return await self.search_text(query, limit)

    async def search_vector_with_threshold(
        self,
        query_embedding: list[float],
        threshold: float = 0.0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search archival memory with vector similarity and threshold filter."""
        resolved_limit = _search_limit(limit)
        has_col = await self._check_superseded_column()
        active_filter = f"AND {ACTIVE_ROW_FILTER}" if has_col else ""
        padded_query = pad_embedding(query_embedding)
        async with get_connection() as conn:
            await init_pgvector(conn)
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata, created_at,
                    1 - (embedding <=> $3::vector) as similarity
                FROM archival_memory
                WHERE session_id = $1
                AND agent_id IS NOT DISTINCT FROM $2
                AND embedding IS NOT NULL
                {active_filter}
                AND (1 - (embedding <=> $3::vector)) >= $4
                ORDER BY embedding <=> $3::vector
                LIMIT $5
                """,
                self.session_id, self.agent_id, padded_query, threshold, resolved_limit,
            )
            return format_rows(rows, extra_fields=["similarity"])

    async def search_vector_with_filter(
        self,
        query_embedding: list[float],
        metadata_filter: dict[str, Any],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search archival memory with vector similarity and metadata filter."""
        resolved_limit = _search_limit(limit)
        has_col = await self._check_superseded_column()
        active_filter = f"AND {ACTIVE_ROW_FILTER}" if has_col else ""
        padded_query = pad_embedding(query_embedding)
        async with get_connection() as conn:
            await init_pgvector(conn)
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata, created_at,
                    1 - (embedding <=> $3::vector) as similarity
                FROM archival_memory
                WHERE session_id = $1
                AND agent_id IS NOT DISTINCT FROM $2
                AND embedding IS NOT NULL
                {active_filter}
                AND metadata @> $4::jsonb
                ORDER BY embedding <=> $3::vector
                LIMIT $5
                """,
                self.session_id, self.agent_id, padded_query,
                json.dumps(metadata_filter), resolved_limit,
            )
            return format_rows(rows, extra_fields=["similarity"])

    async def search_vector_global(
        self,
        query_embedding: list[float],
        threshold: float = 0.92,
        limit: int = 5,
        embedding_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search archival memory globally (all sessions) with similarity threshold.

        ``embedding_model`` (issue #151, round 1 FIX 2): when supplied, the
        cross-session search — used by store-time semantic dedup — is pinned
        to the same embedding space that will be written with the row, so a
        new voyage embedding is never compared against bge rows (a spurious
        cross-space match would silently skip the write). The filter binds at
        $4, after the existing $1/$2/$3. The capability probe drops the filter
        on a pre-migration DB so the SQL stays byte-identical to pre-#151.
        """
        has_col = await self._check_superseded_column()
        active_filter = f"AND {ACTIVE_ROW_FILTER}" if has_col else ""
        padded_query = pad_embedding(query_embedding)
        params: list[Any] = [padded_query, threshold, limit]
        async with get_connection() as conn:
            await init_pgvector(conn)
            model_filter = ""
            if embedding_model and await embedding_model_column_available(conn):
                model_filter = f"AND embedding_model = ${len(params) + 1}"
                params.append(embedding_model)
            rows = await conn.fetch(
                f"""
                SELECT id, session_id, content, metadata, created_at,
                    1 - (embedding <=> $1::vector) as similarity
                FROM archival_memory
                WHERE embedding IS NOT NULL
                {active_filter}
                AND (1 - (embedding <=> $1::vector)) >= $2
                {model_filter}
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                *params,
            )
            return format_rows(
                rows, extra_fields=["session_id", "similarity"]
            )

    async def search_hybrid(
        self,
        text_query: str,
        query_embedding: list[float],
        limit: int | None = None,
        text_weight: float = 0.3,
        vector_weight: float = 0.7,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining full-text search and vector similarity."""
        resolved_limit = _search_limit(limit)
        has_col = await self._check_superseded_column()
        padded_query = pad_embedding(query_embedding)
        sql, params = build_hybrid_search_sql(
            self.session_id, self.agent_id, text_query, padded_query, resolved_limit,
            text_weight=text_weight, vector_weight=vector_weight,
            start_date=start_date, end_date=end_date,
            include_active_filter=has_col,
        )
        async with get_connection() as conn:
            await init_pgvector(conn)
            rows = await conn.fetch(sql, *params)
            return format_rows(
                rows,
                extra_fields=["text_rank", "similarity", "combined_score"],
            )

    async def search_hybrid_rrf(
        self,
        text_query: str,
        query_embedding: list[float],
        limit: int | None = None,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search using Reciprocal Rank Fusion."""
        resolved_limit = _search_limit(limit)
        resolved_k = _rrf_k(k)
        has_col = await self._check_superseded_column()
        active_filter = f"AND {ACTIVE_ROW_FILTER}" if has_col else ""
        padded_query = pad_embedding(query_embedding)
        async with get_connection() as conn:
            await init_pgvector(conn)
            rows = await conn.fetch(
                f"""
                WITH fts_ranked AS (
                    SELECT id,
                        ROW_NUMBER() OVER (
                            ORDER BY ts_rank(
                                to_tsvector('english', content),
                                plainto_tsquery('english', $3)
                            ) DESC
                        ) as fts_rank
                    FROM archival_memory
                    WHERE session_id = $1
                    AND agent_id IS NOT DISTINCT FROM $2
                    AND to_tsvector('english', content) @@ plainto_tsquery('english', $3)
                    {active_filter}
                ),
                vector_ranked AS (
                    SELECT id,
                        ROW_NUMBER() OVER (ORDER BY embedding <=> $4::vector) as vec_rank
                    FROM archival_memory
                    WHERE session_id = $1
                    AND agent_id IS NOT DISTINCT FROM $2
                    AND embedding IS NOT NULL
                    {active_filter}
                ),
                combined AS (
                    SELECT
                        COALESCE(f.id, v.id) as id,
                        COALESCE(1.0 / ($5 + f.fts_rank), 0) +
                        COALESCE(1.0 / ($5 + v.vec_rank), 0) as rrf_score
                    FROM fts_ranked f
                    FULL OUTER JOIN vector_ranked v ON f.id = v.id
                )
                SELECT a.id, a.content, a.metadata, a.created_at, c.rrf_score
                FROM combined c
                JOIN archival_memory a ON a.id = c.id
                ORDER BY c.rrf_score DESC
                LIMIT $6
                """,
                self.session_id, self.agent_id, text_query,
                padded_query, resolved_k, resolved_limit,
            )
            return format_rows(
                rows, extra_fields=["rrf_score"], float_fields=["rrf_score"]
            )

    async def store_with_embedding(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> str:
        """Store a fact with auto-generated embedding."""
        embedding = None
        if embedder is not None:
            embedding = await embedder.embed(content)
        return await self.store(content, metadata, embedding)

    async def delete_archival(self, memory_id: str) -> None:
        """Delete a fact from archival memory (tags cascade)."""
        async with get_connection() as conn:
            await conn.execute(
                """
                DELETE FROM archival_memory
                WHERE id = $1 AND session_id = $2 AND agent_id IS NOT DISTINCT FROM $3
                """,
                memory_id, self.session_id, self.agent_id,
            )

    # ==================== Tag Operations ====================

    async def get_tags(self, memory_id: str) -> list[str]:
        """Get all tags for a memory ID."""
        async with get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT tag FROM memory_tags
                WHERE memory_id = $1 AND session_id = $2
                ORDER BY tag
                """,
                memory_id, self.session_id,
            )
            return [row["tag"] for row in rows]

    async def add_tag(self, memory_id: str, tag: str) -> None:
        """Add a tag to an existing memory."""
        async with get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO memory_tags (memory_id, tag, session_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (memory_id, tag) DO NOTHING
                """,
                memory_id, tag, self.session_id,
            )

    async def remove_tag(self, memory_id: str, tag: str) -> None:
        """Remove a tag from a memory."""
        async with get_connection() as conn:
            await conn.execute(
                """
                DELETE FROM memory_tags
                WHERE memory_id = $1 AND tag = $2 AND session_id = $3
                """,
                memory_id, tag, self.session_id,
            )

    async def get_all_session_tags(self) -> list[str]:
        """Get all unique tags used in this session."""
        async with get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT tag FROM memory_tags
                WHERE session_id = $1
                ORDER BY tag
                """,
                self.session_id,
            )
            return [row["tag"] for row in rows]

    async def search_with_tags(
        self,
        query: str,
        tags: list[str] | None = None,
        tag_match_mode: str = "any",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search archival memory with optional tag filtering."""
        resolved_limit = _search_limit(limit)
        has_col = await self._check_superseded_column()
        active_filter = f"AND a.{ACTIVE_ROW_FILTER}" if has_col else ""
        async with get_connection() as conn:
            if not tags:
                rows = await conn.fetch(
                    f"""
                    SELECT a.id, a.content, a.metadata, a.created_at,
                        ts_rank(
                            to_tsvector('english', a.content),
                            plainto_tsquery('english', $3)
                        ) as score
                    FROM archival_memory a
                    WHERE a.session_id = $1
                    AND a.agent_id IS NOT DISTINCT FROM $2
                    AND to_tsvector('english', a.content)
                        @@ plainto_tsquery('english', $3)
                    {active_filter}
                    ORDER BY score DESC
                    LIMIT $4
                    """,
                    self.session_id, self.agent_id, query, resolved_limit,
                )
            elif tag_match_mode == "all":
                rows = await conn.fetch(
                    f"""
                    SELECT a.id, a.content, a.metadata, a.created_at,
                        ts_rank(
                            to_tsvector('english', a.content),
                            plainto_tsquery('english', $3)
                        ) as score
                    FROM archival_memory a
                    WHERE a.session_id = $1
                    AND a.agent_id IS NOT DISTINCT FROM $2
                    AND to_tsvector('english', a.content)
                        @@ plainto_tsquery('english', $3)
                    {active_filter}
                    AND a.id IN (
                        SELECT memory_id FROM memory_tags
                        WHERE session_id = $1 AND tag = ANY($4)
                        GROUP BY memory_id
                        HAVING COUNT(DISTINCT tag) = $5
                    )
                    ORDER BY score DESC
                    LIMIT $6
                    """,
                    self.session_id, self.agent_id, query,
                    tags, len(tags), resolved_limit,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT a.id, a.content, a.metadata, a.created_at,
                        ts_rank(
                            to_tsvector('english', a.content),
                            plainto_tsquery('english', $3)
                        ) as score
                    FROM archival_memory a
                    WHERE a.session_id = $1
                    AND a.agent_id IS NOT DISTINCT FROM $2
                    AND to_tsvector('english', a.content)
                        @@ plainto_tsquery('english', $3)
                    {active_filter}
                    AND a.id IN (
                        SELECT memory_id FROM memory_tags
                        WHERE session_id = $1 AND tag = ANY($4)
                    )
                    ORDER BY score DESC
                    LIMIT $5
                    """,
                    self.session_id, self.agent_id, query, tags, resolved_limit,
                )

            return format_rows(rows, extra_fields=["score"], float_fields=["score"])

    # ==================== Recall Memory ====================

    async def recall(
        self,
        query: str,
        include_core: bool = True,
        limit: int = 5,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> str:
        """Recall information from all memory sources."""
        core_matches: dict[str, str] = {}
        if include_core:
            core = await self.get_all_core()
            core_matches = filter_core_by_query(core, query)

        archival_results = await self.search_text(
            query, limit=limit, start_date=start_date, end_date=end_date
        )
        return format_recall_text(core_matches, archival_results)

    async def to_context(self, max_archival: int | None = None) -> str:
        """Generate context string for prompt injection."""
        resolved_max_archival = _max_archival(max_archival)
        core = await self.get_all_core()
        has_col = await self._check_superseded_column()
        active_filter = f"AND {ACTIVE_ROW_FILTER}" if has_col else ""

        async with get_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT content FROM archival_memory
                WHERE session_id = $1 AND agent_id IS NOT DISTINCT FROM $2
                {active_filter}
                ORDER BY created_at DESC
                LIMIT $3
                """,
                self.session_id, self.agent_id, resolved_max_archival,
            )

        archival_contents = [row["content"] for row in rows]
        return format_context_string(core, archival_contents)
