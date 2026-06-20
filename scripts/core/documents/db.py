"""Async DB layer for the document-collection RAG tables.

Reuses scripts.core.db.postgres_pool for connection management. The pool's
_init_connection already registers the pgvector codec, so embeddings can be
passed as Python lists.
"""

from __future__ import annotations

from typing import Any

from scripts.core.db.postgres_pool import get_connection, get_transaction
from scripts.core.documents.chunk import Chunk


async def get_document_by_path(collection_name: str, file_path: str) -> dict[str, Any] | None:
    """Return the document row for (collection, path), or None if absent."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT id, file_hash, extraction_status, page_count "
            "FROM documents WHERE collection_name = $1 AND file_path = $2",
            collection_name,
            file_path,
        )
    return dict(row) if row else None


async def upsert_document_with_chunks(
    *,
    collection_name: str,
    scope: str,
    file_path: str,
    file_hash: str,
    file_size_bytes: int | None,
    page_count: int | None,
    extraction_status: str,
    error: str | None,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> str:
    """Insert or replace a document and its chunks atomically.

    On conflict (collection_name, file_path) the document row is updated and
    its old chunks are deleted (ON DELETE CASCADE) before the new ones land.
    This makes re-ingestion of a changed file idempotent.

    Raises:
        ValueError: if len(chunks) != len(embeddings).

    Returns:
        The document UUID as a string.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    async with get_transaction() as conn:
        doc_id = await conn.fetchval(
            """
            INSERT INTO documents (
                collection_name, file_path, file_hash, file_size_bytes,
                page_count, extraction_status, error, extracted_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (collection_name, file_path) DO UPDATE SET
                file_hash = EXCLUDED.file_hash,
                file_size_bytes = EXCLUDED.file_size_bytes,
                page_count = EXCLUDED.page_count,
                extraction_status = EXCLUDED.extraction_status,
                error = EXCLUDED.error,
                extracted_at = NOW()
            RETURNING id
            """,
            collection_name,
            file_path,
            file_hash,
            file_size_bytes,
            page_count,
            extraction_status,
            error,
        )
        # Replace chunks wholesale — simplest correct behaviour for a changed file.
        await conn.execute("DELETE FROM document_chunks WHERE document_id = $1", doc_id)
        if chunks:
            await conn.executemany(
                """
                INSERT INTO document_chunks (
                    document_id, collection_name, scope, chunk_index,
                    content, page_number, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (
                        doc_id,
                        collection_name,
                        scope,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.page_number,
                        embedding,
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )
    return str(doc_id)


async def query_chunks(
    query_embedding: list[float],
    *,
    scope: str,
    collection: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Retrieve the nearest chunks by cosine distance, gated by scope.

    Gating rules:
        collection is not None -> search ONLY that collection (any scope).
        otherwise              -> search every chunk whose scope == `scope`.

    A default query passes scope='global', collection=None, so 'restricted'
    collections never surface unless explicitly targeted by name. There is
    deliberately NO "all scopes" path: restricted collections (medical/legal
    records) are reachable only via an explicit collection name, never through
    a blanket flag.
    """
    if collection is not None:
        where = "dc.collection_name = $2"
        scope_arg = collection
    else:
        where = "dc.scope = $2"
        scope_arg = scope

    # A filtered query over a single global HNSW index can under-return (fewer
    # than LIMIT rows) when the WHERE filter is selective — e.g. a small
    # restricted collection embedded among many global chunks. pgvector 0.8's
    # iterative scan keeps probing until LIMIT rows pass the filter (or the
    # index is exhausted), preserving recall without a per-collection index.
    async with get_transaction() as conn:
        await conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        rows = await conn.fetch(
            f"""
            SELECT
                dc.content,
                dc.page_number,
                dc.collection_name,
                dc.scope,
                d.file_path,
                1 - (dc.embedding <=> $1) AS similarity
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE {where}
            ORDER BY dc.embedding <=> $1
            LIMIT $3
            """,
            query_embedding,
            scope_arg,
            limit,
        )
    return [dict(row) for row in rows]


async def reconcile_collection_scope(collection_name: str, scope: str) -> int:
    """Force every chunk in a collection to `scope`; return rows changed.

    Scope lives on document_chunks, not on the file hash. A folder reclassified
    in the registry (e.g. global -> restricted) for files whose bytes are
    unchanged would otherwise keep its old chunk scope and keep leaking into
    default queries. Ingest calls this ONCE per collection per scan (not per
    file), so a scope change always takes effect cheaply: a single set-based
    UPDATE that touches only mismatched rows and no-ops when scope is unchanged.
    """
    async with get_transaction() as conn:
        result = await conn.execute(
            """
            UPDATE document_chunks SET scope = $1
            WHERE collection_name = $2 AND scope <> $1
            """,
            scope,
            collection_name,
        )
    # asyncpg returns a status string like "UPDATE 3"; parse the row count.
    try:
        return int(result.split()[-1])
    except (AttributeError, ValueError, IndexError):
        return 0


async def delete_document_by_path(collection_name: str, file_path: str) -> int:
    """Delete one document (and its chunks via cascade) by path; return count.

    Used when a file can no longer be represented faithfully — e.g. it grew past
    the size ceiling — so its now-stale chunks must not remain queryable.
    """
    async with get_transaction() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE collection_name = $1 AND file_path = $2",
            collection_name,
            file_path,
        )
    try:
        return int(result.split()[-1])
    except (AttributeError, ValueError, IndexError):
        return 0


async def delete_documents_not_in(collection_name: str, keep_paths: set[str]) -> int:
    """Delete documents (and their chunks via cascade) whose file_path is not in
    keep_paths. Returns the number of documents removed.

    This is the deletion-reconciliation step: a file removed from a tracked
    folder must stop being retrievable. Without it, deleting a local
    medical/legal file would leave its chunks queryable forever. Chunks cascade
    on documents delete (ON DELETE CASCADE).

    Caller MUST pass the set of paths actually observed on disk this scan. The
    ingest path only reaches here after confirming the collection directory
    exists, so a transient unmounted folder fails closed earlier and never
    purges.
    """
    async with get_transaction() as conn:
        if keep_paths:
            result = await conn.execute(
                """
                DELETE FROM documents
                WHERE collection_name = $1
                  AND file_path <> ALL($2::text[])
                """,
                collection_name,
                list(keep_paths),
            )
        else:
            # No files observed at all -> every stored document is an orphan.
            result = await conn.execute(
                "DELETE FROM documents WHERE collection_name = $1",
                collection_name,
            )
    try:
        return int(result.split()[-1])
    except (AttributeError, ValueError, IndexError):
        return 0


async def collection_stats(collection_name: str) -> dict[str, Any]:
    """Return document count, chunk count, and last scan time for a collection."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(DISTINCT d.id) AS document_count,
                COUNT(dc.id) AS chunk_count,
                MAX(d.extracted_at) AS last_scanned_at
            FROM documents d
            LEFT JOIN document_chunks dc ON dc.document_id = d.id
            WHERE d.collection_name = $1
            """,
            collection_name,
        )
    return dict(row)
