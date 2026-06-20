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
        scope == 'all'         -> search every chunk regardless of scope.
        otherwise              -> search every chunk whose scope == `scope`.

    A default query passes scope='global', collection=None, so 'restricted'
    collections never surface unless explicitly targeted.
    """
    if collection is not None:
        where = "dc.collection_name = $2"
        scope_arg = collection
    elif scope == "all":
        where = "TRUE OR $2 = $2"  # $2 referenced to keep arg arity fixed
        scope_arg = scope
    else:
        where = "dc.scope = $2"
        scope_arg = scope

    async with get_connection() as conn:
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
