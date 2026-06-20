"""Tests for the document DB layer. Requires a live Postgres (skipped otherwise)."""

from __future__ import annotations

import os

import pytest

from scripts.core.db.postgres_pool import close_pool, get_connection, reset_pool
from scripts.core.documents.chunk import Chunk
from scripts.core.documents.db import (
    collection_stats,
    get_document_by_path,
    query_chunks,
    upsert_document_with_chunks,
)

_HAS_DB = bool(
    os.getenv("OPC_POSTGRES_URL")
    or os.getenv("CONTINUOUS_CLAUDE_DB_URL")
    or os.getenv("DATABASE_URL")
)
pytestmark = pytest.mark.skipif(not _HAS_DB, reason="no Postgres URL in environment")


@pytest.fixture(autouse=True)
async def _fresh_pool():
    """Bind a fresh connection pool to each test's event loop.

    pytest-asyncio uses a per-function loop; the module-level pool in
    postgres_pool would otherwise be reused across loops, raising
    "Event loop is closed". reset_pool() exists for exactly this case.
    """
    reset_pool()
    yield
    await close_pool()


def _vec() -> list[float]:
    return [0.01] * 1024


async def _cleanup(collection: str) -> None:
    async with get_connection() as conn:
        await conn.execute("DELETE FROM documents WHERE collection_name = $1", collection)


async def test_upsert_then_get_document() -> None:
    col = "test-col-upsert"
    await _cleanup(col)
    chunks = [Chunk(chunk_index=0, content="alpha beta", page_number=1)]
    doc_id = await upsert_document_with_chunks(
        collection_name=col,
        scope="global",
        file_path="/tmp/a.txt",
        file_hash="hash-1",
        file_size_bytes=10,
        page_count=1,
        extraction_status="extracted",
        error=None,
        chunks=chunks,
        embeddings=[_vec()],
    )
    assert doc_id is not None
    fetched = await get_document_by_path(col, "/tmp/a.txt")
    assert fetched is not None
    assert fetched["file_hash"] == "hash-1"
    await _cleanup(col)


async def test_upsert_replaces_chunks_on_rehash() -> None:
    col = "test-col-replace"
    await _cleanup(col)
    await upsert_document_with_chunks(
        collection_name=col,
        scope="global",
        file_path="/tmp/b.txt",
        file_hash="hash-old",
        file_size_bytes=5,
        page_count=1,
        extraction_status="extracted",
        error=None,
        chunks=[Chunk(0, "old content", 1)],
        embeddings=[_vec()],
    )
    await upsert_document_with_chunks(
        collection_name=col,
        scope="global",
        file_path="/tmp/b.txt",
        file_hash="hash-new",
        file_size_bytes=5,
        page_count=1,
        extraction_status="extracted",
        error=None,
        chunks=[Chunk(0, "new content", 1), Chunk(1, "more new", 1)],
        embeddings=[_vec(), _vec()],
    )
    fetched = await get_document_by_path(col, "/tmp/b.txt")
    assert fetched["file_hash"] == "hash-new"
    results = await query_chunks(_vec(), scope="global", collection=col, limit=10)
    contents = {r["content"] for r in results}
    assert contents == {"new content", "more new"}
    assert "old content" not in contents
    await _cleanup(col)


async def test_query_chunks_respects_scope() -> None:
    await _cleanup("test-col-global")
    await _cleanup("test-col-restricted")
    await upsert_document_with_chunks(
        collection_name="test-col-global",
        scope="global",
        file_path="/tmp/g.txt",
        file_hash="hg",
        file_size_bytes=1,
        page_count=1,
        extraction_status="extracted",
        error=None,
        chunks=[Chunk(0, "global secret", 1)],
        embeddings=[_vec()],
    )
    await upsert_document_with_chunks(
        collection_name="test-col-restricted",
        scope="restricted",
        file_path="/tmp/r.txt",
        file_hash="hr",
        file_size_bytes=1,
        page_count=1,
        extraction_status="extracted",
        error=None,
        chunks=[Chunk(0, "restricted secret", 1)],
        embeddings=[_vec()],
    )
    # Default global query must NOT see restricted content.
    global_only = await query_chunks(_vec(), scope="global", collection=None, limit=10)
    seen = {r["content"] for r in global_only}
    assert "restricted secret" not in seen
    # Explicit collection targeting DOES see it.
    targeted = await query_chunks(
        _vec(), scope="global", collection="test-col-restricted", limit=10
    )
    assert "restricted secret" in {r["content"] for r in targeted}
    await _cleanup("test-col-global")
    await _cleanup("test-col-restricted")


async def test_query_chunks_scope_all_sees_restricted() -> None:
    await _cleanup("test-col-all")
    await upsert_document_with_chunks(
        collection_name="test-col-all",
        scope="restricted",
        file_path="/tmp/all.txt",
        file_hash="ha",
        file_size_bytes=1,
        page_count=1,
        extraction_status="extracted",
        error=None,
        chunks=[Chunk(0, "findable everywhere", 1)],
        embeddings=[_vec()],
    )
    results = await query_chunks(_vec(), scope="all", collection=None, limit=10)
    assert "findable everywhere" in {r["content"] for r in results}
    await _cleanup("test-col-all")


async def test_collection_stats() -> None:
    col = "test-col-stats"
    await _cleanup(col)
    await upsert_document_with_chunks(
        collection_name=col,
        scope="global",
        file_path="/tmp/s.txt",
        file_hash="hs",
        file_size_bytes=1,
        page_count=1,
        extraction_status="extracted",
        error=None,
        chunks=[Chunk(0, "x", 1), Chunk(1, "y", 1)],
        embeddings=[_vec(), _vec()],
    )
    stats = await collection_stats(col)
    assert stats["document_count"] == 1
    assert stats["chunk_count"] == 2
    assert stats["last_scanned_at"] is not None
    await _cleanup(col)
