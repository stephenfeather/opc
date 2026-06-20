"""Ingest pipeline: scan a collection folder, extract, chunk, embed, persist.

Incremental and idempotent. Each file's sha256 is compared against the stored
hash; unchanged files are skipped so a cron-driven `scan --all` is cheap.
Files with no extractable text (scans) are still recorded — with status
'skipped_needs_ocr' and zero chunks — so they are not re-attempted every run
and are queryable for a future phase-2 OCR backfill.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from scripts.core.documents.chunk import chunk_pages
from scripts.core.documents.db import (
    delete_document_by_path,
    delete_documents_not_in,
    get_document_by_path,
    reconcile_collection_scope,
    upsert_document_with_chunks,
)
from scripts.core.documents.extract import extract_text
from scripts.core.documents.registry import Collection

_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB read buffer
EMBEDDING_DIM = 1024  # must match document_chunks.embedding vector(1024)
_DEFAULT_MAX_FILE_MB = 25
_DEFAULT_MAX_CHUNKS = 5000
_ERROR_MSG_MAX = 500


def _max_file_bytes() -> int:
    """Per-file size ceiling; larger files are skipped (not embedded), bounding
    memory and embedding-request size. Override with OPC_DOC_MAX_FILE_MB.

    A negative value would make the size check always true and skip every file,
    so negatives fall back to the default.
    """
    try:
        mb = float(os.getenv("OPC_DOC_MAX_FILE_MB", str(_DEFAULT_MAX_FILE_MB)))
        if mb < 0:
            raise ValueError
    except (TypeError, ValueError):
        mb = _DEFAULT_MAX_FILE_MB
    return int(mb * 1024 * 1024)


def _max_chunks_per_file() -> int:
    """Cap on chunks embedded from one file, bounding a single embed_batch /
    insert payload even for an allowed-size but chunk-dense file. Override with
    OPC_DOC_MAX_CHUNKS.

    A negative value would reject every file, so negatives fall back to default.
    """
    try:
        n = int(os.getenv("OPC_DOC_MAX_CHUNKS", str(_DEFAULT_MAX_CHUNKS)))
        if n < 0:
            raise ValueError
        return n
    except (TypeError, ValueError):
        return _DEFAULT_MAX_CHUNKS


class Embedder(Protocol):
    """Minimal embedding interface — satisfied by OPC's EmbeddingService."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class IngestReport:
    """Summary of one ingest run over a collection."""

    collection: str
    ingested: int = 0
    skipped_unchanged: int = 0
    skipped_unsupported: int = 0
    skipped_too_large: int = 0
    needs_ocr: int = 0
    errors: int = 0
    rescoped: int = 0
    purged: int = 0


def compute_file_hash(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _iter_files(root: Path, extensions: list[str]) -> list[Path]:
    """Return every file under root whose suffix is in extensions (recursive).

    Trust boundary: rglob follows symlinks, and the registry path is authored by
    the (single, trusted) user, so tracked folders MUST be trusted and not shared
    with less-trusted writers. If a tracked folder could receive symlinks planted
    by another party, that symlink becomes an arbitrary-file-read into a scope the
    planter influences — add is_symlink()/realpath containment checks before
    ingesting shared directories.
    """
    wanted = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted)


def _validate_embeddings(chunks: list, embeddings: list[list[float]]) -> None:
    """Fail fast (with a clear message) on a mismatched count or wrong vector
    dimension before any DB write, instead of surfacing an opaque pgvector error.

    A misconfigured EMBEDDING_PROVIDER (e.g. one producing 1536-dim vectors)
    would otherwise blow up at insert time mid-transaction.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"embedder returned {len(embeddings)} vectors for {len(chunks)} chunks")
    for vec in embeddings:
        if len(vec) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding dimension {len(vec)} != expected {EMBEDDING_DIM}; "
                "check EMBEDDING_PROVIDER against the vector(1024) column"
            )


async def _process_file(
    collection: Collection,
    embedder: Embedder,
    file_path: Path,
    max_bytes: int,
    max_chunks: int,
    report: IngestReport,
) -> None:
    """Process one file, mutating report. Raises on unexpected failures so the
    caller can isolate them per-file rather than aborting the whole scan."""
    size = file_path.stat().st_size
    if size > max_bytes:
        # Fail closed: if this path was previously ingested (smaller) and has
        # now grown past the ceiling, its stored chunks are stale and must not
        # remain queryable. Delete them; do not leave them indexed.
        await delete_document_by_path(collection.name, str(file_path))
        report.skipped_too_large += 1
        return

    file_hash = compute_file_hash(file_path)
    existing = await get_document_by_path(collection.name, str(file_path))
    if (
        existing is not None
        and existing["file_hash"] == file_hash
        and existing["extraction_status"] != "error"
    ):
        # Bytes unchanged AND last attempt was not an error -> nothing to do.
        # An 'error' row (e.g. a transient embedding-provider timeout) is NOT
        # treated as unchanged, so a later scan after the problem clears will
        # re-extract/re-embed it instead of leaving it empty forever. Scope
        # reconciliation is handled once per collection in ingest_collection.
        report.skipped_unchanged += 1
        return

    result = extract_text(file_path)

    if result.status == "skipped_unsupported":
        # Fail closed: a path that is no longer a supported type (extension
        # dropped, or support removed) must not keep previously-ingested chunks
        # queryable. Remove any existing row for it.
        await delete_document_by_path(collection.name, str(file_path))
        report.skipped_unsupported += 1
        return

    if result.status == "extracted":
        chunks = chunk_pages(result.pages)
        if len(chunks) > max_chunks:
            # Allowed size but chunk-dense: bound the embed_batch / insert
            # payload by refusing it as a terminal error rather than risking a
            # provider failure or memory spike. Recorded with zero chunks.
            await upsert_document_with_chunks(
                collection_name=collection.name,
                scope=collection.scope,
                file_path=str(file_path),
                file_hash=file_hash,
                file_size_bytes=size,
                page_count=result.page_count,
                extraction_status="error",
                error=f"too many chunks ({len(chunks)} > {max_chunks}); raise OPC_DOC_MAX_CHUNKS",
                chunks=[],
                embeddings=[],
            )
            report.errors += 1
            return
        embeddings = await embedder.embed_batch([c.content for c in chunks]) if chunks else []
        _validate_embeddings(chunks, embeddings)
        await upsert_document_with_chunks(
            collection_name=collection.name,
            scope=collection.scope,
            file_path=str(file_path),
            file_hash=file_hash,
            file_size_bytes=size,
            page_count=result.page_count,
            extraction_status="extracted",
            error=None,
            chunks=chunks,
            embeddings=embeddings,
        )
        report.ingested += 1
        return

    # 'skipped_needs_ocr' or 'error' — record the file with zero chunks so it is
    # not re-processed every scan, but carries no embeddings.
    await upsert_document_with_chunks(
        collection_name=collection.name,
        scope=collection.scope,
        file_path=str(file_path),
        file_hash=file_hash,
        file_size_bytes=size,
        page_count=0,
        extraction_status=result.status,
        error=result.error,
        chunks=[],
        embeddings=[],
    )
    if result.status == "skipped_needs_ocr":
        report.needs_ocr += 1
    else:
        report.errors += 1


async def _record_error_best_effort(
    collection: Collection, file_path: Path, exc: Exception
) -> None:
    """Record a file that raised an unexpected error as status='error' so cron
    does not wedge on it forever. Best-effort: if even this write fails (e.g. DB
    down), swallow it — the run already counted the error and will retry next
    scan since no row was persisted."""
    try:
        size: int | None = file_path.stat().st_size
    except OSError:
        size = None
    try:
        file_hash = compute_file_hash(file_path)
    except OSError:
        # File vanished mid-scan; leave hash empty so a future scan reprocesses.
        file_hash = ""
    try:
        await upsert_document_with_chunks(
            collection_name=collection.name,
            scope=collection.scope,
            file_path=str(file_path),
            file_hash=file_hash,
            file_size_bytes=size,
            page_count=0,
            extraction_status="error",
            error=str(exc)[:_ERROR_MSG_MAX],
            chunks=[],
            embeddings=[],
        )
    except Exception:  # noqa: BLE001 - error recording must never raise
        pass


async def ingest_collection(collection: Collection, embedder: Embedder) -> IngestReport:
    """Ingest one collection. Returns a per-run report.

    Per-file failures are isolated: one bad file (provider timeout, wrong
    embedding dimension, corrupt document) is recorded as an error and the scan
    continues. After processing, documents whose files no longer exist on disk
    are purged so deleted (especially restricted) files stop being retrievable.

    Raises:
        FileNotFoundError: if the collection's path does not exist (fails closed
            so a transiently-unmounted folder never triggers a mass purge).
    """
    root = Path(collection.path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"collection path does not exist: {root}")

    report = IngestReport(collection=collection.name)
    max_bytes = _max_file_bytes()
    max_chunks = _max_chunks_per_file()
    seen_paths: set[str] = set()

    # Scope reconciliation, once per collection: a single bulk UPDATE brings any
    # mismatched chunks to the registry scope (cheap no-op when unchanged), so a
    # global -> restricted reclassification takes effect without a per-file write.
    report.rescoped = await reconcile_collection_scope(collection.name, collection.scope)

    for file_path in _iter_files(root, collection.extensions):
        seen_paths.add(str(file_path))
        try:
            await _process_file(collection, embedder, file_path, max_bytes, max_chunks, report)
        except Exception as exc:  # noqa: BLE001 - isolate per-file failures
            report.errors += 1
            await _record_error_best_effort(collection, file_path, exc)

    # Deletion reconciliation: drop rows for files that disappeared from disk.
    report.purged = await delete_documents_not_in(collection.name, seen_paths)
    return report
