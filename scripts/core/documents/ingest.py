"""Ingest pipeline: scan a collection folder, extract, chunk, embed, persist.

Incremental and idempotent. Each file's sha256 is compared against the stored
hash; unchanged files are skipped so a cron-driven `scan --all` is cheap.
Files with no extractable text (scans) are still recorded — with status
'skipped_needs_ocr' and zero chunks — so they are not re-attempted every run
and are queryable for a future phase-2 OCR backfill.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from scripts.core.documents.chunk import chunk_pages
from scripts.core.documents.db import (
    get_document_by_path,
    reconcile_chunk_scope,
    upsert_document_with_chunks,
)
from scripts.core.documents.extract import extract_text
from scripts.core.documents.registry import Collection

_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB read buffer


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
    needs_ocr: int = 0
    errors: int = 0
    rescoped: int = 0


def compute_file_hash(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _iter_files(root: Path, extensions: list[str]) -> list[Path]:
    """Return every file under root whose suffix is in extensions (recursive)."""
    wanted = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted)


async def ingest_collection(collection: Collection, embedder: Embedder) -> IngestReport:
    """Ingest one collection. Returns a per-run report.

    Raises:
        FileNotFoundError: if the collection's path does not exist.
    """
    root = Path(collection.path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"collection path does not exist: {root}")

    report = IngestReport(collection=collection.name)

    for file_path in _iter_files(root, collection.extensions):
        file_hash = compute_file_hash(file_path)
        existing = await get_document_by_path(collection.name, str(file_path))
        if existing is not None and existing["file_hash"] == file_hash:
            # Bytes unchanged, but the registry scope may have changed since the
            # last scan (e.g. a folder reclassified global -> restricted). Scope
            # lives on the chunks, not the hash, so reconcile it explicitly —
            # otherwise stale-scoped chunks keep leaking into default queries.
            report.rescoped += await reconcile_chunk_scope(
                collection.name, str(file_path), collection.scope
            )
            report.skipped_unchanged += 1
            continue

        result = extract_text(file_path)
        size = file_path.stat().st_size

        if result.status == "skipped_unsupported":
            report.skipped_unsupported += 1
            continue

        if result.status == "extracted":
            chunks = chunk_pages(result.pages)
            embeddings = await embedder.embed_batch([c.content for c in chunks]) if chunks else []
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
        else:
            # 'skipped_needs_ocr' or 'error' — record the file with zero chunks
            # so it is not re-processed every scan, but carries no embeddings.
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

    return report
