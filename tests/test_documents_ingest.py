"""Tests for the ingest pipeline. DB layer is mocked; this tests orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.core.documents.ingest import compute_file_hash, ingest_collection
from scripts.core.documents.registry import Collection


def test_compute_file_hash_is_stable(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("content")
    h1 = compute_file_hash(f)
    h2 = compute_file_hash(f)
    assert h1 == h2 and len(h1) == 64


def test_compute_file_hash_changes_with_content(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("a")
    h1 = compute_file_hash(f)
    f.write_text("b")
    assert compute_file_hash(f) != h1


class _FakeEmbedder:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * 1024 for _ in texts]


async def test_ingest_collection_skips_unchanged_files(tmp_path: Path) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("hello there")
    collection = Collection(
        name="c", path=str(tmp_path), scope="restricted", extensions=[".txt"], ocr=False
    )
    with (
        patch(
            "scripts.core.documents.ingest.get_document_by_path",
            new=AsyncMock(return_value={"file_hash": compute_file_hash(doc)}),
        ),
        patch(
            "scripts.core.documents.ingest.upsert_document_with_chunks",
            new=AsyncMock(),
        ) as mock_upsert,
        patch(
            "scripts.core.documents.ingest.reconcile_chunk_scope",
            new=AsyncMock(return_value=2),
        ) as mock_reconcile,
        patch(
            "scripts.core.documents.ingest.delete_documents_not_in",
            new=AsyncMock(return_value=0),
        ),
    ):
        report = await ingest_collection(collection, _FakeEmbedder())
    mock_upsert.assert_not_called()
    # Even when bytes are unchanged, the current registry scope is reconciled
    # so a reclassified folder cannot keep leaking stale-scoped chunks.
    mock_reconcile.assert_awaited_once_with("c", str(doc), "restricted")
    assert report.skipped_unchanged == 1
    assert report.rescoped == 2
    assert report.ingested == 0


async def test_ingest_collection_ingests_new_file(tmp_path: Path) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("hello there")
    collection = Collection(
        name="c", path=str(tmp_path), scope="global", extensions=[".txt"], ocr=False
    )
    with (
        patch(
            "scripts.core.documents.ingest.get_document_by_path",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "scripts.core.documents.ingest.upsert_document_with_chunks",
            new=AsyncMock(return_value="doc-uuid"),
        ) as mock_upsert,
        patch(
            "scripts.core.documents.ingest.delete_documents_not_in",
            new=AsyncMock(return_value=0),
        ),
    ):
        report = await ingest_collection(collection, _FakeEmbedder())
    mock_upsert.assert_awaited_once()
    assert report.ingested == 1
    # scope from the collection must be forwarded to the DB layer.
    assert mock_upsert.await_args.kwargs["scope"] == "global"


async def test_ingest_collection_records_needs_ocr_without_embedding(tmp_path: Path) -> None:
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.1\n%%EOF")  # unreadable as text -> error/needs_ocr path
    collection = Collection(
        name="c", path=str(tmp_path), scope="restricted", extensions=[".pdf"], ocr=True
    )
    with (
        patch(
            "scripts.core.documents.ingest.get_document_by_path",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "scripts.core.documents.ingest.upsert_document_with_chunks",
            new=AsyncMock(return_value="doc-uuid"),
        ) as mock_upsert,
        patch(
            "scripts.core.documents.ingest.delete_documents_not_in",
            new=AsyncMock(return_value=0),
        ),
    ):
        report = await ingest_collection(collection, _FakeEmbedder())
    # The file is still recorded (so we don't re-try it every scan), but with
    # no chunks and a non-'extracted' status.
    mock_upsert.assert_awaited_once()
    assert mock_upsert.await_args.kwargs["chunks"] == []
    assert mock_upsert.await_args.kwargs["extraction_status"] in (
        "skipped_needs_ocr",
        "error",
    )
    assert report.needs_ocr + report.errors == 1


class _BoomEmbedder:
    """Raises for any chunk containing 'boom', succeeds otherwise."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if any("boom" in t for t in texts):
            raise RuntimeError("provider exploded")
        return [[0.01] * 1024 for _ in texts]


async def test_ingest_skips_oversize_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPC_DOC_MAX_FILE_MB", "0")  # any non-empty file is too large
    big = tmp_path / "big.txt"
    big.write_text("this content exceeds the zero-byte ceiling")
    collection = Collection(
        name="c", path=str(tmp_path), scope="global", extensions=[".txt"], ocr=False
    )
    with (
        patch("scripts.core.documents.ingest.get_document_by_path", new=AsyncMock()),
        patch(
            "scripts.core.documents.ingest.upsert_document_with_chunks", new=AsyncMock()
        ) as mock_upsert,
        patch(
            "scripts.core.documents.ingest.delete_documents_not_in",
            new=AsyncMock(return_value=0),
        ),
    ):
        report = await ingest_collection(collection, _FakeEmbedder())
    # Too-large files are skipped without embedding or a DB write.
    mock_upsert.assert_not_called()
    assert report.skipped_too_large == 1
    assert report.ingested == 0


async def test_ingest_isolates_per_file_failure(tmp_path: Path) -> None:
    (tmp_path / "a_boom.txt").write_text("boom")  # sorts first; embedder raises
    (tmp_path / "b_good.txt").write_text("good")  # must still be ingested
    collection = Collection(
        name="c", path=str(tmp_path), scope="global", extensions=[".txt"], ocr=False
    )
    with (
        patch(
            "scripts.core.documents.ingest.get_document_by_path",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "scripts.core.documents.ingest.upsert_document_with_chunks",
            new=AsyncMock(return_value="doc-uuid"),
        ) as mock_upsert,
        patch(
            "scripts.core.documents.ingest.delete_documents_not_in",
            new=AsyncMock(return_value=0),
        ),
    ):
        report = await ingest_collection(collection, _BoomEmbedder())
    # One file blew up but the scan continued and ingested the other.
    assert report.errors == 1
    assert report.ingested == 1
    # The failed file is recorded with status='error' so cron will not wedge.
    statuses = [c.kwargs["extraction_status"] for c in mock_upsert.await_args_list]
    assert "error" in statuses
    assert "extracted" in statuses


async def test_ingest_purges_deleted_files(tmp_path: Path) -> None:
    doc = tmp_path / "keep.txt"
    doc.write_text("still here")
    collection = Collection(
        name="c", path=str(tmp_path), scope="global", extensions=[".txt"], ocr=False
    )
    with (
        patch(
            "scripts.core.documents.ingest.get_document_by_path",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "scripts.core.documents.ingest.upsert_document_with_chunks",
            new=AsyncMock(return_value="doc-uuid"),
        ),
        patch(
            "scripts.core.documents.ingest.delete_documents_not_in",
            new=AsyncMock(return_value=3),
        ) as mock_purge,
    ):
        report = await ingest_collection(collection, _FakeEmbedder())
    # Deletion reconciliation runs with exactly the paths seen on disk.
    mock_purge.assert_awaited_once_with("c", {str(doc)})
    assert report.purged == 3
