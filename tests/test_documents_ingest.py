"""Tests for the ingest pipeline. DB layer is mocked; this tests orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scripts.core.documents.ingest import (
    _DEFAULT_MAX_CHUNKS,
    _DEFAULT_MAX_FILE_MB,
    _max_chunks_per_file,
    _max_file_bytes,
    compute_file_hash,
    ingest_collection,
)
from scripts.core.documents.registry import Collection


def test_max_file_bytes_rejects_negative_env(monkeypatch) -> None:
    monkeypatch.setenv("OPC_DOC_MAX_FILE_MB", "-5")
    assert _max_file_bytes() == int(_DEFAULT_MAX_FILE_MB * 1024 * 1024)


def test_max_file_bytes_rejects_garbage_env(monkeypatch) -> None:
    monkeypatch.setenv("OPC_DOC_MAX_FILE_MB", "not-a-number")
    assert _max_file_bytes() == int(_DEFAULT_MAX_FILE_MB * 1024 * 1024)


def test_max_chunks_rejects_negative_env(monkeypatch) -> None:
    monkeypatch.setenv("OPC_DOC_MAX_CHUNKS", "-1")
    assert _max_chunks_per_file() == _DEFAULT_MAX_CHUNKS


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


class _BoomEmbedder:
    """Raises for any chunk containing 'boom', succeeds otherwise."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if any("boom" in t for t in texts):
            raise RuntimeError("provider exploded")
        return [[0.01] * 1024 for _ in texts]


@pytest.fixture
def db():
    """Patch every DB function ingest reaches, defaulting to the new-file path."""
    with (
        patch(
            "scripts.core.documents.ingest.get_document_by_path",
            new=AsyncMock(return_value=None),
        ) as get_doc,
        patch(
            "scripts.core.documents.ingest.upsert_document_with_chunks",
            new=AsyncMock(return_value="doc-uuid"),
        ) as upsert,
        patch(
            "scripts.core.documents.ingest.reconcile_collection_scope",
            new=AsyncMock(return_value=0),
        ) as reconcile,
        patch(
            "scripts.core.documents.ingest.delete_documents_not_in",
            new=AsyncMock(return_value=0),
        ) as purge,
        patch(
            "scripts.core.documents.ingest.delete_document_by_path",
            new=AsyncMock(return_value=0),
        ) as del_one,
    ):
        yield SimpleNamespace(
            get_doc=get_doc,
            upsert=upsert,
            reconcile=reconcile,
            purge=purge,
            del_one=del_one,
        )


def _collection(tmp_path: Path, scope: str = "global", exts=(".txt",)) -> Collection:
    return Collection(name="c", path=str(tmp_path), scope=scope, extensions=list(exts), ocr=False)


async def test_ingest_collection_skips_unchanged_files(tmp_path: Path, db) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("hello there")
    db.get_doc.return_value = {"file_hash": compute_file_hash(doc)}
    db.reconcile.return_value = 2  # the bulk per-collection reconcile rescoped 2

    report = await ingest_collection(_collection(tmp_path, scope="restricted"), _FakeEmbedder())

    db.upsert.assert_not_called()  # unchanged files do no per-file writes
    # Scope is reconciled once per collection (bulk), not per file.
    db.reconcile.assert_awaited_once_with("c", "restricted")
    assert report.skipped_unchanged == 1
    assert report.rescoped == 2
    assert report.ingested == 0


async def test_ingest_collection_ingests_new_file(tmp_path: Path, db) -> None:
    (tmp_path / "note.txt").write_text("hello there")

    report = await ingest_collection(_collection(tmp_path), _FakeEmbedder())

    db.upsert.assert_awaited_once()
    assert report.ingested == 1
    assert db.upsert.await_args.kwargs["scope"] == "global"


async def test_ingest_collection_records_needs_ocr_without_embedding(tmp_path: Path, db) -> None:
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.1\n%%EOF")  # unreadable as text -> error/needs_ocr path

    report = await ingest_collection(
        _collection(tmp_path, scope="restricted", exts=(".pdf",)), _FakeEmbedder()
    )

    db.upsert.assert_awaited_once()
    assert db.upsert.await_args.kwargs["chunks"] == []
    assert db.upsert.await_args.kwargs["extraction_status"] in ("skipped_needs_ocr", "error")
    assert report.needs_ocr + report.errors == 1


async def test_ingest_skips_oversize_file(tmp_path: Path, db, monkeypatch) -> None:
    monkeypatch.setenv("OPC_DOC_MAX_FILE_MB", "0")  # any non-empty file is too large
    doc = tmp_path / "big.txt"
    doc.write_text("this content exceeds the zero-byte ceiling")

    report = await ingest_collection(_collection(tmp_path), _FakeEmbedder())

    db.upsert.assert_not_called()  # never embedded / written as a chunked doc
    # Fail closed: any stale row for the now-oversize file is purged.
    db.del_one.assert_awaited_once_with("c", str(doc))
    assert report.skipped_too_large == 1
    assert report.ingested == 0


async def test_ingest_caps_chunks_per_file(tmp_path: Path, db, monkeypatch) -> None:
    monkeypatch.setenv("OPC_DOC_MAX_CHUNKS", "1")
    doc = tmp_path / "dense.txt"
    doc.write_text("x" * 1500)  # >1200 chars -> 2 chunks at default chunk size

    report = await ingest_collection(_collection(tmp_path), _FakeEmbedder())

    db.upsert.assert_awaited_once()
    assert db.upsert.await_args.kwargs["extraction_status"] == "error"
    assert db.upsert.await_args.kwargs["chunks"] == []
    assert report.errors == 1
    assert report.ingested == 0


async def test_ingest_isolates_per_file_failure(tmp_path: Path, db) -> None:
    (tmp_path / "a_boom.txt").write_text("boom")  # sorts first; embedder raises
    (tmp_path / "b_good.txt").write_text("good")  # must still be ingested

    report = await ingest_collection(_collection(tmp_path), _BoomEmbedder())

    assert report.errors == 1
    assert report.ingested == 1
    statuses = [c.kwargs["extraction_status"] for c in db.upsert.await_args_list]
    assert "error" in statuses
    assert "extracted" in statuses


async def test_ingest_purges_deleted_files(tmp_path: Path, db) -> None:
    doc = tmp_path / "keep.txt"
    doc.write_text("still here")
    db.purge.return_value = 3

    report = await ingest_collection(_collection(tmp_path), _FakeEmbedder())

    db.purge.assert_awaited_once_with("c", {str(doc)})
    assert report.purged == 3
