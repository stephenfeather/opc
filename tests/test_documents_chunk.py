"""Tests for document chunking."""

from __future__ import annotations

from scripts.core.documents.chunk import Chunk, chunk_pages
from scripts.core.documents.extract import ExtractedPage


def test_short_page_yields_one_chunk() -> None:
    pages = [ExtractedPage(page_number=1, text="short text")]
    chunks = chunk_pages(pages, max_chars=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == Chunk(chunk_index=0, content="short text", page_number=1)


def test_long_page_splits_with_overlap() -> None:
    text = "a" * 2500
    pages = [ExtractedPage(page_number=1, text=text)]
    chunks = chunk_pages(pages, max_chars=1000, overlap=200)
    assert len(chunks) == 3
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert all(c.page_number == 1 for c in chunks)
    # Overlap: chunk 1 starts 200 chars before chunk 0 ends.
    assert len(chunks[0].content) == 1000
    assert chunks[1].content[:200] == chunks[0].content[-200:]


def test_chunk_indexes_are_continuous_across_pages() -> None:
    pages = [
        ExtractedPage(page_number=1, text="page one text"),
        ExtractedPage(page_number=2, text="page two text"),
    ]
    chunks = chunk_pages(pages, max_chars=1000, overlap=100)
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.page_number for c in chunks] == [1, 2]


def test_empty_pages_yield_no_chunks() -> None:
    assert chunk_pages([], max_chars=1000, overlap=100) == []


def test_whitespace_only_page_is_skipped() -> None:
    pages = [ExtractedPage(page_number=1, text="   \n  ")]
    assert chunk_pages(pages, max_chars=1000, overlap=100) == []
