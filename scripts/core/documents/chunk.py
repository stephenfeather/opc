"""Split extracted pages into overlapping, embeddable chunks. Pure module.

Chunking is fixed-width over characters with a configurable overlap so a fact
spanning a boundary still lands whole in at least one chunk. Page numbers are
preserved so a retrieval result can cite "page 47".
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.core.documents.extract import ExtractedPage

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 200


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit. chunk_index is continuous across all pages of a doc."""

    chunk_index: int
    content: str
    page_number: int


def _split_one(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split a single string into <= max_chars windows stepping by max_chars-overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    step = max_chars - overlap
    windows = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            break
        start += step
    return windows


def chunk_pages(
    pages: list[ExtractedPage],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk every page, assigning continuous chunk_index values across the document.

    Raises:
        ValueError: if overlap >= max_chars (would never advance).
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for window in _split_one(page.text, max_chars, overlap):
            chunks.append(Chunk(chunk_index=index, content=window, page_number=page.page_number))
            index += 1
    return chunks
