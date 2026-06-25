"""Canonical content hashing for archival memories.

A memory's ``content_hash`` is the deduplication key for ``archival_memory``
(unique index ``idx_archival_content_hash``). The golden-set re-anchoring in the
rerank benchmark matches results by this same hash, so the normalization MUST be
identical everywhere it is computed. Keeping the definition in one place prevents
the golden labels from silently drifting away from the stored hashes.
"""

from __future__ import annotations

import hashlib


def content_hash(content: str) -> str:
    """Return the canonical SHA-256 hex digest for a memory's content.

    Normalization: strip leading/trailing whitespace, UTF-8 encode, SHA-256.
    This matches the value stored in ``archival_memory.content_hash``.
    """
    return hashlib.sha256(content.strip().encode()).hexdigest()
