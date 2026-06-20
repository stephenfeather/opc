"""Scoped query orchestration: embed the question, retrieve, map to results.

The DB layer enforces the scope gate. The default query is global-only;
restricted collections (medical/legal records) are reachable ONLY by naming
the collection explicitly. There is deliberately no "all scopes" switch — a
single blanket flag surfacing every restricted record at once is too wide a
leak surface, so the only way to reach restricted content is `collection=...`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from scripts.core.documents.db import query_chunks

DEFAULT_LIMIT = 8


class QueryEmbedder(Protocol):
    """Minimal embedding interface for queries."""

    async def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class QueryResult:
    """One retrieved chunk, ready to present or hand to a model as context."""

    content: str
    file_path: str
    page_number: int | None
    collection: str
    similarity: float


async def query_documents(
    query_text: str,
    embedder: QueryEmbedder,
    *,
    collection: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[QueryResult]:
    """Embed query_text and retrieve the nearest chunks under the scope gate.

    Scoping:
        collection set  -> that collection only (the ONLY way to reach a
                           restricted collection).
        collection None -> global collections only (the safe default).
    """
    if not query_text.strip():
        return []

    embedding = await embedder.embed(query_text)
    rows = await query_chunks(embedding, scope="global", collection=collection, limit=limit)
    return [
        QueryResult(
            content=row["content"],
            file_path=row["file_path"],
            page_number=row["page_number"],
            collection=row["collection_name"],
            similarity=row["similarity"],
        )
        for row in rows
    ]
