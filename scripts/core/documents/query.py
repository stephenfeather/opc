"""Scoped query orchestration: embed the question, retrieve, map to results.

The DB layer enforces the scope gate. When scope_all is True the sentinel
'all' is forwarded and db.query_chunks is responsible for treating it as
"no scope filter" — see the WHERE-clause note in db.query_chunks. For v1,
db.query_chunks only matches a literal scope value, so 'all' is wired here
but the DB layer is extended in Step 4 to honour it.
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
    scope_all: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> list[QueryResult]:
    """Embed query_text and retrieve the nearest chunks under the chosen scope.

    Scoping:
        collection set      -> that collection only (overrides scope_all).
        scope_all=True       -> every collection, any scope.
        neither              -> global collections only (the safe default).
    """
    if not query_text.strip():
        return []

    embedding = await embedder.embed(query_text)
    scope = "all" if scope_all else "global"
    rows = await query_chunks(embedding, scope=scope, collection=collection, limit=limit)
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
