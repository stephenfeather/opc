"""Tests for the scoped query orchestration. DB + embedder mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from scripts.core.documents.query import QueryResult, query_documents


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.02] * 1024


async def test_query_documents_default_scope_is_global() -> None:
    with patch(
        "scripts.core.documents.query.query_chunks",
        new=AsyncMock(return_value=[]),
    ) as mock_q:
        await query_documents("who flagged meds", _FakeEmbedder())
    assert mock_q.await_args.kwargs["scope"] == "global"
    assert mock_q.await_args.kwargs["collection"] is None


async def test_query_documents_collection_targeting_forwarded() -> None:
    with patch(
        "scripts.core.documents.query.query_chunks",
        new=AsyncMock(return_value=[]),
    ) as mock_q:
        await query_documents("who flagged meds", _FakeEmbedder(), collection="caleb-records")
    assert mock_q.await_args.kwargs["collection"] == "caleb-records"


async def test_query_documents_scope_all_forwarded() -> None:
    with patch(
        "scripts.core.documents.query.query_chunks",
        new=AsyncMock(return_value=[]),
    ) as mock_q:
        await query_documents("x", _FakeEmbedder(), scope_all=True)
    # scope_all means: do not gate by scope. Sentinel 'all' is passed through.
    assert mock_q.await_args.kwargs["scope"] == "all"


async def test_query_documents_maps_rows_to_results() -> None:
    rows = [
        {
            "content": "RN Bethune documented home meds",
            "page_number": 47,
            "collection_name": "caleb-records",
            "scope": "restricted",
            "file_path": "/docs/march.pdf",
            "similarity": 0.91,
        }
    ]
    with patch(
        "scripts.core.documents.query.query_chunks",
        new=AsyncMock(return_value=rows),
    ):
        results = await query_documents("meds", _FakeEmbedder(), collection="caleb-records")
    assert results == [
        QueryResult(
            content="RN Bethune documented home meds",
            file_path="/docs/march.pdf",
            page_number=47,
            collection="caleb-records",
            similarity=0.91,
        )
    ]


async def test_query_documents_empty_query_returns_empty() -> None:
    with patch("scripts.core.documents.query.query_chunks", new=AsyncMock()) as mock_q:
        results = await query_documents("   ", _FakeEmbedder())
    assert results == []
    mock_q.assert_not_called()
