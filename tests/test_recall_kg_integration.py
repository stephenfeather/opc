"""Phase 3 Commit C: integration tests for query-side KG flow.

Exercises make_recall_context (entity extraction), enrich_with_kg_context
(fetch plumbing from Commit A), and rerank()+kg_overlap (signal from
Commit B) end-to-end. DB boundary is mocked via _fetch_kg_rows so these
run without a live Postgres.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from scripts.core.recall_learnings import (
    enrich_with_kg_context,
    make_recall_context,
)
from scripts.core.reranker import RerankerConfig, rerank


def _mk_result(rid: str, similarity: float, content: str = "") -> dict:
    return {
        "id": rid,
        "session_id": "s",
        "content": content,
        "metadata": {"type": "session_learning"},
        "similarity": similarity,
    }


@pytest.mark.asyncio
async def test_postgres_query_populates_kg_context_and_query_entities():
    """End-to-end: query with known entity -> context.query_entities populated,
    enrichment attaches kg_context, rerank boosts matching result."""
    mid_match = str(uuid.uuid4())
    mid_miss = str(uuid.uuid4())
    results = [
        _mk_result(mid_miss, similarity=0.5, content="generic stuff"),
        _mk_result(mid_match, similarity=0.5, content="uses pytest heavily"),
    ]

    rows = [
        {
            "id": uuid.UUID(mid_match),
            "kg_entities": [
                {"id": "e1", "name": "pytest", "type": "tool", "mention_count": 10}
            ],
            "kg_edges": [],
        }
    ]

    with (
        patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
        patch(
            "scripts.core.recall_learnings._fetch_kg_rows",
            new_callable=AsyncMock,
            return_value=rows,
        ),
    ):
        ctx = make_recall_context(
            project=None, tags=None, retrieval_mode="vector",
            query="how do I use pytest in async tests",
        )
        enriched = await enrich_with_kg_context(results)

    assert ctx.query_entities is not None
    assert any(e["name"] == "pytest" for e in ctx.query_entities)

    matching = next(r for r in enriched if r["id"] == mid_match)
    missing = next(r for r in enriched if r["id"] == mid_miss)
    assert "kg_context" in matching
    assert "kg_context" not in missing

    # With kg_weight > 0, matching outranks missing even at equal similarity.
    config = RerankerConfig(
        project_weight=0.0, recency_weight=0.0, confidence_weight=0.0,
        recall_weight=0.0, type_affinity_weight=0.0, tag_overlap_weight=0.0,
        pattern_weight=0.0, kg_weight=0.3,
    )
    ranked = rerank(enriched, ctx, config=config, k=2)
    assert ranked[0]["id"] == mid_match


@pytest.mark.asyncio
async def test_kg_weight_zero_produces_no_boost():
    """With kg_weight=0, ordering reflects retrieval only (no KG effect)."""
    mid_match = str(uuid.uuid4())
    mid_miss = str(uuid.uuid4())
    results = [
        _mk_result(mid_miss, similarity=0.9),    # higher retrieval
        _mk_result(mid_match, similarity=0.3),   # lower retrieval
    ]

    rows = [
        {
            "id": uuid.UUID(mid_match),
            "kg_entities": [
                {"id": "e1", "name": "pytest", "type": "tool", "mention_count": 1}
            ],
            "kg_edges": [],
        }
    ]

    with (
        patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
        patch(
            "scripts.core.recall_learnings._fetch_kg_rows",
            new_callable=AsyncMock,
            return_value=rows,
        ),
    ):
        ctx = make_recall_context(
            project=None, tags=None, retrieval_mode="vector", query="pytest tricks",
        )
        enriched = await enrich_with_kg_context(results)

    config = RerankerConfig(
        project_weight=0.0, recency_weight=0.0, confidence_weight=0.0,
        recall_weight=0.0, type_affinity_weight=0.0, tag_overlap_weight=0.0,
        pattern_weight=0.0, kg_weight=0.0,
    )
    ranked = rerank(enriched, ctx, config=config, k=2)
    # Retrieval-only: higher similarity wins.
    assert ranked[0]["id"] == mid_miss


@pytest.mark.asyncio
async def test_make_recall_context_caps_query_for_extraction():
    """Aegis LOW-1 fix: make_recall_context must not feed unbounded queries
    to kg_extractor (regex CPU blowup). The cap is applied before extraction
    so the extractor only ever sees up to _KG_QUERY_EXTRACTION_MAX_CHARS."""
    from unittest.mock import MagicMock
    from scripts.core.recall_learnings import (
        _KG_QUERY_EXTRACTION_MAX_CHARS,
        make_recall_context,
    )

    giant_query = "x" * (_KG_QUERY_EXTRACTION_MAX_CHARS * 4) + " pytest"

    fake_extract = MagicMock(return_value=[])
    with (
        patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
        patch(
            "scripts.core.kg_extractor.extract_entities",
            fake_extract,
        ),
    ):
        make_recall_context(
            project=None, tags=None, retrieval_mode="vector", query=giant_query,
        )

    fake_extract.assert_called_once()
    call_arg = fake_extract.call_args[0][0]
    assert len(call_arg) == _KG_QUERY_EXTRACTION_MAX_CHARS
    # The cap truncates from the end, so the tail ("pytest") is dropped.
    assert call_arg == "x" * _KG_QUERY_EXTRACTION_MAX_CHARS


@pytest.mark.asyncio
async def test_sqlite_backend_skips_kg_path_end_to_end():
    """sqlite backend: no query entities extracted, no kg_context attached."""
    mid = str(uuid.uuid4())
    results = [_mk_result(mid, similarity=0.5, content="pytest mention")]

    with patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"):
        ctx = make_recall_context(
            project=None, tags=None, retrieval_mode="sqlite",
            query="anything involving pytest",
        )
        enriched = await enrich_with_kg_context(results)

    assert ctx.query_entities is None
    assert "kg_context" not in enriched[0]
