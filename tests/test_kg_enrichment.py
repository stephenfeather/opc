"""Tests for knowledge-graph enrichment in recall_learnings.

Phase 3 Commit A: fetch + enrichment plumbing. These tests drive the
implementation of _fetch_kg_rows, build_kg_lookup, apply_kg_enrichment,
and enrich_with_kg_context in scripts/core/recall_learnings.py.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# build_kg_lookup -- pure
# ---------------------------------------------------------------------------


def test_build_kg_lookup_empty_rows_returns_empty_dict():
    from scripts.core.recall_learnings import build_kg_lookup

    assert build_kg_lookup([]) == {}


def test_build_kg_lookup_groups_entities_and_edges_by_memory_id():
    from scripts.core.recall_learnings import build_kg_lookup

    mid = uuid.uuid4()
    rows = [
        {
            "id": mid,
            "kg_entities": [
                {"id": "e1", "name": "pytest", "type": "tool", "mention_count": 5},
                {"id": "e2", "name": "asyncpg", "type": "library", "mention_count": 3},
            ],
            "kg_edges": [
                {"source": "pytest", "target": "asyncpg", "relation": "used_with", "weight": 2.0},
            ],
        }
    ]

    lookup = build_kg_lookup(rows)

    assert str(mid) in lookup
    entry = lookup[str(mid)]
    assert len(entry["entities"]) == 2
    assert entry["entities"][0]["name"] == "pytest"
    assert len(entry["edges"]) == 1
    assert entry["edges"][0]["relation"] == "used_with"


def test_build_kg_lookup_caps_edges_at_max_per_memory():
    from scripts.core.recall_learnings import KG_MAX_EDGES_PER_MEMORY, build_kg_lookup

    mid = uuid.uuid4()
    edges = [
        {"source": f"s{i}", "target": f"t{i}", "relation": "uses", "weight": float(i)}
        for i in range(KG_MAX_EDGES_PER_MEMORY + 10)
    ]
    rows = [{"id": mid, "kg_entities": [], "kg_edges": edges}]

    lookup = build_kg_lookup(rows)

    capped = lookup[str(mid)]["edges"]
    assert len(capped) == KG_MAX_EDGES_PER_MEMORY
    # top-N by weight -> highest weights retained (sorted descending)
    assert capped[0]["weight"] == float(KG_MAX_EDGES_PER_MEMORY + 10 - 1)
    assert capped[-1]["weight"] >= 10.0  # 60 - 50 discarded = min kept weight 10


# ---------------------------------------------------------------------------
# apply_kg_enrichment -- pure
# ---------------------------------------------------------------------------


def test_apply_kg_enrichment_adds_kg_context_to_matching_results():
    from scripts.core.recall_learnings import apply_kg_enrichment

    mid = str(uuid.uuid4())
    results = [{"id": mid, "content": "x"}]
    lookup = {
        mid: {
            "entities": [{"name": "pytest", "type": "tool"}],
            "edges": [],
        }
    }

    enriched = apply_kg_enrichment(results, lookup)

    assert "kg_context" in enriched[0]
    assert enriched[0]["kg_context"]["entities"][0]["name"] == "pytest"


def test_apply_kg_enrichment_omits_kg_context_for_non_matches():
    from scripts.core.recall_learnings import apply_kg_enrichment

    results = [{"id": "no-match", "content": "x"}]
    lookup = {str(uuid.uuid4()): {"entities": [], "edges": []}}

    enriched = apply_kg_enrichment(results, lookup)

    assert "kg_context" not in enriched[0]


def test_apply_kg_enrichment_does_not_mutate_input():
    from scripts.core.recall_learnings import apply_kg_enrichment

    mid = str(uuid.uuid4())
    original = [{"id": mid, "content": "x"}]
    lookup = {mid: {"entities": [{"name": "a", "type": "tool"}], "edges": []}}

    _ = apply_kg_enrichment(original, lookup)

    assert "kg_context" not in original[0]


# ---------------------------------------------------------------------------
# enrich_with_kg_context -- orchestrator (integration with get_backend + I/O)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_with_kg_context_empty_results_returns_empty():
    from scripts.core.recall_learnings import enrich_with_kg_context

    out = await enrich_with_kg_context([])

    assert out == []


@pytest.mark.asyncio
async def test_enrich_with_kg_context_sqlite_returns_unchanged():
    from scripts.core.recall_learnings import enrich_with_kg_context

    results = [{"id": str(uuid.uuid4()), "content": "x"}]
    with patch("scripts.core.recall_learnings.get_backend", return_value="sqlite"):
        out = await enrich_with_kg_context(results)

    assert out == results
    assert "kg_context" not in out[0]


@pytest.mark.asyncio
async def test_enrich_with_kg_context_no_ids_returns_unchanged():
    from scripts.core.recall_learnings import enrich_with_kg_context

    results = [{"content": "no id"}]
    with patch("scripts.core.recall_learnings.get_backend", return_value="postgres"):
        out = await enrich_with_kg_context(results)

    assert out == results


@pytest.mark.asyncio
async def test_enrich_with_kg_context_connection_error_non_fatal():
    from scripts.core.recall_learnings import enrich_with_kg_context

    results = [{"id": str(uuid.uuid4()), "content": "x"}]

    with (
        patch("scripts.core.recall_learnings.get_backend", return_value="postgres"),
        patch(
            "scripts.core.recall_learnings._fetch_kg_rows",
            new_callable=AsyncMock,
            side_effect=ConnectionError("KG table unavailable"),
        ),
    ):
        out = await enrich_with_kg_context(results)

    assert out == results
    assert "kg_context" not in out[0]


@pytest.mark.asyncio
async def test_enrich_with_kg_context_populates_matching_results():
    from scripts.core.recall_learnings import enrich_with_kg_context

    mid = str(uuid.uuid4())
    results = [{"id": mid, "content": "pytest + asyncpg"}]
    rows = [
        {
            "id": uuid.UUID(mid),
            "kg_entities": [{"id": "e1", "name": "pytest", "type": "tool", "mention_count": 1}],
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
        out = await enrich_with_kg_context(results)

    assert "kg_context" in out[0]
    assert out[0]["kg_context"]["entities"][0]["name"] == "pytest"
