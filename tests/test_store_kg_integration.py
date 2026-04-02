"""Tests for KG extraction integration in store_learning_v2."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.core.store_learning import store_learning_v2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_memory_service(memory_id: str | None = None):
    """Create a mock memory service that returns a given memory_id."""
    if memory_id is None:
        memory_id = str(uuid.uuid4())
    svc = AsyncMock()
    svc.store.return_value = memory_id
    svc.close.return_value = None
    # search_vector_global returns empty (no duplicates)
    svc.search_vector_global.return_value = []
    return svc, memory_id


def _mock_embedding():
    """Return a mock EmbeddingService that produces a dummy vector."""
    embedder = AsyncMock()
    embedder.embed.return_value = [0.1] * 1024
    embedder._provider = MagicMock()
    embedder._provider.model = "test-model"
    return embedder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_learning_v2_calls_kg_extractor():
    """KG extraction runs after successful store and populates result."""
    svc, mid = _mock_memory_service()
    embedder = _mock_embedding()

    kg_result = {"entities": 3, "edges": 2, "mentions": 3}

    with (
        patch("scripts.core.store_learning.create_memory_service", return_value=svc),
        patch("scripts.core.store_learning.get_default_backend", return_value="postgres"),
        patch("scripts.core.store_learning.EmbeddingService", return_value=embedder),
        patch(
            "scripts.core.kg_extractor.store_entities_and_edges",
            new_callable=AsyncMock,
            return_value=kg_result,
        ) as mock_store_kg,
        patch(
            "scripts.core.kg_extractor.extract_entities",
        ) as mock_extract_ents,
        patch(
            "scripts.core.kg_extractor.extract_relations",
        ) as mock_extract_rels,
        patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"}),
    ):
        # extract_entities returns some entities so KG path runs
        mock_extract_ents.return_value = [MagicMock(name="pytest", entity_type="tool")]
        mock_extract_rels.return_value = []

        result = await store_learning_v2(
            session_id="test-session",
            content="Using pytest for testing with asyncpg",
            learning_type="WORKING_SOLUTION",
        )

        assert result["success"] is True
        assert result["kg_stats"] == kg_result
        mock_extract_ents.assert_called_once()
        mock_store_kg.assert_called_once_with(mid, mock_extract_ents.return_value, [])


@pytest.mark.asyncio
async def test_store_learning_v2_kg_failure_is_nonfatal():
    """KG extraction failure does not break the store path."""
    svc, mid = _mock_memory_service()
    embedder = _mock_embedding()

    with (
        patch("scripts.core.store_learning.create_memory_service", return_value=svc),
        patch("scripts.core.store_learning.get_default_backend", return_value="postgres"),
        patch("scripts.core.store_learning.EmbeddingService", return_value=embedder),
        patch(
            "scripts.core.kg_extractor.extract_entities",
            side_effect=RuntimeError("KG boom"),
        ),
        patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"}),
    ):
        result = await store_learning_v2(
            session_id="test-session",
            content="Using pytest for testing",
            learning_type="WORKING_SOLUTION",
        )

        assert result["success"] is True
        assert result["memory_id"] == mid
        assert "kg_stats" not in result  # KG failed, no kg key


@pytest.mark.asyncio
async def test_store_learning_v2_no_entities_skips_kg():
    """When no entities extracted, KG storage is skipped entirely."""
    svc, mid = _mock_memory_service()
    embedder = _mock_embedding()

    with (
        patch("scripts.core.store_learning.create_memory_service", return_value=svc),
        patch("scripts.core.store_learning.get_default_backend", return_value="postgres"),
        patch("scripts.core.store_learning.EmbeddingService", return_value=embedder),
        patch(
            "scripts.core.kg_extractor.extract_entities",
            return_value=[],  # no entities
        ),
        patch(
            "scripts.core.kg_extractor.store_entities_and_edges",
            new_callable=AsyncMock,
        ) as mock_store_kg,
        patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"}),
    ):
        result = await store_learning_v2(
            session_id="test-session",
            content="just some text with no technical entities",
            learning_type="WORKING_SOLUTION",
        )

        assert result["success"] is True
        assert "kg_stats" not in result
        mock_store_kg.assert_not_called()


@pytest.mark.asyncio
async def test_store_learning_v2_kg_skipped_for_sqlite():
    """KG extraction only runs on postgres backend."""
    svc, mid = _mock_memory_service()
    embedder = _mock_embedding()

    with (
        patch("scripts.core.store_learning.create_memory_service", return_value=svc),
        patch("scripts.core.store_learning.get_default_backend", return_value="sqlite"),
        patch("scripts.core.store_learning.EmbeddingService", return_value=embedder),
        patch(
            "scripts.core.kg_extractor.extract_entities",
        ) as mock_extract,
        patch.dict("os.environ", {}, clear=True),
    ):
        result = await store_learning_v2(
            session_id="test-session",
            content="Using pytest for testing",
            learning_type="WORKING_SOLUTION",
        )

        assert result["success"] is True
        assert "kg_stats" not in result
        mock_extract.assert_not_called()


@pytest.mark.asyncio
async def test_store_learning_v2_dedup_backfills_kg():
    """When content is a duplicate, KG is backfilled for the existing memory."""
    svc = AsyncMock()
    svc.store.return_value = ""  # empty = dedup
    svc.close.return_value = None
    svc.search_vector_global.return_value = []
    embedder = _mock_embedding()

    existing_id = str(uuid.uuid4())
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = existing_id

    # pool.acquire() is a sync call returning an async context manager
    acm = AsyncMock()
    acm.__aenter__.return_value = mock_conn
    acm.__aexit__.return_value = False
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = acm

    kg_result = {"entities": 2, "edges": 1, "mentions": 2}

    with (
        patch("scripts.core.store_learning.create_memory_service", return_value=svc),
        patch("scripts.core.store_learning.get_default_backend", return_value="postgres"),
        patch("scripts.core.store_learning.EmbeddingService", return_value=embedder),
        patch("scripts.core.db.postgres_pool.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        patch(
            "scripts.core.kg_extractor.extract_entities",
            return_value=[MagicMock(name="pytest", entity_type="tool")],
        ),
        patch("scripts.core.kg_extractor.extract_relations", return_value=[]),
        patch(
            "scripts.core.kg_extractor.store_entities_and_edges",
            new_callable=AsyncMock,
            return_value=kg_result,
        ) as mock_store_kg,
        patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"}),
    ):
        result = await store_learning_v2(
            session_id="test-session",
            content="duplicate content with pytest",
            learning_type="WORKING_SOLUTION",
        )

        assert result["skipped"] is True
        mock_store_kg.assert_called_once()
        # Verify it used the existing memory_id, not a new one
        call_args = mock_store_kg.call_args
        assert call_args[0][0] == existing_id


@pytest.mark.asyncio
async def test_store_learning_v2_dedup_kg_backfill_failure_nonfatal():
    """KG backfill failure on dedup does not break the dedup path."""
    svc = AsyncMock()
    svc.store.return_value = ""  # empty = dedup
    svc.close.return_value = None
    svc.search_vector_global.return_value = []
    embedder = _mock_embedding()

    with (
        patch("scripts.core.store_learning.create_memory_service", return_value=svc),
        patch("scripts.core.store_learning.get_default_backend", return_value="postgres"),
        patch("scripts.core.store_learning.EmbeddingService", return_value=embedder),
        patch(
            "scripts.core.db.postgres_pool.get_pool",
            side_effect=RuntimeError("pool boom"),
        ),
        patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"}),
    ):
        result = await store_learning_v2(
            session_id="test-session",
            content="duplicate content",
            learning_type="WORKING_SOLUTION",
        )

        assert result["skipped"] is True
        assert result["success"] is True
