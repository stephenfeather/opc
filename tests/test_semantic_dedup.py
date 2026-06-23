"""Tests for semantic deduplication in store_learning_v2.

Validates that:
1. search_vector_global searches across ALL sessions (no session_id filter)
2. store_learning_v2 uses global search for dedup
3. Near-duplicates (>= 0.92 similarity) are rejected
4. Sufficiently different learnings (< 0.92) are stored
5. Fallback to session-scoped search when global search is unavailable
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.store_learning import DEDUP_THRESHOLD, store_learning_v2

# Patch targets: imports are now at module level in store_learning,
# so we patch at the consuming module.
STORE_MOD = "scripts.core.store_learning"


@pytest.fixture
def mock_embedding():
    """A fake 1024-dim embedding vector."""
    return [0.1] * 1024


@pytest.fixture
def mock_memory_service():
    """Mock MemoryServicePG with search_vector_global."""
    memory = AsyncMock()
    memory.search_vector_global = AsyncMock(return_value=[])
    memory.search_vector = AsyncMock(return_value=[])
    memory.store = AsyncMock(return_value="new-uuid-123")
    memory.close = AsyncMock()
    return memory


@pytest.fixture
def mock_embedder(mock_embedding):
    """Mock EmbeddingService."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=mock_embedding)
    embedder._provider = MagicMock()
    embedder._provider.model = "test-model"
    return embedder


def _patches(mock_memory_service, mock_embedder):
    """Return context managers that patch the lazy imports."""
    mock_create = AsyncMock(return_value=mock_memory_service)
    mock_get_backend = MagicMock(return_value="postgres")
    mock_embed_cls = MagicMock(return_value=mock_embedder)

    return (
        patch(f"{STORE_MOD}.create_memory_service", mock_create),
        patch(f"{STORE_MOD}.get_default_backend", mock_get_backend),
        patch(f"{STORE_MOD}.EmbeddingService", mock_embed_cls),
    )


def test_dedup_threshold_from_config():
    """Threshold should match opc.toml [dedup] threshold (default 0.85)."""
    from scripts.core.config import get_config
    assert DEDUP_THRESHOLD == get_config().dedup.threshold


@pytest.mark.asyncio
async def test_global_dedup_rejects_cross_session_duplicate(
    mock_memory_service, mock_embedder
):
    """A learning that matches an existing one from a DIFFERENT session should be rejected."""
    mock_memory_service.search_vector_global.return_value = [
        {
            "id": "existing-uuid",
            "session_id": "other-session-999",
            "content": "Nearly identical learning",
            "similarity": 0.95,
        }
    ]

    p1, p2, p3 = _patches(mock_memory_service, mock_embedder)
    with p1, p2, p3:
        result = await store_learning_v2(
            session_id="my-session",
            content="Nearly identical learning content",
        )

    assert result["success"] is True
    assert result["skipped"] is True
    assert "duplicate" in result["reason"]
    assert "other-session-999" in result["reason"]
    assert result["existing_id"] == "existing-uuid"
    mock_memory_service.store.assert_not_called()


@pytest.mark.asyncio
async def test_global_dedup_allows_sufficiently_different(
    mock_memory_service, mock_embedder
):
    """A learning below the threshold should be stored."""
    mock_memory_service.search_vector_global.return_value = [
        {
            "id": "existing-uuid",
            "session_id": "other-session",
            "content": "Somewhat related but different",
            "similarity": 0.80,
        }
    ]

    p1, p2, p3 = _patches(mock_memory_service, mock_embedder)
    with p1, p2, p3:
        result = await store_learning_v2(
            session_id="my-session",
            content="A genuinely new learning",
        )

    assert result["success"] is True
    assert "skipped" not in result
    assert result["memory_id"] == "new-uuid-123"
    mock_memory_service.store.assert_called_once()


@pytest.mark.asyncio
async def test_global_dedup_rejects_at_exact_threshold(
    mock_memory_service, mock_embedder
):
    """Similarity exactly at 0.92 should be rejected (>= condition)."""
    mock_memory_service.search_vector_global.return_value = [
        {
            "id": "boundary-uuid",
            "session_id": "other-session",
            "content": "Boundary match",
            "similarity": 0.92,
        }
    ]

    p1, p2, p3 = _patches(mock_memory_service, mock_embedder)
    with p1, p2, p3:
        result = await store_learning_v2(
            session_id="my-session",
            content="Boundary test learning",
        )

    assert result["success"] is True
    assert result["skipped"] is True
    assert "duplicate" in result["reason"]
    mock_memory_service.store.assert_not_called()


@pytest.mark.asyncio
async def test_global_dedup_allows_just_below_threshold(
    mock_memory_service, mock_embedder
):
    """Similarity just below threshold should be stored."""
    mock_memory_service.search_vector_global.return_value = [
        {
            "id": "near-miss-uuid",
            "session_id": "other-session",
            "content": "Almost a match",
            "similarity": DEDUP_THRESHOLD - 0.001,
        }
    ]

    p1, p2, p3 = _patches(mock_memory_service, mock_embedder)
    with p1, p2, p3:
        result = await store_learning_v2(
            session_id="my-session",
            content="Just below threshold learning",
        )

    assert result["success"] is True
    assert "skipped" not in result
    assert result["memory_id"] == "new-uuid-123"
    mock_memory_service.store.assert_called_once()


@pytest.mark.asyncio
async def test_global_dedup_allows_when_no_matches(
    mock_memory_service, mock_embedder
):
    """When no existing memories match at all, the learning should be stored."""
    mock_memory_service.search_vector_global.return_value = []

    p1, p2, p3 = _patches(mock_memory_service, mock_embedder)
    with p1, p2, p3:
        result = await store_learning_v2(
            session_id="my-session",
            content="Brand new learning",
        )

    assert result["success"] is True
    assert result["memory_id"] == "new-uuid-123"


@pytest.mark.asyncio
async def test_fallback_to_session_scoped_search(mock_embedder):
    """When search_vector_global is missing, fall back to search_vector."""
    memory = AsyncMock()
    del memory.search_vector_global
    memory.search_vector = AsyncMock(return_value=[])
    memory.store = AsyncMock(return_value="fallback-uuid")
    memory.close = AsyncMock()

    mock_create = AsyncMock(return_value=memory)
    mock_get_backend = MagicMock(return_value="sqlite")
    mock_embed_cls = MagicMock(return_value=mock_embedder)

    with (
        patch(f"{STORE_MOD}.create_memory_service", mock_create),
        patch(f"{STORE_MOD}.get_default_backend", mock_get_backend),
        patch(f"{STORE_MOD}.EmbeddingService", mock_embed_cls),
    ):
        result = await store_learning_v2(
            session_id="my-session",
            content="Learning on SQLite backend",
        )

    assert result["success"] is True
    assert result["memory_id"] == "fallback-uuid"
    memory.search_vector.assert_called_once()


@pytest.mark.asyncio
async def test_dedup_error_does_not_block_storage(
    mock_memory_service, mock_embedder
):
    """If global dedup search errors out, the learning should still be stored."""
    mock_memory_service.search_vector_global.side_effect = RuntimeError("DB error")

    p1, p2, p3 = _patches(mock_memory_service, mock_embedder)
    with p1, p2, p3:
        result = await store_learning_v2(
            session_id="my-session",
            content="Learning despite DB errors",
        )

    assert result["success"] is True
    assert result["memory_id"] == "new-uuid-123"


@pytest.mark.asyncio
async def test_supersede_target_does_not_block_replacement(
    mock_memory_service, mock_embedder
):
    """A supersede whose only near-dup is the row it replaces must store (issue #235).

    The corrected content is expected to resemble the row being superseded, so
    the dedup gate must not count the supersede target as a blocking duplicate.
    """
    mock_memory_service.search_vector_global.return_value = [
        {
            "id": "bad-row-uuid",
            "session_id": "old-session",
            "content": "The wrong learning being corrected",
            "similarity": 0.97,
        }
    ]

    p1, p2, p3 = _patches(mock_memory_service, mock_embedder)
    with p1, p2, p3, patch(
        f"{STORE_MOD}.detect_backend", MagicMock(return_value="postgres")
    ):
        result = await store_learning_v2(
            session_id="my-session",
            content="The corrected learning",
            supersedes="bad-row-uuid",
        )

    assert result["success"] is True
    assert "skipped" not in result
    assert result["memory_id"] == "new-uuid-123"
    assert result.get("superseded") == "bad-row-uuid"
    mock_memory_service.store.assert_called_once()
    # The supersede target id is passed through to the store layer.
    _, store_kwargs = mock_memory_service.store.call_args
    assert store_kwargs.get("supersedes") == "bad-row-uuid"
    # The dedup probe fetched an extra neighbor so a *different* dup can still surface.
    _, probe_kwargs = mock_memory_service.search_vector_global.call_args
    assert probe_kwargs.get("limit") == 2


@pytest.mark.asyncio
async def test_supersede_still_rejects_a_different_duplicate(
    mock_memory_service, mock_embedder
):
    """Superseding row X must still reject when a DIFFERENT row Y is a near-dup."""
    mock_memory_service.search_vector_global.return_value = [
        {
            "id": "other-dup-uuid",
            "session_id": "other-session",
            "content": "An unrelated near-duplicate",
            "similarity": 0.96,
        },
        {
            "id": "bad-row-uuid",
            "session_id": "old-session",
            "content": "The wrong learning being corrected",
            "similarity": 0.95,
        },
    ]

    p1, p2, p3 = _patches(mock_memory_service, mock_embedder)
    with p1, p2, p3, patch(
        f"{STORE_MOD}.detect_backend", MagicMock(return_value="postgres")
    ):
        result = await store_learning_v2(
            session_id="my-session",
            content="The corrected learning",
            supersedes="bad-row-uuid",
        )

    assert result["success"] is True
    assert result["skipped"] is True
    assert "duplicate" in result["reason"]
    assert result["existing_id"] == "other-dup-uuid"
    mock_memory_service.store.assert_not_called()


@pytest.mark.asyncio
async def test_supersede_on_sqlite_still_blocks_duplicate(mock_embedder):
    """On a non-postgres backend, supersede is NOT persisted, so the dedup gate
    must still reject a matching row instead of silently writing a duplicate.

    Guards against the bypass where excluding the supersede target on a backend
    that never marks the old row replaced would store an orphaned duplicate.
    """
    memory = AsyncMock()
    memory.search_vector_global = AsyncMock(
        return_value=[
            {
                "id": "bad-row-uuid",
                "session_id": "old-session",
                "content": "The wrong learning being corrected",
                "similarity": 0.97,
            }
        ]
    )
    memory.search_vector = AsyncMock(return_value=[])
    memory.store = AsyncMock(return_value="should-not-be-used")
    memory.close = AsyncMock()

    mock_create = AsyncMock(return_value=memory)
    mock_embed_cls = MagicMock(return_value=mock_embedder)

    with (
        patch(f"{STORE_MOD}.create_memory_service", mock_create),
        patch(f"{STORE_MOD}.detect_backend", MagicMock(return_value="sqlite")),
        patch(f"{STORE_MOD}.EmbeddingService", mock_embed_cls),
    ):
        result = await store_learning_v2(
            session_id="my-session",
            content="The corrected learning",
            supersedes="bad-row-uuid",
        )

    assert result["success"] is True
    assert result["skipped"] is True
    assert "duplicate" in result["reason"]
    assert result["existing_id"] == "bad-row-uuid"
    memory.store.assert_not_called()
