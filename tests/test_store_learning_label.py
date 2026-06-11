"""Tests for issue #151: store_learning_v2 derives embedding_model from
provider.model_label (not the buggy .model probe) and threads it to store().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.core.store_learning import store_learning_v2


@pytest.mark.asyncio
async def test_label_comes_from_model_label_property() -> None:
    mock_memory = AsyncMock()
    mock_memory.search_vector_global = AsyncMock(return_value=[])
    mock_memory.store = AsyncMock(return_value="new-uuid")
    mock_memory.close = AsyncMock()

    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    # Provider exposes model_label; legacy .model probe must not be used.
    mock_embedder.model_label = "voyage-code-3"

    with (
        patch(
            "scripts.core.store_learning.create_memory_service",
            new_callable=AsyncMock,
            return_value=mock_memory,
        ),
        patch(
            "scripts.core.store_learning.EmbeddingService",
            return_value=mock_embedder,
        ),
    ):
        result = await store_learning_v2(session_id="s1", content="Test learning")

    assert result["success"] is True
    # store() receives the explicit embedding_model.
    _, kwargs = mock_memory.store.call_args
    assert kwargs.get("embedding_model") == "voyage-code-3"
    # And metadata carries the same label.
    assert kwargs["metadata"]["embedding_model"] == "voyage-code-3"


@pytest.mark.asyncio
async def test_local_label_is_bge_not_model_name() -> None:
    """Regression: LocalEmbeddingProvider exposed .model_name not .model, so
    the old probe wrote None and relied on the column default. model_label
    canonicalizes to 'bge'."""
    mock_memory = AsyncMock()
    mock_memory.search_vector_global = AsyncMock(return_value=[])
    mock_memory.store = AsyncMock(return_value="new-uuid")
    mock_memory.close = AsyncMock()

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    mock_embedder.aclose = AsyncMock()
    mock_embedder.model_label = "bge"

    with (
        patch(
            "scripts.core.store_learning.create_memory_service",
            new_callable=AsyncMock,
            return_value=mock_memory,
        ),
        patch(
            "scripts.core.store_learning.EmbeddingService",
            return_value=mock_embedder,
        ),
    ):
        await store_learning_v2(session_id="s1", content="Local store")

    _, kwargs = mock_memory.store.call_args
    assert kwargs.get("embedding_model") == "bge"
