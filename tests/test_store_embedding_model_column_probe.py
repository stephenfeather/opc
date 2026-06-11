"""Round 1 FIX 1b + FIX 2: store-path column probe and cross-space dedup.

FIX 1b: store() must skip the explicit embedding_model column bind when the
column is absent (pre-migration DB) so the legacy INSERT shape still works.

FIX 2: store_learning_v2's semantic dedup must filter by the same label that
will be written, so a voyage embedding is not spuriously matched against bge
rows (cross-space false-duplicate => silent data loss).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_memory_service_pg import FakeTransaction  # type: ignore


class TestStoreColumnProbe:
    async def test_legacy_insert_shape_when_column_absent(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            AsyncMock(),
        ), patch(
            "scripts.core.db.memory_service_pg."
            "embedding_model_column_available",
            AsyncMock(return_value=False),
        ):
            await svc.store(
                "fact",
                embedding=[0.1] * 1024,
                embedding_model="voyage-code-3",
            )

        insert_call = next(
            c for c in conn.execute.call_args_list
            if "INSERT INTO archival_memory" in str(c)
        )
        # Column absent -> must NOT name embedding_model in the INSERT.
        assert "embedding_model" not in str(insert_call)
        assert "voyage-code-3" not in [str(a) for a in insert_call.args]

    async def test_column_bound_when_present(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            AsyncMock(),
        ), patch(
            "scripts.core.db.memory_service_pg."
            "embedding_model_column_available",
            AsyncMock(return_value=True),
        ):
            await svc.store(
                "fact",
                embedding=[0.1] * 1024,
                embedding_model="voyage-code-3",
            )

        insert_call = next(
            c for c in conn.execute.call_args_list
            if "INSERT INTO archival_memory" in str(c)
        )
        assert "embedding_model" in str(insert_call)
        assert "voyage-code-3" in [str(a) for a in insert_call.args]


class TestSearchVectorGlobalModelFilter:
    async def test_global_dedup_binds_model_label(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeTransaction(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            AsyncMock(),
        ), patch.object(
            svc, "_check_superseded_column",
            AsyncMock(return_value=True),
        ), patch(
            "scripts.core.db.memory_service_pg."
            "embedding_model_column_available",
            AsyncMock(return_value=True),
        ):
            await svc.search_vector_global(
                [0.1] * 1024, embedding_model="voyage-code-3",
            )

        fetch_call = conn.fetch.call_args
        assert "embedding_model" in str(fetch_call)
        assert "voyage-code-3" in [str(a) for a in fetch_call.args]

    async def test_global_dedup_no_filter_when_column_absent(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeTransaction(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            AsyncMock(),
        ), patch.object(
            svc, "_check_superseded_column",
            AsyncMock(return_value=True),
        ), patch(
            "scripts.core.db.memory_service_pg."
            "embedding_model_column_available",
            AsyncMock(return_value=False),
        ):
            await svc.search_vector_global(
                [0.1] * 1024, embedding_model="voyage-code-3",
            )

        fetch_call = conn.fetch.call_args
        assert "embedding_model" not in str(fetch_call)

    async def test_global_dedup_no_filter_when_label_none(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeTransaction(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            AsyncMock(),
        ), patch.object(
            svc, "_check_superseded_column",
            AsyncMock(return_value=True),
        ):
            await svc.search_vector_global([0.1] * 1024)

        fetch_call = conn.fetch.call_args
        assert "embedding_model" not in str(fetch_call)


class TestStoreLearningDedupSameLabel:
    async def test_dedup_uses_same_label_as_write(self):
        """store_learning_v2 must pass the row's embedding_model into the
        dedup probe so a voyage row is compared only against voyage rows."""
        from scripts.core.store_learning import store_learning_v2

        mock_memory = AsyncMock()
        mock_memory.search_vector_global = AsyncMock(return_value=[])
        mock_memory.store = AsyncMock(return_value="new-uuid")
        mock_memory.close = AsyncMock()

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()
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
            await store_learning_v2(session_id="s1", content="A new learning")

        # dedup probe carried the same label that is written with the row.
        _, gkwargs = mock_memory.search_vector_global.call_args
        assert gkwargs.get("embedding_model") == "voyage-code-3"
        _, skwargs = mock_memory.store.call_args
        assert skwargs.get("embedding_model") == "voyage-code-3"

    async def test_cross_space_near_duplicate_does_not_block_store(self):
        """If the dedup probe (correctly filtered) returns no same-space
        match, the store proceeds even though a cross-space near-dup exists."""
        from scripts.core.store_learning import store_learning_v2

        mock_memory = AsyncMock()
        # Filtered probe returns nothing (the only near-dup was bge space).
        mock_memory.search_vector_global = AsyncMock(return_value=[])
        mock_memory.store = AsyncMock(return_value="new-uuid")
        mock_memory.close = AsyncMock()

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_embedder.aclose = AsyncMock()
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
            result = await store_learning_v2(
                session_id="s1", content="Voyage-space learning",
            )

        assert result["success"] is True
        assert result.get("skipped") is not True
        mock_memory.store.assert_awaited_once()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
