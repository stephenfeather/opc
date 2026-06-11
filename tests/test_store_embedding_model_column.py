"""Tests for issue #151: store() writes the embedding_model column explicitly.

Previously memory_service_pg.store() never bound embedding_model, so every
row fell back to the column default 'bge' regardless of the real embedding
space. The model filter at recall time keys on this column, so it must be
written explicitly on the embedding INSERT path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.test_memory_service_pg import FakeTransaction  # type: ignore


class TestStoreWritesEmbeddingModelColumn:
    async def test_embedding_model_bound_on_embedding_insert(self):
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
        # SQL must name the column and the label must be a bound positional arg.
        assert "embedding_model" in str(insert_call)
        assert "voyage-code-3" in [str(a) for a in insert_call.args]

    async def test_embedding_model_omitted_when_not_provided(self):
        # Backward compatible: no embedding_model -> column not in INSERT,
        # DB default ('bge') still applies for legacy callers.
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
        ):
            await svc.store("fact", embedding=[0.1] * 1024)

        insert_call = next(
            c for c in conn.execute.call_args_list
            if "INSERT INTO archival_memory" in str(c)
        )
        assert "embedding_model" not in str(insert_call)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
