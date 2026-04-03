"""Tests for MemoryServicePG — I/O layer that delegates to pure functions.

Tests the async service class methods by mocking the database connection
and verifying they correctly delegate to the pure query functions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch


class FakeTransaction:
    """Async context manager that yields a mock connection."""

    def __init__(self, conn: AsyncMock):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


class FakeConnection:
    """Async context manager that yields a mock connection."""

    def __init__(self, conn: AsyncMock):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


# ==================== Initialization ====================


class TestMemoryServicePGInit:
    """Tests for service construction."""

    def test_default_session_id(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        svc = MemoryServicePG()
        assert svc.session_id == "default"
        assert svc.agent_id is None

    def test_custom_session_and_agent(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        svc = MemoryServicePG(session_id="test-sess", agent_id="agent-1")
        assert svc.session_id == "test-sess"
        assert svc.agent_id == "agent-1"


# ==================== Core Memory ====================


class TestCoreMemory:
    """Tests for core memory CRUD operations."""

    async def test_set_core_executes_delete_then_insert(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ):
            await svc.set_core("persona", "helpful assistant")

        assert conn.execute.call_count == 2
        calls = [str(c) for c in conn.execute.call_args_list]
        assert any("DELETE" in c for c in calls)
        assert any("INSERT" in c for c in calls)

    async def test_get_core_returns_value(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"value": "test value"})
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc.get_core("persona")

        assert result == "test value"

    async def test_get_core_returns_none_when_missing(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc.get_core("missing")

        assert result is None

    async def test_list_core_keys(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"key": "a"}, {"key": "b"}])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc.list_core_keys()

        assert result == ["a", "b"]

    async def test_delete_core(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            await svc.delete_core("persona")

        assert conn.execute.called
        call_sql = str(conn.execute.call_args)
        assert "DELETE" in call_sql

    async def test_get_all_core(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[
                {"key": "persona", "value": "helper"},
                {"key": "task", "value": "coding"},
            ]
        )
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc.get_all_core()

        assert result == {"persona": "helper", "task": "coding"}


# ==================== Store ====================


class TestStore:
    """Tests for archival memory storage."""

    async def test_store_without_embedding(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ):
            result = await svc.store("test fact")

        assert result  # non-empty ID
        assert conn.execute.called

    async def test_store_with_embedding_calls_init_pgvector(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        init_mock = AsyncMock()
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            init_mock,
        ):
            await svc.store("test", embedding=[0.1] * 1024)

        init_mock.assert_called_once_with(conn)

    async def test_store_dedup_returns_empty_string(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 0")
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ):
            result = await svc.store("dup fact", content_hash="abc")

        assert result == ""

    async def test_store_with_tags(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ):
            await svc.store("tagged fact", tags=["a", "b"])

        # Should have INSERT for fact + INSERT for each tag
        tag_inserts = [
            c for c in conn.execute.call_args_list if "memory_tags" in str(c)
        ]
        assert len(tag_inserts) == 2

    async def test_store_deduplicates_tags(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ):
            await svc.store("tagged fact", tags=["a", "a", "b"])

        tag_inserts = [
            c for c in conn.execute.call_args_list if "memory_tags" in str(c)
        ]
        assert len(tag_inserts) == 2  # "a" deduplicated

    async def test_store_with_supersedes(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ):
            await svc.store("new fact", supersedes="old-uuid")

        # Should have INSERT + UPDATE for supersedes
        supersede_calls = [
            c for c in conn.execute.call_args_list if "superseded_by" in str(c)
        ]
        assert len(supersede_calls) == 1


# ==================== Search methods delegate to query builders ====================


class TestSearchText:
    """Tests for text search I/O."""

    async def test_search_text_returns_formatted_results(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "id1",
                    "content": "fact one",
                    "metadata": json.dumps({"type": "test"}),
                    "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                    "rank": 0.5,
                },
            ]
        )
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            results = await svc.search_text("test query")

        assert len(results) == 1
        assert results[0]["content"] == "fact one"
        assert results[0]["metadata"] == {"type": "test"}
        assert results[0]["rank"] == 0.5


class TestSearchVector:
    """Tests for vector search I/O."""

    async def test_search_vector_returns_formatted_results(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "id1",
                    "content": "fact",
                    "metadata": "{}",
                    "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                    "similarity": 0.92,
                },
            ]
        )
        init_mock = AsyncMock()
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            init_mock,
        ):
            results = await svc.search_vector([0.1] * 1024)

        assert len(results) == 1
        assert results[0]["similarity"] == 0.92


# ==================== Recall ====================


class TestRecall:
    """Tests for recall combining core + archival."""

    async def test_recall_with_matching_core_and_archival(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn_core = AsyncMock()
        conn_core.fetch = AsyncMock(
            return_value=[
                {"key": "persona", "value": "assistant"},
            ]
        )
        conn_search = AsyncMock()
        conn_search.fetch = AsyncMock(
            return_value=[
                {
                    "id": "id1",
                    "content": "archival fact",
                    "metadata": "{}",
                    "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                    "rank": 0.3,
                },
            ]
        )

        svc = MemoryServicePG(session_id="s1")

        # Mock both get_all_core and search_text
        with patch.object(
            svc, "get_all_core", return_value={"persona": "assistant"}
        ), patch.object(
            svc,
            "search_text",
            return_value=[{"content": "archival fact"}],
        ):
            result = await svc.recall("persona")

        assert "[Core/persona]" in result
        assert "[Archival]" in result

    async def test_recall_no_results(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        svc = MemoryServicePG(session_id="s1")

        with patch.object(svc, "get_all_core", return_value={}), patch.object(
            svc, "search_text", return_value=[]
        ):
            result = await svc.recall("nothing")

        assert result == "No relevant memories found."


# ==================== Delete ====================


class TestDeleteArchival:
    """Tests for archival memory deletion."""

    async def test_delete_archival_calls_execute(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            await svc.delete_archival("mem-123")

        assert conn.execute.called
        call_sql = str(conn.execute.call_args)
        assert "DELETE" in call_sql


# ==================== Tag Operations ====================


class TestTagOperations:
    """Tests for tag CRUD operations."""

    async def test_get_tags(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"tag": "alpha"}, {"tag": "beta"}])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc.get_tags("mem-1")

        assert result == ["alpha", "beta"]

    async def test_add_tag(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            await svc.add_tag("mem-1", "newtag")

        assert conn.execute.called

    async def test_remove_tag(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            await svc.remove_tag("mem-1", "oldtag")

        assert conn.execute.called
        call_sql = str(conn.execute.call_args)
        assert "DELETE" in call_sql

    async def test_get_all_session_tags(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[{"tag": "alpha"}, {"tag": "beta"}, {"tag": "gamma"}]
        )
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc.get_all_session_tags()

        assert result == ["alpha", "beta", "gamma"]
