"""Tests for MemoryServicePG — I/O layer that delegates to pure functions.

Tests the async service class methods by mocking the database connection
and verifying they correctly delegate to the pure query functions.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_capability_caches():
    """Isolate the class-level capability probes between tests (issue #63 Phase 2b).

    Both ``_has_superseded_column`` and ``_has_archived_at_column`` are process-wide
    caches; leaking one test's value into the next makes filter-wiring assertions
    order-dependent. Reset to None before and after each test.
    """
    from scripts.core.db.memory_service_pg import MemoryServicePG

    MemoryServicePG._has_superseded_column = None
    MemoryServicePG._has_archived_at_column = None
    yield
    MemoryServicePG._has_superseded_column = None
    MemoryServicePG._has_archived_at_column = None


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


class TestCheckSupersededColumn:
    """Tests for schema migration compatibility check."""

    async def test_returns_true_when_column_exists(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = None  # reset cache
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=1)
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc._check_superseded_column()

        assert result is True
        assert MemoryServicePG._has_superseded_column is True

    async def test_returns_false_when_column_missing(self):
        import asyncpg

        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = None  # reset cache
        conn = AsyncMock()
        conn.fetchval = AsyncMock(
            side_effect=asyncpg.UndefinedColumnError("column does not exist")
        )
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc._check_superseded_column()

        assert result is False
        assert MemoryServicePG._has_superseded_column is False

    async def test_uses_cached_result_on_subsequent_calls(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = True  # pre-cached
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
        ) as mock_conn:
            result = await svc._check_superseded_column()

        assert result is True
        mock_conn.assert_not_called()  # no DB hit when cached

    async def test_cache_reset_allows_recheck(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = False  # was false
        MemoryServicePG._has_superseded_column = None  # reset

        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=1)
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc._check_superseded_column()

        assert result is True


class TestSupersededColumnWiring:
    """Tests that search methods wire _check_superseded_column into query builders."""

    async def test_search_text_passes_active_filter_false_when_no_column(self):
        import asyncpg

        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = None
        conn = AsyncMock()
        conn.fetchval = AsyncMock(
            side_effect=asyncpg.UndefinedColumnError("no column")
        )
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.build_text_search_sql",
            wraps=__import__(
                "scripts.core.db.memory_service_queries", fromlist=["build_text_search_sql"]
            ).build_text_search_sql,
        ) as mock_builder:
            await svc.search_text("test")

        _, kwargs = mock_builder.call_args
        assert kwargs.get("include_active_filter") is False

    async def test_search_text_passes_active_filter_true_when_column_exists(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = None
        check_conn = AsyncMock()
        check_conn.fetchval = AsyncMock(return_value=1)
        search_conn = AsyncMock()
        search_conn.fetch = AsyncMock(return_value=[])
        call_count = [0]

        def conn_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeConnection(check_conn)
            return FakeConnection(search_conn)

        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            side_effect=conn_factory,
        ), patch(
            "scripts.core.db.memory_service_pg.build_text_search_sql",
            wraps=__import__(
                "scripts.core.db.memory_service_queries", fromlist=["build_text_search_sql"]
            ).build_text_search_sql,
        ) as mock_builder:
            await svc.search_text("test")

        _, kwargs = mock_builder.call_args
        assert kwargs.get("include_active_filter") is True

    async def test_search_vector_global_omits_filter_when_no_column(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = False  # pre-cached
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            AsyncMock(),
        ):
            await svc.search_vector_global([0.1] * 1024)

        sql = conn.fetch.call_args[0][0]
        assert "superseded_by" not in sql

    async def test_search_with_tags_omits_filter_when_no_column(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = False  # pre-cached
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            await svc.search_with_tags("test")

        sql = conn.fetch.call_args[0][0]
        assert "superseded_by" not in sql

    async def test_search_with_tags_includes_filter_when_column_exists(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = True  # pre-cached
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            await svc.search_with_tags("test")

        sql = conn.fetch.call_args[0][0]
        assert "superseded_by IS NULL" in sql

    async def test_search_vector_global_includes_filter_when_column_exists(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = True  # pre-cached
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            AsyncMock(),
        ):
            await svc.search_vector_global([0.1] * 1024)

        sql = conn.fetch.call_args[0][0]
        assert "superseded_by IS NULL" in sql


class TestCheckArchivedAtColumn:
    """Issue #63 Phase 2b (SF-1): a SECOND capability probe for archived_at.

    The active-row filter gains `AND archived_at IS NULL`, but a DB that has
    superseded_by yet has NOT run the add_archived_at migration would pass the
    superseded probe and then crash on every recall. The archived_at probe must
    independently gate the new clause.
    """

    async def test_returns_true_when_column_exists(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_archived_at_column = None  # reset cache
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=1)
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc._check_archived_at_column()

        assert result is True
        assert MemoryServicePG._has_archived_at_column is True
        # The probe must check archived_at, not superseded_by.
        sql = conn.fetchval.call_args[0][0]
        assert "archived_at" in sql

    async def test_returns_false_when_column_missing(self):
        import asyncpg

        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_archived_at_column = None  # reset cache
        conn = AsyncMock()
        conn.fetchval = AsyncMock(
            side_effect=asyncpg.UndefinedColumnError("column does not exist")
        )
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            result = await svc._check_archived_at_column()

        assert result is False
        assert MemoryServicePG._has_archived_at_column is False

    async def test_uses_cached_result_on_subsequent_calls(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_archived_at_column = True  # pre-cached
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
        ) as mock_conn:
            result = await svc._check_archived_at_column()

        assert result is True
        mock_conn.assert_not_called()

    async def test_archived_probe_is_independent_of_superseded_probe(self):
        """The two probes cache independently — superseded True must NOT imply
        archived True (the exact pre-migration crash SF-1 guards)."""
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = True  # pre-cached True
        MemoryServicePG._has_archived_at_column = False  # pre-cached False
        svc = MemoryServicePG(session_id="s1")
        with patch("scripts.core.db.memory_service_pg.get_connection") as mock_conn:
            assert await svc._check_superseded_column() is True
            assert await svc._check_archived_at_column() is False
            mock_conn.assert_not_called()


class TestArchivedAtFilterWiring:
    """The `archived_at IS NULL` clause is injected ONLY when the archived probe
    passes; on a pre-migration DB (archived absent) recall does NOT crash and the
    clause is omitted."""

    async def test_search_text_no_archived_clause_when_column_absent(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = True  # superseded present
        MemoryServicePG._has_archived_at_column = False  # archived ABSENT (pre-migration)
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            # Must NOT raise — the archived clause is omitted when the column is absent.
            await svc.search_text("test")

        sql = conn.fetch.call_args[0][0]
        assert "superseded_by IS NULL" in sql  # superseded still filtered
        assert "archived_at" not in sql  # but NOT archived (column absent)

    async def test_search_text_adds_archived_clause_when_column_present(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = True
        MemoryServicePG._has_archived_at_column = True
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            await svc.search_text("test")

        sql = conn.fetch.call_args[0][0]
        assert "archived_at IS NULL" in sql

    async def test_search_with_tags_adds_archived_clause_when_present(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = True
        MemoryServicePG._has_archived_at_column = True
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ):
            await svc.search_with_tags("test")

        sql = conn.fetch.call_args[0][0]
        assert "archived_at IS NULL" in sql

    async def test_search_vector_global_adds_archived_clause_when_present(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        MemoryServicePG._has_superseded_column = True
        MemoryServicePG._has_archived_at_column = True
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_connection",
            return_value=FakeConnection(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.init_pgvector",
            AsyncMock(),
        ):
            await svc.search_vector_global([0.1] * 1024)

        sql = conn.fetch.call_args[0][0]
        assert "archived_at IS NULL" in sql


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

    async def test_store_supersede_uses_helper_with_reason_store(self):
        """D1/D2: store() routes through supersede_row with reason='store'."""
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.supersede_row",
            new=AsyncMock(return_value=1),
        ) as mock_helper:
            await svc.store("new fact", supersedes="old-uuid")

        mock_helper.assert_awaited_once()
        kwargs = mock_helper.await_args.kwargs
        assert kwargs["loser_id"] == "old-uuid"
        assert kwargs["reason"] == "store"
        # keeper is the freshly generated memory id (non-empty)
        assert kwargs["keeper_id"]

    async def test_store_supersede_zero_rows_warns(self, caplog):
        """store() owns the 0-row policy: best-effort warning, no raise."""
        import logging

        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        svc = MemoryServicePG(session_id="s1")

        with patch(
            "scripts.core.db.memory_service_pg.get_transaction",
            return_value=FakeTransaction(conn),
        ), patch(
            "scripts.core.db.memory_service_pg.supersede_row",
            new=AsyncMock(return_value=0),
        ), caplog.at_level(
            logging.WARNING, logger="scripts.core.db.memory_service_pg"
        ):
            await svc.store("new fact", supersedes="old-uuid")

        assert any("0 row" in r.getMessage() for r in caplog.records)


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


# ==================== Config Wiring ====================


class TestConfigWiring:
    """Tests that search defaults are read from current config at call time."""

    @staticmethod
    def config(*, limit: int = 37, rrf_k: int = 91, max_archival: int = 23):
        from scripts.core.config.models import DatabaseConfig, OPCConfig, RecallConfig

        return OPCConfig(
            recall=RecallConfig(default_search_limit=limit, rrf_k=rrf_k),
            database=DatabaseConfig(max_archival_context=max_archival),
        )

    async def test_search_text_uses_current_config_default_limit_after_import(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with (
            patch(
                "scripts.core.db.memory_service_pg._get_config",
                return_value=self.config(limit=37),
            ),
            patch.object(
                MemoryServicePG,
                "_check_superseded_column",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "scripts.core.db.memory_service_pg.get_connection",
                return_value=FakeConnection(conn),
            ),
            patch(
                "scripts.core.db.memory_service_pg.build_text_search_sql",
                return_value=("SELECT 1", []),
            ) as build_sql,
        ):
            await svc.search_text("needle")

        assert build_sql.call_args.args[3] == 37

    def test_module_import_does_not_load_config(self):
        sys.modules.pop("scripts.core.db.memory_service_pg", None)

        with patch("scripts.core.config.get_config") as get_config:
            module = importlib.import_module("scripts.core.db.memory_service_pg")

        get_config.assert_not_called()
        assert module.MemoryServicePG is not None

    async def test_search_text_explicit_limit_overrides_config_default(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with (
            patch(
                "scripts.core.db.memory_service_pg._get_config",
                return_value=self.config(limit=37),
            ),
            patch.object(
                MemoryServicePG,
                "_check_superseded_column",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "scripts.core.db.memory_service_pg.get_connection",
                return_value=FakeConnection(conn),
            ),
            patch(
                "scripts.core.db.memory_service_pg.build_text_search_sql",
                return_value=("SELECT 1", []),
            ) as build_sql,
        ):
            await svc.search_text("needle", limit=4)

        assert build_sql.call_args.args[3] == 4

    async def test_search_vector_uses_current_config_default_limit_after_import(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with (
            patch(
                "scripts.core.db.memory_service_pg._get_config",
                return_value=self.config(limit=37),
            ),
            patch.object(
                MemoryServicePG,
                "_check_superseded_column",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "scripts.core.db.memory_service_pg.get_connection",
                return_value=FakeConnection(conn),
            ),
            patch("scripts.core.db.memory_service_pg.init_pgvector", new=AsyncMock()),
            patch(
                "scripts.core.db.memory_service_pg.build_vector_search_sql",
                return_value=("SELECT 1", []),
            ) as build_sql,
        ):
            await svc.search_vector([0.1])

        assert build_sql.call_args.args[3] == 37

    async def test_search_alias_delegates_none_limit_to_search_text(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        svc = MemoryServicePG(session_id="s1")

        with patch.object(svc, "search_text", new=AsyncMock(return_value=[])) as search_text:
            await svc.search("needle")

        search_text.assert_awaited_once_with("needle", None)

    async def test_search_vector_threshold_uses_current_config_default_limit(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with (
            patch(
                "scripts.core.db.memory_service_pg._get_config",
                return_value=self.config(limit=37),
            ),
            patch.object(
                MemoryServicePG,
                "_check_superseded_column",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "scripts.core.db.memory_service_pg.get_connection",
                return_value=FakeConnection(conn),
            ),
            patch("scripts.core.db.memory_service_pg.init_pgvector", new=AsyncMock()),
        ):
            await svc.search_vector_with_threshold([0.1])

        assert conn.fetch.call_args.args[-1] == 37

    async def test_search_vector_filter_uses_current_config_default_limit(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with (
            patch(
                "scripts.core.db.memory_service_pg._get_config",
                return_value=self.config(limit=37),
            ),
            patch.object(
                MemoryServicePG,
                "_check_superseded_column",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "scripts.core.db.memory_service_pg.get_connection",
                return_value=FakeConnection(conn),
            ),
            patch("scripts.core.db.memory_service_pg.init_pgvector", new=AsyncMock()),
        ):
            await svc.search_vector_with_filter([0.1], {"type": "session_learning"})

        assert conn.fetch.call_args.args[-1] == 37

    async def test_search_hybrid_uses_current_config_default_limit(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with (
            patch(
                "scripts.core.db.memory_service_pg._get_config",
                return_value=self.config(limit=37),
            ),
            patch.object(
                MemoryServicePG,
                "_check_superseded_column",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "scripts.core.db.memory_service_pg.get_connection",
                return_value=FakeConnection(conn),
            ),
            patch("scripts.core.db.memory_service_pg.init_pgvector", new=AsyncMock()),
            patch(
                "scripts.core.db.memory_service_pg.build_hybrid_search_sql",
                return_value=("SELECT 1", []),
            ) as build_sql,
        ):
            await svc.search_hybrid("needle", [0.1])

        assert build_sql.call_args.args[4] == 37

    async def test_search_hybrid_rrf_uses_current_config_defaults(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with (
            patch(
                "scripts.core.db.memory_service_pg._get_config",
                return_value=self.config(limit=37, rrf_k=91),
            ),
            patch.object(
                MemoryServicePG,
                "_check_superseded_column",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "scripts.core.db.memory_service_pg.get_connection",
                return_value=FakeConnection(conn),
            ),
            patch("scripts.core.db.memory_service_pg.init_pgvector", new=AsyncMock()),
        ):
            await svc.search_hybrid_rrf("needle", [0.1])

        assert conn.fetch.call_args.args[-2:] == (91, 37)

    async def test_search_with_tags_uses_current_config_default_limit(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with (
            patch(
                "scripts.core.db.memory_service_pg._get_config",
                return_value=self.config(limit=37),
            ),
            patch.object(
                MemoryServicePG,
                "_check_superseded_column",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "scripts.core.db.memory_service_pg.get_connection",
                return_value=FakeConnection(conn),
            ),
        ):
            await svc.search_with_tags("needle")

        assert conn.fetch.call_args.args[-1] == 37

    async def test_to_context_uses_current_config_max_archival_after_import(self):
        from scripts.core.db.memory_service_pg import MemoryServicePG

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        svc = MemoryServicePG(session_id="s1")

        with (
            patch(
                "scripts.core.db.memory_service_pg._get_config",
                return_value=self.config(max_archival=23),
            ),
            patch.object(svc, "get_all_core", new=AsyncMock(return_value={})),
            patch.object(
                MemoryServicePG,
                "_check_superseded_column",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "scripts.core.db.memory_service_pg.get_connection",
                return_value=FakeConnection(conn),
            ),
        ):
            await svc.to_context()

        assert conn.fetch.call_args.args[-1] == 23
