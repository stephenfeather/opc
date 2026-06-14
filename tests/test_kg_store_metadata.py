"""Tests for store_entities_and_edges persistence details (issue #122).

Covers:
- Finding #6: entity metadata is persisted as json.dumps(e.metadata), not "{}".
- Finding #1: the json module is imported and used.
- Finding #7: entity_count reflects genuinely new entities (idempotent), matching
  the documented contract instead of incrementing on every upsert.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.core.kg_extractor import (
    ExtractedEntity,
    store_entities_and_edges,
)


def _mock_pool(conn):
    """Wrap a mock connection in a pool whose acquire() yields it."""
    acm = AsyncMock()
    acm.__aenter__.return_value = conn
    acm.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = acm
    return pool


def _mock_conn_transaction(conn):
    """Give the mock conn a working `async with conn.transaction()`."""
    txn = AsyncMock()
    txn.__aenter__.return_value = None
    txn.__aexit__.return_value = False
    conn.transaction = MagicMock(return_value=txn)


@pytest.mark.asyncio
async def test_metadata_persisted_as_json(monkeypatch):
    """Entity metadata is serialized with json.dumps, not hardcoded '{}'."""
    eid = str(uuid.uuid4())
    conn = AsyncMock()
    _mock_conn_transaction(conn)
    # entity upsert: return id + created flag
    conn.fetchrow.return_value = {"id": eid, "created": True}
    # mention insert + edge insert
    conn.fetchval.return_value = True
    conn.execute.return_value = None

    pool = _mock_pool(conn)
    monkeypatch.setattr(
        "scripts.core.db.postgres_pool.get_pool",
        AsyncMock(return_value=pool),
    )

    memory_id = str(uuid.uuid4())
    entities = [
        ExtractedEntity(
            name="scripts/core",
            display_name="scripts/core",
            entity_type="file",
            metadata={"is_directory": True},
        )
    ]
    await store_entities_and_edges(memory_id, entities, [])

    # The 4th positional arg to the entity INSERT must be json.dumps(metadata).
    insert_call = conn.fetchrow.call_args
    metadata_arg = insert_call[0][4]
    assert metadata_arg == json.dumps({"is_directory": True})
    assert json.loads(metadata_arg) == {"is_directory": True}


@pytest.mark.asyncio
async def test_entity_count_only_genuinely_new(monkeypatch):
    """entity_count counts only newly-created entities (created=True), so
    re-processing the same content does not inflate the count."""
    new_eid = str(uuid.uuid4())
    existing_eid = str(uuid.uuid4())
    conn = AsyncMock()
    _mock_conn_transaction(conn)
    # First entity is newly created, second already existed (ON CONFLICT update).
    conn.fetchrow.side_effect = [
        {"id": new_eid, "created": True},
        {"id": existing_eid, "created": False},
    ]
    # No new mentions/edges for this assertion's purpose.
    conn.fetchval.return_value = None
    conn.execute.return_value = None

    pool = _mock_pool(conn)
    monkeypatch.setattr(
        "scripts.core.db.postgres_pool.get_pool",
        AsyncMock(return_value=pool),
    )

    memory_id = str(uuid.uuid4())
    entities = [
        ExtractedEntity(name="pytest", display_name="pytest", entity_type="tool"),
        ExtractedEntity(name="asyncpg", display_name="asyncpg", entity_type="library"),
    ]
    stats = await store_entities_and_edges(memory_id, entities, [])

    # Only the genuinely-new entity is counted.
    assert stats["entities"] == 1
