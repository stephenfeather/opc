"""Regression test for issue #123.

kg_entities and kg_edges declare ``id UUID PRIMARY KEY DEFAULT gen_random_uuid()``,
which requires the pgcrypto extension. Both the bootstrap schema and the
knowledge-graph migration must enable pgcrypto so a clean Postgres install
does not fail with ``function gen_random_uuid() does not exist`` on the
first KG insert.

This is a static assertion against the SQL files (no live DB required).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_SCHEMA = REPO_ROOT / "docker" / "init-schema.sql"
KG_MIGRATION = REPO_ROOT / "scripts" / "migrations" / "add_knowledge_graph.sql"

_PGCRYPTO_RE = re.compile(
    r"CREATE\s+EXTENSION\s+IF\s+NOT\s+EXISTS\s+pgcrypto\s*;",
    re.IGNORECASE,
)


def _read(path: Path) -> str:
    assert path.exists(), f"Expected SQL file at {path}"
    return path.read_text(encoding="utf-8")


def test_init_schema_enables_pgcrypto() -> None:
    """docker/init-schema.sql must enable pgcrypto for gen_random_uuid()."""
    sql = _read(INIT_SCHEMA)
    assert _PGCRYPTO_RE.search(sql), (
        "docker/init-schema.sql is missing 'CREATE EXTENSION IF NOT EXISTS pgcrypto;'. "
        "kg_entities/kg_edges use gen_random_uuid() which requires pgcrypto."
    )


def test_kg_migration_enables_pgcrypto() -> None:
    """scripts/migrations/add_knowledge_graph.sql must enable pgcrypto."""
    sql = _read(KG_MIGRATION)
    assert _PGCRYPTO_RE.search(sql), (
        "scripts/migrations/add_knowledge_graph.sql is missing "
        "'CREATE EXTENSION IF NOT EXISTS pgcrypto;'. kg_entities/kg_edges use "
        "gen_random_uuid() which requires pgcrypto."
    )


def test_pgcrypto_enabled_before_kg_entities_in_migration() -> None:
    """The pgcrypto CREATE EXTENSION must precede the first kg_entities CREATE TABLE."""
    sql = _read(KG_MIGRATION)
    pgcrypto_match = _PGCRYPTO_RE.search(sql)
    kg_match = re.search(r"CREATE\s+TABLE[^;]*kg_entities", sql, re.IGNORECASE)
    assert pgcrypto_match is not None, "pgcrypto extension not enabled in KG migration"
    assert kg_match is not None, "kg_entities CREATE TABLE not found in KG migration"
    assert pgcrypto_match.start() < kg_match.start(), (
        "pgcrypto extension must be enabled before kg_entities is created so "
        "gen_random_uuid() is available for the column default."
    )
