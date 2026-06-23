"""Schema/migration parity for the archival_memory.archived_at lifecycle column.

Issue #63 Phase 2b Step 3 (stale-archive), review W-4 (PARITY).

A stale learning has NO survivor, so ``superseded_by`` cannot mark it retired.
``archived_at`` is a first-class lifecycle column: NULL = active, a timestamp =
archived (stale-retired). Both the bootstrap schema (``docker/init-schema.sql``,
fresh DB) and the column-provisioning migration (existing DBs) must declare an
IDENTICAL ``archived_at`` column on ``archival_memory`` (same type + nullability)
and a partial active index mirroring ``idx_archival_active``.

Static assertions against the SQL files (no live DB) — same approach as
test_project_lower_index.py / test_pgcrypto_extension.py.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_SCHEMA = REPO_ROOT / "docker" / "init-schema.sql"
ADD_ARCHIVED_AT_MIGRATION = REPO_ROOT / "scripts" / "migrations" / "add_archived_at.sql"

_COMMENT_RE = re.compile(r"--[^\n]*")

# init-schema.sql declares the column inline in CREATE TABLE archival_memory:
#   archived_at TIMESTAMPTZ
_INIT_COLUMN_RE = re.compile(
    r"^\s*archived_at\s+(?P<type>TIMESTAMPTZ)\s*(?P<extra>[^,\n]*),?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# the migration adds it via ALTER TABLE ... ADD COLUMN IF NOT EXISTS:
#   ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ
_MIGRATION_COLUMN_RE = re.compile(
    r"ALTER\s+TABLE\s+archival_memory\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+"
    r"archived_at\s+(?P<type>TIMESTAMPTZ)\s*(?P<extra>[^;\n]*)",
    re.IGNORECASE,
)

# a partial index on the active (NOT archived) rows, mirroring idx_archival_active.
_ACTIVE_ARCHIVED_INDEX_RE = re.compile(
    r"CREATE\s+INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?\w+\s+"
    r"ON\s+archival_memory\s*\(\s*archived_at\s*\)\s*WHERE\s+archived_at\s+IS\s+NULL",
    re.IGNORECASE,
)


def _strip_comments(sql: str) -> str:
    return _COMMENT_RE.sub("", sql)


def _archived_at_section(path: Path) -> str:
    """The relevant `archived_at` table-scope text only (skip the sessions column)."""
    return _strip_comments(path.read_text())


def _init_archived_at_def() -> tuple[str, str] | None:
    """The (type, extra) of the archived_at column declared inside CREATE TABLE
    archival_memory in init-schema.sql. Restricts the search to the text from the
    archival_memory CREATE TABLE up to the next CREATE TABLE, so the unrelated
    sessions.archived_at column never matches."""
    sql = _strip_comments(INIT_SCHEMA.read_text())
    start = re.search(
        r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+archival_memory\b",
        sql,
        re.IGNORECASE,
    )
    if not start:
        return None
    rest = sql[start.end():]
    nxt = re.search(r"CREATE\s+TABLE\b", rest, re.IGNORECASE)
    body = rest[: nxt.start()] if nxt else rest
    col = _INIT_COLUMN_RE.search(body)
    if not col:
        return None
    return col.group("type").upper(), col.group("extra").strip().upper()


def _migration_archived_at_def() -> tuple[str, str] | None:
    sql = _strip_comments(ADD_ARCHIVED_AT_MIGRATION.read_text())
    m = _MIGRATION_COLUMN_RE.search(sql)
    if not m:
        return None
    return m.group("type").upper(), m.group("extra").strip().upper()


def test_init_schema_has_archived_at_column() -> None:
    assert _init_archived_at_def() is not None, (
        "docker/init-schema.sql must declare archived_at TIMESTAMPTZ on the "
        "archival_memory table (the stale lifecycle column)."
    )


def test_migration_has_archived_at_column() -> None:
    assert _migration_archived_at_def() is not None, (
        "scripts/migrations/add_archived_at.sql must ADD COLUMN IF NOT EXISTS "
        "archived_at TIMESTAMPTZ on archival_memory."
    )


def test_archived_at_definition_parity() -> None:
    """W-4 PARITY: init-schema and migration must agree on type AND nullability."""
    init_def = _init_archived_at_def()
    mig_def = _migration_archived_at_def()
    assert init_def is not None and mig_def is not None
    assert init_def[0] == mig_def[0] == "TIMESTAMPTZ", (
        f"archived_at type mismatch: init={init_def[0]!r} migration={mig_def[0]!r}"
    )
    # Nullability: neither side may add NOT NULL — a NOT NULL default-less column
    # would fail on a populated table, and active rows must stay NULL.
    assert "NOT NULL" not in init_def[1], "archived_at must be nullable in init-schema"
    assert "NOT NULL" not in mig_def[1], "archived_at must be nullable in the migration"
    # And neither carries a DEFAULT (active rows are NULL by absence, not by default).
    assert "DEFAULT" not in init_def[1], "archived_at must not carry a DEFAULT in init-schema"
    assert "DEFAULT" not in mig_def[1], "archived_at must not carry a DEFAULT in the migration"


def test_init_schema_has_active_archived_index() -> None:
    sql = _strip_comments(INIT_SCHEMA.read_text())
    assert _ACTIVE_ARCHIVED_INDEX_RE.search(sql), (
        "docker/init-schema.sql must have a partial index on archived_at WHERE "
        "archived_at IS NULL, mirroring idx_archival_active."
    )


def test_migration_has_active_archived_index() -> None:
    sql = _strip_comments(ADD_ARCHIVED_AT_MIGRATION.read_text())
    assert _ACTIVE_ARCHIVED_INDEX_RE.search(sql), (
        "the migration must create the partial archived_at IS NULL index too."
    )


def test_migration_column_and_index_are_idempotent() -> None:
    sql = _strip_comments(ADD_ARCHIVED_AT_MIGRATION.read_text())
    assert re.search(
        r"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+archived_at", sql, re.IGNORECASE
    ), "the migration must use ADD COLUMN IF NOT EXISTS for idempotent reruns."
    assert re.search(
        r"CREATE\s+INDEX\s+(?:CONCURRENTLY\s+)?IF\s+NOT\s+EXISTS", sql, re.IGNORECASE
    ), "the migration index must use IF NOT EXISTS for idempotent reruns."


def test_migration_index_is_concurrent() -> None:
    """aegis MEDIUM-1: a CREATE INDEX on a populated table blocks writes; the
    migration runs against existing DBs, so its archived_at index must be
    CONCURRENTLY (mirrors add_project_column.sql)."""
    sql = _strip_comments(ADD_ARCHIVED_AT_MIGRATION.read_text())
    archival_index = re.search(
        r"CREATE\s+INDEX\s+(CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?\w+\s+"
        r"ON\s+archival_memory\s*\(\s*archived_at\s*\)",
        sql,
        re.IGNORECASE,
    )
    assert archival_index is not None, "expected an archived_at index in the migration"
    assert archival_index.group(1) and archival_index.group(1).strip(), (
        "the migration's archived_at index must use CREATE INDEX CONCURRENTLY "
        "(populated table; avoid a write-blocking SHARE lock)."
    )


def test_init_schema_archived_index_stays_non_concurrent() -> None:
    """init-schema runs on a fresh DB; CONCURRENTLY there cannot run in a txn block."""
    sql = _strip_comments(INIT_SCHEMA.read_text())
    assert "CONCURRENTLY" not in sql.upper(), (
        "docker/init-schema.sql must not use CREATE INDEX CONCURRENTLY."
    )
