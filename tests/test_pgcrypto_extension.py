"""Regression tests for issue #123 + PR #126 reviewer feedback.

kg_entities and kg_edges declare ``id UUID PRIMARY KEY DEFAULT gen_random_uuid()``,
which requires the pgcrypto extension. Both the bootstrap schema and the
knowledge-graph migration must enable pgcrypto so a clean Postgres install
does not fail with ``function gen_random_uuid() does not exist`` on the
first KG insert.

These are static assertions against the SQL files (no live DB required).

PR #126 review hardened two regex helpers:

1. ``_pgcrypto_present`` strips ``--`` line comments before scanning so a
   commented-out ``CREATE EXTENSION`` line does not satisfy the assertion
   (CodeRabbit finding).
2. ``_KG_ENTITIES_TABLE_RE`` is anchored on a word boundary at the table
   name so it does not match ``CREATE TABLE kg_edges (... REFERENCES
   kg_entities (id) ...)`` and similar foreign-key references (Copilot
   finding).
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

# Anchored on \b at the table name so we do not match a foreign-key reference
# from another table's CREATE statement.
_KG_ENTITIES_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?kg_entities\b",
    re.IGNORECASE,
)


def _read(path: Path) -> str:
    assert path.exists(), f"Expected SQL file at {path}"
    return path.read_text(encoding="utf-8")


def _strip_line_comments(sql: str) -> str:
    """Drop ``--``-prefixed line comments (whitespace-tolerant).

    This is intentionally line-oriented and not string-literal-aware: the SQL
    files under test contain no in-string ``--`` sequences, so a simple
    line-by-line filter is sufficient and avoids pulling in a SQL parser for
    the regression suite.
    """
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def _pgcrypto_present(sql: str) -> bool:
    """True iff pgcrypto is enabled in *executable* SQL (comments ignored)."""
    return _PGCRYPTO_RE.search(_strip_line_comments(sql)) is not None


# ---------------------------------------------------------------------------
# Static assertions on the real schema files
# ---------------------------------------------------------------------------


def test_init_schema_enables_pgcrypto() -> None:
    """docker/init-schema.sql must enable pgcrypto for gen_random_uuid()."""
    sql = _read(INIT_SCHEMA)
    assert _pgcrypto_present(sql), (
        "docker/init-schema.sql is missing 'CREATE EXTENSION IF NOT EXISTS pgcrypto;' "
        "outside of a -- comment. gen_random_uuid() defaults are used across "
        "multiple tables in this schema."
    )


def test_kg_migration_enables_pgcrypto() -> None:
    """scripts/migrations/add_knowledge_graph.sql must enable pgcrypto."""
    sql = _read(KG_MIGRATION)
    assert _pgcrypto_present(sql), (
        "scripts/migrations/add_knowledge_graph.sql is missing "
        "'CREATE EXTENSION IF NOT EXISTS pgcrypto;' outside of a -- comment. "
        "kg_entities/kg_edges use gen_random_uuid() which requires pgcrypto."
    )


def test_pgcrypto_enabled_before_kg_entities_in_migration() -> None:
    """The pgcrypto CREATE EXTENSION must precede the kg_entities CREATE TABLE."""
    sql = _strip_line_comments(_read(KG_MIGRATION))
    pgcrypto_match = _PGCRYPTO_RE.search(sql)
    kg_match = _KG_ENTITIES_TABLE_RE.search(sql)
    assert pgcrypto_match is not None, "pgcrypto extension not enabled in KG migration"
    assert kg_match is not None, "kg_entities CREATE TABLE not found in KG migration"
    assert pgcrypto_match.start() < kg_match.start(), (
        "pgcrypto extension must be enabled before kg_entities is created so "
        "gen_random_uuid() is available for the column default."
    )


# ---------------------------------------------------------------------------
# Regex hardening tests (PR #126 review feedback)
# ---------------------------------------------------------------------------


def test_pgcrypto_present_ignores_commented_extension_line() -> None:
    """A commented-out CREATE EXTENSION line must not satisfy the assertion.

    Regression for CodeRabbit's finding on PR #126: the original regex matched
    anywhere in the file, including in ``--`` line comments. With the comment
    stripper in place the function correctly reports the extension as absent.
    """
    sql_with_only_comment = (
        "-- CREATE EXTENSION IF NOT EXISTS pgcrypto;\n"
        "CREATE TABLE foo (id UUID PRIMARY KEY);\n"
    )
    # Sanity: the raw regex still matches the comment line, demonstrating that
    # the comment stripper is the actual fix.
    assert _PGCRYPTO_RE.search(sql_with_only_comment) is not None
    # The hardened helper must return False because the extension is only
    # present in a comment.
    assert _pgcrypto_present(sql_with_only_comment) is False

    sql_with_real_extension = (
        "-- CREATE EXTENSION IF NOT EXISTS pgcrypto;\n"
        "CREATE EXTENSION IF NOT EXISTS pgcrypto;\n"
        "CREATE TABLE foo (id UUID PRIMARY KEY);\n"
    )
    assert _pgcrypto_present(sql_with_real_extension) is True


def test_kg_entities_regex_does_not_match_fk_reference() -> None:
    """The kg_entities CREATE TABLE regex must not match an FK reference.

    Regression for Copilot's finding on PR #126: the original pattern
    ``CREATE\\s+TABLE[^;]*kg_entities`` would match ``CREATE TABLE kg_edges
    ( ... REFERENCES kg_entities(id) )`` because ``[^;]*`` happily consumed
    the body up to the FK clause. The tightened pattern anchors on a word
    boundary right after the table name.
    """
    sql = (
        "CREATE TABLE kg_edges (\n"
        "    source_id UUID REFERENCES kg_entities(id),\n"
        "    target_id UUID REFERENCES kg_entities(id)\n"
        ");\n"
        "CREATE TABLE IF NOT EXISTS kg_entities (\n"
        "    id UUID PRIMARY KEY\n"
        ");\n"
    )

    # The OLD permissive regex would have matched the kg_edges CREATE TABLE
    # because of the FK reference. Spell out the old pattern here as a
    # regression assertion so future readers can see the failure mode.
    old_pattern = re.compile(r"CREATE\s+TABLE[^;]*kg_entities", re.IGNORECASE)
    old_match = old_pattern.search(sql)
    assert old_match is not None
    # The old match starts at the kg_edges CREATE TABLE — i.e. wrong table.
    assert sql[old_match.start():].lstrip().startswith("CREATE TABLE kg_edges")

    # The NEW anchored regex matches only the actual kg_entities CREATE TABLE.
    new_match = _KG_ENTITIES_TABLE_RE.search(sql)
    assert new_match is not None
    assert "kg_edges" not in sql[new_match.start():new_match.end()]
    assert "kg_entities" in sql[new_match.start():new_match.end()]
