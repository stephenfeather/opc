"""Regression tests for issue #139 review round 3.

The fetch-time scoped predicate is case-insensitive
(``AND LOWER(project) = $N`` — recall_backends.project_filter_clause), but the
schema only indexed the raw ``project`` column (``idx_archival_project``), so a
``LOWER(project)`` lookup could degrade to a sequential scan on a large
archival_memory table — paid on top of the global pass.

Both the bootstrap schema and the column-provisioning migration must declare a
functional index on ``LOWER(project)`` so the scoped pass stays index-backed.

These are static assertions against the SQL files (no live DB required) — the
same approach as test_pgcrypto_extension.py.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_SCHEMA = REPO_ROOT / "docker" / "init-schema.sql"
ADD_PROJECT_MIGRATION = (
    REPO_ROOT / "scripts" / "migrations" / "add_project_column.sql"
)

# Matches CREATE INDEX ... ON archival_memory (LOWER(project)) regardless of
# index name, IF NOT EXISTS, or whitespace. The functional expression is what
# matters: Postgres only uses a functional index when the query expression
# matches the indexed expression (LOWER(project)).
_LOWER_PROJECT_INDEX_RE = re.compile(
    r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s+"
    r"ON\s+archival_memory\s*\(\s*LOWER\s*\(\s*project\s*\)\s*\)",
    re.IGNORECASE,
)

_COMMENT_RE = re.compile(r"--[^\n]*")


def _strip_comments(sql: str) -> str:
    """Drop -- line comments so a commented-out index does not satisfy a test."""
    return _COMMENT_RE.sub("", sql)


def _lower_project_index_present(sql: str) -> bool:
    return bool(_LOWER_PROJECT_INDEX_RE.search(_strip_comments(sql)))


def test_init_schema_has_lower_project_index() -> None:
    """docker/init-schema.sql must index LOWER(project) for the case-insensitive
    scoped pass (issue #139 review round 3)."""
    sql = INIT_SCHEMA.read_text()
    assert _lower_project_index_present(sql), (
        "docker/init-schema.sql is missing a functional index on "
        "LOWER(project); the --project-first scoped pass uses "
        "'AND LOWER(project) = $N' and would seq-scan without it."
    )


def test_add_project_migration_has_lower_project_index() -> None:
    """The migration that provisions the project column for existing DBs must
    also create the LOWER(project) functional index (idempotent rerun)."""
    sql = ADD_PROJECT_MIGRATION.read_text()
    assert _lower_project_index_present(sql), (
        "scripts/migrations/add_project_column.sql is missing the "
        "LOWER(project) functional index; existing DBs upgraded via this "
        "migration would seq-scan the --project-first scoped pass."
    )


def test_lower_project_index_is_idempotent() -> None:
    """Both LOWER(project) index statements must use IF NOT EXISTS so reruns
    are safe."""
    idempotent_re = re.compile(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+\w+\s+"
        r"ON\s+archival_memory\s*\(\s*LOWER\s*\(\s*project\s*\)\s*\)",
        re.IGNORECASE,
    )
    for path in (INIT_SCHEMA, ADD_PROJECT_MIGRATION):
        sql = _strip_comments(path.read_text())
        assert idempotent_re.search(sql), (
            f"{path.name}: LOWER(project) index must use "
            "CREATE INDEX IF NOT EXISTS for idempotent reruns."
        )
