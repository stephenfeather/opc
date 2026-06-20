"""Tests for the unified backend / connection-URL resolver (issue #71).

The shared resolver is the single source of truth for:
  - which connection URL to use (CONTINUOUS_CLAUDE_DB_URL > DATABASE_URL > OPC_POSTGRES_URL)
  - which backend to use ("sqlite" | "postgres")

These are pure functions that take an explicit env mapping, so they are
isolated from os.environ and deterministic.
"""

from __future__ import annotations

from scripts.core.db.backend_resolution import resolve_backend, resolve_url


class TestResolveUrl:
    """resolve_url() — pure URL precedence over the three connection vars."""

    def test_prefers_continuous_claude_db_url(self) -> None:
        env = {
            "CONTINUOUS_CLAUDE_DB_URL": "postgresql://canonical/db",
            "DATABASE_URL": "postgresql://fallback/db",
            "OPC_POSTGRES_URL": "postgresql://legacy/db",
        }
        assert resolve_url(env) == "postgresql://canonical/db"

    def test_database_url_when_canonical_absent(self) -> None:
        env = {
            "DATABASE_URL": "postgresql://fallback/db",
            "OPC_POSTGRES_URL": "postgresql://legacy/db",
        }
        assert resolve_url(env) == "postgresql://fallback/db"

    def test_opc_postgres_url_as_last_resort(self) -> None:
        env = {"OPC_POSTGRES_URL": "postgresql://legacy/db"}
        assert resolve_url(env) == "postgresql://legacy/db"

    def test_none_when_no_url_set(self) -> None:
        assert resolve_url({}) is None

    def test_ignores_empty_string_values(self) -> None:
        env = {
            "CONTINUOUS_CLAUDE_DB_URL": "",
            "DATABASE_URL": "",
            "OPC_POSTGRES_URL": "postgresql://legacy/db",
        }
        assert resolve_url(env) == "postgresql://legacy/db"

    def test_all_empty_returns_none(self) -> None:
        env = {"CONTINUOUS_CLAUDE_DB_URL": "", "DATABASE_URL": "", "OPC_POSTGRES_URL": ""}
        assert resolve_url(env) is None


class TestResolveBackend:
    """resolve_backend() — explicit override > URL presence > default."""

    def test_explicit_sqlite_wins_over_url(self) -> None:
        env = {
            "AGENTICA_MEMORY_BACKEND": "sqlite",
            "DATABASE_URL": "postgresql://localhost/test",
        }
        assert resolve_backend(env) == "sqlite"

    def test_explicit_postgres(self) -> None:
        assert resolve_backend({"AGENTICA_MEMORY_BACKEND": "postgres"}) == "postgres"

    def test_explicit_is_case_insensitive(self) -> None:
        assert resolve_backend({"AGENTICA_MEMORY_BACKEND": "PostgreS"}) == "postgres"

    def test_invalid_explicit_is_ignored(self) -> None:
        # An unrecognised value falls through to URL/default resolution.
        env = {"AGENTICA_MEMORY_BACKEND": "redis", "DATABASE_URL": "postgresql://x/y"}
        assert resolve_backend(env) == "postgres"

    def test_url_presence_implies_postgres(self) -> None:
        assert resolve_backend({"DATABASE_URL": "postgresql://localhost/test"}) == "postgres"

    def test_opc_postgres_url_implies_postgres(self) -> None:
        # The split-brain fix: OPC_POSTGRES_URL alone now selects postgres.
        assert resolve_backend({"OPC_POSTGRES_URL": "postgresql://legacy/db"}) == "postgres"

    def test_continuous_claude_db_url_implies_postgres(self) -> None:
        assert resolve_backend({"CONTINUOUS_CLAUDE_DB_URL": "postgresql://x/y"}) == "postgres"

    def test_defaults_to_sqlite(self) -> None:
        assert resolve_backend({}) == "sqlite"

    def test_custom_default_when_undetermined(self) -> None:
        assert resolve_backend({}, default="postgres") == "postgres"

    def test_none_default_when_undetermined(self) -> None:
        assert resolve_backend({}, default=None) is None

    def test_empty_url_does_not_imply_postgres(self) -> None:
        assert resolve_backend({"DATABASE_URL": ""}) == "sqlite"

    def test_explicit_blank_falls_through_to_url(self) -> None:
        env = {"AGENTICA_MEMORY_BACKEND": "  ", "DATABASE_URL": "postgresql://x/y"}
        assert resolve_backend(env) == "postgres"
