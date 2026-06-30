"""Tests for the unified backend / connection-URL resolver (issue #71).

The shared resolver is the single source of truth for:
  - which connection URL to use (CONTINUOUS_CLAUDE_DB_URL > DATABASE_URL > OPC_POSTGRES_URL)
  - which backend to use ("sqlite" | "postgres")

These are pure functions that take an explicit env mapping, so they are
isolated from os.environ and deterministic.
"""

from __future__ import annotations

import pytest

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

    def test_whitespace_only_is_treated_as_unset(self) -> None:
        # Issue #214: a templated/blank value like "   " must not count as a URL,
        # else it would bypass the postgres-without-URL fail-fast in resolve_backend.
        assert resolve_url({"DATABASE_URL": "   "}) is None

    def test_strips_surrounding_whitespace(self) -> None:
        env = {"DATABASE_URL": "  postgresql://x/y  "}
        assert resolve_url(env) == "postgresql://x/y"


class TestResolveBackend:
    """resolve_backend() — explicit override > URL presence > default."""

    def test_explicit_sqlite_wins_over_url(self) -> None:
        env = {
            "AGENTICA_MEMORY_BACKEND": "sqlite",
            "DATABASE_URL": "postgresql://localhost/test",
        }
        assert resolve_backend(env) == "sqlite"

    def test_explicit_postgres_with_url(self) -> None:
        env = {
            "AGENTICA_MEMORY_BACKEND": "postgres",
            "DATABASE_URL": "postgresql://localhost/test",
        }
        assert resolve_backend(env) == "postgres"

    def test_explicit_is_case_insensitive(self) -> None:
        # sqlite needs no URL, so it exercises case-folding without Finding-3.
        assert resolve_backend({"AGENTICA_MEMORY_BACKEND": "SqLite"}) == "sqlite"
        env = {"AGENTICA_MEMORY_BACKEND": "PostgreS", "DATABASE_URL": "postgresql://x/y"}
        assert resolve_backend(env) == "postgres"

    def test_invalid_explicit_raises(self) -> None:
        # Finding 1 (issue #214): an unrecognised override is a hard config error,
        # not silently ignored — it must not fall through to URL/default.
        with pytest.raises(ValueError, match="AGENTICA_MEMORY_BACKEND"):
            resolve_backend({"AGENTICA_MEMORY_BACKEND": "redis"})

    def test_invalid_explicit_with_url_raises(self) -> None:
        # Finding 1 core case: a typo'd override (e.g. "sqllite") together with a
        # leftover URL previously fell through to postgres, silently ignoring the
        # operator's intent. It must now fail fast.
        env = {"AGENTICA_MEMORY_BACKEND": "sqllite", "DATABASE_URL": "postgresql://x/y"}
        with pytest.raises(ValueError, match="sqllite"):
            resolve_backend(env)

    def test_invalid_safe_token_is_echoed(self) -> None:
        # A short alnum value (a plausible backend-name typo) is safe to reflect.
        with pytest.raises(ValueError, match="'sqllite'"):
            resolve_backend({"AGENTICA_MEMORY_BACKEND": "sqllite"})

    def test_invalid_dsn_is_not_reflected(self) -> None:
        # Defense-in-depth (issue #214/#265 r3): a DSN accidentally pasted into
        # the wrong var must NOT have any fragment reflected back. An allowlist
        # of safe shapes beats blacklist redaction — the old `://[^@]+@` only
        # redacted to the first `@` and missed no-scheme credential strings, and
        # this message now reaches user-facing SessionStart output via the TS
        # mirror. Anything non-token becomes a fixed placeholder.
        secretish = "postgresql://user:supersecret@host.example.com/db"
        with pytest.raises(ValueError) as exc:
            resolve_backend({"AGENTICA_MEMORY_BACKEND": secretish})
        msg = str(exc.value)
        assert "<redacted non-token value>" in msg
        assert "supersecret" not in msg
        assert "user" not in msg
        assert "host" not in msg

    def test_invalid_password_with_at_sign_is_not_leaked(self) -> None:
        # The first-`@` blind spot: `://user:p@ssword@h` previously leaked the
        # "ssword" fragment after the first `@`. Must be fully suppressed now.
        with pytest.raises(ValueError) as exc:
            resolve_backend({"AGENTICA_MEMORY_BACKEND": "postgres://user:p@ssword@host/db"})
        msg = str(exc.value)
        assert "<redacted non-token value>" in msg
        assert "ssword" not in msg

    def test_invalid_no_scheme_credential_is_not_leaked(self) -> None:
        # A no-scheme credential string never matched the old `://` regex and was
        # reflected verbatim. The allowlist suppresses it.
        with pytest.raises(ValueError) as exc:
            resolve_backend({"AGENTICA_MEMORY_BACKEND": "user:supersecret@host/db"})
        msg = str(exc.value)
        assert "<redacted non-token value>" in msg
        assert "supersecret" not in msg

    def test_explicit_postgres_without_url_raises(self) -> None:
        # Finding 3 (issue #214): explicitly selecting postgres with no connection
        # URL is a misconfiguration, not a silent fall-back to sqlite.
        with pytest.raises(ValueError, match="no PostgreSQL connection URL"):
            resolve_backend({"AGENTICA_MEMORY_BACKEND": "postgres"})

    def test_explicit_postgres_with_whitespace_url_raises(self) -> None:
        # Finding 3 edge: a whitespace-only URL is not a real URL, so explicit
        # postgres still fails fast rather than passing a blank DSN downstream.
        env = {"AGENTICA_MEMORY_BACKEND": "postgres", "DATABASE_URL": "   "}
        with pytest.raises(ValueError, match="no PostgreSQL connection URL"):
            resolve_backend(env)

    def test_explicit_postgres_without_url_raises_under_none_default(self) -> None:
        # The raise is independent of the caller's `default` (an explicit override
        # is an operator statement, so detect_backend's default=None path raises too).
        with pytest.raises(ValueError, match="no PostgreSQL connection URL"):
            resolve_backend({"AGENTICA_MEMORY_BACKEND": "postgres"}, default=None)

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
