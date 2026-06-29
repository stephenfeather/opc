"""Issue #71: artifact indexing/marking must honor the unified backend resolver.

Both ``artifact_index`` and ``artifact_mark`` previously kept their own
URL/backend detection (CONTINUOUS_CLAUDE_DB_URL/DATABASE_URL + psycopg2), which
ignored OPC_POSTGRES_URL and the explicit AGENTICA_MEMORY_BACKEND=sqlite
override. That let artifacts land in a different backend than store/recall/daemon
(split-brain). These tests pin the unified behavior.
"""

from __future__ import annotations

import json
import sys

import pytest

from scripts.core import artifact_index, artifact_mark

MODULES = pytest.mark.parametrize("module", [artifact_index, artifact_mark])


class TestArtifactIndexCustomDbOverride:
    """Issue #214 (review R2): a --db SQLite override is a purely local operation
    and must not be blocked by the fail-fast backend resolver. Previously
    ``using_pg = use_postgres() and not args.db`` evaluated use_postgres() first,
    so an invalid/misconfigured AGENTICA_MEMORY_BACKEND raised before --db could
    force SQLite. The override now wins before the resolver is consulted.
    """

    def test_custom_db_ignores_invalid_backend(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("AGENTICA_MEMORY_BACKEND", "sqllite")  # typo'd, invalid
        missing = tmp_path / "does-not-exist.md"
        db_path = tmp_path / "artifacts.sqlite"
        monkeypatch.setattr(
            sys,
            "argv",
            ["artifact_index.py", "--file", str(missing), "--db", str(db_path), "--json"],
        )

        # Must NOT raise from the resolver: a missing file yields a JSON failure,
        # proving main() reached the file branch instead of crashing at backend
        # resolution.
        rc = artifact_index.main()

        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is False
        assert "not found" in payload["error"].lower()

    def test_no_db_config_error_yields_json_not_traceback(self, monkeypatch, tmp_path, capsys):
        # Codex P2 (PR #262): without --db, the resolver IS consulted and may
        # fail-fast (issue #214). Under --file --json the machine-caller contract
        # requires exactly one JSON object on stdout, not an uncaught traceback.
        monkeypatch.setenv("AGENTICA_MEMORY_BACKEND", "sqllite")  # invalid -> raises
        missing = tmp_path / "does-not-exist.md"
        monkeypatch.setattr(
            sys,
            "argv",
            ["artifact_index.py", "--file", str(missing), "--json"],  # no --db
        )

        rc = artifact_index.main()

        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is False
        assert "configuration error" in payload["error"].lower()


@MODULES
class TestArtifactBackendResolution:
    def test_get_postgres_url_falls_back_to_opc(self, module, monkeypatch):
        monkeypatch.delenv("CONTINUOUS_CLAUDE_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("OPC_POSTGRES_URL", "postgresql://legacy")
        assert module.get_postgres_url() == "postgresql://legacy"

    def test_get_postgres_url_none_when_unset(self, module, monkeypatch):
        monkeypatch.delenv("CONTINUOUS_CLAUDE_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("OPC_POSTGRES_URL", raising=False)
        assert module.get_postgres_url() is None

    def test_use_postgres_false_under_sqlite_override(self, module, monkeypatch):
        monkeypatch.setenv("AGENTICA_MEMORY_BACKEND", "sqlite")
        monkeypatch.setenv("CONTINUOUS_CLAUDE_DB_URL", "postgresql://canon")
        monkeypatch.setenv("OPC_POSTGRES_URL", "postgresql://legacy")
        assert module.use_postgres() is False

    def test_use_postgres_true_with_url_no_override(self, module, monkeypatch):
        monkeypatch.delenv("AGENTICA_MEMORY_BACKEND", raising=False)
        monkeypatch.setenv("CONTINUOUS_CLAUDE_DB_URL", "postgresql://canon")
        # psycopg2 is installed in the test env
        assert module.use_postgres() is True
