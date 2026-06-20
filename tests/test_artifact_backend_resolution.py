"""Issue #71: artifact indexing/marking must honor the unified backend resolver.

Both ``artifact_index`` and ``artifact_mark`` previously kept their own
URL/backend detection (CONTINUOUS_CLAUDE_DB_URL/DATABASE_URL + psycopg2), which
ignored OPC_POSTGRES_URL and the explicit AGENTICA_MEMORY_BACKEND=sqlite
override. That let artifacts land in a different backend than store/recall/daemon
(split-brain). These tests pin the unified behavior.
"""

from __future__ import annotations

import pytest

from scripts.core import artifact_index, artifact_mark

MODULES = pytest.mark.parametrize("module", [artifact_index, artifact_mark])


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
