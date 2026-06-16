"""Issue #108: extraction subprocess env must not inherit secrets.

Exercises the single unified builder used by the daemon, the CLI
extractor, and backfill. Asserts known secret-named keys never reach
the child process env, while the minimal allowlisted set survives.
"""
import pytest

from scripts.core.memory_daemon_core import build_extraction_env

SECRET_KEYS = [
    "DATABASE_URL",
    "OPENAI_API_KEY",
    "VOYAGE_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GCP_SA_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    # Codex review #108 round 1 (HIGH): secret-bearing UV_ registry vars
    # must NOT survive. Only UV_CACHE_DIR is allowed by exact name; a
    # broad UV_ prefix would leak these into the Bash-capable child.
    "UV_PUBLISH_TOKEN",
    "UV_INDEX_FOO_PASSWORD",
    # Codex review #108 round 3 (HIGH): secret-named CLAUDE_* vars must not
    # ride the broad CLAUDE_ allow prefix into the Bash-capable child.
    "CLAUDE_API_KEY",
    "CLAUDE_TOKEN",
    "CLAUDE_SECRET_DB",
    "CLAUDE_CODE_OAUTH_TOKEN",
]

ALLOWED_KEYS = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/u",
    "CLAUDE_CONFIG_DIR": "/home/u/.claude",
    # A non-secret CLAUDE_* var the extractor child needs must survive
    # deny-before-allow (proves the filter is not over-broad).
    "CLAUDE_OPC_DIR": "/home/u/opc",
    "XDG_CONFIG_HOME": "/home/u/.config",
    "UV_CACHE_DIR": "/home/u/.cache/uv",
}


def _polluted_env():
    env = dict(ALLOWED_KEYS)
    for k in SECRET_KEYS:
        env[k] = f"SECRET::{k}"
    return env


@pytest.mark.parametrize("secret", SECRET_KEYS)
def test_secret_never_in_child_env(secret):
    env = build_extraction_env(_polluted_env(), "/tmp/proj")
    assert secret not in env, f"{secret} leaked into extraction subprocess env"


def test_allowlisted_keys_survive():
    env = build_extraction_env(_polluted_env(), "/tmp/proj")
    for k, v in ALLOWED_KEYS.items():
        assert env[k] == v


def test_daemon_marker_and_project_dir_set():
    env = build_extraction_env(_polluted_env(), "/tmp/proj")
    assert env["CLAUDE_MEMORY_EXTRACTION"] == "1"
    assert env["CLAUDE_PROJECT_DIR"] == "/tmp/proj"


def test_no_secret_values_present_anywhere():
    env = build_extraction_env(_polluted_env(), "/tmp/proj")
    for v in env.values():
        assert not v.startswith("SECRET::"), "a secret value survived filtering"
