"""Static invariant: no hardcoded DB credentials anywhere in the repo.

Addresses GitHub issue #62. Walks the repo tree and asserts that the
literal string ``claude:claude_dev`` appears only in approved locations:

- ``docker/`` — the local-dev credentials legitimately live here
- ``.worktrees/`` — sibling feature branches may still be mid-fix
- ``.git/`` — internal git state; never touched
- ``node_modules/`` — third-party JS deps
- ``thoughts/`` — plans and agent reports may quote the literal during
  design discussion
- ``tests/test_postgres_pool.py`` — one test (``test_redacts_credentials``)
  passes the literal as *input* to verify the sanitizer strips it;
  that's a fixture, not a hardcoded credential the code returns

Test fails if any other file contains the literal, ensuring that future
changes cannot silently reintroduce a hardcoded fallback URL.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories exempted from the scan (per ARCHITECT guidance #104-style
# deferred-fix rule; plans / worktrees / docker credentials).
_EXEMPT_DIRS = {
    ".git",
    ".worktrees",
    ".venv",
    "node_modules",
    "docker",
    "thoughts",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "htmlcov",
}

# Specific files exempted because they use the literal as legitimate test
# input (sanitizer fixtures), not as a credential the code returns.
_EXEMPT_FILES = {
    # Sanitizer fixture — the test *passes* the literal to verify redaction.
    _REPO_ROOT / "tests" / "test_postgres_pool.py",
    # This walker itself references the literal.
    Path(__file__).resolve(),
}

# File extensions scanned (source code + config, not binaries or lockfiles).
_SCAN_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".mjs",
    ".cjs",
    ".sh",
    ".zsh",
    ".bash",
    ".md",
    ".rst",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".env",
    ".example",
    ".json",
}

_CREDENTIAL_LITERAL = "claude:claude_dev"


def _scan_paths() -> list[Path]:
    """Return files to scan, excluding exempted directories and files."""
    paths: list[Path] = []
    for path in _REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path in _EXEMPT_FILES:
            continue
        # Skip anything under an exempted directory at any depth.
        try:
            rel = path.relative_to(_REPO_ROOT)
        except ValueError:
            continue
        if any(part in _EXEMPT_DIRS for part in rel.parts):
            continue
        # Also skip .env.example variants — caught by extension in a moment.
        if path.suffix and path.suffix not in _SCAN_EXTENSIONS:
            continue
        # Unknown extension but named .env* or Dockerfile* — scan anyway.
        if not path.suffix and path.name not in {".env", ".env.example"}:
            continue
        paths.append(path)
    return paths


def test_no_hardcoded_credential_literal_in_code():
    """Assert no file contains the literal `claude:claude_dev`."""
    offenders: list[tuple[Path, int, str]] = []
    for path in _scan_paths():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _CREDENTIAL_LITERAL not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _CREDENTIAL_LITERAL in line:
                offenders.append((path.relative_to(_REPO_ROOT), lineno, line.strip()))

    if offenders:
        lines = [f"  {p}:{n}: {text}" for p, n, text in offenders]
        pytest.fail(
            "Hardcoded credential literal 'claude:claude_dev' found in "
            f"{len(offenders)} location(s) outside exempted paths:\n"
            + "\n".join(lines)
            + "\n\nIssue #62 requires all DB URLs to come from env "
            "(CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, OPC_POSTGRES_URL). "
            "If a new legitimate use exists, add the file to _EXEMPT_FILES "
            "and document why."
        )


def test_env_example_uses_placeholder_not_real_credential():
    """`.env.example` must not ship a real-looking credential."""
    env_example = _REPO_ROOT / ".env.example"
    assert env_example.exists(), ".env.example must exist"
    text = env_example.read_text(encoding="utf-8")
    assert _CREDENTIAL_LITERAL not in text, (
        ".env.example must not contain the literal dev credential. "
        "Use a commented-out placeholder instead."
    )


def test_scan_covers_expected_paths():
    """Meta-test: walker must reach the files we care about."""
    scanned = {p.relative_to(_REPO_ROOT) for p in _scan_paths()}
    # Spot-check a few known files across the tree so a future refactor
    # that accidentally excludes a directory doesn't silently hide hits.
    expected = {
        Path("scripts/core/backfill_sessions.py"),
        Path("scripts/core/db/postgres_pool.py"),
        Path("scripts/core/memory_daemon.py"),
        Path("hooks/src/shared/db-utils-pg.ts"),
        Path(".env.example"),
    }
    missing = expected - scanned
    assert not missing, f"Walker missed expected files: {missing}"
