"""Static invariant: no hardcoded DB credentials anywhere in the repo.

Addresses GitHub issue #62. Walks the repo tree and asserts that no
file contains a hardcoded PostgreSQL connection string of the form
``postgresql://<user>:<pwd>@<host>...``. Both the original
``claude:claude_dev`` family and other families (e.g., ``opc:opc_dev_password``,
``agentica:agentica_dev``) must be absent — Round 1 adversarial review
caught that a literal-only check missed the secondary credential
families. See ``_REGEX`` below.

Approved locations (exempted):

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

import re
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

# Matches hardcoded credentials in any PostgreSQL URL form:
#   postgresql://user:password@host...
# Exclusions encoded directly in the character class:
#   - empty password (peer auth, e.g. embedded postgres) is allowed —
#     that's not a credential leak.
#   - f-string templates where the user or password is a {placeholder}
#     are allowed — the "{" character is excluded from both char
#     classes, so a URL like ``f"postgresql://{user}:{password}@..."``
#     does not match.
# Named literal allow-list below covers legitimate test fixtures and
# placeholders that happen to match the regex shape. Future additions
# go in _ALLOWED_MATCHES — simpler than extending the regex (cycle-1
# Gemini R3).
_CREDENTIAL_REGEX = re.compile(
    r"postgresql://[^:@/\s{]+:[^@/\s{]+@"
)

# Literal substrings that are allowed to match the regex. Each is a
# safe-by-review fixture, placeholder, or sanitizer test input.
_ALLOWED_MATCHES = frozenset({
    "postgresql://USER:PASSWORD@",       # .env.example placeholder
    "postgresql://user:s3cretPass@",      # tests/test_extract_workflow_patterns.py
    "postgresql://u:p@",                   # tests/test_postgres_pool.py sanitizer
    "postgresql://u2:p2@",                 # tests/test_postgres_pool.py sanitizer
    "postgresql://user:secret@",           # tests/test_postgres_pool.py sanitizer
})


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
        # Files named `.env.*` need custom handling — Path.suffix splits
        # on the last dot, so `.env.local` has suffix=".local" (not in the
        # scan list by default), which would silently skip a future
        # `.env.production.example` committed with a real credential
        # (Aegis cycle on #62). Rule: any file whose name starts with
        # `.env` is scanned ONLY if it is exactly `.env.example`.
        # Other `.env*` files (`.env`, `.env.local`, `.env.dev`, etc.)
        # are either dev-local secrets (gitignored) or covered by this
        # same rule when their sibling `.env.example` exists.
        if path.name.startswith(".env"):
            if path.name != ".env.example":
                continue
        else:
            if path.suffix and path.suffix not in _SCAN_EXTENSIONS:
                continue
            # Extensionless files (shell scripts, config) are skipped
            # unless they match a known filename — none for this project.
            if not path.suffix:
                continue
        paths.append(path)
    return paths


def test_no_hardcoded_credential_literal_in_code():
    """Assert no file contains a hardcoded PostgreSQL credential."""
    offenders: list[tuple[Path, int, str]] = []
    for path in _scan_paths():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = _CREDENTIAL_REGEX.search(line)
            if match is None:
                continue
            if match.group(0) in _ALLOWED_MATCHES:
                continue
            offenders.append((path.relative_to(_REPO_ROOT), lineno, line.strip()))

    if offenders:
        lines = [f"  {p}:{n}: {text}" for p, n, text in offenders]
        pytest.fail(
            "Hardcoded PostgreSQL credential found in "
            f"{len(offenders)} location(s) outside exempted paths:\n"
            + "\n".join(lines)
            + "\n\nIssue #62 requires all DB URLs to come from env "
            "(CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, OPC_POSTGRES_URL). "
            "If a new legitimate use exists, add the file to _EXEMPT_FILES, "
            "the regex exemption list, and document why."
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
    # Also assert no other credential family leaked in via copy-paste.
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = _CREDENTIAL_REGEX.search(line)
        if match is None:
            continue
        assert match.group(0) in _ALLOWED_MATCHES, (
            f".env.example line {lineno} contains a real-looking "
            f"credential: {line.strip()!r}. Use USER:PASSWORD placeholder "
            "instead."
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
