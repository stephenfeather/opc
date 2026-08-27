#!/usr/bin/env python3
"""
USAGE: artifact_query.py <query> [--type TYPE] [--outcome OUTCOME] [--limit N] [--db PATH]

Search the Context Graph for relevant precedent.

Examples:
    # Search for authentication-related work
    uv run python scripts/artifact_query.py "authentication OAuth JWT"

    # Search only successful handoffs
    uv run python scripts/artifact_query.py "implement agent" --outcome SUCCEEDED

    # Search plans only
    uv run python scripts/artifact_query.py "API design" --type plans
"""

import argparse
import contextlib
import faulthandler
import json
import os
import re
import secrets
import sqlite3
import stat
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Dual import (package vs by-path execution) — same pattern as artifact_index.py.
try:
    from scripts.core.artifact_index import pg_connect, use_postgres
    from scripts.core.artifact_query_sql import sql_for
except ModuleNotFoundError:
    from artifact_index import pg_connect, use_postgres  # type: ignore[no-redef]
    from artifact_query_sql import sql_for  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Faulthandler (side effect isolated to explicit call)
# ---------------------------------------------------------------------------


_faulthandler_file = None


def _enable_faulthandler() -> None:
    """Enable faulthandler for crash diagnostics — best-effort, idempotent."""
    global _faulthandler_file  # noqa: PLW0603
    if _faulthandler_file is not None:
        return
    try:
        log_dir = Path(os.path.expanduser("~/.claude/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        _faulthandler_file = open(log_dir / "opc_crash.log", "a")  # noqa: SIM115
        faulthandler.enable(file=_faulthandler_file, all_threads=True)
    except OSError:
        pass  # Best-effort: crash logging is not critical


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def get_db_path(custom_path: str | None = None) -> Path:
    """Return database path, using custom_path if provided."""
    if custom_path:
        return Path(custom_path)
    return Path(".claude/cache/artifact-index/context.db")


def escape_fts5_query(query: str) -> str:
    """Escape FTS5 query to prevent syntax errors.

    Splits query into words and joins with OR for flexible matching.
    Each word is quoted to handle special characters.
    """
    words = query.split()
    quoted_words = [f'"{w.replace(chr(34), chr(34) + chr(34))}"' for w in words]
    return " OR ".join(quoted_words)


_SESSION_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def is_safe_artifact_path(candidate: str | Path, allowed_root: Path) -> bool:
    """Return True if ``candidate`` resolves to a path within ``allowed_root``.

    Resolution happens before the containment check so symlinks that point
    outside the root are rejected. Non-existent paths are permitted (the
    caller's own ``.exists()`` check decides whether to read them).
    """
    resolved = Path(candidate).resolve()
    root = allowed_root.resolve()
    return resolved.is_relative_to(root)


def safe_artifact_read_path(
    candidate: str | Path, allowed_root: Path, *, suffixes: tuple[str, ...]
) -> Path | None:
    """Return the validated resolved path, or ``None`` if it is not authorized.

    Authorization policy: the resolved path must live inside
    ``allowed_root`` (resolved) AND have a suffix in ``suffixes``. Resolution
    happens before the containment check so symlinks escaping the root are
    rejected. The returned Path is the exact object the caller must read, so the
    checked target and the read target cannot diverge (no re-construction).
    """
    resolved = Path(candidate).resolve()
    root = allowed_root.resolve()
    if not resolved.is_relative_to(root):
        return None
    if resolved.suffix not in suffixes:
        return None
    return resolved


def is_safe_dir_root(root: Path, repo_root: Path) -> bool:
    """Return True if ``root`` is a trustworthy directory root under ``repo_root``.

    Fails CLOSED: rejects the root if it resolves outside ``repo_root`` OR if any
    path component from ``repo_root`` down to ``root`` (checked on the *unresolved*
    on-disk path) is a symlink. This prevents a symlinked trust root (e.g. a swapped
    ``thoughts/shared/handoffs``) from being blessed as the new authorization
    boundary, since ``root.resolve()`` alone would silently follow it.
    """
    repo = repo_root.resolve()
    if not root.resolve().is_relative_to(repo):
        return False
    # Derive the component chain from the *given* paths (not resolved), so we can
    # walk each on-disk component and reject any that is itself a symlink. Anchor
    # the walk at the resolved repo so macOS /var -> /private/var prefixes match.
    try:
        relative = root.absolute().relative_to(repo_root.absolute())
    except ValueError:
        return False
    current = repo
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    return True


def is_safe_session_name(name: str) -> bool:
    """Return True if ``name`` is a safe session identifier.

    Must consist solely of ``[A-Za-z0-9._-]``, be non-empty, and not be made
    up entirely of dots (rejecting ".", "..", "..." and longer dot runs).
    """
    if not name or not _SESSION_NAME_RE.fullmatch(name):
        return False
    return name.strip(".") != ""


def generate_uuid7(
    timestamp_ms: int | None = None, random_bytes: bytes | bytearray | None = None
) -> str:
    """Return a canonical RFC 9562 UUIDv7 string.

    Layout (128 bits): 48-bit big-endian unix-epoch milliseconds, 4-bit
    version (0b0111), 12-bit ``rand_a``, 2-bit variant (0b10), 62-bit
    ``rand_b``. ``timestamp_ms`` and ``random_bytes`` are injection seams for
    tests; production callers leave them as ``None``.
    """
    if timestamp_ms is None:
        timestamp_ms = int(datetime.now().timestamp() * 1000)
    if random_bytes is None:
        random_bytes = secrets.token_bytes(10)

    # Enforce the contract with deliberate errors (bool is an int subclass — reject it).
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int):
        raise ValueError(f"timestamp_ms must be in [0, 2**48), got {timestamp_ms!r}")
    if not 0 <= timestamp_ms < 2**48:
        raise ValueError(f"timestamp_ms must be in [0, 2**48), got {timestamp_ms!r}")
    if not isinstance(random_bytes, (bytes, bytearray)) or len(random_bytes) != 10:
        raise ValueError("random_bytes must be exactly 10 bytes")

    ts = timestamp_ms.to_bytes(6, "big")

    rand_a = int.from_bytes(random_bytes[0:2], "big") & 0x0FFF
    byte6 = 0x70 | (rand_a >> 8)
    byte7 = rand_a & 0xFF

    rand_b = bytearray(random_bytes[2:10])
    rand_b[0] = (rand_b[0] & 0x3F) | 0x80

    raw = ts + bytes([byte6, byte7]) + bytes(rand_b)
    return str(uuid.UUID(bytes=raw))


# ---------------------------------------------------------------------------
# Backend-agnostic execution (issue #282)
# ---------------------------------------------------------------------------

Backend = str  # "sqlite" | "postgres"
SQLITE = "sqlite"
POSTGRES = "postgres"


def _execute(conn, sql: str, params: list, backend: Backend) -> tuple[list[str], list]:
    """Run ``sql`` on ``conn`` and return ``(column_names, rows)``.

    SQLite connections execute directly; psycopg2 connections have no
    ``execute`` and go through a cursor that is always closed. The SQL must
    already carry the backend's placeholders (``?`` vs ``%s``) — see
    ``artifact_query_sql``.
    """
    if backend == SQLITE:
        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return columns, cursor.fetchall()
    if backend == POSTGRES:
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return columns, cur.fetchall()
        finally:
            cur.close()
    raise ValueError(f"Unknown backend: {backend!r}")


def _rows_as_dicts(columns: list[str], rows: list) -> list[dict]:
    return [dict(zip(columns, row)) for row in rows]


def pg_search_query(query: str) -> str:
    """Build the ``websearch_to_tsquery`` input with the same contract as FTS5.

    Mirrors :func:`escape_fts5_query`: whitespace-split tokens, each quoted and
    OR-joined, so a multi-word search matches artifacts containing *any* term on
    both backends (``plainto_tsquery`` would AND them — a silent recall
    regression). Quoting neutralises websearch operators (``-term``, bare
    ``OR``/``AND``); embedded double quotes are stripped since they cannot be
    escaped inside a websearch phrase. ``websearch_to_tsquery`` never raises on
    malformed input; an empty result simply matches nothing.
    """
    tokens = [w.replace('"', "") for w in query.split()]
    return " OR ".join(f'"{t}"' for t in tokens if t)


def _search_params(query: str, backend: Backend) -> list:
    """Query params bound by the search statements, per backend.

    SQLite FTS5 takes one escaped ``MATCH`` expression; PostgreSQL binds the
    websearch expression twice (``ts_rank`` in the SELECT and the ``@@`` filter).
    """
    if backend == POSTGRES:
        q = pg_search_query(query)
        return [q, q]
    return [escape_fts5_query(query)]


# ---------------------------------------------------------------------------
# DB lookup functions (take conn, return data)
# ---------------------------------------------------------------------------


def get_handoff_by_span_id(conn, root_span_id: str, backend: Backend = SQLITE) -> dict | None:
    """Get a handoff by its Braintrust root_span_id.

    When multiple handoffs share the same root_span_id (e.g. multi-task
    sessions), returns the most recent one by created_at.
    """
    columns, rows = _execute(
        conn, sql_for(backend, "get_handoff_by_span_id"), [root_span_id], backend
    )
    return dict(zip(columns, rows[0])) if rows else None


def get_ledger_for_session(conn, session_name: str, backend: Backend = SQLITE) -> dict | None:
    """Get continuity ledger by session name."""
    columns, rows = _execute(
        conn, sql_for(backend, "get_ledger_for_session"), [session_name], backend
    )
    return dict(zip(columns, rows[0])) if rows else None


# ---------------------------------------------------------------------------
# DB search functions (take conn + query, return list[dict])
# ---------------------------------------------------------------------------


def search_handoffs(
    conn,
    query: str,
    outcome: str | None = None,
    limit: int = 5,
    backend: Backend = SQLITE,
) -> list:
    """Search handoffs — FTS5/BM25 on SQLite, tsvector/ts_rank on PostgreSQL."""
    sql = sql_for(backend, "search_handoffs")
    params = _search_params(query, backend)

    if outcome:
        sql += sql_for(backend, "search_handoffs_outcome_filter")
        params.append(outcome)

    sql += sql_for(backend, "search_handoffs_tail")
    params.append(limit)

    return _rows_as_dicts(*_execute(conn, sql, params, backend))


def search_plans(conn, query: str, limit: int = 3, backend: Backend = SQLITE) -> list:
    """Search plans — FTS5/BM25 on SQLite, tsvector/ts_rank on PostgreSQL."""
    params = [*_search_params(query, backend), limit]
    return _rows_as_dicts(*_execute(conn, sql_for(backend, "search_plans"), params, backend))


def search_continuity(conn, query: str, limit: int = 3, backend: Backend = SQLITE) -> list:
    """Search continuity ledgers — FTS5/BM25 on SQLite, tsvector/ts_rank on PostgreSQL."""
    params = [*_search_params(query, backend), limit]
    return _rows_as_dicts(*_execute(conn, sql_for(backend, "search_continuity"), params, backend))


def search_past_queries(conn, query: str, limit: int = 2, backend: Backend = SQLITE) -> list:
    """Check if similar questions have been asked before.

    The ``queries`` tables exist only in the SQLite schema. On PostgreSQL, and on
    a SQLite database created without them, this returns ``[]`` instead of
    crashing every search (issue #282).
    """
    if backend != SQLITE:
        return []
    sql = sql_for(SQLITE, "search_past_queries")
    try:
        columns, rows = _execute(conn, sql, [escape_fts5_query(query), limit], SQLITE)
    except sqlite3.OperationalError as e:
        if "no such table: queries_fts" in str(e):
            return []
        raise
    return _rows_as_dicts(columns, rows)


# ---------------------------------------------------------------------------
# Formatters (pure functions — data in, string out)
# ---------------------------------------------------------------------------

STATUS_ICONS = {
    "SUCCEEDED": "v",
    "PARTIAL_PLUS": "~+",
    "PARTIAL_MINUS": "~-",
    "FAILED": "x",
}

# Unicode icons used by format_results for richer terminal display
_UNICODE_ICONS = {
    "SUCCEEDED": "\u2713",
    "PARTIAL_PLUS": "\u25d0",
    "PARTIAL_MINUS": "\u25d1",
    "FAILED": "\u2717",
}


def _format_past_queries(items: list) -> str:
    """Format past queries section."""
    output = ["## Previously Asked"]
    for q in items:
        question = q.get("question", "")[:100]
        answer = q.get("answer", "")[:200]
        output.append(f"- **Q:** {question}...")
        output.append(f"  **A:** {answer}...")
    output.append("")
    return "\n".join(output)


def _format_handoffs(items: list) -> str:
    """Format handoffs section."""
    output = ["## Relevant Handoffs"]
    for h in items:
        status_icon = STATUS_ICONS.get(h.get("outcome"), "?")
        session = h.get("session_name", "unknown")
        task = h.get("task_number", "?")
        output.append(f"### {status_icon} {session}/task-{task}")
        summary = h.get("task_summary", "")[:200]
        output.append(f"**Summary:** {summary}")
        what_worked = h.get("what_worked")
        if what_worked:
            output.append(f"**What worked:** {what_worked[:200]}")
        what_failed = h.get("what_failed")
        if what_failed:
            output.append(f"**What failed:** {what_failed[:200]}")
        output.append(f"**File:** `{h.get('file_path', '')}`")
        output.append("")
    return "\n".join(output)


def _format_plans(items: list) -> str:
    """Format plans section."""
    output = ["## Relevant Plans"]
    for p in items:
        title = p.get("title", "Untitled")
        output.append(f"### {title}")
        overview = p.get("overview", "")[:200]
        output.append(f"**Overview:** {overview}")
        output.append(f"**File:** `{p.get('file_path', '')}`")
        output.append("")
    return "\n".join(output)


def _format_continuity(items: list) -> str:
    """Format continuity section."""
    output = ["## Related Sessions"]
    for c in items:
        session = c.get("session_name", "unknown")
        output.append(f"### Session: {session}")
        goal = c.get("goal", "")[:200]
        output.append(f"**Goal:** {goal}")
        key_learnings = c.get("key_learnings")
        if key_learnings:
            output.append(f"**Key learnings:** {key_learnings[:200]}")
        output.append("")
    return "\n".join(output)


_SECTION_FORMATTERS = {
    "past_queries": _format_past_queries,
    "handoffs": _format_handoffs,
    "plans": _format_plans,
    "continuity": _format_continuity,
}


def format_result_section(section_type: str, items: list) -> str:
    """Format a single result section using dispatch table.

    Args:
        section_type: One of 'handoffs', 'plans', 'continuity', 'past_queries'
        items: List of result dicts

    Returns:
        Formatted markdown string
    """
    if not items:
        return ""

    formatter = _SECTION_FORMATTERS.get(section_type)
    return formatter(items) if formatter else ""


def format_results(results: dict, verbose: bool = False) -> str:
    """Format search results for display.

    Uses _UNICODE_ICONS for richer terminal output (distinct from
    format_result_section which uses ASCII STATUS_ICONS).
    """
    output = []

    if results.get("past_queries"):
        output.append("## Previously Asked")
        for q in results["past_queries"]:
            question = q.get("question", "")[:100]
            answer = q.get("answer", "")[:200]
            output.append(f"- **Q:** {question}...")
            output.append(f"  **A:** {answer}...")
        output.append("")

    if results.get("handoffs"):
        output.append("## Relevant Handoffs")
        for h in results["handoffs"]:
            status_icon = _UNICODE_ICONS.get(h.get("outcome"), "?")
            session = h.get("session_name", "unknown")
            task = h.get("task_number", "?")
            output.append(f"### {status_icon} {session}/task-{task}")
            summary = h.get("task_summary", "")[:200]
            output.append(f"**Summary:** {summary}")
            what_worked = h.get("what_worked")
            if what_worked:
                output.append(f"**What worked:** {what_worked[:200]}")
            what_failed = h.get("what_failed")
            if what_failed:
                output.append(f"**What failed:** {what_failed[:200]}")
            output.append(f"**File:** `{h.get('file_path', '')}`")
            output.append("")

    if results.get("plans"):
        output.append("## Relevant Plans")
        for p in results["plans"]:
            title = p.get("title", "Untitled")
            output.append(f"### {title}")
            overview = p.get("overview", "")[:200]
            output.append(f"**Overview:** {overview}")
            output.append(f"**File:** `{p.get('file_path', '')}`")
            output.append("")

    if results.get("continuity"):
        output.append("## Related Sessions")
        for c in results["continuity"]:
            session = c.get("session_name", "unknown")
            output.append(f"### Session: {session}")
            goal = c.get("goal", "")[:200]
            output.append(f"**Goal:** {goal}")
            key_learnings = c.get("key_learnings")
            if key_learnings:
                output.append(f"**Key learnings:** {key_learnings[:200]}")
            output.append("")

    if not any(results.values()):
        output.append("No relevant precedent found.")

    return "\n".join(output)


# ---------------------------------------------------------------------------
# Dispatch (coordination — composes search + format)
# ---------------------------------------------------------------------------


def search_dispatch(
    conn,
    query: str,
    search_type: str = "all",
    outcome: str | None = None,
    limit: int = 5,
    backend: Backend = SQLITE,
) -> dict:
    """Dispatch search to appropriate handlers based on type.

    Uses dispatch table pattern to reduce if/elif chains.

    Args:
        conn: Database connection
        query: Search query string
        search_type: One of 'handoffs', 'plans', 'continuity', 'all'
        outcome: Optional outcome filter for handoffs
        limit: Max results per type
        backend: 'sqlite' or 'postgres' — selects the SQL dialect

    Returns:
        Dict with results keyed by type
    """
    results = {}

    # Always check past queries (empty on PostgreSQL — no queries table there)
    results["past_queries"] = search_past_queries(conn, query, backend=backend)

    search_handlers = {
        "handoffs": lambda: search_handoffs(conn, query, outcome, limit, backend=backend),
        "plans": lambda: search_plans(conn, query, limit, backend=backend),
        "continuity": lambda: search_continuity(conn, query, limit, backend=backend),
    }

    if search_type == "all":
        for key, handler in search_handlers.items():
            results[key] = handler()
    elif search_type in search_handlers:
        results[search_type] = search_handlers[search_type]()

    return results


# ---------------------------------------------------------------------------
# Span ID lookup (I/O at the boundary)
# ---------------------------------------------------------------------------


def read_text_within_root(
    root: Path,
    candidate: str | Path,
    *,
    suffixes: tuple[str, ...],
    anchor: Path | None = None,
) -> str | None:
    """Authorize ``candidate`` under ``root`` and read it via an openat walk.

    Two layers:

    1. **Policy** — :func:`safe_artifact_read_path` resolves ``candidate`` and
       requires it to stay inside ``root`` (rejecting ``..`` escapes and symlinks
       that point out of the root) and to carry an allowed suffix.

    2. **No-TOCTOU read** — the authorized path is then read by opening a single
       pathname-resolved trust ``anchor`` directory and walking *every lexical
       component* from the anchor down to the leaf with ``O_NOFOLLOW`` *relative
       to the previously-opened parent dirfd* (``openat``-style).
       ``read_text_nofollow``'s single final-component ``O_NOFOLLOW`` left every
       *intermediate* directory exposed: an attacker who won a race on one between
       the resolve check and the open could redirect the read (issue #166).

    ``anchor`` is the irreducible, pathname-resolved trust root the fd-relative
    walk starts from; it defaults to ``root``. Callers whose ``root`` is itself a
    nested directory (e.g. ``thoughts/shared/handoffs``) should pass a higher
    ``anchor`` (e.g. the repo root) so that ``root``'s *own* parent components are
    traversed under ``O_NOFOLLOW`` too — otherwise a parent of ``root`` could be
    swapped for a symlink after validation and the walk would anchor inside the
    attacker's tree. Only the anchor open follows the pathname; once its fd is
    held, no component below it — including ``root`` and the leaf — can be a
    symlink (or be swapped for one).

    The walk uses the un-resolved (lexical) component names on purpose: a symlink
    present at open time is refused by ``O_NOFOLLOW`` rather than silently
    canonicalized away as ``Path.resolve()`` would do, so the read target either
    matches the policy-authorized resolved path exactly (no symlinks present) or
    is refused. The final component is ``fstat``-checked on the same fd it is read
    from: it must be a single-link (``st_nlink == 1``) regular file whose
    (device, inode) matches the policy-authorized resolved target. The link-count
    check rejects an attacker's hardlink aliasing an out-of-policy secret into an
    allowed path; the identity check rejects a non-symlink directory swap that
    substitutes a same-named file across the check/read boundary. Never raises:
    returns ``None`` on any rejection or I/O failure.
    """
    safe = safe_artifact_read_path(candidate, root, suffixes=suffixes)
    if safe is None:
        return None

    # Capture the identity (device, inode) of the authorized resolved target so
    # the leaf we ultimately read can be proven to be that same inode. O_NOFOLLOW
    # blocks symlink swaps, but a *non-symlink* directory swap between this check
    # and the walk could otherwise substitute a same-named file; binding identity
    # across the policy/read boundary fails closed on that race too.
    try:
        authorized = os.stat(safe)
    except OSError:
        return None
    authorized_id = (authorized.st_dev, authorized.st_ino)

    walk_anchor = anchor if anchor is not None else root

    # Lexical (un-resolved) components of the candidate below the trust anchor.
    # Path.resolve() in the policy layer follows symlinks; here we deliberately
    # keep the on-disk component names so the O_NOFOLLOW walk inspects reality.
    cand_abs = Path(candidate)
    if not cand_abs.is_absolute():
        cand_abs = Path.cwd() / cand_abs
    try:
        rel_parts = cand_abs.relative_to(walk_anchor.absolute()).parts
    except ValueError:
        return None
    if not rel_parts or any(part in ("..", ".") for part in rel_parts):
        # No component below the anchor, or a traversal component the openat walk
        # would let escape the boundary — fail closed.
        return None

    # Open the irreducible trust anchor by pathname; every component below it
    # (root's own parents, root, intermediates, and the leaf) is O_NOFOLLOW.
    try:
        dir_fd = os.open(walk_anchor, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return None

    open_fds: list[int] = [dir_fd]
    try:
        for part in rel_parts[:-1]:
            try:
                open_fds.append(
                    os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=open_fds[-1],
                    )
                )
            except OSError:
                return None

        try:
            leaf_fd = os.open(rel_parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=open_fds[-1])
        except OSError:
            return None
        # Track leaf_fd in open_fds so the finally closes it until fdopen takes
        # ownership; this avoids a double-close on the read-error path.
        open_fds.append(leaf_fd)

        try:
            leaf_stat = os.fstat(leaf_fd)
        except OSError:
            return None
        if not stat.S_ISREG(leaf_stat.st_mode):
            return None
        # Reject hardlinked content: a legitimate artifact is a single-link file.
        # A multi-link regular file may be an attacker's hardlink aliasing an
        # out-of-policy secret (.env, settings.local.json) into an allowed path;
        # O_NOFOLLOW and the inode binding cannot distinguish that alias because
        # the hardlink and the secret share one inode (review #166 round 3).
        if leaf_stat.st_nlink != 1:
            return None
        # Bind authorization to the opened inode: the leaf must be the exact
        # device/inode that passed policy, defeating non-symlink swap races.
        if (leaf_stat.st_dev, leaf_stat.st_ino) != authorized_id:
            return None

        try:
            handle = os.fdopen(leaf_fd, "r", encoding="utf-8")
        except (OSError, ValueError):
            # io.open may raise ValueError on an unexpected fd/mode edge case;
            # honor the "never raises" contract and fail closed.
            return None
        # fdopen now owns leaf_fd; drop it from manual cleanup so the context
        # manager is its sole closer (no double-close if read() raises).
        open_fds.pop()
        with handle:
            try:
                return handle.read()
            except (OSError, UnicodeError, ValueError):
                return None
    finally:
        for fd in open_fds:
            with contextlib.suppress(OSError):
                os.close(fd)


def handle_span_id_lookup(
    conn, span_id: str, with_content: bool = False, backend: Backend = SQLITE
) -> dict | None:
    """Handle --by-span-id lookup mode.

    Args:
        conn: Database connection
        span_id: Braintrust root_span_id to look up
        with_content: Whether to include full file content
        backend: 'sqlite' or 'postgres'

    Returns:
        Handoff dict or None if not found
    """
    handoff = get_handoff_by_span_id(conn, span_id, backend=backend)

    if not handoff:
        return None

    if with_content and handoff.get("file_path"):
        allowed_root = Path.cwd()
        handoffs_root = allowed_root / "thoughts" / "shared" / "handoffs"

        # Fail closed if the trusted handoffs root is itself reached via a symlink.
        if is_safe_dir_root(handoffs_root, allowed_root):
            # Anchor the openat walk at the repo root (allowed_root) so that the
            # handoffs root's own parents (thoughts/shared/handoffs) are traversed
            # under O_NOFOLLOW too — closing the residual TOCTOU between the
            # is_safe_dir_root check above and the directory open.
            content = read_text_within_root(
                handoffs_root,
                handoff["file_path"],
                suffixes=(".md", ".yaml", ".yml"),
                anchor=allowed_root,
            )
            if content is not None:
                handoff["content"] = content

        session_name = handoff.get("session_name")
        if not session_name and handoff.get("file_path"):
            parts = Path(handoff["file_path"]).parts
            if "handoffs" in parts:
                idx = parts.index("handoffs")
                if idx + 1 < len(parts):
                    session_name = parts[idx + 1]

        if session_name and is_safe_session_name(session_name):
            ledger_path = Path(f"CONTINUITY_CLAUDE-{session_name}.md")
            ledger_content = read_text_within_root(allowed_root, ledger_path, suffixes=(".md",))
            if ledger_content is not None:
                handoff["ledger"] = {
                    "session_name": session_name,
                    "file_path": str(ledger_path),
                    "content": ledger_content,
                }
            else:
                ledger = get_ledger_for_session(conn, session_name, backend=backend)
                if ledger:
                    handoff["ledger"] = ledger

    return handoff


# ---------------------------------------------------------------------------
# DB write (side effect at the boundary)
# ---------------------------------------------------------------------------


def save_query(
    conn: sqlite3.Connection,
    question: str,
    answer: str,
    matches: dict,
    now: datetime | None = None,
) -> None:
    """Save query for compound learning.

    Args:
        conn: Database connection
        question: The search query
        answer: Formatted result text
        matches: Dict of matched items by type
        now: Timestamp override for testability (defaults to datetime.now())
    """
    timestamp = now or datetime.now()
    query_id = generate_uuid7(timestamp_ms=int(timestamp.timestamp() * 1000))

    conn.execute(
        """
        INSERT INTO queries (id, question, answer, handoffs_matched, plans_matched,
                             continuity_matched)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            query_id,
            question,
            answer,
            json.dumps([h["id"] for h in matches.get("handoffs", [])]),
            json.dumps([p["id"] for p in matches.get("plans", [])]),
            json.dumps([c["id"] for c in matches.get("continuity", [])]),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# CLI helpers (I/O at the boundary)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser — pure construction, no side effects."""
    parser = argparse.ArgumentParser(description="Search the Context Graph for relevant precedent")
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("--type", choices=["handoffs", "plans", "continuity", "all"], default="all")
    parser.add_argument(
        "--outcome", choices=["SUCCEEDED", "PARTIAL_PLUS", "PARTIAL_MINUS", "FAILED"]
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--db", type=str, help="Custom database path")
    parser.add_argument("--save", action="store_true", help="Save query for compound learning")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--by-span-id", type=str, help="Get handoff by Braintrust root_span_id")
    parser.add_argument("--with-content", action="store_true", help="Include full file content")
    return parser


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open SQLite database connection with standard pragmas."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _open_pg():
    """Open a PostgreSQL connection via the indexer's resolver (same URL precedence)."""
    return pg_connect()


def _select_backend(args: argparse.Namespace) -> Backend:
    """Pick the backend the way the indexer does (issue #282).

    A custom ``--db`` path always means SQLite and is honored BEFORE consulting
    the resolver, so a mis-set AGENTICA_MEMORY_BACKEND cannot block a purely
    local SQLite query. Otherwise defer to ``use_postgres()``; its ``ValueError``
    (invalid backend config) propagates for the caller to report.
    """
    if args.db:
        return SQLITE
    return POSTGRES if use_postgres() else SQLITE


def _resolve_backend_or_report(args: argparse.Namespace) -> Backend | None:
    """Select the backend, printing a clean diagnostic (not a traceback) on config errors."""
    try:
        return _select_backend(args)
    except ValueError as e:
        print(f"Backend configuration error: {e}", file=sys.stderr)
        return None


def _open_conn(args: argparse.Namespace, backend: Backend):
    """Open the connection for ``backend``; ``None`` (with a message) if SQLite DB is missing."""
    if backend == POSTGRES:
        return _open_pg()
    db_path = get_db_path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run: uv run python scripts/artifact_index.py --all")
        return None
    return _open_db(db_path)


def _is_pg_error(exc: BaseException) -> bool:
    """True for any psycopg2 exception, without importing psycopg2 at module load."""
    return type(exc).__module__.split(".")[0] == "psycopg2"


# Any ``user:password@`` token, not only after ``://`` — a *malformed* DSN (the
# case libpq echoes verbatim) may be missing the scheme separator entirely.
_USERINFO_RE = re.compile(r"[^\s/:@\"']+:[^\s/:@\"']+@")
_KV_PASSWORD_RE = re.compile(r"(password=)\S+")


def redact_credentials(text: str) -> str:
    """Mask credentials in URL/userinfo (``user:pw@``) or key=value (``password=``) form.

    libpq echoes a malformed DSN verbatim in its error text, so anything that
    prints a psycopg2 message must pass through here first.
    """
    text = _USERINFO_RE.sub("***:***@", text)
    return _KV_PASSWORD_RE.sub(r"\1***", text)


def _report_pg_error(exc: BaseException) -> int:
    """Print a concise, actionable PostgreSQL failure to stderr; return exit code 1.

    Connection/auth/timeout failures and missing artifact tables both surface
    here. No automatic SQLite fallback: that would silently reintroduce the
    split-brain read this module exists to fix. Pass ``--db <path>`` to query a
    local SQLite index explicitly.
    """
    print(f"PostgreSQL error: {redact_credentials(str(exc))}", file=sys.stderr)
    if "does not exist" in str(exc):
        print(
            "The artifact tables are missing in this database. "
            "Run: uv run python scripts/core/artifact_index.py --all",
            file=sys.stderr,
        )
    else:
        print(
            "Check the configured URL (CONTINUOUS_CLAUDE_DB_URL / DATABASE_URL / "
            "OPC_POSTGRES_URL) and that the server is reachable; "
            "or pass --db <path> to query a local SQLite index.",
            file=sys.stderr,
        )
    return 1


def _run_span_lookup(args: argparse.Namespace) -> int:
    """Handle --by-span-id CLI mode. Returns a process exit code."""
    backend = _resolve_backend_or_report(args)
    if backend is None:
        return 1
    try:
        conn = _open_conn(args, backend)
        if conn is None:
            return 0

        with contextlib.closing(conn):
            handoff = handle_span_id_lookup(
                conn, args.by_span_id, with_content=args.with_content, backend=backend
            )
    except Exception as e:
        if _is_pg_error(e):
            return _report_pg_error(e)
        raise

    if args.json:
        print(json.dumps(handoff, indent=2, default=str))
    elif handoff:
        print(f"## Handoff: {handoff.get('session_name')}/task-{handoff.get('task_number')}")
        print(f"**Outcome:** {handoff.get('outcome', 'UNKNOWN')}")
        print(f"**File:** {handoff.get('file_path')}")
        if handoff.get("content"):
            print(f"\n{handoff['content']}")
    else:
        print(f"No handoff found for root_span_id: {args.by_span_id}")
    return 0


def _run_search(args: argparse.Namespace, query: str) -> int:
    """Handle regular search CLI mode. Returns a process exit code."""
    backend = _resolve_backend_or_report(args)
    if backend is None:
        return 1
    if args.save and backend != SQLITE:
        # Refuse up front rather than search, print results, and exit 0 while
        # silently dropping the requested compound-learning record.
        print(
            "--save is only supported on the SQLite backend (PostgreSQL has no queries "
            "table). Re-run with --db <path> or without --save.",
            file=sys.stderr,
        )
        return 2

    try:
        conn = _open_conn(args, backend)
        if conn is None:
            return 0

        with contextlib.closing(conn):
            results = search_dispatch(
                conn, query, args.type, args.outcome, args.limit, backend=backend
            )

            if args.json:
                print(json.dumps(results, indent=2, default=str))
            else:
                formatted = format_results(results)
                print(formatted)

                if args.save:
                    save_query(conn, query, formatted, results)
                    print("\n[Query saved for compound learning]")
    except Exception as e:
        if _is_pg_error(e):
            return _report_pg_error(e)
        raise
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for artifact query CLI."""
    _enable_faulthandler()

    parser = _build_parser()
    args = parser.parse_args()

    if args.by_span_id:
        rc = _run_span_lookup(args)
    elif not args.query:
        parser.print_help()
        rc = 0
    else:
        rc = _run_search(args, " ".join(args.query))

    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
