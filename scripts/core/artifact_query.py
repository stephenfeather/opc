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
import faulthandler
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

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
# Path safety (pure)
# ---------------------------------------------------------------------------

_SESSION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _is_safe_path(file_path: Path, base: Path | None = None) -> bool:
    """Check that resolved file_path is under the allowed base directory."""
    base = (base or Path.cwd()).resolve()
    try:
        resolved = file_path.resolve()
        return resolved == base or str(resolved).startswith(str(base) + os.sep)
    except (OSError, ValueError):
        return False


def _safe_read_text(path: Path) -> str | None:
    """Read text from path, returning None on I/O or encoding errors."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


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


# ---------------------------------------------------------------------------
# DB lookup functions (take conn, return data)
# ---------------------------------------------------------------------------


def get_handoff_by_span_id(conn: sqlite3.Connection, root_span_id: str) -> dict | None:
    """Get a handoff by its Braintrust root_span_id.

    When multiple handoffs share the same root_span_id (e.g. multi-task
    sessions), returns the most recent one by created_at.
    """
    sql = """
        SELECT id, session_name, task_number, task_summary,
               outcome, what_worked, what_failed, key_decisions,
               file_path, root_span_id, created_at
        FROM handoffs
        WHERE root_span_id = ?
        ORDER BY datetime(created_at) DESC, task_number DESC, rowid DESC
        LIMIT 1
    """
    cursor = conn.execute(sql, [root_span_id])
    columns = [desc[0] for desc in cursor.description]
    row = cursor.fetchone()
    return dict(zip(columns, row)) if row else None


def get_ledger_for_session(conn: sqlite3.Connection, session_name: str) -> dict | None:
    """Get continuity ledger by session name."""
    sql = """
        SELECT id, session_name, goal, key_learnings, key_decisions,
               state_done, state_now, state_next, created_at
        FROM continuity
        WHERE session_name = ?
        ORDER BY created_at DESC
        LIMIT 1
    """
    cursor = conn.execute(sql, [session_name])
    columns = [desc[0] for desc in cursor.description]
    row = cursor.fetchone()
    return dict(zip(columns, row)) if row else None


# ---------------------------------------------------------------------------
# DB search functions (take conn + query, return list[dict])
# ---------------------------------------------------------------------------


def search_handoffs(
    conn: sqlite3.Connection, query: str, outcome: str | None = None, limit: int = 5
) -> list:
    """Search handoffs using FTS5 with BM25 ranking."""
    sql = """
        SELECT h.id, h.session_name, h.task_number, h.task_summary,
               h.what_worked, h.what_failed, h.key_decisions,
               h.outcome, h.file_path, h.created_at,
               handoffs_fts.rank as score
        FROM handoffs_fts
        JOIN handoffs h ON handoffs_fts.rowid = h.rowid
        WHERE handoffs_fts MATCH ?
    """
    params = [escape_fts5_query(query)]

    if outcome:
        sql += " AND h.outcome = ?"
        params.append(outcome)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    cursor = conn.execute(sql, params)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def search_plans(conn: sqlite3.Connection, query: str, limit: int = 3) -> list:
    """Search plans using FTS5 with BM25 ranking."""
    sql = """
        SELECT p.id, p.title, p.overview, p.approach, p.file_path, p.created_at,
               plans_fts.rank as score
        FROM plans_fts
        JOIN plans p ON plans_fts.rowid = p.rowid
        WHERE plans_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """
    cursor = conn.execute(sql, [escape_fts5_query(query), limit])
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def search_continuity(conn: sqlite3.Connection, query: str, limit: int = 3) -> list:
    """Search continuity ledgers using FTS5 with BM25 ranking."""
    sql = """
        SELECT c.id, c.session_name, c.goal, c.key_learnings, c.key_decisions,
               c.state_now, c.created_at,
               continuity_fts.rank as score
        FROM continuity_fts
        JOIN continuity c ON continuity_fts.rowid = c.rowid
        WHERE continuity_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """
    cursor = conn.execute(sql, [escape_fts5_query(query), limit])
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def search_past_queries(conn: sqlite3.Connection, query: str, limit: int = 2) -> list:
    """Check if similar questions have been asked before."""
    sql = """
        SELECT q.id, q.question, q.answer, q.was_helpful, q.created_at,
               queries_fts.rank as score
        FROM queries_fts
        JOIN queries q ON queries_fts.rowid = q.rowid
        WHERE queries_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """
    cursor = conn.execute(sql, [escape_fts5_query(query), limit])
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


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
    conn: sqlite3.Connection,
    query: str,
    search_type: str = "all",
    outcome: str | None = None,
    limit: int = 5,
) -> dict:
    """Dispatch search to appropriate handlers based on type.

    Uses dispatch table pattern to reduce if/elif chains.

    Args:
        conn: Database connection
        query: Search query string
        search_type: One of 'handoffs', 'plans', 'continuity', 'all'
        outcome: Optional outcome filter for handoffs
        limit: Max results per type

    Returns:
        Dict with results keyed by type
    """
    results = {}

    # Always check past queries
    results["past_queries"] = search_past_queries(conn, query)

    search_handlers = {
        "handoffs": lambda: search_handoffs(conn, query, outcome, limit),
        "plans": lambda: search_plans(conn, query, limit),
        "continuity": lambda: search_continuity(conn, query, limit),
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


def handle_span_id_lookup(
    conn: sqlite3.Connection,
    span_id: str,
    with_content: bool = False,
    allowed_base: Path | None = None,
) -> dict | None:
    """Handle --by-span-id lookup mode.

    Args:
        conn: Database connection
        span_id: Braintrust root_span_id to look up
        with_content: Whether to include full file content
        allowed_base: Root directory for path-traversal checks (defaults to cwd)

    Returns:
        Handoff dict or None if not found
    """
    handoff = get_handoff_by_span_id(conn, span_id)

    if not handoff:
        return None

    if with_content and handoff.get("file_path"):
        file_path = Path(handoff["file_path"])
        if file_path.exists() and _is_safe_path(file_path, base=allowed_base):
            content = _safe_read_text(file_path)
            if content is not None:
                handoff["content"] = content

        session_name = handoff.get("session_name")
        if not session_name and handoff.get("file_path"):
            parts = Path(handoff["file_path"]).parts
            if "handoffs" in parts:
                idx = parts.index("handoffs")
                if idx + 1 < len(parts):
                    session_name = parts[idx + 1]

        if session_name and _SESSION_NAME_RE.match(session_name):
            ledger_path = Path(f"CONTINUITY_CLAUDE-{session_name}.md")
            if ledger_path.exists() and _is_safe_path(ledger_path, base=allowed_base):
                content = _safe_read_text(ledger_path)
                if content is not None:
                    ledger = {
                        "session_name": session_name,
                        "file_path": str(ledger_path),
                        "content": content,
                    }
                    handoff["ledger"] = ledger
            else:
                ledger = get_ledger_for_session(conn, session_name)
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
    query_id = hashlib.sha256(
        f"{question}{timestamp.isoformat()}".encode()
    ).hexdigest()[:12]

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
    parser = argparse.ArgumentParser(
        description="Search the Context Graph for relevant precedent"
    )
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument(
        "--type", choices=["handoffs", "plans", "continuity", "all"], default="all"
    )
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
    """Open database connection with standard pragmas."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _run_span_lookup(args: argparse.Namespace) -> None:
    """Handle --by-span-id CLI mode."""
    db_path = get_db_path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = _open_db(db_path)
    handoff = handle_span_id_lookup(conn, args.by_span_id, with_content=args.with_content)
    conn.close()

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


def _run_search(args: argparse.Namespace, query: str) -> None:
    """Handle regular search CLI mode."""
    db_path = get_db_path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run: uv run python scripts/artifact_index.py --all")
        return

    conn = _open_db(db_path)
    results = search_dispatch(conn, query, args.type, args.outcome, args.limit)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        formatted = format_results(results)
        print(formatted)

        if args.save:
            save_query(conn, query, formatted, results)
            print("\n[Query saved for compound learning]")

    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for artifact query CLI."""
    _enable_faulthandler()

    parser = _build_parser()
    args = parser.parse_args()

    if args.by_span_id:
        _run_span_lookup(args)
        return

    if not args.query:
        parser.print_help()
        return

    query = " ".join(args.query)
    _run_search(args, query)


if __name__ == "__main__":
    main()
