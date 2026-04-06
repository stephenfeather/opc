#!/usr/bin/env python3
"""
USAGE: artifact_index.py [--handoffs] [--plans] [--continuity] [--all] [--file PATH] [--db PATH]

Index handoffs, plans, and continuity ledgers into the Context Graph database.

Examples:
    # Index all handoffs
    uv run python scripts/artifact_index.py --handoffs

    # Index everything
    uv run python scripts/artifact_index.py --all

    # Index a single handoff file (fast, for hooks)
    uv run python scripts/artifact_index.py --file thoughts/shared/handoffs/session/task-01.md

    # Use custom database path
    uv run python scripts/artifact_index.py --all --db /path/to/context.db
"""

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

try:
    from scripts.core.artifact_index_core import (
        adapt_for_postgres as _adapt_for_postgres,
    )
    from scripts.core.artifact_index_core import (
        classify_file,
        parse_continuity_content,
        parse_handoff_content,
        parse_handoff_yaml_content,
        parse_plan_content,
    )
except ModuleNotFoundError:
    from artifact_index_core import (  # type: ignore[no-redef]
        adapt_for_postgres as _adapt_for_postgres,
    )
    from artifact_index_core import (  # type: ignore[no-redef]
        classify_file,
        parse_continuity_content,
        parse_handoff_content,
        parse_handoff_yaml_content,
        parse_plan_content,
    )


def _bootstrap() -> None:
    """Initialize faulthandler and load .env files. Call from main() only."""
    try:
        import faulthandler

        log_path = Path.home() / ".claude" / "logs" / "opc_crash.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        faulthandler.enable(
            file=open(log_path, "a"),  # noqa: SIM115
            all_threads=True,
        )
    except OSError:
        pass  # Best-effort: crash logging not critical

    try:
        from dotenv import load_dotenv

        global_env = Path.home() / ".claude" / ".env"
        if global_env.exists():
            load_dotenv(global_env)
        opc_env = Path(__file__).parent.parent.parent / ".env"
        if opc_env.exists():
            load_dotenv(opc_env, override=True)
    except ImportError:
        pass


# =============================================================================
# DATABASE BACKEND SELECTION
# =============================================================================

def get_postgres_url() -> str | None:
    """Get PostgreSQL URL from environment variables (canonical first)."""
    return os.environ.get("CONTINUOUS_CLAUDE_DB_URL") or os.environ.get("DATABASE_URL")


def use_postgres() -> bool:
    """Check if PostgreSQL should be used as backend."""
    url = get_postgres_url()
    if not url:
        return False
    try:
        import psycopg2  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# SQLITE BACKEND
# =============================================================================

def get_db_path(custom_path: str | None = None) -> Path:
    """Get SQLite database path, creating directory if needed."""
    if custom_path:
        path = Path(custom_path)
    else:
        path = Path(".claude/cache/artifact-index/context.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_sqlite(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    schema_path = Path(__file__).parent / "artifact_schema.sql"
    if schema_path.exists():
        conn.executescript(schema_path.read_text())
    # Idempotent migration for existing databases missing session_uuid
    cols = {row[1] for row in conn.execute("PRAGMA table_info(handoffs)").fetchall()}
    if "session_uuid" not in cols:
        conn.execute("ALTER TABLE handoffs ADD COLUMN session_uuid TEXT")
    return conn


# =============================================================================
# POSTGRESQL BACKEND
# =============================================================================

def pg_connect():
    """Connect to PostgreSQL."""
    import psycopg2
    return psycopg2.connect(get_postgres_url())


def init_postgres():
    """Initialize PostgreSQL connection and ensure schema exists."""
    conn = pg_connect()
    cur = conn.cursor()

    # Create handoffs table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS handoffs (
            id TEXT PRIMARY KEY,
            session_name TEXT,
            session_uuid TEXT,
            task_number INTEGER,
            file_path TEXT,
            task_summary TEXT,
            what_worked TEXT,
            what_failed TEXT,
            key_decisions TEXT,
            files_modified TEXT,
            outcome TEXT DEFAULT 'UNKNOWN',
            outcome_notes TEXT,
            root_span_id TEXT,
            turn_span_id TEXT,
            session_id TEXT,
            braintrust_session_id TEXT,
            created_at TIMESTAMP,
            indexed_at TIMESTAMP DEFAULT NOW(),
            goal TEXT
        )
    """)

    # Idempotent migration for existing databases missing session_uuid
    cur.execute("""
        ALTER TABLE handoffs ADD COLUMN IF NOT EXISTS session_uuid TEXT
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_handoffs_session_uuid ON handoffs(session_uuid)
    """)

    # Create plans table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            title TEXT,
            file_path TEXT,
            overview TEXT,
            approach TEXT,
            phases TEXT,
            constraints TEXT,
            indexed_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # Create continuity table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS continuity (
            id TEXT PRIMARY KEY,
            session_name TEXT,
            goal TEXT,
            state_done TEXT,
            state_now TEXT,
            state_next TEXT,
            key_learnings TEXT,
            key_decisions TEXT,
            snapshot_reason TEXT,
            indexed_at TIMESTAMP DEFAULT NOW()
        )
    """)

    conn.commit()
    return conn


# =============================================================================
# UNIFIED DATABASE INTERFACE
# =============================================================================

class DatabaseConnection:
    """Unified interface for SQLite and PostgreSQL."""

    def __init__(self, use_pg: bool = False, sqlite_path: Path | None = None):
        self.use_pg = use_pg
        self.sqlite_path = sqlite_path
        self.conn = None
        self.cur = None

    def __enter__(self):
        if self.use_pg:
            self.conn = init_postgres()
            self.cur = self.conn.cursor()
        else:
            self.conn = init_sqlite(self.sqlite_path or get_db_path())
            self.cur = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def execute(self, sql: str, params: tuple = ()):
        """Execute SQL with backend-appropriate placeholder conversion."""
        if self.cur is None:
            raise RuntimeError("Database connection not initialized")
        if self.use_pg:
            # Convert ? to %s for PostgreSQL
            sql = sql.replace("?", "%s")
            # Convert INSERT OR REPLACE to ON CONFLICT DO UPDATE
            if "INSERT OR REPLACE INTO" in sql:
                sql = self._convert_upsert(sql)
        self.cur.execute(sql, params)

    def _convert_upsert(self, sql: str) -> str:
        """Convert SQLite INSERT OR REPLACE to PostgreSQL ON CONFLICT."""
        # Extract table name and columns
        match = re.match(r"INSERT OR REPLACE INTO (\w+)\s*\(([^)]+)\)", sql, re.IGNORECASE)
        if not match:
            return sql

        _table = match.group(1)
        columns = [c.strip() for c in match.group(2).split(",")]

        # Build ON CONFLICT clause
        non_pk_cols = [c for c in columns if c != "id"]
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk_cols)

        # Replace INSERT OR REPLACE with INSERT ... ON CONFLICT
        sql = sql.replace("INSERT OR REPLACE INTO", "INSERT INTO")
        sql = sql.rstrip().rstrip(";")
        sql += f" ON CONFLICT (id) DO UPDATE SET {update_clause}"

        return sql

    def commit(self):
        if self.conn:
            self.conn.commit()


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize database with schema (legacy compatibility)."""
    return init_sqlite(db_path)


def db_execute(conn, sql: str, params: tuple = (), table_hint: str = ""):
    """Execute SQL on either SQLite or PostgreSQL connection.

    Handles the difference between SQLite (conn.execute) and
    PostgreSQL (conn.cursor().execute) interfaces.
    """
    # Check if this is a PostgreSQL connection
    is_pg = hasattr(conn, 'cursor') and not isinstance(conn, sqlite3.Connection)

    if is_pg:
        # PostgreSQL: use adapted queries for existing schema
        sql, params = _adapt_for_postgres(sql, params, table_hint)
        cur = conn.cursor()
        cur.execute(sql, params)
        cur.close()
    else:
        # SQLite: use directly
        conn.execute(sql, params)



# --- Helpers imported from artifact_index_core ---
# Module-level imports provide database adaptation, file classification,
# and parse_*_content helpers; the remaining I/O wrappers are defined below.


def parse_handoff(file_path: Path) -> dict:
    """Parse a handoff markdown file into structured data."""
    raw_content = file_path.read_text()
    result = parse_handoff_content(raw_content, file_path)
    if not result["created_at"]:
        result["created_at"] = datetime.now().isoformat()
    return result




def parse_handoff_yaml(file_path: Path) -> dict:
    """Parse a handoff YAML file into structured data."""
    raw_content = file_path.read_text()
    result = parse_handoff_yaml_content(raw_content, file_path)
    if not result["created_at"]:
        result["created_at"] = datetime.now().isoformat()
    return result




def index_handoffs(conn, base_path: Path = Path("thoughts/shared/handoffs")):
    """Index all handoffs (.md and .yaml) into the database."""
    if not base_path.exists():
        print(f"Handoffs directory not found: {base_path}")
        return 0

    count = 0
    # Collect both markdown and YAML handoff files
    handoff_files = list(base_path.rglob("*.md"))
    handoff_files.extend(base_path.rglob("*.yaml"))
    handoff_files.extend(base_path.rglob("*.yml"))

    for handoff_file in handoff_files:
        try:
            if handoff_file.suffix in (".yaml", ".yml"):
                data = parse_handoff_yaml(handoff_file)
            else:
                data = parse_handoff(handoff_file)
            _index_handoff(conn, data)
            count += 1
        except Exception as e:
            print(f"Error indexing {handoff_file}: {e}")

    conn.commit()
    print(f"Indexed {count} handoffs")
    return count


def parse_plan(file_path: Path) -> dict:
    """Parse a plan markdown file into structured data."""
    content = file_path.read_text()
    return parse_plan_content(content, file_path)


def index_plans(conn, base_path: Path = Path("thoughts/shared/plans")):
    """Index all plans into the database."""
    if not base_path.exists():
        print(f"Plans directory not found: {base_path}")
        return 0

    count = 0
    for plan_file in base_path.glob("*.md"):
        try:
            data = parse_plan(plan_file)
            db_execute(
                conn,
                """
                INSERT OR REPLACE INTO plans
                (id, title, file_path, overview, approach, phases, constraints)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["id"],
                    data["title"],
                    data["file_path"],
                    data["overview"],
                    data["approach"],
                    data["phases"],
                    data["constraints"],
                ),
            )
            count += 1
        except Exception as e:
            print(f"Error indexing {plan_file}: {e}")

    conn.commit()
    print(f"Indexed {count} plans")
    return count


def parse_continuity(file_path: Path) -> dict:
    """Parse a continuity ledger into structured data."""
    content = file_path.read_text()
    return parse_continuity_content(content, file_path)


def index_continuity(conn, base_path: Path = Path(".")):
    """Index all continuity ledgers into the database."""
    count = 0
    for ledger_file in base_path.glob("CONTINUITY_CLAUDE-*.md"):
        try:
            data = parse_continuity(ledger_file)
            db_execute(
                conn,
                """
                INSERT OR REPLACE INTO continuity
                (id, session_name, goal, state_done, state_now, state_next,
                 key_learnings, key_decisions, snapshot_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["id"],
                    data["session_name"],
                    data["goal"],
                    data["state_done"],
                    data["state_now"],
                    data["state_next"],
                    data["key_learnings"],
                    data["key_decisions"],
                    data["snapshot_reason"],
                ),
            )
            count += 1
        except Exception as e:
            print(f"Error indexing {ledger_file}: {e}")

    conn.commit()
    print(f"Indexed {count} continuity ledgers")
    return count


def _index_handoff(conn, data: dict) -> None:
    """Write handoff data to database."""
    db_execute(
        conn,
        """
        INSERT OR REPLACE INTO handoffs
        (id, session_name, session_uuid, task_number, file_path, task_summary,
         what_worked, what_failed, key_decisions, files_modified, outcome,
         root_span_id, turn_span_id, session_id, braintrust_session_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            data["id"],
            data["session_name"],
            data["session_uuid"],
            data["task_number"],
            data["file_path"],
            data["task_summary"],
            data["what_worked"],
            data["what_failed"],
            data["key_decisions"],
            data["files_modified"],
            data["outcome"],
            data["root_span_id"],
            data["turn_span_id"],
            data["session_id"],
            data["braintrust_session_id"],
            data["created_at"],
        ),
    )


def _index_plan(conn, data: dict) -> None:
    """Write plan data to database."""
    db_execute(
        conn,
        """
        INSERT OR REPLACE INTO plans
        (id, title, file_path, overview, approach, phases, constraints)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            data["id"],
            data["title"],
            data["file_path"],
            data["overview"],
            data["approach"],
            data["phases"],
            data["constraints"],
        ),
    )


def _index_continuity(conn, data: dict) -> None:
    """Write continuity data to database."""
    db_execute(
        conn,
        """
        INSERT OR REPLACE INTO continuity
        (id, session_name, goal, state_done, state_now, state_next,
         key_learnings, key_decisions, snapshot_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            data["id"],
            data["session_name"],
            data["goal"],
            data["state_done"],
            data["state_now"],
            data["state_next"],
            data["key_learnings"],
            data["key_decisions"],
            data["snapshot_reason"],
        ),
    )


# Dispatch table for file type -> (parser, db_writer, label)
_INDEX_DISPATCH = {
    "handoff": (parse_handoff, _index_handoff, "handoff"),
    "handoff_yaml": (parse_handoff_yaml, _index_handoff, "handoff"),
    "plan": (parse_plan, _index_plan, "plan"),
    "continuity": (parse_continuity, _index_continuity, "continuity"),
}


def index_single_file(conn, file_path: Path) -> bool:
    """Index a single file based on its location/type.

    Returns True if indexed successfully, False otherwise.
    """
    file_path = Path(file_path).resolve()
    file_type = classify_file(file_path)

    if file_type is None:
        print(f"Unknown file type, skipping: {file_path}")
        return False

    parser, writer, label = _INDEX_DISPATCH[file_type]
    try:
        data = parser(file_path)
        writer(conn, data)
        conn.commit()
        print(f"Indexed {label}: {file_path.name}")
        return True
    except Exception as e:
        print(f"Error indexing {label} {file_path}: {e}")
        return False


def main() -> int:
    """Entry point for artifact indexing CLI."""
    _bootstrap()

    parser = argparse.ArgumentParser(description="Index context graph artifacts")
    parser.add_argument("--handoffs", action="store_true", help="Index handoffs")
    parser.add_argument("--plans", action="store_true", help="Index plans")
    parser.add_argument("--continuity", action="store_true", help="Index continuity ledgers")
    parser.add_argument("--all", action="store_true", help="Index everything")
    parser.add_argument("--file", type=str, help="Index a single file (fast, for hooks)")
    parser.add_argument("--db", type=str, help="Custom database path (SQLite only)")

    args = parser.parse_args()

    # Determine backend
    using_pg = use_postgres() and not args.db  # Custom db path forces SQLite
    db_type = "PostgreSQL" if using_pg else "SQLite"

    # Handle single file indexing (fast path for hooks)
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return 1

        if using_pg:
            conn = init_postgres()
        else:
            conn = init_sqlite(get_db_path(args.db))

        success = index_single_file(conn, file_path)
        conn.close()
        return 0 if success else 1

    if not any([args.handoffs, args.plans, args.continuity, args.all]):
        parser.print_help()
        return 0

    # Initialize connection
    if using_pg:
        conn = init_postgres()
        pg_url = get_postgres_url() or ""
        # Redact credentials from URL before printing
        safe_url = re.sub(r"://[^@]+@", "://***@", pg_url)
        print(f"Using database: {db_type} ({safe_url})")
    else:
        db_path = get_db_path(args.db)
        conn = init_sqlite(db_path)
        print(f"Using database: {db_type} ({db_path})")

    if args.all or args.handoffs:
        index_handoffs(conn)

    if args.all or args.plans:
        index_plans(conn)

    if args.all or args.continuity:
        index_continuity(conn)

    # FTS5 operations are SQLite-specific
    if not using_pg:
        print("Rebuilding FTS5 indexes...")
        conn.execute("INSERT INTO handoffs_fts(handoffs_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO plans_fts(plans_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO continuity_fts(continuity_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO queries_fts(queries_fts) VALUES('rebuild')")

        # Configure BM25 column weights
        conn.execute(
            "INSERT OR REPLACE INTO handoffs_fts(handoffs_fts, rank) "
            "VALUES('rank', 'bm25(10.0, 5.0, 3.0, 3.0, 1.0)')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO plans_fts(plans_fts, rank) "
            "VALUES('rank', 'bm25(10.0, 5.0, 3.0, 3.0, 1.0)')"
        )

        print("Optimizing indexes...")
        conn.execute("INSERT INTO handoffs_fts(handoffs_fts) VALUES('optimize')")
        conn.execute("INSERT INTO plans_fts(plans_fts) VALUES('optimize')")
        conn.execute("INSERT INTO continuity_fts(continuity_fts) VALUES('optimize')")
        conn.execute("INSERT INTO queries_fts(queries_fts) VALUES('optimize')")

    conn.commit()
    conn.close()
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
