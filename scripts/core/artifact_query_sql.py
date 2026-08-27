"""Per-backend SQL for artifact_query.py (issue #282).

The artifact indexer writes to PostgreSQL when it is the active backend, but
the PostgreSQL tables were designed upstream with a different shape from the
indexer's own SQLite schema (``artifact_schema.sql``):

- ``handoffs``: ``goal`` instead of ``task_summary``; no ``task_number`` column
  (derived here from the ``task-NN`` filename convention); ``id`` is a uuid.
- ``plans`` / ``continuity``: ``indexed_at`` only, no ``created_at``.
- Full-text search is ``tsvector``/``tsquery``, not FTS5 ``MATCH``.

The SQLite statements are the original ones, unchanged. Both backends return
the same column names so formatters and callers are backend-agnostic.

Placeholders: SQLite uses ``?``, PostgreSQL (psycopg2) uses ``%s``. Search
statements bind the query text twice on PostgreSQL (rank + filter) followed by
the optional outcome and the limit; see ``artifact_query.py`` for param order.
"""

BACKENDS: tuple[str, ...] = ("sqlite", "postgres")

# --- PostgreSQL document expressions -----------------------------------------
#
# Built from IMMUTABLE functions only (COALESCE, textcat, to_tsvector with an
# explicit regconfig) so the identical expression can back a GIN expression
# index. ``concat_ws`` is STABLE and cannot be indexed. The planner only uses an
# expression index when the query expression matches it exactly, so the search
# statements and ``PG_FTS_INDEX_DDL`` are generated from the same source.

HANDOFF_DOC_COLUMNS: tuple[str, ...] = ("goal", "what_worked", "what_failed", "key_decisions")
PLAN_DOC_COLUMNS: tuple[str, ...] = ("title", "overview", "approach", "phases")
CONTINUITY_DOC_COLUMNS: tuple[str, ...] = ("goal", "key_learnings", "key_decisions", "state_now")


def pg_document_expression(columns: tuple[str, ...], prefix: str = "") -> str:
    """Return the tsvector expression over ``columns`` (optionally ``alias.``-prefixed)."""
    parts = " || ' ' || ".join(f"COALESCE({prefix}{c}, '')" for c in columns)
    return f"to_tsvector('english', {parts})"


_PG_HANDOFF_DOC = pg_document_expression(HANDOFF_DOC_COLUMNS, "h.")
_PG_PLAN_DOC = pg_document_expression(PLAN_DOC_COLUMNS, "p.")
_PG_CONTINUITY_DOC = pg_document_expression(CONTINUITY_DOC_COLUMNS, "c.")
# Bounded digit run: an unbounded [0-9]+ on a poisoned path would overflow ::int
# and abort the whole statement (aegis review, #282).
_PG_TASK_NUMBER_EXPR = "substring(h.file_path from 'task-([0-9]{1,6})')::int"
_PG_TASK_NUMBER = f"{_PG_TASK_NUMBER_EXPR} AS task_number"

# Idempotent DDL the indexer applies on init so the searches above are
# index-backed. Searches still work (as sequential scans) without these.
PG_FTS_INDEX_DDL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_handoffs_search_fts ON handoffs "
    f"USING gin({pg_document_expression(HANDOFF_DOC_COLUMNS)})",
    "CREATE INDEX IF NOT EXISTS idx_plans_search_fts ON plans "
    f"USING gin({pg_document_expression(PLAN_DOC_COLUMNS)})",
    "CREATE INDEX IF NOT EXISTS idx_continuity_search_fts ON continuity "
    f"USING gin({pg_document_expression(CONTINUITY_DOC_COLUMNS)})",
)

_SQL: dict[str, dict[str, str]] = {
    "sqlite": {
        "search_handoffs": """
        SELECT h.id, h.session_name, h.task_number, h.task_summary,
               h.what_worked, h.what_failed, h.key_decisions,
               h.outcome, h.file_path, h.created_at,
               handoffs_fts.rank as score
        FROM handoffs_fts
        JOIN handoffs h ON handoffs_fts.rowid = h.rowid
        WHERE handoffs_fts MATCH ?
    """,
        "search_handoffs_outcome_filter": " AND h.outcome = ?",
        "search_handoffs_tail": " ORDER BY rank LIMIT ?",
        "search_plans": """
        SELECT p.id, p.title, p.overview, p.approach, p.file_path, p.created_at,
               plans_fts.rank as score
        FROM plans_fts
        JOIN plans p ON plans_fts.rowid = p.rowid
        WHERE plans_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """,
        "search_continuity": """
        SELECT c.id, c.session_name, c.goal, c.key_learnings, c.key_decisions,
               c.state_now, c.created_at,
               continuity_fts.rank as score
        FROM continuity_fts
        JOIN continuity c ON continuity_fts.rowid = c.rowid
        WHERE continuity_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """,
        "search_past_queries": """
        SELECT q.id, q.question, q.answer, q.was_helpful, q.created_at,
               queries_fts.rank as score
        FROM queries_fts
        JOIN queries q ON queries_fts.rowid = q.rowid
        WHERE queries_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """,
        "get_handoff_by_span_id": """
        SELECT id, session_name, task_number, task_summary,
               outcome, what_worked, what_failed, key_decisions,
               file_path, root_span_id, created_at
        FROM handoffs
        WHERE root_span_id = ?
        ORDER BY datetime(created_at) DESC, task_number DESC, rowid DESC
        LIMIT 1
    """,
        "get_ledger_for_session": """
        SELECT id, session_name, goal, key_learnings, key_decisions,
               state_done, state_now, state_next, created_at
        FROM continuity
        WHERE session_name = ?
        ORDER BY created_at DESC
        LIMIT 1
    """,
    },
    "postgres": {
        "search_handoffs": f"""
        SELECT h.id::text AS id, h.session_name,
               {_PG_TASK_NUMBER},
               h.goal AS task_summary,
               h.what_worked, h.what_failed, h.key_decisions,
               h.outcome, h.file_path, h.created_at,
               ts_rank({_PG_HANDOFF_DOC}, websearch_to_tsquery('english', %s)) AS score
        FROM handoffs h
        WHERE {_PG_HANDOFF_DOC} @@ websearch_to_tsquery('english', %s)
    """,
        "search_handoffs_outcome_filter": " AND h.outcome = %s",
        "search_handoffs_tail": " ORDER BY score DESC, h.created_at DESC LIMIT %s",
        "search_plans": f"""
        SELECT p.id, p.title, p.overview, p.approach, p.file_path,
               p.indexed_at AS created_at,
               ts_rank({_PG_PLAN_DOC}, websearch_to_tsquery('english', %s)) AS score
        FROM plans p
        WHERE {_PG_PLAN_DOC} @@ websearch_to_tsquery('english', %s)
        ORDER BY score DESC, p.indexed_at DESC
        LIMIT %s
    """,
        "search_continuity": f"""
        SELECT c.id, c.session_name, c.goal, c.key_learnings, c.key_decisions,
               c.state_now, c.indexed_at AS created_at,
               ts_rank({_PG_CONTINUITY_DOC}, websearch_to_tsquery('english', %s)) AS score
        FROM continuity c
        WHERE {_PG_CONTINUITY_DOC} @@ websearch_to_tsquery('english', %s)
        ORDER BY score DESC, c.indexed_at DESC
        LIMIT %s
    """,
        "get_handoff_by_span_id": f"""
        SELECT h.id::text AS id, h.session_name,
               {_PG_TASK_NUMBER},
               h.goal AS task_summary,
               h.outcome, h.what_worked, h.what_failed, h.key_decisions,
               h.file_path, h.root_span_id, h.created_at
        FROM handoffs h
        WHERE h.root_span_id = %s
        ORDER BY h.created_at DESC NULLS LAST,
                 {_PG_TASK_NUMBER_EXPR} DESC NULLS LAST,
                 h.file_path DESC
        LIMIT 1
    """,
        "get_ledger_for_session": """
        SELECT id, session_name, goal, key_learnings, key_decisions,
               state_done, state_now, state_next, indexed_at AS created_at
        FROM continuity
        WHERE session_name = %s
        ORDER BY indexed_at DESC NULLS LAST
        LIMIT 1
    """,
    },
}


def sql_for(backend: str, name: str) -> str:
    """Return the SQL statement ``name`` for ``backend``.

    Raises ``KeyError`` for an unknown backend or statement name — a programming
    error, not a runtime condition.
    """
    return _SQL[backend][name]
