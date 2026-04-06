"""Tests for artifact_query.py — TDD+FP compliance.

Tests cover:
- Pure functions: escape_fts5_query, format_result_section, formatters, format_results
- DB search functions: search_handoffs, search_plans, search_continuity, search_past_queries
- DB lookup functions: get_handoff_by_span_id, get_ledger_for_session
- Dispatch: search_dispatch
- save_query
- handle_span_id_lookup (with filesystem mock)
- main() CLI integration
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "scripts" / "core" / "artifact_schema.sql"

from scripts.core.artifact_query import (
    STATUS_ICONS,
    _format_continuity,
    _format_handoffs,
    _format_past_queries,
    _format_plans,
    escape_fts5_query,
    format_result_section,
    format_results,
    get_db_path,
    get_handoff_by_span_id,
    get_ledger_for_session,
    handle_span_id_lookup,
    save_query,
    search_continuity,
    search_dispatch,
    search_handoffs,
    search_past_queries,
    search_plans,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_conn():
    """In-memory SQLite database using the production artifact schema."""
    conn = sqlite3.connect(":memory:")
    schema_sql = _SCHEMA_PATH.read_text()
    conn.executescript(schema_sql)
    return conn


@pytest.fixture
def populated_db(db_conn):
    """Database with sample data in all tables.

    Uses named columns matching the production schema. FTS is synced
    automatically via triggers defined in artifact_schema.sql.
    """
    # Handoffs (triggers auto-populate handoffs_fts)
    db_conn.execute(
        """INSERT INTO handoffs
           (id, session_name, task_number, file_path, task_summary, what_worked,
            what_failed, key_decisions, outcome, root_span_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "h1", "auth-session", 1,
            "thoughts/shared/handoffs/auth-session/task-1.yaml",
            "Implement OAuth login flow", "Token refresh worked",
            None, None, "SUCCEEDED", "span-abc123", "2026-01-01T00:00:00",
        ),
    )
    db_conn.execute(
        """INSERT INTO handoffs
           (id, session_name, task_number, file_path, task_summary, what_worked,
            what_failed, key_decisions, outcome, root_span_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "h2", "debug-session", 2,
            "thoughts/shared/handoffs/debug-session/task-2.yaml",
            "Fix database connection timeout", None,
            "Connection pool exhausted", None, "FAILED",
            "span-def456", "2026-01-02T00:00:00",
        ),
    )

    # Plans (triggers auto-populate plans_fts)
    db_conn.execute(
        """INSERT INTO plans
           (id, title, overview, approach, file_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            "p1", "Auth Redesign", "Redesign authentication system",
            "Use JWT with refresh tokens", "plans/auth.md", "2026-01-01",
        ),
    )

    # Continuity (triggers auto-populate continuity_fts)
    db_conn.execute(
        """INSERT INTO continuity
           (id, session_name, goal, key_learnings, key_decisions,
            state_done, state_now, state_next, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "c1", "auth-session", "Complete OAuth implementation",
            "JWT tokens work well", "Use refresh tokens",
            "Setup done", "In progress", "Deploy", "2026-01-01",
        ),
    )

    # Queries (triggers auto-populate queries_fts)
    db_conn.execute(
        """INSERT INTO queries
           (id, question, answer, was_helpful, handoffs_matched,
            plans_matched, continuity_matched)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("q1", "How does auth work?", "It uses JWT tokens", 1, "[]", "[]", "[]"),
    )

    db_conn.commit()
    return db_conn


# ===========================================================================
# Pure function tests
# ===========================================================================


class TestEscapeFts5Query:
    """Tests for escape_fts5_query — pure function."""

    def test_single_word(self):
        assert escape_fts5_query("hello") == '"hello"'

    def test_multiple_words_joined_with_or(self):
        result = escape_fts5_query("hello world")
        assert result == '"hello" OR "world"'

    def test_empty_string(self):
        result = escape_fts5_query("")
        assert result == ""

    def test_special_characters_escaped(self):
        result = escape_fts5_query('test"quote')
        assert '""' in result  # double-quote escaping

    def test_preserves_words_with_hyphens(self):
        result = escape_fts5_query("auth-session")
        assert result == '"auth-session"'


class TestGetDbPath:
    """Tests for get_db_path — pure function."""

    def test_default_path(self):
        from pathlib import Path
        result = get_db_path(None)
        assert result == Path(".claude/cache/artifact-index/context.db")

    def test_custom_path(self):
        from pathlib import Path
        result = get_db_path("/tmp/test.db")
        assert result == Path("/tmp/test.db")


class TestStatusIcons:
    """STATUS_ICONS dispatch table has expected keys."""

    def test_all_outcomes_present(self):
        expected = {"SUCCEEDED", "PARTIAL_PLUS", "PARTIAL_MINUS", "FAILED"}
        assert set(STATUS_ICONS.keys()) == expected


# ===========================================================================
# Formatter tests (pure functions)
# ===========================================================================


class TestFormatPastQueries:
    """Tests for _format_past_queries — pure function."""

    def test_formats_question_and_answer(self):
        items = [{"question": "How does X work?", "answer": "It uses Y"}]
        result = _format_past_queries(items)
        assert "## Previously Asked" in result
        assert "**Q:**" in result
        assert "**A:**" in result

    def test_truncates_long_text(self):
        items = [{"question": "x" * 200, "answer": "y" * 300}]
        result = _format_past_queries(items)
        # Question truncated to 100 chars
        assert "x" * 100 in result
        assert "x" * 101 not in result

    def test_empty_list(self):
        # _format_past_queries is only called when items is non-empty
        # but test defensive behavior
        result = _format_past_queries([])
        assert "## Previously Asked" in result


class TestFormatHandoffs:
    """Tests for _format_handoffs — pure function."""

    def test_formats_with_status_icon(self):
        items = [{
            "outcome": "SUCCEEDED", "session_name": "sess1",
            "task_number": 1, "task_summary": "Did stuff",
            "file_path": "path/to/file",
        }]
        result = _format_handoffs(items)
        assert "## Relevant Handoffs" in result
        assert "v" in result  # STATUS_ICONS["SUCCEEDED"]
        assert "sess1/task-1" in result

    def test_includes_what_worked_and_failed(self):
        items = [{
            "outcome": "FAILED", "session_name": "s",
            "task_number": 1, "task_summary": "x",
            "what_worked": "caching", "what_failed": "timeout",
            "file_path": "f",
        }]
        result = _format_handoffs(items)
        assert "**What worked:** caching" in result
        assert "**What failed:** timeout" in result

    def test_unknown_outcome_shows_question_mark(self):
        items = [{
            "outcome": "WEIRD", "session_name": "s",
            "task_number": 1, "task_summary": "x", "file_path": "f",
        }]
        result = _format_handoffs(items)
        assert "? s/task-1" in result

    def test_missing_optional_fields(self):
        items = [{
            "outcome": "SUCCEEDED", "session_name": "s",
            "task_number": 1, "task_summary": "x", "file_path": "f",
        }]
        result = _format_handoffs(items)
        assert "What worked" not in result
        assert "What failed" not in result


class TestFormatPlans:
    """Tests for _format_plans — pure function."""

    def test_formats_plan(self):
        items = [{"title": "My Plan", "overview": "Do things", "file_path": "plan.md"}]
        result = _format_plans(items)
        assert "## Relevant Plans" in result
        assert "### My Plan" in result
        assert "**Overview:** Do things" in result

    def test_missing_title_defaults(self):
        items = [{"overview": "x", "file_path": "f"}]
        result = _format_plans(items)
        assert "### Untitled" in result


class TestFormatContinuity:
    """Tests for _format_continuity — pure function."""

    def test_formats_session(self):
        items = [{
            "session_name": "my-session", "goal": "Build feature",
            "key_learnings": "Learned stuff",
        }]
        result = _format_continuity(items)
        assert "## Related Sessions" in result
        assert "### Session: my-session" in result
        assert "**Goal:** Build feature" in result
        assert "**Key learnings:** Learned stuff" in result

    def test_no_key_learnings(self):
        items = [{"session_name": "s", "goal": "g"}]
        result = _format_continuity(items)
        assert "Key learnings" not in result


class TestFormatResultSection:
    """Tests for format_result_section dispatch — pure function."""

    def test_dispatches_to_correct_formatter(self):
        items = [{"question": "Q?", "answer": "A"}]
        result = format_result_section("past_queries", items)
        assert "## Previously Asked" in result

    def test_empty_items_returns_empty(self):
        assert format_result_section("handoffs", []) == ""

    def test_unknown_type_returns_empty(self):
        assert format_result_section("nonexistent", [{"data": 1}]) == ""

    def test_all_known_types_dispatch(self):
        for section_type in ("past_queries", "handoffs", "plans", "continuity"):
            # Should not raise — formatters handle missing keys with .get()
            result = format_result_section(section_type, [{}])
            assert isinstance(result, str)


class TestFormatResults:
    """Tests for format_results — pure function."""

    def test_empty_results(self):
        result = format_results({})
        assert "No relevant precedent found." in result

    def test_all_empty_lists(self):
        result = format_results({"handoffs": [], "plans": [], "continuity": []})
        assert "No relevant precedent found." in result

    def test_formats_handoffs_section(self):
        results = {
            "handoffs": [{
                "outcome": "SUCCEEDED", "session_name": "s",
                "task_number": 1, "task_summary": "summary", "file_path": "f",
            }],
        }
        result = format_results(results)
        assert "## Relevant Handoffs" in result
        assert "summary" in result

    def test_formats_plans_section(self):
        results = {"plans": [{"title": "P", "overview": "O", "file_path": "f"}]}
        result = format_results(results)
        assert "## Relevant Plans" in result

    def test_formats_continuity_section(self):
        results = {"continuity": [{"session_name": "s", "goal": "g"}]}
        result = format_results(results)
        assert "## Related Sessions" in result

    def test_formats_past_queries_section(self):
        results = {"past_queries": [{"question": "Q?", "answer": "A"}]}
        result = format_results(results)
        assert "## Previously Asked" in result


# ===========================================================================
# DB search function tests
# ===========================================================================


class TestSearchHandoffs:
    """Tests for search_handoffs — DB function."""

    def test_finds_matching_handoff(self, populated_db):
        results = search_handoffs(populated_db, "OAuth")
        assert len(results) >= 1
        assert results[0]["session_name"] == "auth-session"

    def test_filters_by_outcome(self, populated_db):
        results = search_handoffs(populated_db, "connection", outcome="FAILED")
        assert len(results) >= 1
        assert all(r["outcome"] == "FAILED" for r in results)

    def test_respects_limit(self, populated_db):
        results = search_handoffs(populated_db, "session", limit=1)
        assert len(results) <= 1

    def test_no_results_for_unmatched_query(self, populated_db):
        results = search_handoffs(populated_db, "zzzznonexistent")
        assert results == []


class TestSearchPlans:
    """Tests for search_plans — DB function."""

    def test_finds_matching_plan(self, populated_db):
        results = search_plans(populated_db, "authentication")
        assert len(results) >= 1
        assert results[0]["title"] == "Auth Redesign"

    def test_no_results(self, populated_db):
        results = search_plans(populated_db, "zzzznothing")
        assert results == []


class TestSearchContinuity:
    """Tests for search_continuity — DB function."""

    def test_finds_matching_continuity(self, populated_db):
        results = search_continuity(populated_db, "OAuth")
        assert len(results) >= 1
        assert results[0]["session_name"] == "auth-session"

    def test_no_results(self, populated_db):
        results = search_continuity(populated_db, "zzzznothing")
        assert results == []


class TestSearchPastQueries:
    """Tests for search_past_queries — DB function."""

    def test_finds_matching_query(self, populated_db):
        results = search_past_queries(populated_db, "auth")
        assert len(results) >= 1
        assert "JWT" in results[0]["answer"]

    def test_no_results(self, populated_db):
        results = search_past_queries(populated_db, "zzzznothing")
        assert results == []


# ===========================================================================
# DB lookup function tests
# ===========================================================================


class TestGetHandoffBySpanId:
    """Tests for get_handoff_by_span_id — DB function."""

    def test_finds_by_span_id(self, populated_db):
        result = get_handoff_by_span_id(populated_db, "span-abc123")
        assert result is not None
        assert result["session_name"] == "auth-session"

    def test_returns_none_for_missing(self, populated_db):
        result = get_handoff_by_span_id(populated_db, "nonexistent")
        assert result is None

    def test_returns_most_recent_for_shared_span_id(self, populated_db):
        """Multiple handoffs with same root_span_id returns newest."""
        populated_db.execute(
            """INSERT INTO handoffs
               (id, session_name, task_number, file_path, task_summary,
                outcome, what_worked, root_span_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "h3", "auth-session", 3,
                "thoughts/shared/handoffs/auth-session/task-3.yaml",
                "Follow-up task", "SUCCEEDED", "All good",
                "span-abc123", "2026-01-05T00:00:00",
            ),
        )
        populated_db.commit()

        result = get_handoff_by_span_id(populated_db, "span-abc123")
        assert result is not None
        assert result["id"] == "h3"  # newer one
        assert result["task_number"] == 3

    def test_picks_newest_with_mixed_timestamp_formats(self, populated_db):
        """Regression: isoformat (T separator) vs CURRENT_TIMESTAMP (space separator)."""
        # h1 has "2026-01-01T00:00:00" (isoformat)
        # Insert one with space-separated timestamp that is actually newer
        populated_db.execute(
            """INSERT INTO handoffs
               (id, session_name, task_number, file_path, task_summary,
                outcome, root_span_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "h4", "auth-session", 4, "f4", "Newer task",
                "SUCCEEDED", "span-abc123", "2026-01-10 12:00:00",
            ),
        )
        populated_db.commit()

        result = get_handoff_by_span_id(populated_db, "span-abc123")
        assert result is not None
        assert result["id"] == "h4"  # space-format but actually newer

    def test_includes_key_decisions_field(self, populated_db):
        """Regression: key_decisions must be in returned dict."""
        result = get_handoff_by_span_id(populated_db, "span-abc123")
        assert result is not None
        assert "key_decisions" in result


class TestGetLedgerForSession:
    """Tests for get_ledger_for_session — DB function."""

    def test_finds_by_session_name(self, populated_db):
        result = get_ledger_for_session(populated_db, "auth-session")
        assert result is not None
        assert result["goal"] == "Complete OAuth implementation"

    def test_returns_none_for_missing(self, populated_db):
        result = get_ledger_for_session(populated_db, "nonexistent")
        assert result is None


# ===========================================================================
# Dispatch tests
# ===========================================================================


class TestSearchDispatch:
    """Tests for search_dispatch — coordination function."""

    def test_all_type_searches_everything(self, populated_db):
        results = search_dispatch(populated_db, "auth", search_type="all")
        assert "past_queries" in results
        assert "handoffs" in results
        assert "plans" in results
        assert "continuity" in results

    def test_handoffs_type_only(self, populated_db):
        results = search_dispatch(populated_db, "auth", search_type="handoffs")
        assert "handoffs" in results
        assert "plans" not in results
        assert "continuity" not in results
        # past_queries always included
        assert "past_queries" in results

    def test_plans_type_only(self, populated_db):
        results = search_dispatch(populated_db, "auth", search_type="plans")
        assert "plans" in results
        assert "handoffs" not in results

    def test_continuity_type_only(self, populated_db):
        results = search_dispatch(populated_db, "auth", search_type="continuity")
        assert "continuity" in results
        assert "handoffs" not in results

    def test_unknown_type_returns_only_past_queries(self, populated_db):
        results = search_dispatch(populated_db, "auth", search_type="unknown")
        assert "past_queries" in results
        assert len(results) == 1


# ===========================================================================
# save_query tests
# ===========================================================================


class TestSaveQuery:
    """Tests for save_query — DB write function."""

    def test_saves_query_to_database(self, populated_db):
        matches = {
            "handoffs": [{"id": "h1"}],
            "plans": [{"id": "p1"}],
            "continuity": [{"id": "c1"}],
        }
        save_query(populated_db, "test question", "test answer", matches)

        cursor = populated_db.execute("SELECT * FROM queries WHERE question = 'test question'")
        row = cursor.fetchone()
        assert row is not None

    def test_serializes_match_ids_as_json(self, populated_db):
        matches = {
            "handoffs": [{"id": "h1"}, {"id": "h2"}],
            "plans": [],
            "continuity": [{"id": "c1"}],
        }
        save_query(populated_db, "q", "a", matches)

        cursor = populated_db.execute(
            "SELECT handoffs_matched, plans_matched, continuity_matched "
            "FROM queries WHERE question = 'q'"
        )
        row = cursor.fetchone()
        assert json.loads(row[0]) == ["h1", "h2"]
        assert json.loads(row[1]) == []
        assert json.loads(row[2]) == ["c1"]

    def test_empty_matches(self, populated_db):
        save_query(populated_db, "q", "a", {})

        cursor = populated_db.execute("SELECT * FROM queries WHERE question = 'q'")
        row = cursor.fetchone()
        assert row is not None


# ===========================================================================
# handle_span_id_lookup tests
# ===========================================================================


class TestHandleSpanIdLookup:
    """Tests for handle_span_id_lookup — coordination + I/O function."""

    def test_returns_none_for_missing_span(self, populated_db):
        result = handle_span_id_lookup(populated_db, "nonexistent")
        assert result is None

    def test_returns_handoff_without_content(self, populated_db):
        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=False)
        assert result is not None
        assert result["session_name"] == "auth-session"
        assert "content" not in result

    def test_with_content_reads_file(self, populated_db, tmp_path):
        # Create a fake handoff file
        handoff_file = tmp_path / "task-1.yaml"
        handoff_file.write_text("handoff content here")

        # Update the DB to point to our tmp file
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (str(handoff_file),),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert result["content"] == "handoff content here"

    def test_with_content_missing_file_no_crash(self, populated_db):
        # file_path points to a non-existent file — should not crash
        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert "content" not in result

    def test_with_content_missing_file_preserves_key_decisions(self, populated_db):
        """Regression: key_decisions available even when file is missing."""
        populated_db.execute(
            "UPDATE handoffs SET key_decisions = 'Use JWT' WHERE id = 'h1'"
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert result["key_decisions"] == "Use JWT"


# ===========================================================================
# main() CLI tests
# ===========================================================================


class TestMain:
    """Tests for main() CLI entry point."""

    def test_no_args_prints_help(self, capsys):
        from scripts.core.artifact_query import main

        with patch("sys.argv", ["artifact_query.py"]):
            main()
        captured = capsys.readouterr()
        # No query and no --by-span-id prints help
        assert "Search the Context Graph" in captured.out or captured.out == ""

    def test_by_span_id_json_output(self, populated_db, capsys, tmp_path):
        from scripts.core.artifact_query import main

        db_file = tmp_path / "test.db"

        file_conn = sqlite3.connect(str(db_file))
        file_conn.executescript(_SCHEMA_PATH.read_text())
        file_conn.execute(
            """INSERT INTO handoffs
               (id, session_name, task_number, file_path, task_summary,
                outcome, root_span_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("h1", "sess", 1, "f", "sum", "SUCCEEDED", "span-x", "2026-01-01"),
        )
        file_conn.commit()
        file_conn.close()

        with patch("sys.argv", [
            "artifact_query.py", "--by-span-id", "span-x", "--json", "--db", str(db_file),
        ]):
            main()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["session_name"] == "sess"

    def test_search_json_output(self, tmp_path, capsys):
        from scripts.core.artifact_query import main

        db_file = tmp_path / "test.db"

        file_conn = sqlite3.connect(str(db_file))
        file_conn.executescript(_SCHEMA_PATH.read_text())
        file_conn.commit()
        file_conn.close()

        with patch("sys.argv", [
            "artifact_query.py", "test", "query", "--json", "--db", str(db_file),
        ]):
            main()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict)

    def test_missing_db_prints_error(self, capsys, tmp_path):
        from scripts.core.artifact_query import main

        with patch("sys.argv", [
            "artifact_query.py", "query", "--db", str(tmp_path / "nonexistent.db"),
        ]):
            main()

        captured = capsys.readouterr()
        assert "Database not found" in captured.out
