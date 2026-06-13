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
    generate_uuid7,
    get_db_path,
    get_handoff_by_span_id,
    get_ledger_for_session,
    handle_span_id_lookup,
    is_safe_artifact_path,
    is_safe_dir_root,
    is_safe_session_name,
    read_text_nofollow,
    safe_artifact_read_path,
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
    yield conn
    conn.close()


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

    def test_with_content_reads_file(self, populated_db, tmp_path, monkeypatch):
        # Create a fake handoff file under the indexed handoffs dir (the guard
        # requires the resolved file to live within thoughts/shared/handoffs/).
        monkeypatch.chdir(tmp_path)
        handoffs_dir = tmp_path / "thoughts" / "shared" / "handoffs" / "auth-session"
        handoffs_dir.mkdir(parents=True)
        handoff_file = handoffs_dir / "task-1.yaml"
        handoff_file.write_text("handoff content here")

        # Update the DB to point to our handoff file (relative to cwd)
        rel = "thoughts/shared/handoffs/auth-session/task-1.yaml"
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (rel,),
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

    @pytest.fixture(autouse=True)
    def _patch_faulthandler(self):
        """Prevent main() from writing to ~/.claude/logs during tests."""
        with patch("scripts.core.artifact_query._enable_faulthandler"):
            yield

    def test_no_args_prints_help(self, capsys):
        from scripts.core.artifact_query import main

        with patch("sys.argv", ["artifact_query.py"]):
            main()
        captured = capsys.readouterr()
        assert "Search the Context Graph" in captured.out

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


# ===========================================================================
# Finding 1 — Path traversal guards (pure helpers)
# ===========================================================================


class TestIsSafeArtifactPath:
    """Tests for is_safe_artifact_path — pure containment helper."""

    def test_path_inside_root_is_safe(self, tmp_path):
        target = tmp_path / "sub" / "file.md"
        assert is_safe_artifact_path(target, tmp_path) is True

    def test_relative_path_inside_root_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert is_safe_artifact_path("file.md", tmp_path) is True

    def test_absolute_path_outside_root_is_unsafe(self, tmp_path):
        assert is_safe_artifact_path("/etc/hosts", tmp_path) is False

    def test_relative_traversal_outside_root_is_unsafe(self, tmp_path):
        assert is_safe_artifact_path(tmp_path / ".." / ".." / "x", tmp_path) is False

    def test_symlink_escaping_root_is_unsafe(self, tmp_path):
        outside_dir = tmp_path.parent / "outside_secret_dir"
        outside_dir.mkdir()
        secret = outside_dir / "secret.txt"
        secret.write_text("top secret")
        root = tmp_path / "root"
        root.mkdir()
        link = root / "link.txt"
        link.symlink_to(secret)
        # Resolution must happen BEFORE containment so the symlink is rejected.
        assert is_safe_artifact_path(link, root) is False

    def test_nonexistent_path_inside_root_is_safe(self, tmp_path):
        assert is_safe_artifact_path(tmp_path / "does-not-exist.md", tmp_path) is True


class TestSafeArtifactReadPath:
    """Tests for safe_artifact_read_path — path-returning resolver."""

    _SUFFIXES = (".md", ".yaml", ".yml")

    def test_returns_resolved_path_for_valid(self, tmp_path):
        target = tmp_path / "task-1.md"
        target.write_text("x")
        result = safe_artifact_read_path(target, tmp_path, suffixes=self._SUFFIXES)
        assert result == target.resolve()

    def test_relative_path_resolved_under_root(self, tmp_path, monkeypatch):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "task-1.yaml").write_text("x")
        monkeypatch.chdir(tmp_path)
        result = safe_artifact_read_path(
            "sub/task-1.yaml", tmp_path / "sub", suffixes=self._SUFFIXES
        )
        assert result == (sub / "task-1.yaml").resolve()

    def test_returns_none_for_outside_dir(self, tmp_path):
        outside = tmp_path.parent / "outside.md"
        outside.write_text("x")
        assert (
            safe_artifact_read_path(outside, tmp_path, suffixes=self._SUFFIXES) is None
        )

    def test_returns_none_for_wrong_suffix(self, tmp_path):
        target = tmp_path / "task-1.txt"
        target.write_text("x")
        assert (
            safe_artifact_read_path(target, tmp_path, suffixes=self._SUFFIXES) is None
        )

    def test_returns_none_for_escaping_symlink(self, tmp_path):
        outside_dir = tmp_path.parent / "outside_secret_dir2"
        outside_dir.mkdir()
        secret = outside_dir / "secret.md"
        secret.write_text("top secret")
        root = tmp_path / "root"
        root.mkdir()
        link = root / "task-1.md"
        link.symlink_to(secret)
        assert safe_artifact_read_path(link, root, suffixes=self._SUFFIXES) is None


class TestReadTextNofollow:
    """Tests for read_text_nofollow — no-follow, no-TOCTOU I/O helper."""

    def test_reads_regular_file(self, tmp_path):
        target = tmp_path / "file.md"
        target.write_text("hello content")
        assert read_text_nofollow(target) == "hello content"

    def test_symlink_final_component_returns_none(self, tmp_path):
        real = tmp_path / "real.md"
        real.write_text("secret")
        link = tmp_path / "link.md"
        link.symlink_to(real)
        # O_NOFOLLOW must refuse the symlinked final component.
        assert read_text_nofollow(link) is None

    def test_directory_returns_none(self, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        assert read_text_nofollow(d) is None

    def test_missing_path_returns_none(self, tmp_path):
        assert read_text_nofollow(tmp_path / "nope.md") is None


class TestIsSafeDirRoot:
    """Tests for is_safe_dir_root — fail-closed trust-root validation."""

    def test_normal_dir_under_cwd_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        root = tmp_path / "thoughts" / "shared" / "handoffs"
        root.mkdir(parents=True)
        assert is_safe_dir_root(root, tmp_path) is True

    def test_symlinked_leaf_component_is_unsafe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real_handoffs"
        real.mkdir()
        shared = tmp_path / "thoughts" / "shared"
        shared.mkdir(parents=True)
        link = shared / "handoffs"
        link.symlink_to(real, target_is_directory=True)
        assert is_safe_dir_root(link, tmp_path) is False

    def test_symlinked_parent_component_is_unsafe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        real_shared = tmp_path / "real_shared"
        (real_shared / "handoffs").mkdir(parents=True)
        thoughts = tmp_path / "thoughts"
        thoughts.mkdir()
        # thoughts/shared is a symlink to real_shared
        (thoughts / "shared").symlink_to(real_shared, target_is_directory=True)
        root = thoughts / "shared" / "handoffs"
        assert is_safe_dir_root(root, tmp_path) is False

    def test_root_outside_repo_is_unsafe(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        assert is_safe_dir_root(outside, repo) is False


class TestIsSafeSessionName:
    """Tests for is_safe_session_name — pure validation helper."""

    def test_simple_name_is_safe(self):
        assert is_safe_session_name("auth-session") is True

    def test_name_with_allowed_punctuation_is_safe(self):
        assert is_safe_session_name("auth.session_2-v3") is True

    def test_empty_name_is_unsafe(self):
        assert is_safe_session_name("") is False

    def test_name_with_slash_is_unsafe(self):
        assert is_safe_session_name("../etc/passwd") is False

    def test_name_with_dotdot_is_unsafe(self):
        assert is_safe_session_name("..") is False

    def test_name_of_only_dots_is_unsafe(self):
        assert is_safe_session_name("...") is False

    def test_single_dot_is_unsafe(self):
        assert is_safe_session_name(".") is False


_HANDOFFS_REL = "thoughts/shared/handoffs/auth-session"


def _make_handoff_file(tmp_path: Path, name: str, content: str) -> str:
    """Create a handoff file under the indexed handoffs dir; return rel path."""
    handoffs_dir = tmp_path / "thoughts" / "shared" / "handoffs" / "auth-session"
    handoffs_dir.mkdir(parents=True, exist_ok=True)
    (handoffs_dir / name).write_text(content)
    return f"{_HANDOFFS_REL}/{name}"


class TestHandleSpanIdLookupPathTraversal:
    """Tests for handle_span_id_lookup path-traversal hardening."""

    def test_absolute_path_outside_cwd_not_read(
        self, populated_db, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        secret = tmp_path.parent / "secret_outside.txt"
        secret.write_text("SECRET")
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (str(secret),),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert "content" not in result

    def test_relative_traversal_not_read(self, populated_db, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("SECRET")
        monkeypatch.chdir(work)
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            ("../secret.txt",),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert "content" not in result

    def test_symlink_inside_handoffs_pointing_outside_not_read(
        self, populated_db, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        handoffs_dir = tmp_path / "thoughts" / "shared" / "handoffs" / "auth-session"
        handoffs_dir.mkdir(parents=True)
        secret = tmp_path / "secret.txt"
        secret.write_text("SECRET")
        link = handoffs_dir / "task-1.yaml"
        link.symlink_to(secret)
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (f"{_HANDOFFS_REL}/task-1.yaml",),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert "content" not in result

    def test_dotenv_under_cwd_not_read(self, populated_db, tmp_path, monkeypatch):
        """Poisoned row pointing at .env (under repo root) must not be read."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("SECRET_KEY=topsecret")
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (".env",),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert "content" not in result

    def test_settings_local_under_cwd_not_read(
        self, populated_db, tmp_path, monkeypatch
    ):
        """Poisoned row pointing at .claude/settings.local.json must not be read."""
        monkeypatch.chdir(tmp_path)
        settings = tmp_path / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text('{"secret": true}')
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (".claude/settings.local.json",),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert "content" not in result

    def test_md_file_outside_handoffs_dir_not_read(
        self, populated_db, tmp_path, monkeypatch
    ):
        """A .md file under cwd but outside handoffs dir proves the dir constraint."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "README.md").write_text("# secret readme")
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            ("README.md",),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert "content" not in result

    def test_disallowed_suffix_in_handoffs_dir_not_read(
        self, populated_db, tmp_path, monkeypatch
    ):
        """A non-allowlisted suffix even inside handoffs dir is rejected."""
        monkeypatch.chdir(tmp_path)
        rel = _make_handoff_file(tmp_path, "task-1.txt", "secret txt")
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (rel,),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert "content" not in result

    def test_symlinked_handoffs_root_skips_content(
        self, populated_db, tmp_path, monkeypatch
    ):
        """If thoughts/shared/handoffs is a symlink, content enrichment is skipped
        even when file_path resolves under the symlink target."""
        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real_handoffs" / "auth-session"
        real.mkdir(parents=True)
        (real / "task-1.yaml").write_text("symlink-target content")
        shared = tmp_path / "thoughts" / "shared"
        shared.mkdir(parents=True)
        (shared / "handoffs").symlink_to(
            tmp_path / "real_handoffs", target_is_directory=True
        )
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            ("thoughts/shared/handoffs/auth-session/task-1.yaml",),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert "content" not in result

    def test_unsafe_session_name_skips_ledger(
        self, populated_db, tmp_path, monkeypatch
    ):
        # Poison the session_name with traversal; legit handoff file under handoffs dir.
        monkeypatch.chdir(tmp_path)
        rel = _make_handoff_file(tmp_path, "task-1.yaml", "legit content")
        populated_db.execute(
            "UPDATE handoffs SET file_path = ?, session_name = ? WHERE id = 'h1'",
            (rel, "../../etc/passwd"),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert result["content"] == "legit content"
        assert "ledger" not in result

    def test_legit_md_under_handoffs_returns_content(
        self, populated_db, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        rel = _make_handoff_file(tmp_path, "task-01.md", "happy path content")
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (rel,),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert result["content"] == "happy path content"

    def test_legit_yaml_under_handoffs_returns_content(
        self, populated_db, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        rel = _make_handoff_file(tmp_path, "task-1.yaml", "yaml content")
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (rel,),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert result["content"] == "yaml content"

    def test_legit_yml_under_handoffs_returns_content(
        self, populated_db, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        rel = _make_handoff_file(tmp_path, "task-1.yml", "yml content")
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (rel,),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert result["content"] == "yml content"

    def test_legit_ledger_under_cwd_returned(
        self, populated_db, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        rel = _make_handoff_file(tmp_path, "task-1.yaml", "handoff content")
        ledger_file = tmp_path / "CONTINUITY_CLAUDE-auth-session.md"
        ledger_file.write_text("ledger content")
        populated_db.execute(
            "UPDATE handoffs SET file_path = ? WHERE id = 'h1'",
            (rel,),
        )
        populated_db.commit()

        result = handle_span_id_lookup(populated_db, "span-abc123", with_content=True)
        assert result is not None
        assert result["ledger"]["content"] == "ledger content"


# ===========================================================================
# Finding 2 — generate_uuid7 pure helper + save_query integration
# ===========================================================================


class TestGenerateUuid7:
    """Tests for generate_uuid7 — pure RFC 9562 UUIDv7 helper."""

    def test_returns_36_char_canonical_string(self):
        result = generate_uuid7()
        assert isinstance(result, str)
        assert len(result) == 36
        assert result.count("-") == 4

    def test_version_nibble_is_7(self):
        result = generate_uuid7(timestamp_ms=0, random_bytes=b"\x00" * 10)
        # Version nibble is first char of 3rd group (index 14 overall).
        assert result[14] == "7"

    def test_variant_bits_are_10(self):
        result = generate_uuid7(timestamp_ms=0, random_bytes=b"\xff" * 10)
        # Variant nibble is first char of 4th group; high two bits must be 10.
        variant_nibble = int(result[19], 16)
        assert (variant_nibble >> 2) == 0b10

    def test_deterministic_given_seeds(self):
        a = generate_uuid7(timestamp_ms=123456789, random_bytes=b"\x01" * 10)
        b = generate_uuid7(timestamp_ms=123456789, random_bytes=b"\x01" * 10)
        assert a == b

    def test_timestamp_ordering_sorts_lexicographically(self):
        early = generate_uuid7(timestamp_ms=1000, random_bytes=b"\x00" * 10)
        late = generate_uuid7(timestamp_ms=2000, random_bytes=b"\x00" * 10)
        assert early < late

    def test_lowercase_hex(self):
        result = generate_uuid7(timestamp_ms=0xABCDEF, random_bytes=b"\xab" * 10)
        assert result == result.lower()


class TestSaveQueryUuid7:
    """Tests for save_query using uuid7 ids."""

    def test_stores_36_char_id(self, populated_db):
        save_query(populated_db, "uuid question", "answer", {})
        cursor = populated_db.execute(
            "SELECT id FROM queries WHERE question = 'uuid question'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert len(row[0]) == 36
        assert row[0].count("-") == 4

    def test_same_question_and_now_different_ids(self, populated_db):
        from datetime import datetime

        now = datetime(2026, 1, 1, 12, 0, 0)
        save_query(populated_db, "dup question", "a", {}, now=now)
        save_query(populated_db, "dup question", "a", {}, now=now)
        cursor = populated_db.execute(
            "SELECT id FROM queries WHERE question = 'dup question'"
        )
        ids = [r[0] for r in cursor.fetchall()]
        assert len(ids) == 2
        assert ids[0] != ids[1]


# ===========================================================================
# Finding 3 — connection lifetime in _run_span_lookup / _run_search
# ===========================================================================


class _CloseTrackingConn:
    """Wrapper recording close() calls while delegating to a real connection."""

    def __init__(self, real):
        self._real = real
        self.close_count = 0

    def close(self):
        self.close_count += 1
        self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestConnectionLifetime:
    """_run_span_lookup / _run_search must close connections even on error."""

    def _make_db(self, tmp_path):
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))
        conn.executescript(_SCHEMA_PATH.read_text())
        conn.commit()
        conn.close()
        return db_file

    def test_span_lookup_closes_on_success(self, tmp_path, monkeypatch, capsys):
        from scripts.core import artifact_query as aq

        db_file = self._make_db(tmp_path)
        tracker = {}

        def fake_open(path):
            tracker["conn"] = _CloseTrackingConn(sqlite3.connect(str(path)))
            return tracker["conn"]

        monkeypatch.setattr(aq, "_open_db", fake_open)
        args = aq._build_parser().parse_args(["--by-span-id", "nope", "--db", str(db_file)])
        aq._run_span_lookup(args)
        assert tracker["conn"].close_count == 1

    def test_span_lookup_closes_on_error(self, tmp_path, monkeypatch):
        from scripts.core import artifact_query as aq

        db_file = self._make_db(tmp_path)
        tracker = {}

        def fake_open(path):
            tracker["conn"] = _CloseTrackingConn(sqlite3.connect(str(path)))
            return tracker["conn"]

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(aq, "_open_db", fake_open)
        monkeypatch.setattr(aq, "handle_span_id_lookup", boom)
        args = aq._build_parser().parse_args(["--by-span-id", "x", "--db", str(db_file)])
        with pytest.raises(RuntimeError):
            aq._run_span_lookup(args)
        assert tracker["conn"].close_count == 1

    def test_search_closes_on_success(self, tmp_path, monkeypatch):
        from scripts.core import artifact_query as aq

        db_file = self._make_db(tmp_path)
        tracker = {}

        def fake_open(path):
            tracker["conn"] = _CloseTrackingConn(sqlite3.connect(str(path)))
            return tracker["conn"]

        monkeypatch.setattr(aq, "_open_db", fake_open)
        args = aq._build_parser().parse_args(["query", "--json", "--db", str(db_file)])
        aq._run_search(args, "query")
        assert tracker["conn"].close_count == 1

    def test_search_closes_on_error(self, tmp_path, monkeypatch):
        from scripts.core import artifact_query as aq

        db_file = self._make_db(tmp_path)
        tracker = {}

        def fake_open(path):
            tracker["conn"] = _CloseTrackingConn(sqlite3.connect(str(path)))
            return tracker["conn"]

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(aq, "_open_db", fake_open)
        monkeypatch.setattr(aq, "search_dispatch", boom)
        args = aq._build_parser().parse_args(["query", "--json", "--db", str(db_file)])
        with pytest.raises(RuntimeError):
            aq._run_search(args, "query")
        assert tracker["conn"].close_count == 1
