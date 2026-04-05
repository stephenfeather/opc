"""Tests for scripts/core/backfill_sessions.py - TDD+FP refactor."""

import argparse
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from scripts.core.backfill_sessions import (
    _build_session_info,
    _date_string,
    _positive_int,
    _read_first_line,
    build_session_record,
    compute_fake_pid,
    decode_project_path_pure,
    filter_sessions_by_date,
    find_unregistered_sessions,
    format_dry_run_line,
    get_pg_url,
    insert_sessions,
    is_daemon_extraction_content,
    is_subagent_file,
    main,
    naive_decode_path,
    select_batch,
    sort_sessions_by_mtime,
)

# --- _positive_int ---


class TestPositiveInt:
    def test_valid(self):
        assert _positive_int("5") == 5

    def test_zero_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("0")

    def test_negative_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("-1")

    def test_non_int_raises(self):
        with pytest.raises(ValueError):
            _positive_int("abc")


# --- _date_string ---


class TestDateString:
    def test_valid(self):
        assert _date_string("2026-03-15") == "2026-03-15"

    def test_invalid_format_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _date_string("03-15-2026")

    def test_nonsense_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _date_string("not-a-date")


# --- get_pg_url ---


class TestGetPgUrl:
    def test_returns_continuous_claude_db_url_first(self):
        env = {
            "CONTINUOUS_CLAUDE_DB_URL": "postgresql://a",
            "DATABASE_URL": "postgresql://b",
        }
        with patch.dict("os.environ", env, clear=True):
            assert get_pg_url() == "postgresql://a"

    def test_falls_back_to_database_url(self):
        env = {"DATABASE_URL": "postgresql://b"}
        with patch.dict("os.environ", env, clear=True):
            assert get_pg_url() == "postgresql://b"

    def test_falls_back_to_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_pg_url() == "postgresql://claude:claude_dev@localhost:5432/continuous_claude"


# --- naive_decode_path ---


class TestNaiveDecodePath:
    def test_simple_path(self):
        assert naive_decode_path("Users-foo-bar") == "/Users/foo/bar"

    def test_strips_leading_dash(self):
        assert naive_decode_path("-Users-foo-bar") == "/Users/foo/bar"

    def test_empty_string(self):
        assert naive_decode_path("") == "/"

    def test_single_segment(self):
        assert naive_decode_path("home") == "/home"


# --- decode_project_path_pure ---


class TestDecodeProjectPathPure:
    def test_uses_resolver_result_when_found(self):
        def resolver(candidate: str) -> bool:
            return candidate == "/Users/stephenfeather/Dev/foo"

        result = decode_project_path_pure("-Users-stephenfeather-Dev-foo", resolver)
        assert result == "/Users/stephenfeather/Dev/foo"

    def test_falls_back_to_naive_when_resolver_finds_nothing(self):
        def resolver(candidate: str) -> bool:
            return False

        result = decode_project_path_pure("-Users-stephenfeather-Dev-foo", resolver)
        assert result == "/Users/stephenfeather/Dev/foo"

    def test_hyphenated_segment_resolved(self):
        """When a directory name contains hyphens, the resolver disambiguates."""
        # Simulate: /Users/step/my-project exists
        def resolver(candidate: str) -> bool:
            return candidate in {"/Users", "/Users/step", "/Users/step/my-project"}

        result = decode_project_path_pure("-Users-step-my-project", resolver)
        assert result == "/Users/step/my-project"


# --- is_subagent_file ---


class TestIsSubagentFile:
    def test_subagent_in_path(self):
        assert is_subagent_file("/home/.claude/projects/foo/subagents/abc.jsonl") is True

    def test_agent_prefix_stem(self):
        assert is_subagent_file("/home/.claude/projects/foo/agent-a1b2.jsonl") is True

    def test_normal_session(self):
        assert is_subagent_file("/home/.claude/projects/foo/abc123.jsonl") is False


# --- is_daemon_extraction_content ---


class TestIsDaemonExtractionContent:
    def test_matches_extraction_content(self):
        line = '{"content": "Extract learnings from session abc"}'
        assert is_daemon_extraction_content(line) is True

    def test_normal_content(self):
        assert is_daemon_extraction_content('{"content": "hello world"}') is False

    def test_empty_string(self):
        assert is_daemon_extraction_content("") is False


# --- filter_sessions_by_date ---


class TestFilterSessionsByDate:
    def setup_method(self):
        self.sessions = [
            {"uuid": "a", "mtime": datetime(2026, 1, 10)},
            {"uuid": "b", "mtime": datetime(2026, 2, 15)},
            {"uuid": "c", "mtime": datetime(2026, 3, 20)},
        ]

    def test_no_filter_returns_all(self):
        result = filter_sessions_by_date(self.sessions, after_date=None)
        assert len(result) == 3

    def test_filters_before_cutoff(self):
        result = filter_sessions_by_date(self.sessions, after_date="2026-02-01")
        assert [s["uuid"] for s in result] == ["b", "c"]

    def test_filters_all(self):
        result = filter_sessions_by_date(self.sessions, after_date="2026-12-01")
        assert result == []

    def test_does_not_mutate_input(self):
        original_len = len(self.sessions)
        filter_sessions_by_date(self.sessions, after_date="2026-02-01")
        assert len(self.sessions) == original_len


# --- sort_sessions_by_mtime ---


class TestSortSessionsByMtime:
    def test_sorts_ascending(self):
        sessions = [
            {"uuid": "c", "mtime": datetime(2026, 3, 1)},
            {"uuid": "a", "mtime": datetime(2026, 1, 1)},
            {"uuid": "b", "mtime": datetime(2026, 2, 1)},
        ]
        result = sort_sessions_by_mtime(sessions)
        assert [s["uuid"] for s in result] == ["a", "b", "c"]

    def test_does_not_mutate_input(self):
        sessions = [
            {"uuid": "b", "mtime": datetime(2026, 2, 1)},
            {"uuid": "a", "mtime": datetime(2026, 1, 1)},
        ]
        original = list(sessions)
        sort_sessions_by_mtime(sessions)
        assert sessions == original

    def test_empty_list(self):
        assert sort_sessions_by_mtime([]) == []


# --- build_session_record ---


class TestBuildSessionRecord:
    def test_builds_correct_record(self):
        session = {
            "uuid": "abc-123",
            "project": "/Users/foo/bar",
            "mtime": datetime(2026, 3, 15, 10, 30),
            "jsonl_path": "/home/.claude/projects/foo/abc-123.jsonl",
        }
        result = build_session_record(session, fake_pid=900005)
        assert result == {
            "id": "abc-123",
            "project": "/Users/foo/bar",
            "working_on": "backfill",
            "started_at": datetime(2026, 3, 15, 10, 30),
            "last_heartbeat": datetime(2026, 3, 15, 10, 30),
            "exited_at": datetime(2026, 3, 15, 10, 30),
            "pid": 900005,
            "transcript_path": "/home/.claude/projects/foo/abc-123.jsonl",
        }

    def test_missing_jsonl_path_defaults_to_empty(self):
        session = {
            "uuid": "abc-123",
            "project": "/Users/foo/bar",
            "mtime": datetime(2026, 3, 15, 10, 30),
        }
        result = build_session_record(session, fake_pid=900005)
        assert result["transcript_path"] == ""

    def test_exited_at_matches_mtime_for_crash_recovery_safety(self):
        """Backfilled rows must have exited_at set so crash-recovery skips them."""
        session = {
            "uuid": "abc-123",
            "project": "/Users/foo/bar",
            "mtime": datetime(2026, 3, 15, 10, 30),
        }
        result = build_session_record(session, fake_pid=900005)
        assert result["exited_at"] == session["mtime"]
        assert result["exited_at"] is not None


# --- compute_fake_pid ---


class TestComputeFakePid:
    def test_base_offset(self):
        assert compute_fake_pid(0) == 900000
        assert compute_fake_pid(5) == 900005

    def test_custom_base(self):
        assert compute_fake_pid(3, base=800000) == 800003


# --- format_dry_run_line ---


class TestFormatDryRunLine:
    def test_format_output(self):
        session = {
            "uuid": "abcdef12-3456-7890-abcd-ef1234567890",
            "mtime": datetime(2026, 3, 15, 10, 30),
            "size": 10240,
            "project": "/Users/foo/my-project",
        }
        line = format_dry_run_line(session)
        assert "abcdef12..." in line
        assert "2026-03-15 10:30" in line
        assert "10KB" in line or "10" in line
        assert "my-project" in line


# --- select_batch ---


class TestSelectBatch:
    def test_select_all(self):
        sessions = [{"uuid": "a"}, {"uuid": "b"}, {"uuid": "c"}]
        result = select_batch(sessions, batch_size=10, select_all=True)
        assert len(result) == 3

    def test_select_batch_size(self):
        sessions = [{"uuid": "a"}, {"uuid": "b"}, {"uuid": "c"}]
        result = select_batch(sessions, batch_size=2, select_all=False)
        assert len(result) == 2
        assert result[0]["uuid"] == "a"

    def test_batch_larger_than_sessions(self):
        sessions = [{"uuid": "a"}]
        result = select_batch(sessions, batch_size=10, select_all=False)
        assert len(result) == 1

    def test_does_not_mutate_input(self):
        sessions = [{"uuid": "a"}, {"uuid": "b"}]
        result = select_batch(sessions, batch_size=1, select_all=False)
        assert len(sessions) == 2
        assert len(result) == 1


# --- _read_first_line ---


class TestReadFirstLine:
    def test_reads_first_line(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"content": "hello"}\n{"content": "world"}\n')
        assert _read_first_line(p) == '{"content": "hello"}\n'

    def test_returns_empty_on_missing_file(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        assert _read_first_line(p) == ""

    def test_returns_empty_on_permission_error(self, tmp_path):
        p = tmp_path / "noperm.jsonl"
        with patch("builtins.open", side_effect=PermissionError("access denied")):
            result = _read_first_line(p)
        assert result == ""


# --- _build_session_info ---


class TestBuildSessionInfo:
    def test_builds_info_from_jsonl(self, tmp_path):
        p = tmp_path / "abc-123.jsonl"
        p.write_text('{"content": "test"}\n')
        result = _build_session_info(p, "/Users/foo/project")
        assert result["uuid"] == "abc-123"
        assert result["project"] == "/Users/foo/project"
        assert isinstance(result["mtime"], datetime)
        assert result["size"] > 0
        assert result["jsonl_path"] == str(p)


# --- find_unregistered_sessions (mocked I/O) ---


class TestFindUnregisteredSessions:
    def _make_projects_dir(self, tmp_path):
        """Create a fake projects dir with JSONL files."""
        proj_dir = tmp_path / ".claude" / "projects" / "Users-foo-bar"
        proj_dir.mkdir(parents=True)
        (proj_dir / "session-aaa.jsonl").write_text('{"type": "user"}\n')
        (proj_dir / "session-bbb.jsonl").write_text('{"type": "user"}\n')
        return tmp_path / ".claude"

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_finds_unregistered_sessions(self, mock_pg, tmp_path):
        config_dir = self._make_projects_dir(tmp_path)
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [("session-aaa",)]  # aaa is registered
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(config_dir)}):
            result = find_unregistered_sessions()

        assert len(result) == 1
        assert result[0]["uuid"] == "session-bbb"

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_returns_empty_when_all_registered(self, mock_pg, tmp_path):
        config_dir = self._make_projects_dir(tmp_path)
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [("session-aaa",), ("session-bbb",)]
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(config_dir)}):
            result = find_unregistered_sessions()

        assert result == []

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_skips_subagent_files(self, mock_pg, tmp_path):
        config_dir = tmp_path / ".claude"
        proj_dir = config_dir / "projects" / "Users-foo-bar"
        proj_dir.mkdir(parents=True)
        # agent- prefix at glob-matched depth (*/*.jsonl)
        (proj_dir / "agent-xyz.jsonl").write_text('{"type": "assistant"}\n')

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(config_dir)}):
            result = find_unregistered_sessions()

        assert result == []

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_skips_daemon_extraction_files(self, mock_pg, tmp_path):
        config_dir = tmp_path / ".claude"
        proj_dir = config_dir / "projects" / "Users-foo-bar"
        proj_dir.mkdir(parents=True)
        (proj_dir / "sess-1.jsonl").write_text(
            '{"content": "Extract learnings from session abc"}\n'
        )

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(config_dir)}):
            result = find_unregistered_sessions()

        assert result == []

    def test_returns_empty_when_no_projects_dir(self, tmp_path):
        config_dir = tmp_path / ".claude"
        config_dir.mkdir()
        # no projects/ subdir
        with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(config_dir)}):
            result = find_unregistered_sessions()
        assert result == []

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_after_date_filtering(self, mock_pg, tmp_path):
        config_dir = tmp_path / ".claude"
        proj_dir = config_dir / "projects" / "Users-foo-bar"
        proj_dir.mkdir(parents=True)
        old = proj_dir / "old-sess.jsonl"
        old.write_text('{"type": "user"}\n')
        # Set mtime to 2025-01-01
        os.utime(old, (datetime(2025, 1, 1).timestamp(), datetime(2025, 1, 1).timestamp()))
        new = proj_dir / "new-sess.jsonl"
        new.write_text('{"type": "user"}\n')

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(config_dir)}):
            result = find_unregistered_sessions(after_date="2026-01-01")

        uuids = [s["uuid"] for s in result]
        assert "old-sess" not in uuids
        assert "new-sess" in uuids


# --- insert_sessions (mocked I/O) ---


class TestInsertSessions:
    def _sample_sessions(self):
        return [
            {"uuid": "aaa", "project": "/foo", "mtime": datetime(2026, 3, 1),
             "size": 1024, "jsonl_path": "/tmp/aaa.jsonl"},
            {"uuid": "bbb", "project": "/bar", "mtime": datetime(2026, 3, 2),
             "size": 2048, "jsonl_path": "/tmp/bbb.jsonl"},
        ]

    def test_dry_run_does_not_connect_and_returns_zero(self, capsys):
        sessions = self._sample_sessions()
        with patch("scripts.core.backfill_sessions.psycopg2") as mock_pg:
            result = insert_sessions(sessions, dry_run=True)
            mock_pg.connect.assert_not_called()
        assert result == 0
        output = capsys.readouterr().out
        assert "Dry run" in output
        assert "aaa" in output

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_inserts_records_and_returns_count(self, mock_pg):
        sessions = self._sample_sessions()
        mock_conn = MagicMock()
        mock_cur = MagicMock()

        def track_rowcount(sql, *args):
            if sql.strip().startswith("INSERT"):
                mock_cur.rowcount = 1
            else:
                mock_cur.rowcount = 0

        mock_cur.execute.side_effect = track_rowcount
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        result = insert_sessions(sessions)

        assert result == 2
        # Each row: SAVEPOINT + INSERT + RELEASE = 3 calls per row
        assert mock_cur.execute.call_count == 6
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_on_conflict_do_nothing_does_not_increment(self, mock_pg):
        """When ON CONFLICT fires, rowcount=0 and inserted should not increment."""
        sessions = self._sample_sessions()[:1]
        mock_conn = MagicMock()
        mock_cur = MagicMock()

        def track_rowcount(sql, *args):
            # INSERT hits ON CONFLICT — no row inserted
            mock_cur.rowcount = 0

        mock_cur.execute.side_effect = track_rowcount
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        result = insert_sessions(sessions)

        assert result == 0

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_inserts_transcript_path_and_exited_at(self, mock_pg):
        sessions = self._sample_sessions()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        insert_sessions(sessions)

        # Check that the INSERT call includes transcript_path and exited_at
        insert_calls = [
            c for c in mock_cur.execute.call_args_list
            if c[0][0].strip().startswith("INSERT")
        ]
        assert len(insert_calls) == 2
        params = insert_calls[0][0][1]
        assert params[6] == "/tmp/aaa.jsonl"  # transcript_path
        assert params[7] == datetime(2026, 3, 1)  # exited_at matches mtime

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_handles_insert_error_with_savepoint_rollback(self, mock_pg, capsys):
        sessions = self._sample_sessions()
        mock_conn = MagicMock()
        mock_cur = MagicMock()

        call_count = 0

        def side_effect(sql, *args):
            nonlocal call_count
            call_count += 1
            # Fail on the 2nd call (first INSERT), succeed on everything else
            if call_count == 2:
                raise Exception("duplicate key")

        mock_cur.execute.side_effect = side_effect
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        insert_sessions(sessions)

        # Still commits and closes despite one error
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()
        output = capsys.readouterr().out
        assert "Error" in output

    @patch("scripts.core.backfill_sessions.psycopg2")
    def test_rollback_and_release_savepoint_on_error(self, mock_pg):
        """After a failed INSERT, ROLLBACK TO + RELEASE SAVEPOINT are called."""
        sessions = [self._sample_sessions()[0]]  # just one session
        mock_conn = MagicMock()
        mock_cur = MagicMock()

        call_count = 0

        def side_effect(sql, *args):
            nonlocal call_count
            call_count += 1
            # call 1 = SAVEPOINT, call 2 = INSERT (fails)
            if call_count == 2:
                raise Exception("db error")

        mock_cur.execute.side_effect = side_effect
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        insert_sessions(sessions)

        # After error: ROLLBACK TO SAVEPOINT + RELEASE SAVEPOINT
        rollback_calls = [
            c for c in mock_cur.execute.call_args_list
            if "ROLLBACK TO SAVEPOINT" in str(c)
        ]
        release_after_rollback = [
            c for c in mock_cur.execute.call_args_list
            if "RELEASE SAVEPOINT" in str(c)
        ]
        assert len(rollback_calls) == 1
        assert len(release_after_rollback) == 1


# --- main (mocked I/O) ---


class TestMain:
    @patch("scripts.core.backfill_sessions._bootstrap")
    @patch("scripts.core.backfill_sessions.find_unregistered_sessions", return_value=[])
    def test_no_sessions_returns_zero(self, mock_find, mock_boot, capsys):
        with patch("sys.argv", ["backfill_sessions.py"]):
            result = main()
        assert result == 0
        output = capsys.readouterr().out
        assert "No unregistered sessions" in output

    @patch("scripts.core.backfill_sessions._bootstrap")
    @patch("scripts.core.backfill_sessions.insert_sessions")
    @patch("scripts.core.backfill_sessions.find_unregistered_sessions")
    def test_dry_run_passes_flag(self, mock_find, mock_insert, mock_boot):
        mock_find.return_value = [
            {"uuid": "a", "project": "/foo", "mtime": datetime(2026, 1, 1),
             "size": 100, "jsonl_path": "/tmp/a.jsonl"},
        ]
        with patch("sys.argv", ["backfill_sessions.py", "--dry-run"]):
            result = main()
        assert result == 0
        mock_insert.assert_called_once()
        _, kwargs = mock_insert.call_args
        assert kwargs.get("dry_run") is True or mock_insert.call_args[0][1] is True

    @patch("scripts.core.backfill_sessions._bootstrap")
    @patch("scripts.core.backfill_sessions.insert_sessions", return_value=2)
    @patch("scripts.core.backfill_sessions.find_unregistered_sessions")
    def test_batch_size_limits_insertion(self, mock_find, mock_insert, mock_boot, capsys):
        mock_find.return_value = [
            {"uuid": f"s{i}", "project": "/foo", "mtime": datetime(2026, 1, i + 1),
             "size": 100, "jsonl_path": f"/tmp/s{i}.jsonl"}
            for i in range(5)
        ]
        with patch("sys.argv", ["backfill_sessions.py", "--batch-size", "2"]):
            result = main()
        assert result == 0
        batch_passed = mock_insert.call_args[0][0]
        assert len(batch_passed) == 2
        output = capsys.readouterr().out
        assert "3 sessions remaining" in output

    @patch("scripts.core.backfill_sessions._bootstrap")
    @patch("scripts.core.backfill_sessions.insert_sessions", return_value=5)
    @patch("scripts.core.backfill_sessions.find_unregistered_sessions")
    def test_all_flag_inserts_everything(self, mock_find, mock_insert, mock_boot):
        mock_find.return_value = [
            {"uuid": f"s{i}", "project": "/foo", "mtime": datetime(2026, 1, i + 1),
             "size": 100, "jsonl_path": f"/tmp/s{i}.jsonl"}
            for i in range(5)
        ]
        with patch("sys.argv", ["backfill_sessions.py", "--all"]):
            result = main()
        assert result == 0
        batch_passed = mock_insert.call_args[0][0]
        assert len(batch_passed) == 5
