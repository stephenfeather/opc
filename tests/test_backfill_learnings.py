"""Tests for scripts/core/backfill_learnings.py — TDD+FP refactor (S16)."""

import argparse
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.core.backfill_learnings import (
    _positive_int,
    _process_one,
    build_extraction_cmd,
    build_extraction_env,
    claim_session,
    classify_session,
    download_and_decompress,
    format_dry_run_line,
    format_summary,
    get_pg_url,
    get_s3_bucket,
    is_session_extracted,
    list_s3_keys,
    load_agent_prompt,
    log_extraction_result,
    lookup_session_id,
    main,
    parse_extraction_output,
    parse_s3_listing,
    run_extraction,
    select_batch,
    strip_yaml_frontmatter,
)

# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestPositiveInt:
    def test_valid(self):
        assert _positive_int("5") == 5

    def test_one(self):
        assert _positive_int("1") == 1

    def test_zero_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("0")

    def test_negative_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("-1")

    def test_non_int_raises(self):
        with pytest.raises(ValueError):
            _positive_int("abc")


class TestGetPgUrl:
    def test_database_url(self):
        with patch.dict("os.environ", {"DATABASE_URL": "pg://a"}, clear=True):
            assert get_pg_url() == "pg://a"

    def test_postgres_url_fallback(self):
        with patch.dict("os.environ", {"POSTGRES_URL": "pg://b"}, clear=True):
            assert get_pg_url() == "pg://b"

    def test_database_url_takes_precedence(self):
        with patch.dict(
            "os.environ", {"DATABASE_URL": "pg://a", "POSTGRES_URL": "pg://b"}, clear=True
        ):
            assert get_pg_url() == "pg://a"

    def test_missing_returns_none(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_pg_url() is None


class TestGetS3Bucket:
    def test_present(self):
        with patch.dict("os.environ", {"CLAUDE_SESSION_ARCHIVE_BUCKET": "my-bucket"}, clear=True):
            assert get_s3_bucket() == "my-bucket"

    def test_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_s3_bucket() is None


class TestParseS3Listing:
    SAMPLE_OUTPUT = (
        "2026-03-30 14:04:34     192400 sessions/-Users-stephenfeather-opc/"
        "d0f60cd7-65e8-4a30-a1fc-345ec418a1ec.jsonl.zst\n"
        "2026-03-31 10:22:01      54000 sessions/-Users-stephenfeather-opc/"
        "abc12345-0000-0000-0000-000000000001.jsonl.zst\n"
        "2026-04-01 08:00:00     100000 sessions/-Users-stephenfeather-Development-myproj/"
        "fedcba98-1111-2222-3333-444444444444.jsonl.zst\n"
    )

    def test_parses_normal_output(self):
        result = parse_s3_listing(self.SAMPLE_OUTPUT, "my-bucket", project_filter=None)
        assert len(result) == 3
        assert result[0]["uuid"] == "d0f60cd7-65e8-4a30-a1fc-345ec418a1ec"
        assert result[0]["project"] == "-Users-stephenfeather-opc"
        assert result[0]["s3_key"].startswith("s3://my-bucket/")
        assert result[0]["s3_key"].endswith(".jsonl.zst")

    def test_project_filter(self):
        result = parse_s3_listing(self.SAMPLE_OUTPUT, "b", project_filter="opc")
        assert len(result) == 2
        assert all("opc" in s["project"] for s in result)

    def test_project_filter_no_match(self):
        result = parse_s3_listing(self.SAMPLE_OUTPUT, "b", project_filter="nonexistent")
        assert result == []

    def test_empty_output(self):
        assert parse_s3_listing("", "b", project_filter=None) == []

    def test_skips_malformed_lines(self):
        result = parse_s3_listing("bad line\n\n  \n", "b", project_filter=None)
        assert result == []

    def test_skips_non_jsonl_zst(self):
        output = "2026-03-30 14:04:34 100 sessions/proj/file.txt\n"
        assert parse_s3_listing(output, "b", project_filter=None) == []


class TestBuildExtractionCmd:
    def test_builds_correct_argv(self):
        cmd = build_extraction_cmd(
            jsonl_path=Path("/tmp/abc.jsonl"),
            session_id="s-abc123",
            agent_prompt="Extract learnings",
            model="sonnet",
            max_turns=15,
        )
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "sonnet"
        assert "--dangerously-skip-permissions" in cmd
        assert "--allowedTools" in cmd
        assert "--max-turns" in cmd
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "15"
        assert "--append-system-prompt" in cmd
        # Final arg is the extraction instruction
        assert "s-abc123" in cmd[-1]
        assert "/tmp/abc.jsonl" in cmd[-1]

    def test_different_model(self):
        cmd = build_extraction_cmd(
            jsonl_path=Path("/tmp/x.jsonl"),
            session_id="s-x",
            agent_prompt="prompt",
            model="haiku",
            max_turns=5,
        )
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "haiku"


class TestBuildExtractionEnv:
    def test_sets_extraction_flag(self):
        env = build_extraction_env(project_dir="/some/project")
        assert env["CLAUDE_MEMORY_EXTRACTION"] == "1"

    def test_sets_project_dir(self):
        env = build_extraction_env(project_dir="/some/project")
        assert env["CLAUDE_PROJECT_DIR"] == "/some/project"

    def test_none_project_dir_does_not_override(self):
        with patch.dict("os.environ", {}, clear=True):
            env = build_extraction_env(project_dir=None)
            assert "CLAUDE_PROJECT_DIR" not in env

    def test_excludes_sensitive_keys_via_allowlist(self):
        with patch.dict(
            "os.environ",
            {
                "AWS_SECRET_ACCESS_KEY": "secret",
                "VOYAGE_API_KEY": "vk",
                "OPENAI_API_KEY": "ok",
                "DATABASE_URL": "pg://creds",
                "POSTGRES_URL": "pg://creds2",
                "GITHUB_TOKEN": "tok",
                "PGPASSWORD": "pass",
                "PATH": "/usr/bin",
                "HOME": "/home/test",
                "CLAUDE_CONFIG_DIR": "/tmp/claude",
            },
            clear=True,
        ):
            env = build_extraction_env(project_dir=None)
            # Sensitive keys excluded (not in allowlist)
            assert "AWS_SECRET_ACCESS_KEY" not in env
            assert "VOYAGE_API_KEY" not in env
            assert "OPENAI_API_KEY" not in env
            assert "DATABASE_URL" not in env
            assert "POSTGRES_URL" not in env
            assert "GITHUB_TOKEN" not in env
            assert "PGPASSWORD" not in env
            # Safe keys included (in allowlist)
            assert env["PATH"] == "/usr/bin"
            assert env["HOME"] == "/home/test"
            assert env["CLAUDE_CONFIG_DIR"] == "/tmp/claude"

    def test_passes_allowlisted_vars(self):
        with patch.dict("os.environ", {"PATH": "/usr/bin", "HOME": "/home/test"}, clear=True):
            env = build_extraction_env(project_dir=None)
            assert env["PATH"] == "/usr/bin"
            assert env["HOME"] == "/home/test"

    def test_blocks_non_allowlisted_vars(self):
        with patch.dict("os.environ", {"MY_SECRET": "val", "PATH": "/usr/bin"}, clear=True):
            env = build_extraction_env(project_dir=None)
            assert "MY_SECRET" not in env
            assert "PATH" in env


class TestParseExtractionOutput:
    def test_parses_learnings_and_dupes(self):
        output = "Some text\nLearnings stored: 7\nDuplicates skipped: 3\nDone.\n"
        result = parse_extraction_output(output)
        assert result["learnings"] == 7
        assert result["duplicates"] == 3

    def test_missing_lines_default_zero(self):
        result = parse_extraction_output("No relevant lines here\n")
        assert result["learnings"] == 0
        assert result["duplicates"] == 0

    def test_empty_output(self):
        result = parse_extraction_output("")
        assert result["learnings"] == 0
        assert result["duplicates"] == 0

    def test_malformed_numbers(self):
        output = "Learnings stored: abc\nDuplicates skipped: xyz\n"
        result = parse_extraction_output(output)
        assert result["learnings"] == 0
        assert result["duplicates"] == 0

    def test_whitespace_around_values(self):
        output = "  Learnings stored:  12  \n  Duplicates skipped:  4  \n"
        result = parse_extraction_output(output)
        assert result["learnings"] == 12
        assert result["duplicates"] == 4


class TestClassifySession:
    def test_has_db_id(self):
        sid, reason = classify_session("uuid-1", session_id="s-abc", skip_no_db=False)
        assert sid == "s-abc"
        assert reason == ""

    def test_no_db_id_fallback_to_uuid(self):
        sid, reason = classify_session("uuid-1", session_id=None, skip_no_db=False)
        assert sid == "uuid-1"
        assert reason == ""

    def test_no_db_id_skip_when_flag_set(self):
        sid, reason = classify_session("uuid-1", session_id=None, skip_no_db=True)
        assert sid is None
        assert "skip" in reason.lower() or "no db" in reason.lower()


class TestFormatDryRunLine:
    def test_format(self):
        session = {
            "session_id": "s-abc123",
            "uuid": "d0f60cd7-65e8-4a30-a1fc-345ec418a1ec",
            "project": "-Users-stephenfeather-opc",
        }
        line = format_dry_run_line(session)
        assert "s-abc123" in line
        assert "d0f60cd7" in line
        assert "opc" in line


class TestFormatSummary:
    def test_format(self):
        text = format_summary(processed=10, learnings=42, dupes=5, errors=2, elapsed=12.3)
        assert "10" in text
        assert "42" in text
        assert "5" in text
        assert "2" in text
        assert "12.3" in text


class TestSelectBatch:
    def test_limit_zero_returns_all(self):
        items = [{"a": 1}, {"a": 2}, {"a": 3}]
        result = select_batch(items, limit=0)
        assert len(result) == 3

    def test_limit_smaller_than_len(self):
        items = [{"a": 1}, {"a": 2}, {"a": 3}]
        result = select_batch(items, limit=2)
        assert len(result) == 2

    def test_limit_larger_than_len(self):
        items = [{"a": 1}]
        result = select_batch(items, limit=10)
        assert len(result) == 1

    def test_does_not_mutate_input(self):
        items = [{"a": 1}, {"a": 2}]
        original_len = len(items)
        select_batch(items, limit=1)
        assert len(items) == original_len

    def test_returns_new_list(self):
        items = [{"a": 1}]
        result = select_batch(items, limit=0)
        assert result is not items


class TestStripYamlFrontmatter:
    def test_with_frontmatter(self):
        content = "---\ntitle: Agent\n---\nBody text here"
        assert strip_yaml_frontmatter(content) == "Body text here"

    def test_without_frontmatter(self):
        content = "Just plain text"
        assert strip_yaml_frontmatter(content) == "Just plain text"

    def test_empty(self):
        assert strip_yaml_frontmatter("") == ""

    def test_frontmatter_only(self):
        content = "---\ntitle: Agent\n---\n"
        assert strip_yaml_frontmatter(content).strip() == ""


# ---------------------------------------------------------------------------
# I/O function tests
# ---------------------------------------------------------------------------


class TestListS3Keys:
    @patch("scripts.core.backfill_learnings.subprocess")
    def test_success(self, mock_sub):
        mock_sub.run.return_value = MagicMock(
            returncode=0, stdout="2026-01-01 00:00:00 100 sessions/proj/a.jsonl.zst\n"
        )
        result = list_s3_keys("my-bucket")
        assert "sessions/proj/a.jsonl.zst" in result
        mock_sub.run.assert_called_once()

    @patch("scripts.core.backfill_learnings.subprocess")
    def test_failure_returns_empty(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=1, stderr="access denied")
        result = list_s3_keys("my-bucket")
        assert result == ""

    @patch("scripts.core.backfill_learnings.subprocess")
    def test_timeout_returns_empty(self, mock_sub):
        mock_sub.run.side_effect = subprocess.TimeoutExpired(cmd="aws", timeout=60)
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        result = list_s3_keys("my-bucket")
        assert result == ""


class TestLookupSessionId:
    _VALID_UUID = "d0f60cd7-65e8-4a30-a1fc-345ec418a1ec"

    def test_found(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("s-abc123",)
        mock_conn.cursor.return_value = mock_cur
        assert lookup_session_id(self._VALID_UUID, mock_conn) == "s-abc123"

    def test_not_found(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cur
        assert lookup_session_id(self._VALID_UUID, mock_conn) is None

    def test_error_rollback(self):
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("db error")
        assert lookup_session_id(self._VALID_UUID, mock_conn) is None
        mock_conn.rollback.assert_called_once()

    def test_invalid_uuid_returns_none(self):
        mock_conn = MagicMock()
        assert lookup_session_id("not-a-uuid", mock_conn) is None
        mock_conn.cursor.assert_not_called()


class TestIsSessionExtracted:
    def test_found_ok(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("ok",)
        mock_conn.cursor.return_value = mock_cur
        assert is_session_extracted("uuid-1", mock_conn) is True

    def test_not_found(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cur
        assert is_session_extracted("uuid-1", mock_conn) is False

    def test_failed_status_is_retryable(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("extraction_failed",)
        mock_conn.cursor.return_value = mock_cur
        assert is_session_extracted("uuid-1", mock_conn) is False

    def test_timeout_status_is_retryable(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("extraction_timeout",)
        mock_conn.cursor.return_value = mock_cur
        assert is_session_extracted("uuid-1", mock_conn) is False

    def test_in_progress_is_not_retryable(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("in_progress",)
        mock_conn.cursor.return_value = mock_cur
        assert is_session_extracted("uuid-1", mock_conn) is True

    def test_error_returns_false(self):
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("db error")
        assert is_session_extracted("uuid-1", mock_conn) is False


class TestClaimSession:
    def test_claim_succeeds(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("uuid-1",)
        mock_conn.cursor.return_value = mock_cur
        assert claim_session("uuid-1", "s-abc", "opc", mock_conn) is True
        mock_conn.commit.assert_called_once()

    def test_claim_already_held(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None  # RETURNING yielded nothing
        mock_conn.cursor.return_value = mock_cur
        assert claim_session("uuid-1", "s-abc", "opc", mock_conn) is False

    def test_claim_error_rolls_back(self):
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("db error")
        assert claim_session("uuid-1", "s-abc", "opc", mock_conn) is False
        mock_conn.rollback.assert_called_once()


class TestLogExtractionResult:
    def test_insert_success(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        result = {
            "uuid": "abc-123",
            "session_id": "s-abc",
            "project": "-Users-opc",
            "status": "ok",
            "learnings": 5,
            "file_size": 1024,
        }
        log_extraction_result(result, mock_conn)
        mock_cur.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    def test_upsert_updates_on_conflict(self):
        """A retry that succeeds should overwrite a prior failed row."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        result = {
            "uuid": "abc-123",
            "session_id": "s-abc",
            "project": "-Users-opc",
            "status": "ok",
            "learnings": 5,
            "file_size": 1024,
        }
        log_extraction_result(result, mock_conn)
        sql = mock_cur.execute.call_args[0][0]
        assert "DO UPDATE" in sql
        assert "DO NOTHING" not in sql

    def test_error_rolls_back(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.execute.side_effect = Exception("constraint")
        mock_conn.cursor.return_value = mock_cur
        result = {
            "uuid": "abc-123",
            "session_id": "s-abc",
            "project": "-Users-opc",
            "status": "ok",
            "learnings": 5,
            "file_size": 0,
        }
        log_extraction_result(result, mock_conn)
        mock_conn.rollback.assert_called_once()


class TestDownloadAndDecompress:
    @patch("scripts.core.backfill_learnings.subprocess")
    def test_success(self, mock_sub, tmp_path):
        # Make the JSONL file appear after "decompression"
        jsonl_file = tmp_path / "abc.jsonl"
        jsonl_file.write_text('{"type": "test"}\n')

        mock_sub.run.return_value = MagicMock(returncode=0)
        result = download_and_decompress(
            "s3://bucket/key.jsonl.zst", tmp_path, "abc"
        )
        assert result is not None
        assert mock_sub.run.call_count == 2  # download + decompress

    @patch("scripts.core.backfill_learnings.subprocess")
    def test_download_failure(self, mock_sub, tmp_path):
        mock_sub.run.return_value = MagicMock(returncode=1, stderr=b"denied")
        result = download_and_decompress(
            "s3://bucket/key.jsonl.zst", tmp_path, "abc"
        )
        assert result is None

    @patch("scripts.core.backfill_learnings.subprocess")
    def test_decompress_failure(self, mock_sub, tmp_path):
        # First call (download) succeeds, second (decompress) fails
        mock_sub.run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=1, stderr=b"corrupt"),
        ]
        result = download_and_decompress(
            "s3://bucket/key.jsonl.zst", tmp_path, "abc"
        )
        assert result is None


class TestRunExtraction:
    @patch("scripts.core.backfill_learnings.subprocess")
    def test_success(self, mock_sub):
        mock_sub.run.return_value = MagicMock(
            returncode=0,
            stdout="Learnings stored: 3\nDuplicates skipped: 1\n",
            stderr="",
        )
        result = run_extraction(
            jsonl_path=Path("/tmp/abc.jsonl"),
            session_id="s-abc",
            agent_prompt="Extract",
            model="sonnet",
            max_turns=15,
            timeout=300,
            project_dir=None,
        )
        assert result["status"] == "ok"
        assert result["learnings"] == 3
        assert result["duplicates"] == 1

    @patch("scripts.core.backfill_learnings.subprocess")
    def test_failure(self, mock_sub):
        mock_sub.run.return_value = MagicMock(
            returncode=1, stdout="", stderr="crash"
        )
        result = run_extraction(
            jsonl_path=Path("/tmp/abc.jsonl"),
            session_id="s-abc",
            agent_prompt="Extract",
            model="sonnet",
            max_turns=15,
            timeout=300,
            project_dir=None,
        )
        assert result["status"] == "extraction_failed"

    @patch("scripts.core.backfill_learnings.subprocess")
    def test_timeout(self, mock_sub):
        mock_sub.run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=300)
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        result = run_extraction(
            jsonl_path=Path("/tmp/abc.jsonl"),
            session_id="s-abc",
            agent_prompt="Extract",
            model="sonnet",
            max_turns=15,
            timeout=300,
            project_dir=None,
        )
        assert result["status"] == "extraction_timeout"


class TestLoadAgentPrompt:
    def test_with_frontmatter(self, tmp_path):
        agent_file = tmp_path / "memory-extractor.md"
        agent_file.write_text("---\ntitle: Agent\n---\nExtract learnings here")
        result = load_agent_prompt(agent_file)
        assert result == "Extract learnings here"

    def test_without_frontmatter(self, tmp_path):
        agent_file = tmp_path / "memory-extractor.md"
        agent_file.write_text("Just a plain prompt")
        result = load_agent_prompt(agent_file)
        assert result == "Just a plain prompt"

    def test_missing_file(self, tmp_path):
        agent_file = tmp_path / "missing.md"
        result = load_agent_prompt(agent_file)
        assert len(result) > 0  # returns fallback prompt


class TestMain:
    @patch("scripts.core.backfill_learnings._bootstrap")
    @patch("scripts.core.backfill_learnings.get_s3_bucket")
    def test_no_bucket_exits(self, mock_bucket, mock_boot):
        mock_bucket.return_value = None
        with patch("sys.argv", ["backfill_learnings.py"]):
            result = main()
        assert result != 0

    @patch("scripts.core.backfill_learnings._bootstrap")
    @patch("scripts.core.backfill_learnings.get_s3_bucket")
    @patch("scripts.core.backfill_learnings.get_pg_url")
    def test_no_db_blocks_non_dry_run(self, mock_pg, mock_bucket, mock_boot):
        mock_bucket.return_value = "my-bucket"
        mock_pg.return_value = None
        with patch("sys.argv", ["backfill_learnings.py"]):
            result = main()
        assert result != 0

    @patch("scripts.core.backfill_learnings._bootstrap")
    @patch("scripts.core.backfill_learnings.get_s3_bucket")
    @patch("scripts.core.backfill_learnings.get_pg_url")
    @patch("scripts.core.backfill_learnings.list_s3_keys")
    def test_dry_run(self, mock_list, mock_pg, mock_bucket, mock_boot, capsys):
        mock_bucket.return_value = "my-bucket"
        mock_pg.return_value = None
        mock_list.return_value = (
            "2026-01-01 00:00:00 100 sessions/-Users-opc/"
            "abc12345-0000-0000-0000-000000000001.jsonl.zst\n"
        )
        with patch("sys.argv", ["backfill_learnings.py", "--dry-run"]):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "abc12345" in captured.out

    @patch("scripts.core.backfill_learnings._bootstrap")
    @patch("scripts.core.backfill_learnings.get_s3_bucket")
    @patch("scripts.core.backfill_learnings.get_pg_url")
    @patch("scripts.core.backfill_learnings.list_s3_keys")
    @patch("scripts.core.backfill_learnings.load_agent_prompt")
    @patch("scripts.core.backfill_learnings._process_one")
    @patch("scripts.core.backfill_learnings.log_extraction_result")
    @patch("scripts.core.backfill_learnings.is_session_extracted")
    @patch("scripts.core.backfill_learnings.lookup_session_id")
    def test_real_run_processes_sessions(
        self,
        mock_lookup,
        mock_is_ext,
        mock_log_result,
        mock_process,
        mock_prompt,
        mock_list,
        mock_pg,
        mock_bucket,
        mock_boot,
        capsys,
    ):
        mock_bucket.return_value = "my-bucket"
        mock_pg.return_value = "pg://test"
        mock_prompt.return_value = "Extract"
        mock_lookup.return_value = None  # fall back to UUID
        mock_is_ext.return_value = False
        mock_list.return_value = (
            "2026-01-01 00:00:00 100 sessions/-Users-opc/"
            "abc12345-0000-0000-0000-000000000001.jsonl.zst\n"
        )
        mock_process.return_value = {
            "status": "ok", "learnings": 3, "duplicates": 1,
            "uuid": "abc12345-0000-0000-0000-000000000001",
            "session_id": "abc12345-0000-0000-0000-000000000001",
        }
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        with (
            patch("sys.argv", ["backfill_learnings.py", "--limit", "1", "--workers", "1"]),
            patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        ):
            result = main()
        assert result == 0
        mock_process.assert_called_once()
        captured = capsys.readouterr()
        assert "OK" in captured.out

    @patch("scripts.core.backfill_learnings._bootstrap")
    @patch("scripts.core.backfill_learnings.get_s3_bucket")
    @patch("scripts.core.backfill_learnings.get_pg_url")
    @patch("scripts.core.backfill_learnings.list_s3_keys")
    def test_no_sessions_found(self, mock_list, mock_pg, mock_bucket, mock_boot, capsys):
        mock_bucket.return_value = "my-bucket"
        mock_pg.return_value = "pg://test"
        mock_list.return_value = ""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.return_value = MagicMock()
        with (
            patch("sys.argv", ["backfill_learnings.py"]),
            patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        ):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "Found 0" in captured.out


class TestProcessOne:
    @patch("scripts.core.backfill_learnings.run_extraction")
    @patch("scripts.core.backfill_learnings.download_and_decompress")
    def test_download_failure(self, mock_dl, mock_run):
        mock_dl.return_value = None
        session = {
            "uuid": "abc", "session_id": "s-abc",
            "s3_key": "s3://b/k", "project": "opc",
        }
        result = _process_one(session, "prompt", "sonnet", 15, 300)
        assert result["status"] == "download_failed"
        mock_run.assert_not_called()

    @patch("scripts.core.backfill_learnings.run_extraction")
    @patch("scripts.core.backfill_learnings.download_and_decompress")
    def test_success(self, mock_dl, mock_run, tmp_path):
        jsonl = tmp_path / "abc.jsonl"
        jsonl.write_text('{"type": "test"}\n')
        mock_dl.return_value = jsonl
        mock_run.return_value = {"status": "ok", "learnings": 5, "duplicates": 2}
        session = {
            "uuid": "abc", "session_id": "s-abc",
            "s3_key": "s3://b/k", "project": "opc",
        }
        result = _process_one(session, "prompt", "sonnet", 15, 300)
        assert result["status"] == "ok"
        assert result["learnings"] == 5
        assert result["file_size"] > 0

    @patch("scripts.core.backfill_learnings.run_extraction")
    @patch("scripts.core.backfill_learnings.download_and_decompress")
    def test_does_not_write_db(self, mock_dl, mock_run, tmp_path):
        """_process_one never touches the DB — caller serializes writes."""
        jsonl = tmp_path / "abc.jsonl"
        jsonl.write_text('{"type": "test"}\n')
        mock_dl.return_value = jsonl
        mock_run.return_value = {"status": "ok", "learnings": 3, "duplicates": 0}
        session = {
            "uuid": "abc", "session_id": "s-abc",
            "s3_key": "s3://b/k", "project": "opc",
        }
        result = _process_one(session, "prompt", "sonnet", 15, 300)
        # Result is returned but no DB interaction occurred
        assert result["status"] == "ok"

    @patch("scripts.core.backfill_learnings.run_extraction")
    @patch("scripts.core.backfill_learnings.download_and_decompress")
    def test_passes_project_to_extraction(self, mock_dl, mock_run, tmp_path):
        jsonl = tmp_path / "abc.jsonl"
        jsonl.write_text('{"type": "test"}\n')
        mock_dl.return_value = jsonl
        mock_run.return_value = {"status": "ok", "learnings": 1, "duplicates": 0}
        session = {
            "uuid": "abc", "session_id": "s-abc",
            "s3_key": "s3://b/k", "project": "-Users-myproj",
        }
        _process_one(session, "prompt", "sonnet", 15, 300)
        # Verify project was passed as project_dir to run_extraction
        _, kwargs = mock_run.call_args
        # run_extraction is called positionally
        args = mock_run.call_args[0]
        assert args[6] == "-Users-myproj"  # project_dir argument
