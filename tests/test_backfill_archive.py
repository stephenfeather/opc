"""Tests for scripts/core/backfill_archive.py - TDD+FP refactor (S15)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.core.backfill_archive import (
    _safe_restore,
    archive_jsonl,
    build_archive_summary,
    build_s3_key,
    build_zst_path,
    compress_file,
    decompress_file,
    filter_recent_jsonls,
    find_archivable_jsonls,
    format_dry_run_info,
    get_pg_url,
    main,
    mark_archived_in_db,
    upload_to_s3,
)
from scripts.core.config.models import ArchivalConfig

# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestGetPgUrl:
    def test_returns_continuous_claude_db_url_first(self):
        env = {
            "CONTINUOUS_CLAUDE_DB_URL": "postgresql://cc",
            "DATABASE_URL": "postgresql://a",
        }
        with patch.dict("os.environ", env, clear=True):
            assert get_pg_url() == "postgresql://cc"

    def test_falls_back_to_database_url(self):
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://a"}, clear=True):
            assert get_pg_url() == "postgresql://a"

    def test_falls_back_to_postgres_url(self):
        with patch.dict("os.environ", {"POSTGRES_URL": "postgresql://b"}, clear=True):
            assert get_pg_url() == "postgresql://b"

    def test_returns_none_when_not_set(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_pg_url() is None


class TestBuildS3Key:
    def test_builds_correct_key(self):
        result = build_s3_key("my-bucket", "Users-foo-bar", "abc-123")
        assert result == "s3://my-bucket/sessions/Users-foo-bar/abc-123.jsonl.zst"

    def test_special_characters_in_project(self):
        result = build_s3_key("b", "my-proj_v2", "sess-1")
        assert result == "s3://b/sessions/my-proj_v2/sess-1.jsonl.zst"


class TestBuildZstPath:
    def test_appends_zst_suffix(self):
        result = build_zst_path(Path("/tmp/foo.jsonl"))
        assert result == Path("/tmp/foo.jsonl.zst")

    def test_preserves_parent_directory(self):
        result = build_zst_path(Path("/home/user/.claude/projects/proj/sess.jsonl"))
        assert result.parent == Path("/home/user/.claude/projects/proj")

    def test_returns_path_type(self):
        assert isinstance(build_zst_path(Path("/tmp/x.jsonl")), Path)


class TestFormatDryRunInfo:
    def test_expected_keys(self):
        result = format_dry_run_info(Path("/tmp/proj/sess.jsonl"), 1024)
        assert set(result.keys()) == {"path", "name", "size_mb", "project", "session_id"}

    def test_size_conversion(self):
        result = format_dry_run_info(Path("/tmp/p/s.jsonl"), 2 * 1024 * 1024)
        assert abs(result["size_mb"] - 2.0) < 0.01

    def test_extracts_project_and_session(self):
        result = format_dry_run_info(Path("/tmp/Users-foo/abc-123.jsonl"), 0)
        assert result["project"] == "Users-foo"
        assert result["session_id"] == "abc-123"


class TestFilterRecentJsonls:
    def test_filters_recent_files(self):
        now = datetime.now()
        files = [
            {"path": Path("/a.jsonl"), "mtime": now - timedelta(minutes=30)},
            {"path": Path("/b.jsonl"), "mtime": now - timedelta(minutes=5)},
        ]
        cutoff = now - timedelta(minutes=10)
        result = filter_recent_jsonls(files, cutoff)
        assert len(result) == 1
        assert result[0]["path"] == Path("/a.jsonl")

    def test_keeps_old_files(self):
        now = datetime.now()
        files = [
            {"path": Path("/a.jsonl"), "mtime": now - timedelta(hours=1)},
            {"path": Path("/b.jsonl"), "mtime": now - timedelta(hours=2)},
        ]
        result = filter_recent_jsonls(files, now - timedelta(minutes=10))
        assert len(result) == 2

    def test_all_recent_returns_empty(self):
        now = datetime.now()
        files = [{"path": Path("/a.jsonl"), "mtime": now - timedelta(seconds=30)}]
        result = filter_recent_jsonls(files, now - timedelta(minutes=10))
        assert result == []

    def test_empty_input_returns_empty(self):
        assert filter_recent_jsonls([], datetime.now()) == []

    def test_does_not_mutate_input(self):
        now = datetime.now()
        files = [{"path": Path("/a.jsonl"), "mtime": now - timedelta(minutes=5)}]
        original_len = len(files)
        filter_recent_jsonls(files, now - timedelta(minutes=10))
        assert len(files) == original_len


class TestBuildArchiveSummary:
    def test_returns_all_fields(self):
        result = build_archive_summary(total=10, skipped=3, success=5, failed=2)
        assert set(result.keys()) == {"total", "skipped", "archived", "failed"}

    def test_values_match_inputs(self):
        result = build_archive_summary(total=100, skipped=5, success=90, failed=5)
        assert result["total"] == 100
        assert result["skipped"] == 5
        assert result["archived"] == 90
        assert result["failed"] == 5


# ---------------------------------------------------------------------------
# I/O function tests (mocked)
# ---------------------------------------------------------------------------


class TestCompressFile:
    @patch("scripts.core.backfill_archive.subprocess.run")
    def test_calls_zstd_with_correct_args(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        compress_file(Path("/tmp/sess.jsonl"), timeout=300)
        mock_run.assert_called_once_with(
            ["zstd", "-q", "--rm", "/tmp/sess.jsonl"],
            capture_output=True,
            timeout=300,
        )

    @patch("scripts.core.backfill_archive.subprocess.run")
    def test_passes_timeout(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        compress_file(Path("/tmp/x.jsonl"), timeout=42)
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 42


class TestUploadToS3:
    @patch("scripts.core.backfill_archive.subprocess.run")
    def test_calls_aws_s3_cp(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        upload_to_s3(Path("/tmp/sess.jsonl.zst"), "s3://b/key", timeout=300)
        mock_run.assert_called_once_with(
            ["aws", "s3", "cp", "/tmp/sess.jsonl.zst", "s3://b/key", "--quiet"],
            capture_output=True,
            timeout=300,
        )

    @patch("scripts.core.backfill_archive.subprocess.run")
    def test_passes_timeout(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        upload_to_s3(Path("/tmp/x.zst"), "s3://b/k", timeout=99)
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 99


class TestDecompressFile:
    @patch("scripts.core.backfill_archive.subprocess.run")
    def test_calls_zstd_decompress(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        decompress_file(Path("/tmp/sess.jsonl.zst"), timeout=300)
        mock_run.assert_called_once_with(
            ["zstd", "-d", "-q", "--rm", "/tmp/sess.jsonl.zst"],
            capture_output=True,
            timeout=300,
        )

    @patch("scripts.core.backfill_archive.subprocess.run")
    def test_passes_timeout(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        decompress_file(Path("/tmp/x.zst"), timeout=77)
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 77


class TestMarkArchivedInDb:
    @patch("scripts.core.backfill_archive.psycopg2")
    def test_updates_sessions_table(self, mock_pg):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        result = mark_archived_in_db("sess-1", "s3://b/path", "postgresql://test")
        assert result == 1

    @patch("scripts.core.backfill_archive.psycopg2")
    def test_updates_archival_memory_when_rows_affected(self, mock_pg):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        mark_archived_in_db("sess-1", "s3://b/path", "postgresql://test")
        assert mock_cur.execute.call_count == 2

    @patch("scripts.core.backfill_archive.psycopg2")
    def test_skips_archival_memory_when_no_rows(self, mock_pg):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 0
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        result = mark_archived_in_db("no-match", "s3://b/path", "postgresql://test")
        assert result == 0
        assert mock_cur.execute.call_count == 1

    @patch("scripts.core.backfill_archive.psycopg2")
    def test_commits_and_closes(self, mock_pg):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 0
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        mark_archived_in_db("sess-1", "s3://b/path", "postgresql://test")
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_returns_none_when_no_url(self):
        result = mark_archived_in_db("sess-1", "s3://b/path", None)
        assert result is None

    @patch("scripts.core.backfill_archive.psycopg2")
    def test_handles_db_error_gracefully(self, mock_pg, capsys):
        mock_pg.connect.side_effect = Exception("connection refused")
        result = mark_archived_in_db("sess-1", "s3://b/path", "postgresql://bad")
        assert result == 0
        output = capsys.readouterr().out
        assert "postgresql://" not in output
        assert "Exception" in output

    @patch("scripts.core.backfill_archive.psycopg2")
    def test_closes_connection_on_error(self, mock_pg):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.execute.side_effect = Exception("query failed")
        mock_conn.cursor.return_value = mock_cur
        mock_pg.connect.return_value = mock_conn

        mark_archived_in_db("sess-1", "s3://b/path", "postgresql://test")
        mock_conn.close.assert_called_once()


class TestFindArchivableJsonls:
    def test_finds_jsonl_files(self, tmp_path):
        proj_dir = tmp_path / "projects" / "proj"
        proj_dir.mkdir(parents=True)
        (proj_dir / "a.jsonl").write_text('{"test": true}\n')
        (proj_dir / "b.jsonl").write_text('{"test": true}\n')

        result = find_archivable_jsonls(tmp_path)
        assert len(result) == 2
        assert all("path" in f and "mtime" in f and "size" in f for f in result)

    def test_returns_empty_for_missing_dir(self, tmp_path):
        result = find_archivable_jsonls(tmp_path / "nonexistent")
        assert result == []

    def test_skips_symlinks(self, tmp_path):
        proj_dir = tmp_path / "projects" / "proj"
        proj_dir.mkdir(parents=True)
        real = proj_dir / "real.jsonl"
        real.write_text('{"test": true}\n')
        link = proj_dir / "link.jsonl"
        link.symlink_to(real)

        result = find_archivable_jsonls(tmp_path)
        paths = [f["path"].name for f in result]
        assert "real.jsonl" in paths
        assert "link.jsonl" not in paths

    def test_sorted_by_mtime(self, tmp_path):
        import os
        import time

        proj_dir = tmp_path / "projects" / "proj"
        proj_dir.mkdir(parents=True)
        old = proj_dir / "old.jsonl"
        old.write_text("old\n")
        os.utime(old, (1000000, 1000000))
        time.sleep(0.01)
        new = proj_dir / "new.jsonl"
        new.write_text("new\n")

        result = find_archivable_jsonls(tmp_path)
        assert result[0]["path"].name == "old.jsonl"
        assert result[1]["path"].name == "new.jsonl"


class TestSafeRestore:
    @patch("scripts.core.backfill_archive.decompress_file")
    def test_returns_true_on_success(self, mock_decompress):
        mock_decompress.return_value = MagicMock(returncode=0)
        assert _safe_restore(Path("/tmp/x.jsonl.zst"), 60) is True

    @patch("scripts.core.backfill_archive.decompress_file")
    def test_returns_false_on_nonzero_return(self, mock_decompress, capsys):
        mock_decompress.return_value = MagicMock(returncode=1)
        assert _safe_restore(Path("/tmp/x.jsonl.zst"), 60) is False
        assert "WARNING" in capsys.readouterr().out

    @patch("scripts.core.backfill_archive.decompress_file", side_effect=Exception("boom"))
    def test_returns_false_on_exception(self, mock_decompress, capsys):
        assert _safe_restore(Path("/tmp/x.jsonl.zst"), 60) is False
        assert "WARNING" in capsys.readouterr().out


class TestArchiveJsonl:
    def _make_jsonl(self, tmp_path):
        p = tmp_path / "proj" / "sess-abc.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"test": true}\n')
        return p

    def _cfg(self):
        return ArchivalConfig(compress_timeout=60, upload_timeout=60, skip_recent_minutes=10)

    def test_dry_run_returns_true_without_subprocess(self, tmp_path):
        p = self._make_jsonl(tmp_path)
        with patch("scripts.core.backfill_archive.compress_file") as mock_c:
            result = archive_jsonl(p, "my-bucket", self._cfg(), None, dry_run=True)
            mock_c.assert_not_called()
        assert result is True

    @patch("scripts.core.backfill_archive.mark_archived_in_db", return_value=1)
    @patch("scripts.core.backfill_archive.upload_to_s3")
    @patch("scripts.core.backfill_archive.compress_file")
    def test_success_flow(self, mock_compress, mock_upload, mock_mark, tmp_path):
        p = self._make_jsonl(tmp_path)
        zst = p.with_suffix(".jsonl.zst")
        zst.write_bytes(b"compressed")
        mock_compress.return_value = MagicMock(returncode=0)
        mock_upload.return_value = MagicMock(returncode=0)

        result = archive_jsonl(p, "my-bucket", self._cfg(), "postgresql://test")

        assert result is True
        mock_compress.assert_called_once()
        mock_upload.assert_called_once()
        mock_mark.assert_called_once()

    @patch("scripts.core.backfill_archive.compress_file")
    def test_compress_failure_returns_false(self, mock_compress, tmp_path):
        p = self._make_jsonl(tmp_path)
        mock_compress.return_value = MagicMock(returncode=1, stderr=b"error")

        result = archive_jsonl(p, "my-bucket", self._cfg(), None)
        assert result is False

    @patch("scripts.core.backfill_archive.decompress_file")
    @patch("scripts.core.backfill_archive.upload_to_s3")
    @patch("scripts.core.backfill_archive.compress_file")
    def test_upload_failure_triggers_decompress(
        self, mock_compress, mock_upload, mock_decompress, tmp_path
    ):
        p = self._make_jsonl(tmp_path)
        mock_compress.return_value = MagicMock(returncode=0)
        mock_upload.return_value = MagicMock(returncode=1, stderr=b"upload fail")

        result = archive_jsonl(p, "my-bucket", self._cfg(), None)

        assert result is False
        mock_decompress.assert_called_once()

    @patch("scripts.core.backfill_archive.decompress_file")
    @patch("scripts.core.backfill_archive.compress_file")
    def test_timeout_restores_when_zst_exists(self, mock_compress, mock_decompress, tmp_path):
        p = self._make_jsonl(tmp_path)
        zst = p.with_suffix(".jsonl.zst")
        zst.write_bytes(b"compressed")
        # Remove the original to simulate zstd --rm having run
        p.unlink()
        mock_compress.side_effect = subprocess.TimeoutExpired(cmd="zstd", timeout=60)

        result = archive_jsonl(p, "my-bucket", self._cfg(), None)

        assert result is False
        mock_decompress.assert_called_once()

    @patch("scripts.core.backfill_archive.mark_archived_in_db")
    @patch("scripts.core.backfill_archive.upload_to_s3")
    @patch("scripts.core.backfill_archive.compress_file")
    def test_skips_db_when_no_url(self, mock_compress, mock_upload, mock_mark, tmp_path):
        p = self._make_jsonl(tmp_path)
        zst = p.with_suffix(".jsonl.zst")
        zst.write_bytes(b"compressed")
        mock_compress.return_value = MagicMock(returncode=0)
        mock_upload.return_value = MagicMock(returncode=0)

        result = archive_jsonl(p, "my-bucket", self._cfg(), db_url=None)

        assert result is True
        mock_mark.assert_not_called()

    @patch("scripts.core.backfill_archive.mark_archived_in_db", return_value=0)
    @patch("scripts.core.backfill_archive.upload_to_s3")
    @patch("scripts.core.backfill_archive.compress_file")
    def test_db_failure_is_best_effort_after_upload(
        self, mock_compress, mock_upload, mock_mark, tmp_path, capsys
    ):
        p = self._make_jsonl(tmp_path)
        zst = p.with_suffix(".jsonl.zst")
        zst.write_bytes(b"compressed")
        mock_compress.return_value = MagicMock(returncode=0)
        mock_upload.return_value = MagicMock(returncode=0)

        result = archive_jsonl(p, "my-bucket", self._cfg(), "postgresql://test")

        assert result is True  # S3 upload succeeded, DB is best-effort
        assert "Warning: DB mark failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Main tests (integration, mocked I/O)
# ---------------------------------------------------------------------------


class TestMain:
    @patch("scripts.core.backfill_archive._bootstrap")
    def test_no_bucket_returns_1(self, mock_boot, capsys):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("CLAUDE_SESSION_ARCHIVE_BUCKET", None)
            with patch("sys.argv", ["backfill_archive.py"]):
                result = main()
        assert result == 1
        assert "CLAUDE_SESSION_ARCHIVE_BUCKET" in capsys.readouterr().out

    @patch("scripts.core.backfill_archive._bootstrap")
    @patch("scripts.core.backfill_archive.archive_jsonl", return_value=True)
    @patch("scripts.core.backfill_archive.find_archivable_jsonls")
    def test_dry_run_passes_flag(self, mock_find, mock_archive, mock_boot, tmp_path):
        now = datetime.now()
        mock_find.return_value = [
            {"path": Path("/tmp/sess.jsonl"), "mtime": now - timedelta(hours=1), "size": 1024},
        ]
        env = {
            "CLAUDE_SESSION_ARCHIVE_BUCKET": "test-bucket",
            "CLAUDE_CONFIG_DIR": str(tmp_path),
        }
        with patch.dict("os.environ", env):
            with patch("sys.argv", ["backfill_archive.py", "--dry-run"]):
                result = main()

        assert result == 0
        mock_archive.assert_called_once()
        call_kwargs = mock_archive.call_args
        # dry_run is the last positional or keyword arg
        assert call_kwargs[1].get("dry_run") is True or call_kwargs[0][-1] is True

    @patch("scripts.core.backfill_archive._bootstrap")
    @patch("scripts.core.backfill_archive.archive_jsonl")
    @patch("scripts.core.backfill_archive.find_archivable_jsonls")
    def test_archives_and_returns_0(self, mock_find, mock_archive, mock_boot, tmp_path):
        now = datetime.now()
        mock_find.return_value = [
            {"path": Path("/tmp/a.jsonl"), "mtime": now - timedelta(hours=1), "size": 100},
            {"path": Path("/tmp/b.jsonl"), "mtime": now - timedelta(hours=2), "size": 200},
        ]
        mock_archive.side_effect = [True, True]
        env = {
            "CLAUDE_SESSION_ARCHIVE_BUCKET": "test-bucket",
            "CLAUDE_CONFIG_DIR": str(tmp_path),
        }
        with patch.dict("os.environ", env):
            with patch("sys.argv", ["backfill_archive.py"]):
                result = main()

        assert result == 0
        assert mock_archive.call_count == 2

    @patch("scripts.core.backfill_archive._bootstrap")
    @patch("scripts.core.backfill_archive.archive_jsonl")
    @patch("scripts.core.backfill_archive.find_archivable_jsonls")
    def test_returns_nonzero_on_failures(self, mock_find, mock_archive, mock_boot, tmp_path):
        now = datetime.now()
        mock_find.return_value = [
            {"path": Path("/tmp/a.jsonl"), "mtime": now - timedelta(hours=1), "size": 100},
        ]
        mock_archive.return_value = False
        env = {
            "CLAUDE_SESSION_ARCHIVE_BUCKET": "test-bucket",
            "CLAUDE_CONFIG_DIR": str(tmp_path),
        }
        with patch.dict("os.environ", env):
            with patch("sys.argv", ["backfill_archive.py"]):
                result = main()

        assert result == 1

    @patch("scripts.core.backfill_archive._bootstrap")
    @patch("scripts.core.backfill_archive.archive_jsonl")
    @patch("scripts.core.backfill_archive.find_archivable_jsonls")
    def test_prints_summary(self, mock_find, mock_archive, mock_boot, tmp_path, capsys):
        now = datetime.now()
        mock_find.return_value = [
            {"path": Path("/tmp/a.jsonl"), "mtime": now - timedelta(hours=1), "size": 100},
            {"path": Path("/tmp/b.jsonl"), "mtime": now - timedelta(hours=2), "size": 200},
            {"path": Path("/tmp/c.jsonl"), "mtime": now - timedelta(hours=3), "size": 300},
        ]
        mock_archive.side_effect = [True, False, True]
        env = {
            "CLAUDE_SESSION_ARCHIVE_BUCKET": "test-bucket",
            "CLAUDE_CONFIG_DIR": str(tmp_path),
        }
        with patch.dict("os.environ", env):
            with patch("sys.argv", ["backfill_archive.py"]):
                main()

        output = capsys.readouterr().out
        assert "2 archived" in output
        assert "1 failed" in output
