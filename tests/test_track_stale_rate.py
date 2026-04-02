"""Tests for scripts/core/track_stale_rate.py — pure functions and I/O handlers."""

import csv as csv_mod
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.core.track_stale_rate import (
    _build_upserted_rows,
    _cli_main,
    compute_stale_stats,
    fetch_stale_counts,
    format_stale_line,
    main,
    upsert_csv_row,
)

# --- compute_stale_stats (pure) ---


class TestComputeStaleStats:
    def test_normal_values(self):
        total, stale, date_str = 100, 25, "2026-04-02"
        result = compute_stale_stats(total, stale, date_str)
        assert result == {
            "date": "2026-04-02",
            "total": 100,
            "stale": 25,
            "stale_pct": 25.0,
        }

    def test_zero_total_returns_zero_pct(self):
        result = compute_stale_stats(0, 0, "2026-04-02")
        assert result["stale_pct"] == 0.0
        assert result["total"] == 0

    def test_all_stale(self):
        result = compute_stale_stats(50, 50, "2026-01-01")
        assert result["stale_pct"] == 100.0

    def test_none_stale(self):
        result = compute_stale_stats(200, 0, "2026-06-15")
        assert result["stale_pct"] == 0.0

    def test_rounding(self):
        result = compute_stale_stats(3, 1, "2026-04-02")
        assert result["stale_pct"] == 33.3

    def test_stale_exceeds_total_still_computes(self):
        result = compute_stale_stats(10, 15, "2026-04-02")
        assert result["stale_pct"] == 150.0


# --- format_stale_line (pure) ---


class TestFormatStaleLine:
    def test_normal_format(self):
        stats = {"date": "2026-04-02", "total": 100, "stale": 25, "stale_pct": 25.0}
        result = format_stale_line(stats)
        assert result == "2026-04-02: 25/100 stale (25.0%)"

    def test_zero_values(self):
        stats = {"date": "2026-04-02", "total": 0, "stale": 0, "stale_pct": 0.0}
        result = format_stale_line(stats)
        assert result == "2026-04-02: 0/0 stale (0.0%)"


# --- upsert_csv_row (I/O handler — atomic upsert) ---


class TestUpsertCsvRow:
    def test_inserts_to_new_file(self, tmp_path: Path):
        log_path = tmp_path / "stale_rate_log.csv"
        stats = {"date": "2026-04-02", "total": 100, "stale": 25, "stale_pct": 25.0}

        action = upsert_csv_row(stats, log_path)

        assert action == "inserted"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "date,total,stale,stale_pct"
        assert lines[1] == "2026-04-02,100,25,25.0"

    def test_inserts_different_dates(self, tmp_path: Path):
        log_path = tmp_path / "stale_rate_log.csv"
        stats1 = {"date": "2026-04-01", "total": 90, "stale": 20, "stale_pct": 22.2}
        stats2 = {"date": "2026-04-02", "total": 100, "stale": 25, "stale_pct": 25.0}

        upsert_csv_row(stats1, log_path)
        upsert_csv_row(stats2, log_path)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert lines[0] == "date,total,stale,stale_pct"

    def test_creates_parent_directories(self, tmp_path: Path):
        log_path = tmp_path / "nested" / "dir" / "log.csv"
        stats = {"date": "2026-04-02", "total": 10, "stale": 1, "stale_pct": 10.0}

        upsert_csv_row(stats, log_path)

        assert log_path.exists()

    def test_unchanged_when_same_data(self, tmp_path: Path):
        log_path = tmp_path / "stale_rate_log.csv"
        stats = {"date": "2026-04-02", "total": 100, "stale": 25, "stale_pct": 25.0}

        first = upsert_csv_row(stats, log_path)
        second = upsert_csv_row(stats, log_path)

        assert first == "inserted"
        assert second == "unchanged"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 row (not duplicated)

    def test_updates_when_data_changes(self, tmp_path: Path):
        log_path = tmp_path / "stale_rate_log.csv"
        stats_v1 = {"date": "2026-04-02", "total": 100, "stale": 25, "stale_pct": 25.0}
        stats_v2 = {"date": "2026-04-02", "total": 110, "stale": 20, "stale_pct": 18.2}

        first = upsert_csv_row(stats_v1, log_path)
        second = upsert_csv_row(stats_v2, log_path)

        assert first == "inserted"
        assert second == "updated"
        reader = csv_mod.DictReader(io.StringIO(log_path.read_text()))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["total"] == "110"
        assert rows[0]["stale"] == "20"

    def test_preserves_other_dates_on_update(self, tmp_path: Path):
        log_path = tmp_path / "stale_rate_log.csv"
        day1 = {"date": "2026-04-01", "total": 90, "stale": 20, "stale_pct": 22.2}
        day2_v1 = {"date": "2026-04-02", "total": 100, "stale": 25, "stale_pct": 25.0}
        day2_v2 = {"date": "2026-04-02", "total": 105, "stale": 22, "stale_pct": 21.0}

        upsert_csv_row(day1, log_path)
        upsert_csv_row(day2_v1, log_path)
        upsert_csv_row(day2_v2, log_path)

        reader = csv_mod.DictReader(io.StringIO(log_path.read_text()))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["date"] == "2026-04-01"
        assert rows[0]["total"] == "90"
        assert rows[1]["date"] == "2026-04-02"
        assert rows[1]["total"] == "105"


    def test_collapses_duplicate_dates(self, tmp_path: Path):
        log_path = tmp_path / "stale_rate_log.csv"
        # Pre-populate with duplicate dates (simulating prior corruption)
        log_path.write_text(
            "date,total,stale,stale_pct\n"
            "2026-04-02,100,25,25.0\n"
            "2026-04-02,100,25,25.0\n"
        )
        stats = {"date": "2026-04-02", "total": 100, "stale": 25, "stale_pct": 25.0}

        action = upsert_csv_row(stats, log_path)

        # Should collapse duplicates even though data matches
        assert action == "unchanged"
        reader = csv_mod.DictReader(io.StringIO(log_path.read_text()))
        rows = list(reader)
        assert len(rows) == 1

    def test_collapses_duplicates_on_update(self, tmp_path: Path):
        log_path = tmp_path / "stale_rate_log.csv"
        log_path.write_text(
            "date,total,stale,stale_pct\n"
            "2026-04-01,80,15,18.8\n"
            "2026-04-02,100,25,25.0\n"
            "2026-04-02,100,25,25.0\n"
        )
        stats = {"date": "2026-04-02", "total": 110, "stale": 20, "stale_pct": 18.2}

        action = upsert_csv_row(stats, log_path)

        assert action == "updated"
        reader = csv_mod.DictReader(io.StringIO(log_path.read_text()))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[1]["total"] == "110"


# --- _build_upserted_rows (pure) ---


class TestBuildUpsertedRows:
    def test_insert_into_empty(self):
        new_row = {"date": "2026-04-02", "total": "100", "stale": "25", "stale_pct": "25.0"}
        rows, action = _build_upserted_rows([], new_row, "2026-04-02")
        assert action == "inserted"
        assert len(rows) == 1
        assert rows[0] == new_row

    def test_unchanged(self):
        existing = [{"date": "2026-04-02", "total": "100", "stale": "25", "stale_pct": "25.0"}]
        new_row = {"date": "2026-04-02", "total": "100", "stale": "25", "stale_pct": "25.0"}
        rows, action = _build_upserted_rows(existing, new_row, "2026-04-02")
        assert action == "unchanged"
        assert len(rows) == 1

    def test_update(self):
        existing = [{"date": "2026-04-02", "total": "100", "stale": "25", "stale_pct": "25.0"}]
        new_row = {"date": "2026-04-02", "total": "110", "stale": "20", "stale_pct": "18.2"}
        rows, action = _build_upserted_rows(existing, new_row, "2026-04-02")
        assert action == "updated"
        assert len(rows) == 1
        assert rows[0]["total"] == "110"

    def test_collapses_duplicates(self):
        existing = [
            {"date": "2026-04-02", "total": "100", "stale": "25", "stale_pct": "25.0"},
            {"date": "2026-04-02", "total": "100", "stale": "25", "stale_pct": "25.0"},
        ]
        new_row = {"date": "2026-04-02", "total": "100", "stale": "25", "stale_pct": "25.0"}
        rows, action = _build_upserted_rows(existing, new_row, "2026-04-02")
        assert action == "unchanged"
        assert len(rows) == 1


# --- fetch_stale_counts (I/O handler, async) ---


class FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *args):
        return None


class TestFetchStaleCounts:
    async def test_returns_total_and_stale(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"total": 150, "stale": 30})
        pool = MagicMock()
        pool.acquire.return_value = FakeAcquire(conn)

        total, stale = await fetch_stale_counts(pool)

        assert total == 150
        assert stale == 30
        assert conn.fetchrow.call_count == 1


# --- main (orchestrator, async) ---


class TestMain:
    @patch("scripts.core.track_stale_rate.get_pool")
    async def test_main_orchestrates_all_steps(self, mock_get_pool, tmp_path: Path):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"total": 100, "stale": 25})

        pool = MagicMock()
        pool.acquire.return_value = FakeAcquire(conn)
        mock_get_pool.return_value = pool

        log_path = tmp_path / "stale_rate_log.csv"

        with patch("scripts.core.track_stale_rate.LOG_PATH", log_path):
            await main()

        assert log_path.exists()
        reader = csv_mod.DictReader(io.StringIO(log_path.read_text()))
        row = next(reader)
        assert row["total"] == "100"
        assert row["stale"] == "25"
        assert row["stale_pct"] == "25.0"


# --- _cli_main (lifecycle) ---


class TestCliMain:
    @patch("scripts.core.track_stale_rate.close_pool", new_callable=AsyncMock)
    @patch("scripts.core.track_stale_rate.main", new_callable=AsyncMock)
    async def test_closes_pool_on_success(self, mock_main, mock_close_pool):
        await _cli_main()

        mock_main.assert_awaited_once()
        mock_close_pool.assert_awaited_once()

    @patch("scripts.core.track_stale_rate.close_pool", new_callable=AsyncMock)
    @patch("scripts.core.track_stale_rate.main", new_callable=AsyncMock)
    async def test_closes_pool_on_error(self, mock_main, mock_close_pool):
        mock_main.side_effect = RuntimeError("db failure")

        with pytest.raises(RuntimeError, match="db failure"):
            await _cli_main()

        mock_close_pool.assert_awaited_once()
