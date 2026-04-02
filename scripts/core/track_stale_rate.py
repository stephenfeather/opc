#!/usr/bin/env python3
"""Track stale learning rate over time with date-keyed CSV upserts."""

import asyncio
import csv
import fcntl
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.core.db.postgres_pool import close_pool, get_pool  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = PROJECT_ROOT / "thoughts" / "shared" / "stale_rate_log.csv"


# --- Pure functions ---


def compute_stale_stats(total: int, stale: int, date_str: str) -> dict[str, str | int | float]:
    stale_pct = round(stale / total * 100, 1) if total > 0 else 0.0
    return {
        "date": date_str,
        "total": total,
        "stale": stale,
        "stale_pct": stale_pct,
    }


def format_stale_line(stats: dict[str, str | int | float]) -> str:
    return f"{stats['date']}: {stats['stale']}/{stats['total']} stale ({stats['stale_pct']}%)"


# --- I/O handlers ---


async def fetch_stale_counts(pool) -> tuple[int, int]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT"
            " COUNT(*) AS total,"
            " COUNT(*) FILTER (WHERE recall_count = 0) AS stale"
            " FROM archival_memory WHERE superseded_by IS NULL"
        )
    return int(row["total"]), int(row["stale"])


FIELDNAMES = ["date", "total", "stale", "stale_pct"]


def _build_upserted_rows(
    existing_rows: list[dict[str, str]], new_row: dict[str, str], date_str: str
) -> tuple[list[dict[str, str]], str]:
    """Collapse duplicates and upsert by date. Returns (rows, action)."""
    found = False
    action = "inserted"
    updated_rows: list[dict[str, str]] = []

    for row in existing_rows:
        if row.get("date") == date_str:
            if not found:
                found = True
                updated_rows.append(new_row)
                action = "unchanged" if row == new_row else "updated"
            # Skip duplicate rows for same date (collapse them)
        else:
            updated_rows.append(row)

    if not found:
        updated_rows.append(new_row)
        action = "inserted"

    return updated_rows, action


def upsert_csv_row(
    stats: dict[str, str | int | float], log_path: Path
) -> str:
    """Atomically upsert a row keyed by date. Returns 'inserted', 'updated', or 'unchanged'."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = log_path.with_suffix(".lock")

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            existing_rows: list[dict[str, str]] = []
            if log_path.exists() and log_path.stat().st_size > 0:
                with open(log_path, newline="") as f:
                    existing_rows = list(csv.DictReader(f))

            date_str = str(stats["date"])
            new_row = {k: str(v) for k, v in stats.items()}
            updated_rows, action = _build_upserted_rows(existing_rows, new_row, date_str)

            if action == "unchanged" and len(existing_rows) == len(updated_rows):
                return action

            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(log_path.parent), suffix=".csv.tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(updated_rows)
                Path(tmp_path).replace(log_path)
            except BaseException:
                Path(tmp_path).unlink(missing_ok=True)
                raise

            return action
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


# --- Orchestrator ---


async def main() -> None:
    pool = await get_pool()
    total, stale = await fetch_stale_counts(pool)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    stats = compute_stale_stats(total, stale, date_str)
    action = upsert_csv_row(stats, LOG_PATH)
    print(f"{format_stale_line(stats)} [{action}]")


async def _cli_main() -> None:
    try:
        await main()
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_cli_main())
