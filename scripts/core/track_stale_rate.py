#!/usr/bin/env python3
"""Track stale learning rate over time. Appends daily readings to a CSV log."""

import asyncio
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.core.db.postgres_pool import get_pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = PROJECT_ROOT / "thoughts" / "shared" / "stale_rate_log.csv"


async def get_stale_stats() -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM archival_memory WHERE superseded_by IS NULL"
        )
        stale = await conn.fetchval(
            "SELECT COUNT(*) FROM archival_memory WHERE superseded_by IS NULL AND recall_count = 0"
        )
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "total": total,
        "stale": stale,
        "stale_pct": round(stale / total * 100, 1) if total > 0 else 0.0,
    }


async def main():
    stats = await get_stale_stats()

    write_header = not LOG_PATH.exists()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "total", "stale", "stale_pct"])
        if write_header:
            writer.writeheader()
        writer.writerow(stats)

    print(f"{stats['date']}: {stats['stale']}/{stats['total']} stale ({stats['stale_pct']}%)")


if __name__ == "__main__":
    asyncio.run(main())
