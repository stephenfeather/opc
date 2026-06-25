"""Append-only journal for recall-tuning experiments.

autoresearch's loop keeps a ``results.tsv`` of every experiment — config, metric,
keep/discard, why — so dead ends are never re-tried. This is OPC's equivalent: a
tab-separated, untracked log (see .gitignore) with one row per evaluated config.

Columns: timestamp · config_hash · ndcg@5 · p@5 · mrr · p95_latency_ms · status ·
description, where status ∈ {keep, discard, crash}.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.core.content_hash import content_hash

DEFAULT_JOURNAL = Path("scripts/benchmarks/results/journal.tsv")

HEADER = [
    "timestamp",
    "config_hash",
    "ndcg@5",
    "p@5",
    "mrr",
    "p95_latency_ms",
    "status",
    "description",
]

VALID_STATUS = {"keep", "discard", "crash"}


def config_hash(config: Any) -> str:
    """Stable short id for a reranker config (a frozen dataclass).

    Hashes the sorted field/value map so the same weights always map to the same
    id — the journal key that ties a row to a reproducible configuration.
    """
    payload = json.dumps(asdict(config), sort_keys=True)
    return content_hash(payload)[:12]


def _clean(text: str) -> str:
    """Make a value safe for a single TSV cell."""
    return text.replace("\t", " ").replace("\n", " ").strip()


def append_entry(
    *,
    timestamp: str,
    cfg_hash: str,
    ndcg: float,
    p_at_k: float,
    mrr: float,
    p95_latency_ms: float,
    status: str,
    description: str,
    path: Path = DEFAULT_JOURNAL,
) -> None:
    """Append one experiment row, writing the header if the file is new."""
    if status not in VALID_STATUS:
        raise ValueError(f"status must be one of {sorted(VALID_STATUS)}, got {status!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    row = [
        _clean(timestamp),
        _clean(cfg_hash),
        f"{ndcg:.4f}",
        f"{p_at_k:.4f}",
        f"{mrr:.4f}",
        f"{p95_latency_ms:.3f}",
        _clean(status),
        _clean(description),
    ]
    with path.open("a", encoding="utf-8") as f:
        if new_file:
            f.write("\t".join(HEADER) + "\n")
        f.write("\t".join(row) + "\n")


def read_entries(path: Path = DEFAULT_JOURNAL) -> list[dict]:
    """Parse the journal into a list of dict rows (empty if the file is absent)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]
