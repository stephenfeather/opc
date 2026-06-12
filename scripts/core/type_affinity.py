"""Type-affinity wiring for the recall reranker (issue #54).

The reranker's ``type_match`` signal scores a result by how well its
``learning_type`` matches a per-query soft type distribution. Before this
module that distribution was never computed — ``type_match`` returned a neutral
0.5 for every result on every path. This module connects the dead wiring:

1. Server-side, model-filtered type centroids are computed by pgvector
   (``avg(embedding)`` grouped by ``learning_type``) and cached to a JSON file
   with a ``model_label`` + ``computed_at`` envelope and a TTL.
2. The query embedding is surfaced (without a second embed call) by
   ``recall_backends.SearchCapture``.
3. ``compute_type_probabilities`` infers the distribution via the reranker's
   temperature-sharpened softmax.

Every failure mode — no embedding, no model label, DB error, no rows, stale or
unreadable cache, or a cache whose label disagrees with the query's embedding
space (never cosine across spaces, issue #151) — collapses to ``None``. The
reranker then keeps its neutral 0.5 ``type_match`` behavior, so this feature is
strictly additive and degrades safely.

Pure logic (inference, freshness) is separated from I/O (cache read/write, the
DB aggregate) per the project's FP conventions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.core.config.models import RerankerConfig
    from scripts.core.recall_backends import SearchCapture

logger = logging.getLogger(__name__)

# Default cache TTL (24h). Centroids over 6k rows shift slowly, so a daily
# refresh keeps the aggregate cost off the recall hot path while staying
# current. Override via the ``ttl_seconds`` argument / config if needed.
DEFAULT_CENTROID_TTL_SECONDS = 24 * 60 * 60

# Hard cap on the centroid cache file size (finding 2, round 1). The envelope is
# a handful of type centroids (7 types x ~1k floats ~= a few hundred KB worst
# case); anything above 4 MB is treated as corrupt/hostile and rejected before
# json.load so a runaway file can't burn the recall budget parsing megabytes.
MAX_CACHE_BYTES = 4 * 1024 * 1024

# Server-side centroid aggregate. Model-filtered (issue #151 single-space
# contract): never average embeddings across embedding spaces. ``::text`` casts
# the pgvector ``avg`` result to a JSON-array-like string the caller parses.
#
# Finding 1 (round 1 adversarial review): the centroids MUST be trained on
# exactly the corpus the RRF recall path can return, otherwise type affinity is
# computed from rows that never appear in results. The authoritative recall
# predicates (recall_backends.py CTEs / text tails) are:
#   metadata->>'type' = 'session_learning'   (session learnings only)
#   superseded_by IS NULL                     (the "chain filter")
# We also drop null learning_type rows server-side (they cannot match any
# result's type anyway). ``{chain_filter}`` lets us degrade exactly like recall
# on a pre-migration DB that lacks the superseded_by column.
_CENTROID_SQL_TEMPLATE = """
SELECT metadata->>'learning_type' AS ltype, avg(embedding)::text AS centroid
FROM archival_memory
WHERE embedding IS NOT NULL
    AND embedding_model = $1
    AND metadata->>'type' = 'session_learning'
    AND metadata->>'learning_type' IS NOT NULL{chain_filter}
GROUP BY 1
"""

# The full-predicate aggregate (mirrors recall's chain=True CTE).
_CENTROID_SQL = _CENTROID_SQL_TEMPLATE.format(
    chain_filter="\n    AND superseded_by IS NULL"
)
# Degraded aggregate for a pre-migration DB without the superseded_by column,
# mirroring recall's chain=False fallback. Only used after the full query
# raises UndefinedColumnError.
_CENTROID_SQL_NO_CHAIN = _CENTROID_SQL_TEMPLATE.format(chain_filter="")


# ---------------------------------------------------------------------------
# Cache envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CentroidCache:
    """A cached centroid set tagged with its embedding space and compute time."""

    model_label: str
    computed_at: datetime
    centroids: dict[str, list[float]]


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


def infer_type_probabilities(
    query_embedding: list[float] | None,
    centroids: dict[str, list[float]] | None,
    *,
    temperature: float | None,
) -> dict[str, float] | None:
    """Infer a type distribution from a query embedding and type centroids.

    Pure wrapper over ``reranker.infer_query_type`` that returns ``None``
    (rather than an empty dict) whenever inputs are missing, so callers can
    treat "no signal" uniformly. ``temperature`` sharpens the otherwise
    near-uniform distribution (issue #54).
    """
    if not query_embedding or not centroids:
        return None

    from scripts.core.reranker import infer_query_type

    probs = infer_query_type(query_embedding, centroids, temperature=temperature)
    return probs or None


def is_cache_fresh(
    cache: CentroidCache | None,
    *,
    model_label: str,
    ttl_seconds: float,
    now: datetime | None = None,
) -> bool:
    """Return True only when the cache is non-None, within TTL, and label-matched.

    A label mismatch is treated as stale even when the timestamp is fresh:
    cosine similarity is only meaningful within a single embedding space
    (issue #151), so centroids from another space must be recomputed.
    """
    if cache is None:
        return False
    if cache.model_label != model_label:
        return False
    current = now if now is not None else datetime.now(UTC)
    computed = cache.computed_at
    if computed.tzinfo is None:
        computed = computed.replace(tzinfo=UTC)
    age_seconds = (current - computed).total_seconds()
    return 0.0 <= age_seconds <= ttl_seconds


# ---------------------------------------------------------------------------
# Cache I/O (side effects)
# ---------------------------------------------------------------------------


def default_cache_path() -> Path:
    """Resolve the centroid cache file path under the user config dir."""
    return Path.home() / ".config" / "opc" / "type_centroids.json"


def read_centroid_cache(path: str | Path) -> CentroidCache | None:
    """Read a centroid cache envelope. Returns None if missing/corrupt/partial.

    An oversized file (> ``MAX_CACHE_BYTES``) is rejected as corrupt before
    json.load so a runaway/hostile cache can't burn the recall budget parsing
    megabytes (finding 2). This function is synchronous; callers on the recall
    hot path must run it off the event loop (see ``load_or_compute_centroids``).
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.stat().st_size > MAX_CACHE_BYTES:
            logger.debug("centroid cache oversized (%d bytes); rejecting", p.stat().st_size)
            return None
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    label = data.get("model_label")
    computed_raw = data.get("computed_at")
    centroids = data.get("centroids")
    if not label or not computed_raw or not isinstance(centroids, dict):
        return None
    try:
        computed_at = datetime.fromisoformat(computed_raw)
    except (ValueError, TypeError):
        return None
    return CentroidCache(
        model_label=str(label), computed_at=computed_at, centroids=centroids,
    )


def write_centroid_cache(
    path: str | Path,
    *,
    model_label: str,
    centroids: dict[str, list[float]],
    now: datetime | None = None,
) -> None:
    """Persist centroids with a model_label + computed_at envelope.

    Best-effort: a write failure (e.g. unwritable config dir) is logged and
    swallowed so it can never abort recall. Writes atomically via a temp file
    + replace so a concurrent reader never observes a half-written file.
    """
    p = Path(path)
    timestamp = (now if now is not None else datetime.now(UTC)).isoformat()
    envelope = {
        "model_label": model_label,
        "computed_at": timestamp,
        "centroids": centroids,
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(envelope, f)
        tmp.replace(p)
    except OSError:
        logger.debug("type-centroid cache write failed", exc_info=True)


# ---------------------------------------------------------------------------
# DB aggregate (side effects)
# ---------------------------------------------------------------------------


def _parse_centroid_rows(rows: list) -> dict[str, list[float]]:
    """Parse (ltype, centroid-text) rows into a centroids dict.

    pgvector's ``avg(embedding)::text`` is a JSON-array-like string. Rows with
    a null type or an unparseable centroid are skipped rather than aborting the
    whole aggregate.
    """
    out: dict[str, list[float]] = {}
    for row in rows:
        ltype = row["ltype"] if not hasattr(row, "get") else row.get("ltype")
        centroid_text = (
            row["centroid"] if not hasattr(row, "get") else row.get("centroid")
        )
        if not ltype or centroid_text is None:
            continue
        try:
            vec = json.loads(centroid_text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(vec, list) and vec:
            out[str(ltype)] = [float(x) for x in vec]
    return out


def _is_missing_column_error(exc: Exception) -> bool:
    """True for the asyncpg UndefinedColumnError raised when superseded_by is
    absent on a pre-migration DB. Imported lazily so type_affinity stays usable
    without asyncpg installed (e.g. unit tests that never hit the DB path)."""
    try:
        from asyncpg.exceptions import UndefinedColumnError
    except ImportError:  # pragma: no cover - asyncpg always present in prod
        return False
    return isinstance(exc, UndefinedColumnError)


async def fetch_type_centroids(model_label: str) -> dict[str, list[float]] | None:
    """Compute model-filtered type centroids server-side. None on any failure.

    The aggregate mirrors the RRF recall corpus exactly (session_learning,
    non-superseded, non-null learning_type) so centroids reflect only rows that
    can appear in results (finding 1). One query over ~6k rows (fast). On a
    pre-migration DB lacking superseded_by it retries without that predicate,
    mirroring recall's chain-filter fallback. Any other DB error, an empty
    result, or a fully unparseable result returns None so the caller degrades
    to neutral reranking instead of crashing recall.
    """
    try:
        from scripts.core.db.postgres_pool import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch(_CENTROID_SQL, model_label)
            except Exception as exc:  # noqa: BLE001 - only the column case retries
                if not _is_missing_column_error(exc):
                    raise
                logger.debug(
                    "centroid aggregate: superseded_by absent, degrading",
                    exc_info=True,
                )
                rows = await conn.fetch(_CENTROID_SQL_NO_CHAIN, model_label)
    except Exception:  # noqa: BLE001 - degrade, never crash recall
        logger.debug("type-centroid aggregate failed", exc_info=True)
        return None

    centroids = _parse_centroid_rows(rows)
    return centroids or None


# ---------------------------------------------------------------------------
# Cache-aware centroid resolution (lazy refresh)
# ---------------------------------------------------------------------------


async def load_or_compute_centroids(
    model_label: str,
    *,
    cache_path: str | Path | None = None,
    ttl_seconds: float = DEFAULT_CENTROID_TTL_SECONDS,
) -> dict[str, list[float]] | None:
    """Return centroids from a fresh cache, else recompute and persist.

    Lazy refresh: a cache hit (fresh + label-matched) is a single file read; a
    miss / stale / label-mismatch runs the aggregate and writes the result.
    Returns None when the cache is unusable AND the recompute fails, so the
    caller degrades to neutral reranking.
    """
    path = Path(cache_path) if cache_path is not None else default_cache_path()

    # Finding 2 (round 1): read the cache off the event loop so the caller's
    # asyncio.wait_for deadline can preempt a slow/locked/oversized file. A
    # synchronous read here would block before the first await, making the
    # TYPE_AFFINITY_TIMEOUT unenforceable.
    cache = await asyncio.to_thread(read_centroid_cache, path)
    if is_cache_fresh(cache, model_label=model_label, ttl_seconds=ttl_seconds):
        # mypy: is_cache_fresh guarantees cache is not None here.
        return cache.centroids  # type: ignore[union-attr]

    centroids = await fetch_type_centroids(model_label)
    if centroids is None:
        return None

    write_centroid_cache(path, model_label=model_label, centroids=centroids)
    return centroids


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def compute_type_probabilities(
    capture: SearchCapture | None,
    *,
    config: RerankerConfig | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, float] | None:
    """End-to-end: SearchCapture -> model-filtered centroids -> type distribution.

    Returns None (neutral reranking) unless the capture carries both a query
    embedding AND a model-space label, the centroids load/compute succeeds, and
    the resulting distribution is non-empty. The label gate enforces the
    single-space contract (#151): without it we would risk cosine across
    embedding spaces.
    """
    if capture is None:
        return None
    query_embedding = capture.query_embedding
    model_label = capture.model_label
    if not query_embedding or not model_label:
        return None

    if config is None:
        from scripts.core.reranker import _default_config

        config = _default_config()
    ttl_seconds = float(getattr(config, "centroid_cache_ttl_seconds", DEFAULT_CENTROID_TTL_SECONDS))
    temperature = getattr(config, "type_softmax_temperature", None)

    centroids = await load_or_compute_centroids(
        model_label, cache_path=cache_path, ttl_seconds=ttl_seconds,
    )
    if not centroids:
        return None

    return infer_type_probabilities(
        query_embedding, centroids, temperature=temperature,
    )
