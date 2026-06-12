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

Hot-path design (round 3 adversarial review — perf/process-budget):

- The recall path NEVER runs the expensive ``avg(embedding)`` aggregate inline.
  ``resolve_centroids`` does a synchronous, size-capped local-file read (no
  thread machinery — a capped read is microseconds, and ``asyncio.to_thread``
  left a worker running past a cancelled ``wait_for``, defeating the process
  budget) and decides:
    * fresh + label-matched envelope  -> use it;
    * stale (expired) but label-matched -> STALE-WHILE-REVALIDATE: serve the
      stale centroids this call AND trigger an out-of-process refresh;
    * cold / label-mismatch / unreadable -> return ``None`` (neutral) this call
      and trigger a background refresh; the next recall is warm.
- The refresh runs in a DETACHED subprocess (``python -m scripts.core.type_affinity
  --refresh``) guarded by a single-flight lockfile so that at TTL expiry only
  ONE process recomputes the aggregate (with a server-side statement_timeout),
  validates all-or-nothing, and atomically publishes the cache.
- TTL carries a deterministic ±10% jitter (keyed on label+path) so a fleet of
  hosts does not all expire on the same tick.

Every failure mode — no embedding, no model label, DB error, no rows, stale or
unreadable cache, or a cache whose label disagrees with the query's embedding
space (never cosine across spaces, issue #151) — collapses to ``None``. The
reranker then keeps its neutral 0.5 ``type_match`` behavior, so this feature is
strictly additive and degrades safely.

First run on a cold cache is neutral by design (the refresh fires in the
background). To warm the cache deterministically — e.g. before an acceptance
check — invoke the refresh entrypoint once and wait for it to finish::

    python -m scripts.core.type_affinity --refresh --model-label <label>

Pure logic (inference, freshness) is separated from I/O (cache read/write, the
DB aggregate) per the project's FP conventions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
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

# Hard cap on the centroid cache file size (round 3: lowered to 1 MB). A handful
# of type centroids (7 types x ~1024 floats ~= 100 KB) never approaches this;
# anything larger is corrupt/hostile and is rejected before json.load. The cap
# is what makes the now-synchronous read safe on the event-loop thread.
MAX_CACHE_BYTES = 1024 * 1024

# A refresh lock older than this is treated as stale and reclaimed: the holder
# (a detached subprocess) likely died before releasing it. 5 minutes comfortably
# exceeds a healthy aggregate + write (sub-second to a few seconds at 50k rows).
REFRESH_LOCK_STALE_SECONDS = 5 * 60

# Server-side statement timeout for the refresh aggregate, so a pathological
# plan or lock contention can't run unbounded in the background process.
REFRESH_STATEMENT_TIMEOUT = "10s"

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


def validate_centroids(centroids: object) -> dict[str, list[float]] | None:
    """All-or-nothing validation of a centroid set (finding 2, round 2).

    Returns a clean ``{type: vector}`` dict only when EVERY entry is a non-empty
    list of finite numbers AND all vectors share one dimensionality. Any single
    defect (non-list value, empty vector, NaN/Inf, mixed dims) rejects the whole
    set → ``None``. A partial set would let ``type_match`` penalize results of
    the dropped type by the full type weight, biasing ranking — the opposite of
    the fail-to-neutral contract. The empty set is also rejected (no signal).
    """
    if not isinstance(centroids, dict) or not centroids:
        return None
    out: dict[str, list[float]] = {}
    dim: int | None = None
    for ltype, vec in centroids.items():
        if not ltype or not isinstance(vec, list) or not vec:
            return None
        clean: list[float] = []
        for x in vec:
            # bool is an int subclass; reject it explicitly. Strings/None raise.
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return None
            fx = float(x)
            if not math.isfinite(fx):
                return None
            clean.append(fx)
        if dim is None:
            dim = len(clean)
        elif len(clean) != dim:
            return None
        out[str(ltype)] = clean
    return out


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


def _ttl_jitter_factor(key: str) -> float:
    """Deterministic ±10% TTL jitter factor in [0.9, 1.1] for ``key``.

    Round 3 finding 2: jittering the effective TTL spreads cache expiry across a
    fleet so hosts don't all recompute on the same tick (thundering herd). The
    jitter must be DETERMINISTIC (same key -> same factor) so tests are stable
    and a given host's expiry is consistent — hence a hash of the key, not
    ``random``. The key is typically ``"<model_label>:<cache_path>"``.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Map the first 4 bytes to [0, 1), then to [-0.1, +0.1].
    fraction = int.from_bytes(digest[:4], "big") / 2**32
    return 1.0 + (fraction * 0.2 - 0.1)


def is_cache_fresh(
    cache: CentroidCache | None,
    *,
    model_label: str,
    ttl_seconds: float,
    now: datetime | None = None,
    jitter_key: str | None = None,
) -> bool:
    """Return True only when the cache is non-None, within TTL, and label-matched.

    A label mismatch is treated as stale even when the timestamp is fresh:
    cosine similarity is only meaningful within a single embedding space
    (issue #151), so centroids from another space must be recomputed.

    When ``jitter_key`` is given, the effective TTL is scaled by a deterministic
    ±10% factor (round 3 finding 2) so concurrent hosts don't expire in lockstep.
    Without it, the bare TTL is used (unchanged behavior for existing callers).
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
    effective_ttl = ttl_seconds
    if jitter_key is not None:
        effective_ttl = ttl_seconds * _ttl_jitter_factor(jitter_key)
    return 0.0 <= age_seconds <= effective_ttl


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
    raw_centroids = data.get("centroids")
    if not label or not computed_raw:
        return None
    # All-or-nothing centroid validation (finding 2, round 2): a partial or
    # corrupt cached set is rejected wholesale so it can never be read as a
    # partial distribution that biases ranking against a missing type.
    centroids = validate_centroids(raw_centroids)
    if centroids is None:
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

    Concurrency-safe atomic publish (finding 1, round 2). Two cold-cache recall
    processes can race here, so a FIXED ``<path>.tmp`` sibling is unsafe — both
    would open the same file and produce a torn/poisoned cache. Instead each
    writer gets a UNIQUE temp file (``mkstemp`` in the target directory),
    write + flush + fsync + close, then ``os.replace`` (atomic rename on POSIX).
    A reader therefore only ever sees a fully-written file, and the last writer
    to ``replace`` wins cleanly. The unique temp is removed on any failure so no
    stray ``.tmp`` files accumulate.

    Best-effort: a write failure (unwritable dir, disk full) is logged and
    swallowed so it can never abort recall. Synchronous by design; the recall
    hot path runs it off the event loop (see ``load_or_compute_centroids``).
    """
    p = Path(path)
    timestamp = (now if now is not None else datetime.now(UTC)).isoformat()
    envelope = {
        "model_label": model_label,
        "computed_at": timestamp,
        "centroids": centroids,
    }
    tmp_name: str | None = None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp in the SAME directory so os.replace is an atomic rename
        # (cross-filesystem rename is not atomic).
        fd, tmp_name = tempfile.mkstemp(
            dir=str(p.parent), prefix=p.name + ".", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(envelope, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, p)
            tmp_name = None  # successfully published; nothing to clean up
        finally:
            # Remove the unique temp if replace never happened (error path).
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    logger.debug("temp centroid cache cleanup failed", exc_info=True)
    except OSError:
        logger.debug("type-centroid cache write failed", exc_info=True)


# ---------------------------------------------------------------------------
# DB aggregate (side effects)
# ---------------------------------------------------------------------------


def _parse_centroid_rows(rows: list) -> dict[str, list[float]] | None:
    """Parse (ltype, centroid-text) rows into a validated centroids dict.

    All-or-nothing (finding 2, round 2): pgvector's ``avg(embedding)::text`` is
    a JSON-array-like string. A single null type, unparseable centroid, or a row
    that fails ``validate_centroids`` (non-finite, inconsistent dims) rejects the
    WHOLE aggregate → ``None``, so the caller degrades to neutral reranking
    instead of producing a partial distribution that biases ranking against the
    dropped type. An empty result is also ``None`` (no signal).
    """
    raw: dict[str, list[float]] = {}
    for row in rows:
        ltype = row["ltype"] if not hasattr(row, "get") else row.get("ltype")
        centroid_text = (
            row["centroid"] if not hasattr(row, "get") else row.get("centroid")
        )
        if not ltype or centroid_text is None:
            return None
        try:
            vec = json.loads(centroid_text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(vec, list):
            return None
        raw[str(ltype)] = vec
    # validate_centroids enforces finiteness, non-emptiness, and uniform dims.
    return validate_centroids(raw)


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
            # Round 3 finding 2: bound the aggregate with a server-side
            # statement_timeout so a pathological plan or lock contention can't
            # run unbounded even in the background refresh process. SET LOCAL is
            # scoped to the transaction, so it cannot leak to other pooled users.
            async with conn.transaction():
                await conn.execute(
                    f"SET LOCAL statement_timeout = '{REFRESH_STATEMENT_TIMEOUT}'"
                )
                try:
                    rows = await conn.fetch(_CENTROID_SQL, model_label)
                except Exception as exc:  # noqa: BLE001 - only column case retries
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
# Single-flight background refresh (lockfile + detached subprocess)
# ---------------------------------------------------------------------------


def _refresh_lock_path(cache_path: str | Path) -> Path:
    """Sibling lockfile guarding the single-flight refresh for ``cache_path``."""
    p = Path(cache_path)
    return p.with_name(p.name + ".refresh.lock")


def _trigger_background_refresh(
    model_label: str, *, cache_path: str | Path,
) -> None:
    """Spawn ONE detached refresh subprocess, guarded by a single-flight lock.

    Round 3 finding 2: at TTL expiry many concurrent recall processes would each
    run the expensive ``avg(embedding)`` aggregate. Instead the first caller to
    create the lockfile (``os.open`` with ``O_CREAT | O_EXCL`` — atomic) wins and
    spawns a detached subprocess to recompute; concurrent callers find the lock
    held and return immediately. A lock older than ``REFRESH_LOCK_STALE_SECONDS``
    is reclaimed (the previous holder likely died). The detached child removes
    the lock in a ``finally`` (see ``_run_refresh``); if the spawn itself fails
    we release the lock so the next call can retry. Best-effort throughout — any
    error here leaves recall on the neutral path.
    """
    lock_path = _refresh_lock_path(cache_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Reclaim a stale lock (holder died before releasing).
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age > REFRESH_LOCK_STALE_SECONDS:
                lock_path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass

        # Atomically acquire: O_EXCL fails if the lock already exists.
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return  # another process is already refreshing
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
    except OSError:
        logger.debug("refresh lock acquisition failed", exc_info=True)
        return

    # Lock held; spawn the detached refresh worker. On spawn failure, release
    # the lock so a later recall can retry instead of waiting out the stale TTL.
    try:
        subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable, "-m", "scripts.core.type_affinity",
                "--refresh",
                "--model-label", model_label,
                "--cache-path", str(cache_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:  # noqa: BLE001 - spawn failure must not abort recall
        logger.debug("refresh subprocess spawn failed; releasing lock", exc_info=True)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("refresh lock release failed", exc_info=True)


async def _run_refresh(model_label: str, *, cache_path: str | Path) -> None:
    """Recompute, validate, and atomically publish centroids; always unlock.

    The detached refresh entrypoint's body. Runs the (statement-timeout-bounded)
    aggregate, and on success atomically writes the validated envelope. The
    single-flight lock is ALWAYS released in the ``finally`` so a crash mid-way
    never wedges future refreshes (a stale lock would also be reclaimed by age,
    but releasing promptly is cleaner).
    """
    lock_path = _refresh_lock_path(cache_path)
    try:
        centroids = await fetch_type_centroids(model_label)
        if centroids is None:
            logger.debug("refresh: aggregate returned no valid centroids")
            return
        write_centroid_cache(
            cache_path, model_label=model_label, centroids=centroids,
        )
    except Exception:  # noqa: BLE001 - background worker must not crash loudly
        logger.debug("centroid refresh failed", exc_info=True)
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("refresh lock release failed", exc_info=True)


# ---------------------------------------------------------------------------
# Hot-path centroid resolution (synchronous read + stale-while-revalidate)
# ---------------------------------------------------------------------------


def resolve_centroids(
    model_label: str,
    *,
    cache_path: str | Path | None = None,
    ttl_seconds: float = DEFAULT_CENTROID_TTL_SECONDS,
) -> dict[str, list[float]] | None:
    """Resolve centroids for the recall hot path WITHOUT running the aggregate.

    Synchronous by design (round 3): the cache read is a size-capped local-file
    read — microseconds — so it runs directly on the loop thread; ``to_thread``
    is avoided because a cancelled ``wait_for`` left its worker running and the
    default executor is joined at interpreter shutdown, defeating the hook's
    process budget.

    Decision table (never runs ``fetch_type_centroids`` inline):
      * fresh + label-matched   -> return the centroids;
      * stale + label-matched   -> STALE-WHILE-REVALIDATE: return the stale
                                    centroids AND trigger a background refresh;
      * cold / label-mismatch / unreadable -> trigger a background refresh and
                                    return ``None`` (neutral) for this call.
    """
    path = Path(cache_path) if cache_path is not None else default_cache_path()
    jitter_key = f"{model_label}:{path}"

    cache = read_centroid_cache(path)

    if is_cache_fresh(
        cache, model_label=model_label, ttl_seconds=ttl_seconds, jitter_key=jitter_key,
    ):
        return cache.centroids  # type: ignore[union-attr]

    # Not fresh. If we have a label-matched (but expired) envelope, serve it
    # while a background refresh runs (stale-while-revalidate). Otherwise the
    # cache is cold / wrong-space / unreadable -> neutral this call.
    _trigger_background_refresh(model_label, cache_path=path)
    if cache is not None and cache.model_label == model_label:
        return cache.centroids
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def compute_type_probabilities(
    capture: SearchCapture | None,
    *,
    config: RerankerConfig | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, float] | None:
    """End-to-end: SearchCapture -> cached centroids -> type distribution.

    Returns None (neutral reranking) unless the capture carries both a query
    embedding AND a model-space label, ``resolve_centroids`` yields a usable
    (fresh or stale-but-valid) cache, and the resulting distribution is
    non-empty. The label gate enforces the single-space contract (#151).

    ``resolve_centroids`` NEVER runs the DB aggregate inline — a cold/stale cache
    triggers an out-of-process refresh and this call stays neutral — so this
    coroutine is cheap (a capped local read + softmax) and keeps the ``async``
    signature only for its existing call site. Remains a coroutine for API
    stability with recall_learnings' ``await``.
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
    ttl_seconds = float(
        getattr(config, "centroid_cache_ttl_seconds", DEFAULT_CENTROID_TTL_SECONDS)
    )
    temperature = getattr(config, "type_softmax_temperature", None)

    centroids = resolve_centroids(
        model_label, cache_path=cache_path, ttl_seconds=ttl_seconds,
    )
    if not centroids:
        return None

    return infer_type_probabilities(
        query_embedding, centroids, temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Refresh entrypoint (run in a detached subprocess by _trigger_background_refresh)
# ---------------------------------------------------------------------------


def _parse_refresh_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.core.type_affinity",
        description=(
            "Recompute and cache type-affinity centroids out-of-process. "
            "Invoked detached by the recall path at cache TTL expiry; also "
            "runnable manually to warm a cold cache before an acceptance check."
        ),
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Run the centroid refresh (recompute aggregate, validate, cache).",
    )
    parser.add_argument(
        "--model-label", required=True,
        help="Embedding-space label to filter the centroid corpus by (#151).",
    )
    parser.add_argument(
        "--cache-path", default=None,
        help="Cache file path (defaults to the user config-dir location).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Module entrypoint: ``python -m scripts.core.type_affinity --refresh``."""
    args = _parse_refresh_args(argv)
    if not args.refresh:
        # Nothing else to do without --refresh; keep the surface minimal.
        print("nothing to do (pass --refresh)", file=sys.stderr)
        return 2
    cache_path = args.cache_path or default_cache_path()
    asyncio.run(_run_refresh(args.model_label, cache_path=cache_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
