"""Tests for issue #54 — type-affinity wiring (scripts/core/type_affinity.py).

The reranker's ``type_match`` signal needs a per-query type distribution. This
module computes it from server-side, model-filtered type centroids (cached to a
JSON file with a TTL + label guard) and the query embedding surfaced by
``SearchCapture``. All failure modes (no embedding, no label, DB error, label
mismatch, stale cache) collapse to ``None`` so the reranker keeps its neutral
0.5 ``type_match`` behavior.

Pure scoring logic is tested directly; the centroid-cache I/O and the DB
aggregate are exercised with fakes (no real DB, no network).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Pure inference wrapper
# ---------------------------------------------------------------------------


class TestInferTypeProbabilities:
    def test_returns_distribution_with_temperature(self):
        from scripts.core.type_affinity import infer_type_probabilities

        centroids = {"A": [1.0, 0.1], "B": [1.0, 0.4]}
        probs = infer_type_probabilities([1.0, 0.0], centroids, temperature=0.05)
        assert probs is not None
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        # Sharpened toward the closer centroid A.
        assert probs["A"] > probs["B"]

    def test_none_embedding_returns_none(self):
        from scripts.core.type_affinity import infer_type_probabilities

        assert infer_type_probabilities(None, {"A": [1.0]}, temperature=0.05) is None

    def test_empty_centroids_returns_none(self):
        from scripts.core.type_affinity import infer_type_probabilities

        assert infer_type_probabilities([1.0, 0.0], {}, temperature=0.05) is None

    def test_none_centroids_returns_none(self):
        from scripts.core.type_affinity import infer_type_probabilities

        assert infer_type_probabilities([1.0, 0.0], None, temperature=0.05) is None


# ---------------------------------------------------------------------------
# Centroid cache envelope (model_label + computed_at + centroids)
# ---------------------------------------------------------------------------


class TestCentroidCacheRoundTrip:
    def test_write_then_read(self, tmp_path):
        from scripts.core.type_affinity import read_centroid_cache, write_centroid_cache

        path = tmp_path / "centroids.json"
        centroids = {"USER_PREFERENCE": [0.1, 0.2], "ERROR_FIX": [0.3, 0.4]}
        write_centroid_cache(path, model_label="voyage-code-3", centroids=centroids)

        cache = read_centroid_cache(path)
        assert cache is not None
        assert cache.model_label == "voyage-code-3"
        assert cache.centroids == centroids
        assert isinstance(cache.computed_at, datetime)

    def test_read_missing_file_returns_none(self, tmp_path):
        from scripts.core.type_affinity import read_centroid_cache

        assert read_centroid_cache(tmp_path / "nope.json") is None

    def test_read_corrupt_file_returns_none(self, tmp_path):
        from scripts.core.type_affinity import read_centroid_cache

        path = tmp_path / "bad.json"
        path.write_text("{ not valid json")
        assert read_centroid_cache(path) is None

    def test_read_missing_envelope_keys_returns_none(self, tmp_path):
        from scripts.core.type_affinity import read_centroid_cache

        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"centroids": {"A": [1.0]}}))  # no label/computed_at
        assert read_centroid_cache(path) is None

    def test_read_oversized_file_returns_none(self, tmp_path):
        """Finding 2 (round 1): an oversized cache file is rejected as corrupt
        before json.load, so a runaway/hostile file can't burn the recall
        budget parsing megabytes."""
        from scripts.core.type_affinity import MAX_CACHE_BYTES, read_centroid_cache

        path = tmp_path / "huge.json"
        # Valid JSON envelope, but padded past the size cap.
        padding = "x" * (MAX_CACHE_BYTES + 1)
        path.write_text(
            json.dumps(
                {
                    "model_label": "voyage-code-3",
                    "computed_at": "2026-01-01T00:00:00+00:00",
                    "centroids": {"A": [1.0]},
                    "_pad": padding,
                }
            )
        )
        assert path.stat().st_size > MAX_CACHE_BYTES
        assert read_centroid_cache(path) is None

    def test_read_at_size_cap_still_parses(self, tmp_path):
        """A normal-sized cache (well under the cap) parses fine."""
        from scripts.core.type_affinity import (
            MAX_CACHE_BYTES,
            read_centroid_cache,
            write_centroid_cache,
        )

        path = tmp_path / "ok.json"
        write_centroid_cache(path, model_label="voyage-code-3", centroids={"A": [1.0]})
        assert path.stat().st_size < MAX_CACHE_BYTES
        assert read_centroid_cache(path) is not None


# ---------------------------------------------------------------------------
# Cache freshness: TTL + label guard
# ---------------------------------------------------------------------------


class TestCacheFreshness:
    def _cache(self, *, label: str, age_seconds: float):
        from scripts.core.type_affinity import CentroidCache

        return CentroidCache(
            model_label=label,
            computed_at=datetime.now(UTC) - timedelta(seconds=age_seconds),
            centroids={"A": [1.0, 0.0]},
        )

    def test_fresh_when_recent_and_label_matches(self):
        from scripts.core.type_affinity import is_cache_fresh

        cache = self._cache(label="voyage-code-3", age_seconds=10)
        assert is_cache_fresh(
            cache, model_label="voyage-code-3", ttl_seconds=86400,
        ) is True

    def test_stale_when_past_ttl(self):
        from scripts.core.type_affinity import is_cache_fresh

        cache = self._cache(label="voyage-code-3", age_seconds=90000)
        assert is_cache_fresh(
            cache, model_label="voyage-code-3", ttl_seconds=86400,
        ) is False

    def test_stale_when_label_mismatch(self):
        """Never cosine across embedding spaces (#151): a label mismatch is
        stale even if the timestamp is fresh."""
        from scripts.core.type_affinity import is_cache_fresh

        cache = self._cache(label="voyage-3", age_seconds=10)
        assert is_cache_fresh(
            cache, model_label="voyage-code-3", ttl_seconds=86400,
        ) is False

    def test_none_cache_is_not_fresh(self):
        from scripts.core.type_affinity import is_cache_fresh

        assert is_cache_fresh(
            None, model_label="voyage-code-3", ttl_seconds=86400,
        ) is False


# ---------------------------------------------------------------------------
# DB aggregate: model-filtered, failures -> None
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, rows: list[dict[str, Any]] | Exception):
        self._rows = rows
        self.captured_sql: str | None = None
        self.captured_args: tuple = ()

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.captured_sql = sql
        self.captured_args = args
        if isinstance(self._rows, Exception):
            raise self._rows
        return self._rows


class _FakeAcquire:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._conn)


class TestFetchCentroids:
    async def test_query_is_model_filtered(self, monkeypatch):
        from scripts.core import type_affinity as ta

        conn = _FakeConn(
            rows=[
                {"ltype": "USER_PREFERENCE", "centroid": "[0.1, 0.2]"},
                {"ltype": "ERROR_FIX", "centroid": "[0.3, 0.4]"},
            ]
        )

        async def fake_get_pool():
            return _FakePool(conn)

        monkeypatch.setattr(
            "scripts.core.db.postgres_pool.get_pool", fake_get_pool
        )

        centroids = await ta.fetch_type_centroids("voyage-code-3")

        assert centroids == {
            "USER_PREFERENCE": [0.1, 0.2],
            "ERROR_FIX": [0.3, 0.4],
        }
        # The model label is bound as a parameter (model-filtered aggregate).
        assert "voyage-code-3" in conn.captured_args
        assert "embedding_model" in (conn.captured_sql or "")

    async def test_aggregate_mirrors_recall_corpus_predicates(self, monkeypatch):
        """Finding 1 (round 1): centroids must be trained on exactly the rows
        the RRF recall corpus can return — session_learning, non-superseded,
        non-null learning_type — so they never reflect rows that never appear
        in results."""
        from scripts.core import type_affinity as ta

        conn = _FakeConn(rows=[{"ltype": "USER_PREFERENCE", "centroid": "[0.1, 0.2]"}])

        async def fake_get_pool():
            return _FakePool(conn)

        monkeypatch.setattr("scripts.core.db.postgres_pool.get_pool", fake_get_pool)

        await ta.fetch_type_centroids("voyage-code-3")

        sql = conn.captured_sql or ""
        # Recall's authoritative predicates (recall_backends.py CTEs / tails).
        assert "session_learning" in sql
        assert "superseded_by IS NULL" in sql
        # Null learning_type rows are excluded server-side, not just in parsing.
        assert "learning_type" in sql
        assert "IS NOT NULL" in sql

    async def test_missing_superseded_column_degrades(self, monkeypatch):
        """Finding 1: a pre-migration DB without superseded_by must not crash
        the aggregate — it retries without that predicate, mirroring recall's
        chain-filter fallback, instead of returning None."""
        from asyncpg.exceptions import UndefinedColumnError

        from scripts.core import type_affinity as ta

        class _ChainFallbackConn:
            """First fetch (with superseded_by) raises UndefinedColumnError;
            the retry (without it) succeeds."""

            def __init__(self) -> None:
                self.calls: list[str] = []

            async def fetch(self, sql: str, *args: Any) -> list[Any]:
                self.calls.append(sql)
                if "superseded_by" in sql:
                    raise UndefinedColumnError("column superseded_by does not exist")
                return [{"ltype": "ERROR_FIX", "centroid": "[0.5, 0.6]"}]

        conn = _ChainFallbackConn()

        async def fake_get_pool():
            return _FakePool(conn)

        monkeypatch.setattr("scripts.core.db.postgres_pool.get_pool", fake_get_pool)

        centroids = await ta.fetch_type_centroids("voyage-code-3")

        assert centroids == {"ERROR_FIX": [0.5, 0.6]}
        # It tried the full-predicate SQL first, then the degraded one.
        assert len(conn.calls) == 2
        assert "superseded_by" in conn.calls[0]
        assert "superseded_by" not in conn.calls[1]

    async def test_db_failure_returns_none(self, monkeypatch):
        from scripts.core import type_affinity as ta

        conn = _FakeConn(rows=RuntimeError("connection refused"))

        async def fake_get_pool():
            return _FakePool(conn)

        monkeypatch.setattr(
            "scripts.core.db.postgres_pool.get_pool", fake_get_pool
        )

        assert await ta.fetch_type_centroids("voyage-code-3") is None

    async def test_empty_rows_returns_none(self, monkeypatch):
        from scripts.core import type_affinity as ta

        conn = _FakeConn(rows=[])

        async def fake_get_pool():
            return _FakePool(conn)

        monkeypatch.setattr(
            "scripts.core.db.postgres_pool.get_pool", fake_get_pool
        )

        assert await ta.fetch_type_centroids("voyage-code-3") is None

    async def test_unparseable_row_makes_whole_aggregate_none(self, monkeypatch):
        """Finding 2 (round 2): all-or-nothing. A single malformed centroid must
        reject the WHOLE aggregate (return None), not silently drop one type and
        return the rest — a partial distribution biases ranking against the
        missing type."""
        from scripts.core import type_affinity as ta

        conn = _FakeConn(
            rows=[
                {"ltype": "USER_PREFERENCE", "centroid": "[0.1, 0.2]"},
                {"ltype": "ERROR_FIX", "centroid": "not valid json"},
            ]
        )

        async def fake_get_pool():
            return _FakePool(conn)

        monkeypatch.setattr("scripts.core.db.postgres_pool.get_pool", fake_get_pool)
        assert await ta.fetch_type_centroids("voyage-code-3") is None

    async def test_inconsistent_dimensions_make_aggregate_none(self, monkeypatch):
        """Vectors of differing dimensionality cannot be compared by cosine —
        reject the whole aggregate."""
        from scripts.core import type_affinity as ta

        conn = _FakeConn(
            rows=[
                {"ltype": "USER_PREFERENCE", "centroid": "[0.1, 0.2]"},
                {"ltype": "ERROR_FIX", "centroid": "[0.3, 0.4, 0.5]"},
            ]
        )

        async def fake_get_pool():
            return _FakePool(conn)

        monkeypatch.setattr("scripts.core.db.postgres_pool.get_pool", fake_get_pool)
        assert await ta.fetch_type_centroids("voyage-code-3") is None

    async def test_non_finite_value_makes_aggregate_none(self, monkeypatch):
        """A NaN/Inf component poisons cosine similarity — reject the aggregate."""
        from scripts.core import type_affinity as ta

        conn = _FakeConn(
            rows=[
                {"ltype": "USER_PREFERENCE", "centroid": "[0.1, 0.2]"},
                {"ltype": "ERROR_FIX", "centroid": "[NaN, 0.4]"},
            ]
        )

        async def fake_get_pool():
            return _FakePool(conn)

        monkeypatch.setattr("scripts.core.db.postgres_pool.get_pool", fake_get_pool)
        assert await ta.fetch_type_centroids("voyage-code-3") is None


class TestCachedEnvelopeValidation:
    """Finding 2 (round 2): a cached envelope with a corrupt/partial centroid
    set must be rejected wholesale (None), never read as a partial distribution
    that biases ranking. read_centroid_cache does the all-or-nothing check."""

    def test_non_list_centroid_rejected(self, tmp_path):
        import json

        from scripts.core.type_affinity import read_centroid_cache

        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "model_label": "voyage-code-3",
                    "computed_at": "2026-01-01T00:00:00+00:00",
                    "centroids": {"A": [1.0, 2.0], "B": "not-a-vector"},
                }
            )
        )
        assert read_centroid_cache(path) is None

    def test_inconsistent_dims_rejected(self, tmp_path):
        import json

        from scripts.core.type_affinity import read_centroid_cache

        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "model_label": "voyage-code-3",
                    "computed_at": "2026-01-01T00:00:00+00:00",
                    "centroids": {"A": [1.0, 2.0], "B": [1.0, 2.0, 3.0]},
                }
            )
        )
        assert read_centroid_cache(path) is None

    def test_non_finite_rejected(self, tmp_path):

        from scripts.core.type_affinity import read_centroid_cache

        path = tmp_path / "c.json"
        # json allows NaN/Infinity by default on load.
        path.write_text(
            '{"model_label": "voyage-code-3", '
            '"computed_at": "2026-01-01T00:00:00+00:00", '
            '"centroids": {"A": [1.0, 2.0], "B": [NaN, 2.0]}}'
        )
        assert read_centroid_cache(path) is None

    def test_empty_centroids_rejected(self, tmp_path):
        import json

        from scripts.core.type_affinity import read_centroid_cache

        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "model_label": "voyage-code-3",
                    "computed_at": "2026-01-01T00:00:00+00:00",
                    "centroids": {},
                }
            )
        )
        assert read_centroid_cache(path) is None

    def test_valid_consistent_envelope_accepted(self, tmp_path):
        from scripts.core.type_affinity import read_centroid_cache, write_centroid_cache

        path = tmp_path / "c.json"
        write_centroid_cache(
            path, model_label="voyage-code-3",
            centroids={"A": [1.0, 2.0], "B": [3.0, 4.0]},
        )
        cache = read_centroid_cache(path)
        assert cache is not None
        assert cache.centroids == {"A": [1.0, 2.0], "B": [3.0, 4.0]}


# ---------------------------------------------------------------------------
# Orchestrator: compute_type_probabilities
# ---------------------------------------------------------------------------


class TestComputeTypeProbabilities:
    async def test_none_capture_returns_none(self):
        from scripts.core.type_affinity import compute_type_probabilities

        assert await compute_type_probabilities(None) is None

    async def test_capture_without_embedding_returns_none(self):
        from scripts.core.recall_backends import SearchCapture
        from scripts.core.type_affinity import compute_type_probabilities

        cap = SearchCapture(query_embedding=None, model_label="voyage-code-3")
        assert await compute_type_probabilities(cap) is None

    async def test_capture_without_label_returns_none(self):
        from scripts.core.recall_backends import SearchCapture
        from scripts.core.type_affinity import compute_type_probabilities

        cap = SearchCapture(query_embedding=[0.1, 0.2], model_label=None)
        assert await compute_type_probabilities(cap) is None

    async def test_centroid_failure_returns_none(self, monkeypatch):
        from scripts.core import type_affinity as ta
        from scripts.core.recall_backends import SearchCapture

        async def fake_load_or_compute(*_a, **_kw):
            return None  # DB failure / no rows / unreadable cache

        monkeypatch.setattr(ta, "load_or_compute_centroids", fake_load_or_compute)

        cap = SearchCapture(query_embedding=[1.0, 0.0], model_label="voyage-code-3")
        assert await ta.compute_type_probabilities(cap) is None

    async def test_happy_path_returns_distribution(self, monkeypatch):
        from scripts.core import type_affinity as ta
        from scripts.core.recall_backends import SearchCapture

        async def fake_load_or_compute(model_label, **_kw):
            assert model_label == "voyage-code-3"
            return {"USER_PREFERENCE": [1.0, 0.05], "ERROR_FIX": [1.0, 0.5]}

        monkeypatch.setattr(ta, "load_or_compute_centroids", fake_load_or_compute)

        cap = SearchCapture(query_embedding=[1.0, 0.0], model_label="voyage-code-3")
        probs = await ta.compute_type_probabilities(cap)

        assert probs is not None
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        # The query aligns with USER_PREFERENCE's centroid -> higher mass.
        assert probs["USER_PREFERENCE"] > probs["ERROR_FIX"]


# ---------------------------------------------------------------------------
# load_or_compute_centroids: cache-hit vs miss/stale
# ---------------------------------------------------------------------------


class TestLoadOrComputeCentroids:
    async def test_fresh_cache_hit_skips_db(self, monkeypatch, tmp_path):
        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"
        ta.write_centroid_cache(
            path, model_label="voyage-code-3", centroids={"A": [1.0, 0.0]},
        )

        async def boom(*_a, **_kw):
            raise AssertionError("fresh cache must not hit the DB")

        monkeypatch.setattr(ta, "fetch_type_centroids", boom)

        out = await ta.load_or_compute_centroids(
            "voyage-code-3", cache_path=path, ttl_seconds=86400,
        )
        assert out == {"A": [1.0, 0.0]}

    async def test_miss_recomputes_and_writes(self, monkeypatch, tmp_path):
        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"  # does not exist yet

        async def fake_fetch(model_label):
            return {"B": [0.5, 0.5]}

        monkeypatch.setattr(ta, "fetch_type_centroids", fake_fetch)

        out = await ta.load_or_compute_centroids(
            "voyage-code-3", cache_path=path, ttl_seconds=86400,
        )
        assert out == {"B": [0.5, 0.5]}
        # The recomputed centroids were persisted with the right label.
        cache = ta.read_centroid_cache(path)
        assert cache is not None
        assert cache.model_label == "voyage-code-3"
        assert cache.centroids == {"B": [0.5, 0.5]}

    async def test_stale_label_forces_recompute(self, monkeypatch, tmp_path):
        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"
        ta.write_centroid_cache(
            path, model_label="voyage-3", centroids={"OLD": [9.0]},
        )

        async def fake_fetch(model_label):
            assert model_label == "voyage-code-3"
            return {"NEW": [0.5, 0.5]}

        monkeypatch.setattr(ta, "fetch_type_centroids", fake_fetch)

        out = await ta.load_or_compute_centroids(
            "voyage-code-3", cache_path=path, ttl_seconds=86400,
        )
        assert out == {"NEW": [0.5, 0.5]}

    async def test_db_failure_on_miss_returns_none(self, monkeypatch, tmp_path):
        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"

        async def fake_fetch(model_label):
            return None  # DB down

        monkeypatch.setattr(ta, "fetch_type_centroids", fake_fetch)

        out = await ta.load_or_compute_centroids(
            "voyage-code-3", cache_path=path, ttl_seconds=86400,
        )
        assert out is None

    async def test_blocking_cache_read_is_preemptible(self, monkeypatch, tmp_path):
        """Finding 2 (round 1): the synchronous cache read must run off the
        event loop so asyncio.wait_for can preempt a slow/locked file. We block
        the read for longer than the deadline and assert wait_for times out
        promptly instead of waiting for the blocking read to finish."""
        import asyncio
        import time

        from scripts.core import type_affinity as ta

        block_seconds = 5.0  # far longer than the deadline below
        deadline = 0.3

        def blocking_read(_path):
            time.sleep(block_seconds)  # sync block (simulates a locked/slow file)
            return None

        monkeypatch.setattr(ta, "read_centroid_cache", blocking_read)

        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                ta.load_or_compute_centroids(
                    "voyage-code-3", cache_path=tmp_path / "c.json", ttl_seconds=86400,
                ),
                timeout=deadline,
            )
        elapsed = time.monotonic() - start
        # Generous bound (flake lesson): the deadline must fire well before the
        # 5s block — proving the read was awaited off-thread and is preemptible.
        assert elapsed < 3.0

    async def test_write_failure_still_returns_centroids(self, monkeypatch, tmp_path):
        """Finding 1 (round 2): a cache write failure is not an inference
        failure. The freshly computed centroids must still be returned even if
        persisting them raises."""
        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"

        async def fake_fetch(model_label):
            return {"USER_PREFERENCE": [0.1, 0.2]}

        def boom_write(*_a, **_kw):
            raise OSError("disk full")

        monkeypatch.setattr(ta, "fetch_type_centroids", fake_fetch)
        monkeypatch.setattr(ta, "write_centroid_cache", boom_write)

        out = await ta.load_or_compute_centroids(
            "voyage-code-3", cache_path=path, ttl_seconds=86400,
        )
        assert out == {"USER_PREFERENCE": [0.1, 0.2]}

    async def test_blocking_cache_write_is_preemptible(self, monkeypatch, tmp_path):
        """Finding 1 (round 2): the cache write must also run off the event loop
        so a slow/blocking write stays inside the wait_for deadline."""
        import asyncio
        import time

        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"
        block_seconds = 5.0
        deadline = 0.3

        async def fake_fetch(model_label):
            return {"USER_PREFERENCE": [0.1, 0.2]}

        def blocking_write(*_a, **_kw):
            time.sleep(block_seconds)

        monkeypatch.setattr(ta, "fetch_type_centroids", fake_fetch)
        monkeypatch.setattr(ta, "write_centroid_cache", blocking_write)

        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                ta.load_or_compute_centroids(
                    "voyage-code-3", cache_path=path, ttl_seconds=86400,
                ),
                timeout=deadline,
            )
        elapsed = time.monotonic() - start
        assert elapsed < 3.0


# ---------------------------------------------------------------------------
# Finding 1 (round 2): concurrency-safe atomic cache write
# ---------------------------------------------------------------------------


class TestConcurrentCacheWrite:
    def test_write_uses_unique_temp_not_fixed_sibling(self, monkeypatch, tmp_path):
        """The temp file must be unique per writer (mkstemp/NamedTemporaryFile),
        never the fixed '<path>.tmp' sibling that two cold-cache processes would
        both open and corrupt."""
        import scripts.core.type_affinity as ta

        path = tmp_path / "centroids.json"
        fixed_sibling = path.with_suffix(path.suffix + ".tmp")

        created_tmps: list[str] = []
        real_replace = ta.os.replace

        def spy_replace(src, dst):
            created_tmps.append(str(src))
            return real_replace(src, dst)

        monkeypatch.setattr(ta.os, "replace", spy_replace)

        ta.write_centroid_cache(
            path, model_label="voyage-code-3", centroids={"A": [1.0, 2.0]},
        )

        assert len(created_tmps) == 1
        # The atomic source was NOT the predictable fixed sibling.
        assert created_tmps[0] != str(fixed_sibling)
        # Final file is the complete envelope.
        cache = ta.read_centroid_cache(path)
        assert cache is not None and cache.centroids == {"A": [1.0, 2.0]}

    def test_concurrent_writers_leave_one_valid_envelope(self, tmp_path):
        """Two writers racing on the same target must leave exactly one
        complete, valid envelope — never a torn/poisoned file. Unique temp
        files + os.replace make each publish atomic."""
        import threading

        import scripts.core.type_affinity as ta

        path = tmp_path / "centroids.json"
        barrier = threading.Barrier(2)

        def writer(label: str, vec: list[float]):
            barrier.wait()  # maximize overlap
            for _ in range(20):
                ta.write_centroid_cache(
                    path, model_label=label, centroids={"A": vec},
                )

        t1 = threading.Thread(target=writer, args=("voyage-code-3", [1.0, 2.0]))
        t2 = threading.Thread(target=writer, args=("voyage-3", [3.0, 4.0]))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # No leftover temp files in the directory (all replaced/cleaned up).
        leftovers = [
            p for p in tmp_path.iterdir()
            if p.name != "centroids.json"
        ]
        assert leftovers == [], f"stray temp files: {leftovers}"

        # The final file is exactly one complete, valid envelope (one writer
        # won the last replace; the other did not corrupt it).
        cache = ta.read_centroid_cache(path)
        assert cache is not None
        assert cache.model_label in ("voyage-code-3", "voyage-3")
        assert cache.centroids in ({"A": [1.0, 2.0]}, {"A": [3.0, 4.0]})
