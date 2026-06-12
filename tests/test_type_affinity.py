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
