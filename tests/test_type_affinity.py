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
import os
import time
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

    def test_ttl_jitter_is_deterministic_and_bounded(self):
        """Round 3 finding 2: the effective TTL carries a deterministic ±10%
        jitter keyed on a stable string (NOT random at import), so concurrent
        hosts don't all expire on the same tick (thundering herd). Same key ->
        same factor; the factor stays within [0.9, 1.1]."""
        from scripts.core.type_affinity import _ttl_jitter_factor

        f1 = _ttl_jitter_factor("voyage-code-3:/home/u/.config/opc/type_centroids.json")
        f2 = _ttl_jitter_factor("voyage-code-3:/home/u/.config/opc/type_centroids.json")
        assert f1 == f2  # deterministic
        assert 0.9 <= f1 <= 1.1
        other = _ttl_jitter_factor("voyage-3:/different/path.json")
        assert 0.9 <= other <= 1.1

    def test_jitter_key_shifts_freshness_boundary(self):
        """is_cache_fresh applies the jitter when a jitter_key is supplied: an
        age just past the bare TTL can still be fresh (or not) depending on the
        deterministic factor, but the bare-TTL behavior is unchanged when no
        key is given (existing callers/tests)."""
        from scripts.core.type_affinity import _ttl_jitter_factor, is_cache_fresh

        ttl = 1000.0
        key = "voyage-code-3:/x.json"
        factor = _ttl_jitter_factor(key)
        effective = ttl * factor
        # An age strictly inside the jittered window is fresh.
        cache = self._cache(label="voyage-code-3", age_seconds=effective - 1)
        assert is_cache_fresh(
            cache, model_label="voyage-code-3", ttl_seconds=ttl, jitter_key=key,
        ) is True
        # An age strictly past it is stale.
        cache2 = self._cache(label="voyage-code-3", age_seconds=effective + 1)
        assert is_cache_fresh(
            cache2, model_label="voyage-code-3", ttl_seconds=ttl, jitter_key=key,
        ) is False


class TestCacheSizeCap:
    def test_max_cache_bytes_is_1mb(self):
        """Round 3 redesign: cap lowered to 1 MB. The read is now synchronous on
        the loop thread, so a tight cap keeps it microsecond-fast (7 types x
        ~1024 floats ~= 100 KB)."""
        from scripts.core.type_affinity import MAX_CACHE_BYTES

        assert MAX_CACHE_BYTES == 1024 * 1024


# ---------------------------------------------------------------------------
# DB aggregate: model-filtered, failures -> None
# ---------------------------------------------------------------------------


class _FakeTransaction:
    """Async-context-manager stand-in for asyncpg's conn.transaction()."""

    async def __aenter__(self) -> _FakeTransaction:
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


class _FakeConn:
    def __init__(self, rows: list[dict[str, Any]] | Exception):
        self._rows = rows
        self.captured_sql: str | None = None
        self.captured_args: tuple = ()
        self.executed: list[str] = []

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def execute(self, sql: str, *_args: Any) -> str:
        # Captures SET LOCAL statement_timeout (round 3) — not a row fetch.
        self.executed.append(sql)
        return "SET"

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
        # Round 3 finding 2: the aggregate is bounded by a server-side
        # statement_timeout set within the transaction.
        assert any("statement_timeout" in s for s in conn.executed)

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
        # Issue #63 Phase 2b: also excludes stale-archived rows.
        assert "archived_at IS NULL" in sql
        # Null learning_type rows are excluded server-side, not just in parsing.
        assert "learning_type" in sql
        assert "IS NOT NULL" in sql

    async def test_missing_archived_column_keeps_superseded_filter(self, monkeypatch):
        """Issue #63 Phase 2b (W-1): a DB with superseded_by but NOT archived_at
        must degrade to the superseded-only aggregate (middle tier), NOT all rows."""
        from asyncpg.exceptions import UndefinedColumnError

        from scripts.core import type_affinity as ta

        class _ArchivedMissingConn:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def transaction(self) -> _FakeTransaction:
                return _FakeTransaction()

            async def execute(self, sql: str, *_args: Any) -> str:
                return "SET"

            async def fetch(self, sql: str, *args: Any) -> list[Any]:
                self.calls.append(sql)
                # Only the archived_at clause is missing; superseded_by exists.
                if "archived_at" in sql:
                    raise UndefinedColumnError("column archived_at does not exist")
                return [{"ltype": "ERROR_FIX", "centroid": "[0.5, 0.6]"}]

        conn = _ArchivedMissingConn()

        async def fake_get_pool():
            return _FakePool(conn)

        monkeypatch.setattr("scripts.core.db.postgres_pool.get_pool", fake_get_pool)

        centroids = await ta.fetch_type_centroids("voyage-code-3")

        assert centroids == {"ERROR_FIX": [0.5, 0.6]}
        # Two attempts: full (raises on archived_at) then no-archive (succeeds,
        # keeps superseded_by). It never reaches the no-chain (all-rows) SQL.
        assert len(conn.calls) == 2
        assert "archived_at" in conn.calls[0]
        assert "superseded_by IS NULL" in conn.calls[1]
        assert "archived_at" not in conn.calls[1]

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

            def transaction(self) -> _FakeTransaction:
                return _FakeTransaction()

            async def execute(self, sql: str, *_args: Any) -> str:
                return "SET"

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
        # Issue #63 Phase 2b: 3-tier cascade when superseded_by is wholly absent —
        # full (superseded+archived) and no-archive (superseded only) both
        # reference superseded_by and raise; only the no-chain SQL succeeds.
        assert len(conn.calls) == 3
        assert "superseded_by" in conn.calls[0]
        assert "archived_at" in conn.calls[0]
        assert "superseded_by" in conn.calls[1]
        assert "archived_at" not in conn.calls[1]
        assert "superseded_by" not in conn.calls[2]
        assert "archived_at" not in conn.calls[2]

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

        def fake_resolve(*_a, **_kw):
            return None  # cold cache / unreadable cache

        monkeypatch.setattr(ta, "resolve_centroids", fake_resolve)

        cap = SearchCapture(query_embedding=[1.0, 0.0], model_label="voyage-code-3")
        assert await ta.compute_type_probabilities(cap) is None

    async def test_happy_path_returns_distribution(self, monkeypatch):
        from scripts.core import type_affinity as ta
        from scripts.core.recall_backends import SearchCapture

        def fake_resolve(model_label, **_kw):
            assert model_label == "voyage-code-3"
            return {"USER_PREFERENCE": [1.0, 0.05], "ERROR_FIX": [1.0, 0.5]}

        monkeypatch.setattr(ta, "resolve_centroids", fake_resolve)

        cap = SearchCapture(query_embedding=[1.0, 0.0], model_label="voyage-code-3")
        probs = await ta.compute_type_probabilities(cap)

        assert probs is not None
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        # The query aligns with USER_PREFERENCE's centroid -> higher mass.
        assert probs["USER_PREFERENCE"] > probs["ERROR_FIX"]


# ---------------------------------------------------------------------------
# Round 3 redesign: resolve_centroids — synchronous read, stale-while-revalidate,
# single-flight background refresh (NEVER runs the aggregate inline on recall).
# ---------------------------------------------------------------------------


class TestResolveCentroids:
    def test_fresh_cache_returns_centroids_no_refresh(self, monkeypatch, tmp_path):
        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"
        ta.write_centroid_cache(path, model_label="voyage-code-3", centroids={"A": [1.0, 0.0]})

        def boom(*_a, **_kw):
            raise AssertionError("fresh cache must not trigger a refresh")

        monkeypatch.setattr(ta, "_trigger_background_refresh", boom)

        out = ta.resolve_centroids(
            "voyage-code-3", cache_path=path, ttl_seconds=86400,
        )
        assert out == {"A": [1.0, 0.0]}

    def test_stale_label_match_serves_stale_and_refreshes(self, monkeypatch, tmp_path):
        """Stale-while-revalidate: an expired-but-label-matched envelope is
        USED for this call AND a background refresh is triggered exactly once."""
        from datetime import UTC, datetime, timedelta

        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"
        old = datetime.now(UTC) - timedelta(seconds=999999)  # well past TTL
        ta.write_centroid_cache(
            path, model_label="voyage-code-3", centroids={"A": [1.0, 0.0]}, now=old,
        )

        triggers: list[str] = []
        monkeypatch.setattr(
            ta, "_trigger_background_refresh",
            lambda model_label, cache_path: triggers.append(model_label),
        )

        out = ta.resolve_centroids(
            "voyage-code-3", cache_path=path, ttl_seconds=86400,
        )
        # Served the stale-but-valid centroids this call.
        assert out == {"A": [1.0, 0.0]}
        # Refresh fired exactly once.
        assert triggers == ["voyage-code-3"]

    def test_cold_cache_returns_none_and_refreshes(self, monkeypatch, tmp_path):
        """No envelope at all -> neutral (None) this call, refresh in background."""
        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"  # does not exist
        triggers: list[str] = []
        monkeypatch.setattr(
            ta, "_trigger_background_refresh",
            lambda model_label, cache_path: triggers.append(model_label),
        )

        out = ta.resolve_centroids(
            "voyage-code-3", cache_path=path, ttl_seconds=86400,
        )
        assert out is None
        assert triggers == ["voyage-code-3"]

    def test_label_mismatch_returns_none_and_refreshes(self, monkeypatch, tmp_path):
        """A label-mismatched envelope must NOT be served (no cosine across
        spaces, #151) -> None + refresh."""
        from scripts.core import type_affinity as ta

        path = tmp_path / "centroids.json"
        ta.write_centroid_cache(path, model_label="voyage-3", centroids={"A": [1.0, 0.0]})

        triggers: list[str] = []
        monkeypatch.setattr(
            ta, "_trigger_background_refresh",
            lambda model_label, cache_path: triggers.append(model_label),
        )

        out = ta.resolve_centroids(
            "voyage-code-3", cache_path=path, ttl_seconds=86400,
        )
        assert out is None
        assert triggers == ["voyage-code-3"]

    def test_resolve_never_runs_aggregate_inline(self, monkeypatch, tmp_path):
        """The recall hot path must NEVER call the DB aggregate inline."""
        from scripts.core import type_affinity as ta

        async def boom(*_a, **_kw):
            raise AssertionError("aggregate must not run on the recall path")

        monkeypatch.setattr(ta, "fetch_type_centroids", boom)
        monkeypatch.setattr(ta, "_trigger_background_refresh", lambda *a, **k: None)

        # cold path
        assert ta.resolve_centroids(
            "voyage-code-3", cache_path=tmp_path / "nope.json", ttl_seconds=86400,
        ) is None

    def test_resolve_is_synchronous(self):
        """resolve_centroids is a plain sync function (no coroutine) — the read
        is a capped local-file read on the loop thread, no to_thread."""
        import inspect

        from scripts.core import type_affinity as ta

        assert not inspect.iscoroutinefunction(ta.resolve_centroids)


# ---------------------------------------------------------------------------
# Round 3: single-flight background refresh (lockfile + detached subprocess)
# ---------------------------------------------------------------------------


class TestBackgroundRefreshSingleFlight:
    def test_winner_spawns_detached_subprocess(self, monkeypatch, tmp_path):
        from scripts.core import type_affinity as ta

        cache_path = tmp_path / "centroids.json"
        spawned: list[dict] = []

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                spawned.append({"cmd": cmd, "kwargs": kwargs})

        monkeypatch.setattr(ta.subprocess, "Popen", FakePopen)

        ta._trigger_background_refresh("voyage-code-3", cache_path=cache_path)

        assert len(spawned) == 1
        cmd = spawned[0]["cmd"]
        kwargs = spawned[0]["kwargs"]
        # Detached: new session, devnull streams.
        assert kwargs.get("start_new_session") is True
        assert kwargs.get("stdin") == ta.subprocess.DEVNULL
        assert kwargs.get("stdout") == ta.subprocess.DEVNULL
        assert kwargs.get("stderr") == ta.subprocess.DEVNULL
        # Runs the module refresh entrypoint with the label + cache path.
        assert "-m" in cmd
        assert "scripts.core.type_affinity" in cmd
        assert "--refresh" in cmd
        assert "voyage-code-3" in cmd

    def test_refresh_env_is_minimized_whitelist(self, monkeypatch, tmp_path):
        """Security MEDIUM-1: the detached refresh only needs the DB; it must
        NOT inherit the full parent env (no embedding-provider API keys). The
        child gets an explicit whitelist: DB-resolution vars are forwarded,
        secret API keys are dropped."""
        from scripts.core import type_affinity as ta

        cache_path = tmp_path / "centroids.json"

        # Parent holds both DB config and provider secrets.
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-secret-should-not-leak")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret-should-not-leak")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        captured: dict = {}

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["env"] = kwargs.get("env")

        monkeypatch.setattr(ta.subprocess, "Popen", FakePopen)

        ta._trigger_background_refresh("voyage-code-3", cache_path=cache_path)

        env = captured["env"]
        # An explicit env was passed (not None -> not full inheritance).
        assert env is not None
        # The refresh resolves the DB from DATABASE_URL: forwarded.
        assert env.get("DATABASE_URL") == "postgresql://u:p@localhost:5432/db"
        # PATH is needed to find the interpreter: forwarded.
        assert env.get("PATH") == "/usr/bin:/bin"
        # Provider secrets the refresh never uses: dropped.
        assert "VOYAGE_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env

    def test_refresh_env_forwards_db_and_config_resolution_vars(
        self, monkeypatch, tmp_path
    ):
        """All DB/config-resolution vars the child needs are forwarded when set;
        absent ones are simply omitted (no None values)."""
        from scripts.core import type_affinity as ta

        cache_path = tmp_path / "centroids.json"
        monkeypatch.setenv("CONTINUOUS_CLAUDE_DB_URL", "postgresql://c/db")
        monkeypatch.setenv("OPC_CONFIG", "/tmp/opc.toml")
        monkeypatch.setenv("AGENTICA_MAX_POOL_SIZE", "4")
        # A var that is NOT in the whitelist must be dropped.
        monkeypatch.setenv("SOME_UNRELATED_VAR", "x")

        captured: dict = {}

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["env"] = kwargs.get("env")

        monkeypatch.setattr(ta.subprocess, "Popen", FakePopen)
        ta._trigger_background_refresh("voyage-code-3", cache_path=cache_path)

        env = captured["env"]
        assert env.get("CONTINUOUS_CLAUDE_DB_URL") == "postgresql://c/db"
        assert env.get("OPC_CONFIG") == "/tmp/opc.toml"
        assert env.get("AGENTICA_MAX_POOL_SIZE") == "4"
        assert "SOME_UNRELATED_VAR" not in env
        # No key maps to None (whitelist omits unset vars).
        assert all(v is not None for v in env.values())

    def test_second_caller_while_lock_held_does_not_spawn(self, monkeypatch, tmp_path):
        from scripts.core import type_affinity as ta

        cache_path = tmp_path / "centroids.json"
        spawned: list[int] = []

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                spawned.append(1)

        monkeypatch.setattr(ta.subprocess, "Popen", FakePopen)

        # First caller acquires the lock and spawns.
        ta._trigger_background_refresh("voyage-code-3", cache_path=cache_path)
        # Second caller finds the lock held (fresh) -> no spawn.
        ta._trigger_background_refresh("voyage-code-3", cache_path=cache_path)

        assert sum(spawned) == 1

    def test_stale_lock_is_reclaimed(self, monkeypatch, tmp_path):
        from scripts.core import type_affinity as ta

        cache_path = tmp_path / "centroids.json"
        lock_path = ta._refresh_lock_path(cache_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a lock and backdate it past the stale threshold.
        lock_path.write_text("99999")
        old = time.time() - (ta.REFRESH_LOCK_STALE_SECONDS + 60)
        os.utime(lock_path, (old, old))

        spawned: list[int] = []

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                spawned.append(1)

        monkeypatch.setattr(ta.subprocess, "Popen", FakePopen)

        ta._trigger_background_refresh("voyage-code-3", cache_path=cache_path)
        # The stale lock was reclaimed and a fresh refresh spawned.
        assert sum(spawned) == 1

    def test_spawn_failure_releases_lock(self, monkeypatch, tmp_path):
        """If Popen raises, the lock must be released so the next call retries."""
        from scripts.core import type_affinity as ta

        cache_path = tmp_path / "centroids.json"
        lock_path = ta._refresh_lock_path(cache_path)

        class BoomPopen:
            def __init__(self, cmd, **kwargs):
                raise OSError("spawn failed")

        monkeypatch.setattr(ta.subprocess, "Popen", BoomPopen)
        ta._trigger_background_refresh("voyage-code-3", cache_path=cache_path)
        # Lock was released after the failed spawn.
        assert not lock_path.exists()


# ---------------------------------------------------------------------------
# Round 3: refresh entrypoint (computes aggregate, validates, writes, unlocks)
# ---------------------------------------------------------------------------


class TestRefreshEntrypoint:
    async def test_refresh_writes_validated_centroids_and_unlocks(
        self, monkeypatch, tmp_path
    ):
        from scripts.core import type_affinity as ta

        cache_path = tmp_path / "centroids.json"
        lock_path = ta._refresh_lock_path(cache_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("123")  # simulate the lock the spawner created

        async def fake_fetch(model_label):
            assert model_label == "voyage-code-3"
            return {"A": [1.0, 0.0], "B": [0.0, 1.0]}

        monkeypatch.setattr(ta, "fetch_type_centroids", fake_fetch)

        await ta._run_refresh("voyage-code-3", cache_path=cache_path)

        cache = ta.read_centroid_cache(cache_path)
        assert cache is not None
        assert cache.model_label == "voyage-code-3"
        assert cache.centroids == {"A": [1.0, 0.0], "B": [0.0, 1.0]}
        # Lock released in finally.
        assert not lock_path.exists()

    async def test_refresh_db_failure_unlocks_and_no_write(self, monkeypatch, tmp_path):
        from scripts.core import type_affinity as ta

        cache_path = tmp_path / "centroids.json"
        lock_path = ta._refresh_lock_path(cache_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("123")

        async def fake_fetch(model_label):
            return None  # DB down / invalid aggregate

        monkeypatch.setattr(ta, "fetch_type_centroids", fake_fetch)

        await ta._run_refresh("voyage-code-3", cache_path=cache_path)

        assert ta.read_centroid_cache(cache_path) is None  # nothing written
        assert not lock_path.exists()  # lock still released


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
