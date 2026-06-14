"""Tests for issue #153: bounded ANN candidate set in the RRF vector leg.

The ``vector_ranked`` CTE used to rank *every* filtered row with no inner
``LIMIT`` so pgvector's HNSW index could never accelerate it. This reshapes
the leg into a bounded ANN inner subquery
(``WHERE ... ORDER BY embedding <=> $q LIMIT $candidates``) ranked by an outer
``ROW_NUMBER()``.

Two invariants pin the change:

1. SQL shape — when ``candidate_param`` is given, ``build_rrf_cte`` emits the
   bounded inner subquery with ``ORDER BY embedding <=> $2::vector LIMIT $N``;
   when it is ``None`` the legacy byte-identical CTE is preserved.
2. Param numbering — the new candidate ``LIMIT`` value binds as the LAST
   positional arg (after project and model), mirroring #139/#151. The
   pre-existing ``$3/$4/$5`` binds keep their numbers.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _reset_recall_probe_caches():
    """Reset module-level recall probe caches before each test (issue #153
    round-2 test-isolation). project / embedding_model / hnsw.iterative_scan
    are process-global; leaving them warm makes the fetch-counting and
    GUC-probe tests order-dependent."""
    from scripts.core import recall_backends as rb

    rb.reset_project_column_cache()
    rb.reset_embedding_model_column_cache()
    rb.reset_hnsw_iterative_scan_cache()
    yield
    rb.reset_project_column_cache()
    rb.reset_embedding_model_column_cache()
    rb.reset_hnsw_iterative_scan_cache()


# ==================== build_rrf_cte SQL shape (no DB) ====================


class TestBuildRrfCteBoundedAnn:
    def test_legacy_shape_when_candidate_param_none(self):
        """candidate_param=None must reproduce the pre-#153 unbounded leg
        byte-for-byte so existing callers/tests stay green."""
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=True)
        # The legacy leg ranks straight off the table with no inner subquery.
        assert (
            "ROW_NUMBER() OVER (ORDER BY embedding <=> $2::vector) as vec_rank"
            in sql
        )
        # No inner LIMIT / candidate subquery in the legacy path.
        assert "LIMIT $" not in sql
        assert "cand" not in sql

    def test_bounded_inner_subquery_when_candidate_param_given(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=True, candidate_param=6)
        # Inner bounded ANN scan: ORDER BY distance, LIMIT $N.
        assert "ORDER BY embedding <=> $2::vector" in sql
        assert "LIMIT $6" in sql

    def test_outer_row_number_ranks_candidates(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=True, candidate_param=6)
        # The outer ROW_NUMBER ranks the (already-bounded) candidate set,
        # not the full table, so it no longer references the raw distance
        # expression directly in the window.
        assert "ROW_NUMBER() OVER (ORDER BY dist)" in sql
        assert (
            "ROW_NUMBER() OVER (ORDER BY embedding <=> $2::vector)" not in sql
        )

    def test_candidate_limit_renders_requested_index(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=True, candidate_param=9)
        assert "LIMIT $9" in sql
        assert "LIMIT $6" not in sql

    def test_bounded_leg_keeps_filters_inside_inner_subquery(self):
        """chain / project / model predicates must shrink the pool *before*
        the LIMIT, otherwise the ANN candidate set is filtered post-hoc and
        under-fills."""
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(
            chain_filter=True,
            project_filter="AND LOWER(project) = $7",
            model_filter="AND embedding_model = $8",
            candidate_param=9,
        )
        _, _, vector_segment = sql.partition("vector_ranked AS")
        inner = vector_segment.split("ORDER BY embedding")[0]
        assert "superseded_by IS NULL" in inner
        assert "AND LOWER(project) = $7" in inner
        assert "AND embedding_model = $8" in inner
        # The bounded LIMIT lives after the inner ORDER BY.
        assert "LIMIT $9" in vector_segment

    def test_fts_leg_has_no_candidate_limit(self):
        """The candidate LIMIT is a vector-leg concept only; the FTS leg must
        be untouched."""
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=True, candidate_param=6)
        fts_segment, _, _ = sql.partition("vector_ranked AS")
        assert "LIMIT $" not in fts_segment

    def test_plain_variant_bounded(self):
        from scripts.core.recall_backends import build_rrf_cte

        sql = build_rrf_cte(chain_filter=False, candidate_param=5)
        assert "LIMIT $5" in sql
        assert "superseded_by IS NULL" not in sql


# ==================== _fetch_all param binding ====================


_SUPPORTED_VECTOR_VERSION = "0.8.1"
# Sentinel so a test can request "no extension row" (NULL) distinctly from the
# default supported version.
_NO_VECTOR_ROW = object()


class _CapturingConn:
    """Capturing conn for the bare-fetch RRF cascade (issue #153 round-3).

    Round-3 removed per-attempt transactions: the session-level
    ``hnsw.iterative_scan`` GUC is SET once on connection acquire (autocommit),
    and the cascade is bare ``conn.fetch``. This conn records ``fetch`` and
    ``execute`` calls so tests can assert the SET is issued and the cascade
    degrades.

    PR review fix: iterative-scan support is detected from the pgvector
    extension VERSION (``SELECT extversion FROM pg_extension WHERE extname =
    'vector'`` via ``fetchval``), NOT from SET success — PostgreSQL accepts any
    two-part custom GUC as a placeholder, so a SET succeeding proves nothing.
    ``vector_version`` controls what the extversion query returns (default
    supported 0.8.1; pass an older string, ``None``, or ``_NO_VECTOR_ROW`` to
    simulate unsupported / missing). ``execute_error`` still lets a test force a
    transient SET failure.
    """

    def __init__(
        self,
        row_provider=None,
        *,
        execute_error: Exception | None = None,
        vector_version: Any = _SUPPORTED_VECTOR_VERSION,
    ) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.executed: list[tuple[str, tuple]] = []
        self.fetchvals: list[tuple[str, tuple]] = []
        self.tx_depth = 0
        self.max_tx_depth = 0
        self._row_provider = row_provider or (lambda sql, args: [])
        self._execute_error = execute_error
        self._vector_version = vector_version

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.calls.append((sql, args))
        return self._row_provider(sql, args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.fetchvals.append((sql, args))
        if "extversion" in sql and "pg_extension" in sql:
            if self._vector_version is _NO_VECTOR_ROW:
                return None
            return self._vector_version
        return None

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        if self._execute_error is not None and "iterative_scan" in sql:
            raise self._execute_error
        return "SET"

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
        return {"cnt": 0}

    def transaction(self):
        # Retained for any caller that still opens a transaction; the round-3
        # RRF cascade does NOT, so this should stay at depth 0 during recall.
        conn = self

        class _Tx:
            async def __aenter__(self):
                conn.tx_depth += 1
                conn.max_tx_depth = max(conn.max_tx_depth, conn.tx_depth)
                return conn

            async def __aexit__(self, *exc):
                conn.tx_depth -= 1
                return False

        return _Tx()


class _CapturingPool:
    def __init__(self, conn: _CapturingConn) -> None:
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Acquire:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Acquire()


def _patch_rrf_env(monkeypatch, conn: _CapturingConn) -> None:
    """Wire a capturing pool + a no-network embedder into the hybrid path."""
    import scripts.core.db.embedding_service as emb_mod
    import scripts.core.db.postgres_pool as pool_mod
    from scripts.core import recall_backends as rb

    # Each test controls the iterative-scan probe outcome; clear any cached
    # result so the conn's transaction()/execute() actually drives the probe.
    rb.reset_hnsw_iterative_scan_cache()

    async def fake_get_pool():
        return _CapturingPool(conn)

    async def fake_init_pgvector(_conn: Any) -> None:
        return None

    class FakeEmbedder:
        def __init__(self, *a: Any, **kw: Any) -> None: ...

        async def embed(self, *_a: Any, **_kw: Any) -> list[float]:
            return [0.1] * 8

        async def aclose(self) -> None: ...

    monkeypatch.setattr(pool_mod, "get_pool", fake_get_pool)
    monkeypatch.setattr(pool_mod, "init_pgvector", fake_init_pgvector)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbedder)


class TestRrfCandidateBinding:
    async def test_candidate_limit_bound_last_and_equals_k_times_multiplier(
        self, monkeypatch,
    ):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        # Pre-migration-free DB: project column present, no missing columns.
        conn = _CapturingConn()
        _patch_rrf_env(monkeypatch, conn)

        mult = rb._recall_cfg.vector_candidate_multiplier
        k = 3
        await rb.search_learnings_hybrid_rrf("query terms", k=k, expand=False)

        # Skip the project-column capability probe; the RRF query is the one
        # that builds the vector_ranked CTE.
        rrf_calls = [c for c in conn.calls if "vector_ranked" in c[0]]
        assert rrf_calls, "expected RRF SQL to run"
        # The first RRF query is the boosted+chain CTE.
        sql, args = rrf_calls[0]
        # Boosted base params: text, embedding, rrf_k, k*2, boost = 5 args.
        # With no project/model active, the candidate LIMIT binds at $6.
        assert "LIMIT $6" in sql
        assert len(args) == 6, args
        assert args[-1] == k * mult

    async def test_plain_args_candidate_last(self, monkeypatch):
        """The plain (no-boost) variant binds the candidate at $5 — one slot
        earlier than boosted because it has no boost param."""
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()

        # Force the boosted pass to fail (missing decay column) so the plain
        # pass runs and we can inspect its args.
        def provider(sql, args):
            from asyncpg.exceptions import UndefinedColumnError

            if "recall_count" in sql:
                raise UndefinedColumnError('column "recall_count" does not exist')
            return []

        conn = _CapturingConn(provider)
        _patch_rrf_env(monkeypatch, conn)

        mult = rb._recall_cfg.vector_candidate_multiplier
        k = 4
        await rb.search_learnings_hybrid_rrf("query terms", k=k, expand=False)

        plain_calls = [
            c for c in conn.calls
            if "vector_ranked" in c[0] and "recall_count" not in c[0]
        ]
        assert plain_calls, "expected a plain-tail query to run"
        sql, args = plain_calls[0]
        # Plain base params: text, embedding, rrf_k, k*2 = 4 args; candidate $5.
        assert "LIMIT $5" in sql
        assert len(args) == 5, args
        assert args[-1] == k * mult

    async def test_existing_positional_params_unchanged(self, monkeypatch):
        """$1..$4 keep their meaning: text, embedding, rrf_k, k*2."""
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = _CapturingConn()
        _patch_rrf_env(monkeypatch, conn)

        k = 5
        await rb.search_learnings_hybrid_rrf("hello world", k=k, expand=False)

        rrf_calls = [c for c in conn.calls if "vector_ranked" in c[0]]
        assert rrf_calls, "expected RRF SQL to run"
        sql, args = rrf_calls[0]
        assert args[0] == "hello world"  # $1 text_query
        assert args[1] == str([0.1] * 8)  # $2 embedding str
        assert args[2] == rb._recall_cfg.rrf_k  # $3 rrf_k
        assert args[3] == k * 2  # $4 tail LIMIT


# === Round-3 finding 1: session-level GUC, no per-attempt transactions ===


class TestRrfIterativeScanGuc:
    """Round-3 redesign: the ``hnsw.iterative_scan = strict_order`` GUC is a
    SESSION SET issued ONCE per acquired connection (autocommit, no SET LOCAL,
    no per-attempt transaction), and the RRF cascade is bare ``conn.fetch``.
    This removes the round-2 per-attempt BEGIN/SET LOCAL/COMMIT overhead while
    still benefiting every cascade fetch, and a failed SET on older pgvector
    poisons nothing because no transaction is ever opened.
    """

    async def test_session_set_issued_once_per_acquire(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = _CapturingConn()
        _patch_rrf_env(monkeypatch, conn)

        await rb.search_learnings_hybrid_rrf("query terms", k=3, expand=False)

        # SESSION SET (not SET LOCAL), strict_order, issued exactly once for the
        # acquired connection.
        guc_stmts = [s for s, _ in conn.executed if "iterative_scan" in s]
        assert len(guc_stmts) == 1, (
            f"expected one session SET per acquire, got {guc_stmts}"
        )
        stmt = guc_stmts[0]
        assert "SET LOCAL" not in stmt.upper(), "must be a SESSION SET, not LOCAL"
        assert "strict_order" in stmt

    async def test_no_transaction_wraps_the_cascade(self, monkeypatch):
        """Round-3: the cascade must NOT open any transaction (no per-attempt
        BEGIN/COMMIT). The capturing conn's transaction() depth must stay 0."""
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = _CapturingConn()
        _patch_rrf_env(monkeypatch, conn)

        await rb.search_learnings_hybrid_rrf("query terms", k=3, expand=False)

        assert conn.max_tx_depth == 0, (
            "RRF cascade must use bare fetches with no transaction wrapping"
        )

    async def test_session_set_before_rrf_fetch_in_order(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        events: list[str] = []

        class _OrderConn(_CapturingConn):
            async def execute(self, sql: str, *args: Any) -> str:
                if "iterative_scan" in sql:
                    events.append("guc")
                return await super().execute(sql, *args)

            async def fetch(self, sql: str, *args: Any) -> list[Any]:
                if "vector_ranked" in sql:
                    events.append("fetch")
                return await super().fetch(sql, *args)

        conn = _OrderConn()
        _patch_rrf_env(monkeypatch, conn)

        await rb.search_learnings_hybrid_rrf("query terms", k=3, expand=False)

        assert "guc" in events and "fetch" in events
        assert events.index("guc") < events.index("fetch")

    async def test_happy_path_one_set_plus_one_fetch(self, monkeypatch):
        """Round-3 perf: the happy path (boosted succeeds) is exactly one
        session SET + one RRF fetch — no extra round trips."""
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        # Pin column caches so no capability-probe fetches inflate the count.
        rb._set_project_column_cache_for_tests(False)
        rb._set_embedding_model_column_cache_for_tests(False)
        # Boosted fetch "succeeds" (returns no rows) so no fallback runs; the
        # point is to count round trips, not result shape.
        conn = _CapturingConn(lambda sql, args: [])
        _patch_rrf_env(monkeypatch, conn)

        await rb.search_learnings_hybrid_rrf("query terms", k=3, expand=False)

        guc_stmts = [s for s, _ in conn.executed if "iterative_scan" in s]
        rrf_fetches = [c for c in conn.calls if "vector_ranked" in c[0]]
        assert len(guc_stmts) == 1, "exactly one session SET per acquire"
        assert len(rrf_fetches) == 1, (
            "happy path must issue exactly one RRF fetch (no fallbacks)"
        )

    async def test_fallback_cascade_degrades_with_bare_fetches(
        self, monkeypatch,
    ):
        """Round-3: the boosted fetch fails (missing recall_count); with bare
        fetches (no transaction) the plain fallback runs and succeeds. No
        transaction means no 'aborted transaction' poisoning is possible."""
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()

        def provider(sql, args):
            from asyncpg.exceptions import UndefinedColumnError

            if "recall_count" in sql:
                raise UndefinedColumnError('column "recall_count" does not exist')
            return []

        conn = _CapturingConn(provider)
        _patch_rrf_env(monkeypatch, conn)

        results = await rb.search_learnings_hybrid_rrf(
            "query terms", k=3, expand=False,
        )
        assert results == []
        rrf_calls = [c for c in conn.calls if "vector_ranked" in c[0]]
        assert any("recall_count" in c[0] for c in rrf_calls), (
            "boosted attempt should have been tried"
        )
        assert any("recall_count" not in c[0] for c in rrf_calls), (
            "plain fallback must run after the boosted attempt fails"
        )
        # No transaction was ever opened around the cascade.
        assert conn.max_tx_depth == 0

    async def test_unsupported_version_fetch_runs_and_warns_once(
        self, monkeypatch, caplog,
    ):
        """PR review fix: on pgvector < 0.8 the version check detects no support,
        the SET is NOT issued (placeholder GUC would silently 'succeed'), recall
        still runs, and a warning is logged."""
        import logging

        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = _CapturingConn(vector_version="0.7.4")
        _patch_rrf_env(monkeypatch, conn)

        with caplog.at_level(logging.WARNING):
            results = await rb.search_learnings_hybrid_rrf(
                "query terms", k=3, expand=False,
            )
        assert results == []
        rrf_calls = [c for c in conn.calls if "vector_ranked" in c[0]]
        assert rrf_calls, "RRF fetch must still run when iterative scan absent"
        # The SET must NOT be issued on an unsupported version.
        assert not any("iterative_scan" in s for s, _ in conn.executed), (
            "must not issue the GUC SET on pgvector < 0.8"
        )
        assert any(
            "iterative" in r.message.lower() and r.levelno >= logging.WARNING
            for r in caplog.records
        ), "expected a warning that bounded ANN runs without iterative scan"

    # --- PR review fix: version-based capability detection ---

    async def test_supported_version_caches_true_and_sets(self, monkeypatch):
        """pgvector >= 0.8.0 -> cache True AND issue the session SET."""
        from scripts.core import recall_backends as rb

        for ver in ("0.8.0", "0.8.1", "0.9.0", "1.0.0"):
            rb.reset_hnsw_iterative_scan_cache()
            conn = _CapturingConn(vector_version=ver)
            assert await rb.ensure_hnsw_iterative_scan(conn) is True, ver
            assert rb._hnsw_iterative_scan_cache is True
            assert any("iterative_scan" in s for s, _ in conn.executed), ver

    async def test_old_version_caches_false_no_set(self, monkeypatch):
        """pgvector < 0.8.0 -> cache False and NEVER issue the SET (a placeholder
        SET would silently succeed and mask the under-fill)."""
        from scripts.core import recall_backends as rb

        for ver in ("0.7.4", "0.5.1", "0.6.0"):
            rb.reset_hnsw_iterative_scan_cache()
            conn = _CapturingConn(vector_version=ver)
            assert await rb.ensure_hnsw_iterative_scan(conn) is False, ver
            assert rb._hnsw_iterative_scan_cache is False
            assert not any("iterative_scan" in s for s, _ in conn.executed), ver

    async def test_missing_extension_row_conservative_false(self, monkeypatch):
        """No pg_extension row (NULL) -> conservatively unsupported, no SET."""
        from scripts.core import recall_backends as rb

        rb.reset_hnsw_iterative_scan_cache()
        conn = _CapturingConn(vector_version=_NO_VECTOR_ROW)
        assert await rb.ensure_hnsw_iterative_scan(conn) is False
        assert rb._hnsw_iterative_scan_cache is False
        assert not any("iterative_scan" in s for s, _ in conn.executed)

    async def test_unparseable_version_conservative_false(self, monkeypatch):
        """An unparseable extversion string -> conservatively unsupported."""
        from scripts.core import recall_backends as rb

        for ver in ("garbage", "", "v-unknown", "..."):
            rb.reset_hnsw_iterative_scan_cache()
            conn = _CapturingConn(vector_version=ver)
            assert await rb.ensure_hnsw_iterative_scan(conn) is False, repr(ver)
            assert rb._hnsw_iterative_scan_cache is False
            assert not any("iterative_scan" in s for s, _ in conn.executed)

    async def test_version_with_suffix_parsed(self, monkeypatch):
        """Tolerant parse handles trailing non-numeric components."""
        from scripts.core import recall_backends as rb

        # Leading numeric components drive the comparison; 0.8.x-dev >= 0.8.0.
        rb.reset_hnsw_iterative_scan_cache()
        conn = _CapturingConn(vector_version="0.8.0-dev")
        assert await rb.ensure_hnsw_iterative_scan(conn) is True

    async def test_ensure_caches_definitive_result(self, monkeypatch):
        """Supported -> cache True; unsupported -> cache False; known-False skips
        the whole probe (no version query, no SET) on later connections."""
        from scripts.core import recall_backends as rb

        # Supported -> cached True, SET issued.
        rb.reset_hnsw_iterative_scan_cache()
        ok_conn = _CapturingConn(vector_version="0.8.1")
        assert await rb.ensure_hnsw_iterative_scan(ok_conn) is True
        assert rb._hnsw_iterative_scan_cache is True

        # Unsupported -> cached False (definitive miss).
        rb.reset_hnsw_iterative_scan_cache()
        bad_conn = _CapturingConn(vector_version="0.7.4")
        assert await rb.ensure_hnsw_iterative_scan(bad_conn) is False
        assert rb._hnsw_iterative_scan_cache is False

        # Known-False: neither the version query nor the SET runs again.
        skip_conn = _CapturingConn(vector_version="0.8.1")
        assert await rb.ensure_hnsw_iterative_scan(skip_conn) is False
        assert not skip_conn.fetchvals, "must skip version query when cached"
        assert not skip_conn.executed, "must skip SET when known unsupported"

    async def test_ensure_no_transaction_opened(self, monkeypatch):
        """Version detection + the session SET run OUTSIDE any transaction."""
        from scripts.core import recall_backends as rb

        rb.reset_hnsw_iterative_scan_cache()
        conn = _CapturingConn(vector_version="0.7.4")

        await rb.ensure_hnsw_iterative_scan(conn)

        assert conn.max_tx_depth == 0, "detection must not open a transaction"

    async def test_transient_version_query_failure_not_cached_warns_once(
        self, monkeypatch, caplog,
    ):
        """A transient failure of the version query is NOT cached (retries next
        connection), warns at most once per process, and never dumps a stack
        trace (no exc_info) on the hot path."""
        import logging

        from scripts.core import recall_backends as rb

        rb.reset_hnsw_iterative_scan_cache()
        transient = TimeoutError("statement timeout")

        class _TransientConn(_CapturingConn):
            async def fetchval(self, sql: str, *args: Any) -> Any:
                self.fetchvals.append((sql, args))
                if "extversion" in sql:
                    raise transient
                return None

        conn1 = _TransientConn()
        with caplog.at_level(logging.WARNING):
            assert await rb.ensure_hnsw_iterative_scan(conn1) is False
            assert rb._hnsw_iterative_scan_cache is None  # not cached
            conn2 = _TransientConn()
            assert await rb.ensure_hnsw_iterative_scan(conn2) is False
            assert rb._hnsw_iterative_scan_cache is None

        transient_warnings = [
            r for r in caplog.records
            if "transient" in r.message.lower() and r.levelno >= logging.WARNING
        ]
        assert len(transient_warnings) == 1, (
            f"expected one rate-limited transient warning, got "
            f"{len(transient_warnings)}"
        )
        assert all(r.exc_info is None for r in transient_warnings), (
            "transient warning must not include exc_info on the hot path"
        )


# ====== Round-1 finding 2: bounded vs exact RRF fusion divergence ======


def _rrf_score(rrf_k: int, fts_rank: int | None, vec_rank: int | None) -> float:
    """RRF fusion mirroring the combined CTE: a missing rank contributes 0."""
    score = 0.0
    if fts_rank is not None:
        score += 1.0 / (rrf_k + fts_rank)
    if vec_rank is not None:
        score += 1.0 / (rrf_k + vec_rank)
    return score


def _topk_ids(
    rows: dict[str, tuple[int | None, int | None]],
    *,
    rrf_k: int,
    candidate_count: int | None,
    k: int,
) -> list[str]:
    """Rank ids by RRF score.

    ``candidate_count`` truncates the vector leg: a row whose vec_rank exceeds
    the candidate pool loses its vector term (matches the bounded inner
    ``ORDER BY ... LIMIT``). ``None`` = exact (unbounded) RRF.
    """
    scored: list[tuple[float, str]] = []
    for rid, (fts_rank, vec_rank) in rows.items():
        eff_vec = vec_rank
        if (
            candidate_count is not None
            and vec_rank is not None
            and vec_rank > candidate_count
        ):
            eff_vec = None  # dropped from the truncated vector leg
        if fts_rank is None and eff_vec is None:
            continue  # row absent from both legs
        scored.append((_rrf_score(rrf_k, fts_rank, eff_vec), rid))
    # Deterministic tie-break by id so the test is repeatable.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [rid for _score, rid in scored[:k]]


class TestBoundedVsExactRrf:
    """The candidate cap is the recall-quality lever: a row with a strong
    fts_rank but vec_rank just outside the candidate pool loses its vector RRF
    term and can drop from the final top-k that exact RRF would include.
    These tests pin that tradeoff deterministically and show that raising the
    multiplier recovers the boundary row.
    """

    def test_boundary_row_dropped_at_small_cap(self):
        rrf_k = 60
        k = 3
        # 'boundary' is strong on FTS (rank 1) but its vec_rank (45) sits just
        # outside a cap of 40. Three filler rows are solidly inside both legs.
        # Fillers are vector-only (single RRF term). 'boundary' carries BOTH a
        # moderate fts term and a vector term just outside a cap of 40: with
        # both terms it outranks every filler (exact), but once the cap drops
        # its vector term its lone fts term loses to all fillers (bounded).
        rows = {
            "a": (None, 1),
            "b": (None, 2),
            "c": (None, 3),
            "boundary": (20, 45),
        }
        exact = _topk_ids(rows, rrf_k=rrf_k, candidate_count=None, k=k)
        bounded = _topk_ids(rows, rrf_k=rrf_k, candidate_count=40, k=k)
        # Exact RRF ranks 'boundary' into the top-k (fts term dominates).
        assert "boundary" in exact
        # Bounded (cap 40) drops its vector term and pushes it out of top-k.
        assert "boundary" not in bounded

    def test_raising_multiplier_recovers_boundary_row(self):
        rrf_k = 60
        k = 3
        # Fillers are vector-only (single RRF term). 'boundary' carries BOTH a
        # moderate fts term and a vector term just outside a cap of 40: with
        # both terms it outranks every filler (exact), but once the cap drops
        # its vector term its lone fts term loses to all fillers (bounded).
        rows = {
            "a": (None, 1),
            "b": (None, 2),
            "c": (None, 3),
            "boundary": (20, 45),
        }
        # default multiplier 8 -> cap 40 drops it; multiplier 10 -> cap 50
        # includes vec_rank 45, recovering exact behavior.
        small = _topk_ids(rows, rrf_k=rrf_k, candidate_count=5 * 8, k=k)
        large = _topk_ids(rows, rrf_k=rrf_k, candidate_count=5 * 10, k=k)
        exact = _topk_ids(rows, rrf_k=rrf_k, candidate_count=None, k=k)
        assert "boundary" not in small
        assert "boundary" in large
        assert large == exact

    def test_divergence_bounded_when_all_vec_ranks_inside_cap(self):
        """When every candidate's vec_rank is within the cap, bounded RRF
        equals exact RRF exactly (zero divergence) — the documented
        no-regression regime."""
        rrf_k = 60
        k = 5
        rows = {
            f"r{i}": (i + 1, i + 1) for i in range(20)
        }  # all vec_ranks 1..20, well inside cap 40
        exact = _topk_ids(rows, rrf_k=rrf_k, candidate_count=None, k=k)
        bounded = _topk_ids(rows, rrf_k=rrf_k, candidate_count=40, k=k)
        assert bounded == exact

    def test_default_cap_overlap_within_documented_bound(self):
        """At multiplier=8 (cap=40) with realistic rank spread, the bounded
        top-k overlaps exact top-k by at least k-1 of k (documented bound:
        at most one boundary displacement for these fixtures)."""
        rrf_k = 60
        k = 5
        rows = {f"r{i}": (i + 1, i + 1) for i in range(60)}
        # One adversarial boundary row: strong fts, vec just outside cap.
        rows["edge"] = (1, 41)
        exact = set(_topk_ids(rows, rrf_k=rrf_k, candidate_count=None, k=k))
        bounded = set(_topk_ids(rows, rrf_k=rrf_k, candidate_count=40, k=k))
        overlap = len(exact & bounded)
        assert overlap >= k - 1, (overlap, exact, bounded)
