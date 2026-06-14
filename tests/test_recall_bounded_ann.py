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


class _AbortedTransactionError(Exception):
    """Models Postgres 'current transaction is aborted' (round-2 finding 1).

    Once a statement fails inside a transaction, every subsequent statement in
    that SAME transaction raises this until the transaction ends. A fresh
    transaction() clears the aborted state.
    """


class _CapturingConn:
    """Capturing conn modelling Postgres aborted-transaction semantics.

    Records ``fetch``/``execute`` calls and ``transaction()`` enter/exit so
    tests can assert the per-attempt-transaction design (issue #153 round-2).
    After any statement raises inside a transaction, further statements in the
    same transaction raise ``_AbortedTransactionError`` — proving the cascade
    must open a fresh transaction per fallback attempt. ``execute_error``
    simulates an older pgvector that rejects the ``hnsw.iterative_scan`` GUC
    (the error aborts whatever transaction issues the SET).
    """

    def __init__(
        self, row_provider=None, *, execute_error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.executed: list[tuple[str, tuple]] = []
        self.tx_depth = 0
        self.max_tx_depth = 0
        # Records (sql, tx_depth_at_call_time) for every fetch so tests can
        # assert the RRF fetch happened inside an open transaction.
        self.fetch_tx_depth: list[tuple[str, int]] = []
        self._row_provider = row_provider or (lambda sql, args: [])
        self._execute_error = execute_error
        # True while the current transaction is in the aborted state.
        self._aborted = False

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        if self.tx_depth > 0 and self._aborted:
            raise _AbortedTransactionError(
                "current transaction is aborted, commands ignored until "
                "end of transaction block"
            )
        self.calls.append((sql, args))
        self.fetch_tx_depth.append((sql, self.tx_depth))
        try:
            return self._row_provider(sql, args)
        except Exception:
            # A failing statement poisons the rest of THIS transaction.
            if self.tx_depth > 0:
                self._aborted = True
            raise

    async def execute(self, sql: str, *args: Any) -> str:
        if self.tx_depth > 0 and self._aborted:
            raise _AbortedTransactionError(
                "current transaction is aborted, commands ignored until "
                "end of transaction block"
            )
        self.executed.append((sql, args))
        if self._execute_error is not None and "iterative_scan" in sql:
            if self.tx_depth > 0:
                self._aborted = True
            raise self._execute_error
        return "SET"

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
        return {"cnt": 0}

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self):
                conn.tx_depth += 1
                conn.max_tx_depth = max(conn.max_tx_depth, conn.tx_depth)
                return conn

            async def __aexit__(self, *exc):
                conn.tx_depth -= 1
                # Leaving the transaction (commit OR rollback) clears the
                # aborted state — the next transaction starts clean.
                if conn.tx_depth == 0:
                    conn._aborted = False
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


# ====== Round-1/2 finding 1: filtered-HNSW under-fill GUC + cascade ======


class TestRrfIterativeScanGuc:
    """The bounded vector leg under-fills when pgvector applies WHERE filters
    after the HNSW scan. ``hnsw.iterative_scan = strict_order`` makes the leg
    return true nearest-k under filters. Round-2 redesign: the GUC is PROBED
    once in its own rolled-back transaction (so version skew never poisons a
    fetch), and EACH fetch attempt runs in its OWN transaction (so a failing
    fallback rolls back cleanly instead of aborting the rest of the cascade).
    """

    async def test_set_local_iterative_scan_issued_for_fetch(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = _CapturingConn()
        _patch_rrf_env(monkeypatch, conn)

        await rb.search_learnings_hybrid_rrf("query terms", k=3, expand=False)

        # SET LOCAL (transaction-scoped), strict_order, is issued (probe + the
        # fetch attempt both run it).
        guc_stmts = [s for s, _ in conn.executed if "iterative_scan" in s]
        assert guc_stmts, "expected a SET LOCAL hnsw.iterative_scan statement"
        stmt = guc_stmts[0]
        assert "SET LOCAL" in stmt.upper()
        assert "strict_order" in stmt

    async def test_rrf_fetch_runs_inside_transaction(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = _CapturingConn()
        _patch_rrf_env(monkeypatch, conn)

        await rb.search_learnings_hybrid_rrf("query terms", k=3, expand=False)

        rrf_fetches = [
            depth for sql, depth in conn.fetch_tx_depth
            if "vector_ranked" in sql
        ]
        assert rrf_fetches, "expected an RRF fetch"
        assert all(d >= 1 for d in rrf_fetches), (
            "RRF fetch must run inside an open transaction (SET LOCAL scope)"
        )

    async def test_guc_set_before_rrf_fetch_in_order(self, monkeypatch):
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

    async def test_fallback_cascade_survives_aborted_transaction(
        self, monkeypatch,
    ):
        """Round-2 finding 1: the boosted fetch fails (missing recall_count) and
        aborts ITS transaction. With per-attempt transactions the plain fetch
        opens a FRESH transaction and succeeds. The single-transaction design
        would have died here with 'current transaction is aborted'."""
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()

        def provider(sql, args):
            from asyncpg.exceptions import UndefinedColumnError

            if "recall_count" in sql:
                raise UndefinedColumnError('column "recall_count" does not exist')
            return []

        conn = _CapturingConn(provider)
        _patch_rrf_env(monkeypatch, conn)

        # Must not raise _AbortedTransactionError; the plain fallback reaches a
        # fresh transaction and returns.
        results = await rb.search_learnings_hybrid_rrf(
            "query terms", k=3, expand=False,
        )
        assert results == []
        # Both the boosted (failing) and a plain (succeeding) RRF query ran.
        rrf_calls = [c for c in conn.calls if "vector_ranked" in c[0]]
        assert any("recall_count" in c[0] for c in rrf_calls), (
            "boosted attempt should have been tried"
        )
        assert any("recall_count" not in c[0] for c in rrf_calls), (
            "plain fallback must reach a fresh transaction and run"
        )

    async def test_unsupported_guc_probe_false_fetch_runs_no_set_local(
        self, monkeypatch, caplog,
    ):
        """Round-2 finding 2: on older pgvector the probe returns False, no SET
        LOCAL is issued on the fetch attempts (so nothing aborts the fetch
        transaction), recall still runs, and a warning is logged."""
        import logging

        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        boom = RuntimeError('unrecognized configuration parameter '
                            '"hnsw.iterative_scan"')
        conn = _CapturingConn(execute_error=boom)
        _patch_rrf_env(monkeypatch, conn)

        with caplog.at_level(logging.WARNING):
            results = await rb.search_learnings_hybrid_rrf(
                "query terms", k=3, expand=False,
            )
        assert results == []

        # The RRF fetch still ran.
        rrf_calls = [c for c in conn.calls if "vector_ranked" in c[0]]
        assert rrf_calls, "RRF fetch must still run when the GUC is unsupported"

        # Exactly the probe issued the SET LOCAL (and failed); no SET LOCAL was
        # issued on the fetch attempts (use_guc is False), so no fetch
        # transaction was ever poisoned by the GUC.
        guc_attempts = [s for s, _ in conn.executed if "iterative_scan" in s]
        assert len(guc_attempts) == 1, (
            "only the probe should attempt the GUC; fetches must skip it"
        )

        # Operators get a warning-level signal, not just a debug line.
        assert any(
            "iterative" in r.message.lower() and r.levelno >= logging.WARNING
            for r in caplog.records
        ), "expected a warning that bounded ANN runs without iterative scan"

    async def test_probe_caches_definitive_result(self, monkeypatch):
        """The probe caches True on success and False on unsupported, so the
        own-transaction probe runs once per process (mirrors the column
        probes)."""
        from scripts.core import recall_backends as rb

        # Success path -> cached True.
        rb.reset_hnsw_iterative_scan_cache()
        ok_conn = _CapturingConn()
        assert await rb.hnsw_iterative_scan_available(ok_conn) is True
        assert rb._hnsw_iterative_scan_cache is True

        # Unsupported path -> cached False (definitive miss).
        rb.reset_hnsw_iterative_scan_cache()
        boom = RuntimeError('unrecognized configuration parameter '
                            '"hnsw.iterative_scan"')
        bad_conn = _CapturingConn(execute_error=boom)
        assert await rb.hnsw_iterative_scan_available(bad_conn) is False
        assert rb._hnsw_iterative_scan_cache is False

    async def test_probe_rolls_back_its_own_transaction(self, monkeypatch):
        """The probe must run inside a transaction it always exits (rollback),
        so an unsupported GUC never leaves an aborted transaction behind."""
        from scripts.core import recall_backends as rb

        rb.reset_hnsw_iterative_scan_cache()
        boom = RuntimeError('unrecognized configuration parameter '
                            '"hnsw.iterative_scan"')
        conn = _CapturingConn(execute_error=boom)

        await rb.hnsw_iterative_scan_available(conn)

        # The probe opened and fully exited its transaction (depth back to 0)
        # and the aborted state was cleared on exit.
        assert conn.tx_depth == 0
        assert conn._aborted is False
        assert conn.max_tx_depth >= 1, "probe must use its own transaction"


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
