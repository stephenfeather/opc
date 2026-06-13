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


class _CapturingConn:
    """Records (sql, args) for every fetch and returns canned rows."""

    def __init__(self, row_provider=None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._row_provider = row_provider or (lambda sql, args: [])

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.calls.append((sql, args))
        return self._row_provider(sql, args)

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
        return {"cnt": 0}


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
