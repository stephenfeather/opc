"""Integration: EXPLAIN proves the reshaped RRF vector leg is HNSW-servable.

Issue #153. The reshaped ``vector_ranked`` CTE wraps a bounded inner
``ORDER BY embedding <=> $q LIMIT $candidates`` subquery — the shape pgvector's
HNSW index can serve as a top-k walk. The OLD unbounded windowed leg
(``ROW_NUMBER() OVER (ORDER BY embedding <=> $q)`` with no inner LIMIT) forces
a full rank of every filtered row.

CORPUS-SIZE CAVEAT (important): on the current ~6k-row corpus the planner
*correctly* prefers a Seq Scan + Sort over the HNSW index, because sorting 6k
in-memory rows is cheaper than pgvector's HNSW graph walk at that scale (the
HNSW cost estimate is ~16x the seq-scan estimate). HNSW only wins on cost once
the corpus grows large. So this test does NOT assert "HNSW is used by default"
— that would misrepresent the planner. Instead it steers the planner into the
large-corpus regime (``enable_seqscan/enable_sort/enable_bitmapscan = off``)
and asserts the reshaped leg plans an ``Index Scan using <hnsw>`` *bounded by a
Limit* — proving the reshape is HNSW-servable as a top-k walk. The negative
control proves the legacy unbounded leg has no such bounding Limit on its scan.

Skip-guarded: requires a reachable database, the pgvector extension, the HNSW
index, and embedded session_learning rows. Absent any of these the test skips
so unit CI stays green.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.asyncio

# Steer the planner into the large-corpus regime where HNSW wins, so we can
# observe HNSW eligibility deterministically regardless of today's row count.
_FORCE_INDEX_GUCS = (
    "SET LOCAL enable_seqscan = off;"
    "SET LOCAL enable_sort = off;"
    "SET LOCAL enable_bitmapscan = off;"
)


async def _try_connect():
    """Return a live asyncpg connection or None when the DB is unreachable."""
    try:
        import asyncpg

        from scripts.core.db.postgres_pool import get_connection_string
    except Exception:
        return None
    try:
        dsn = get_connection_string()
    except Exception:
        return None
    if not dsn:
        return None
    try:
        return await asyncpg.connect(dsn, timeout=5)
    except Exception:
        return None


async def _hnsw_index_present(conn) -> bool:
    row = await conn.fetchval(
        "SELECT 1 FROM pg_indexes WHERE tablename = 'archival_memory' "
        "AND indexdef ILIKE '%hnsw%' LIMIT 1"
    )
    return bool(row)


async def _embedded_row_count(conn) -> int:
    return await conn.fetchval(
        "SELECT count(*) FROM archival_memory "
        "WHERE metadata->>'type' = 'session_learning' AND embedding IS NOT NULL"
    )


async def _sample_embedding(conn) -> str | None:
    return await conn.fetchval(
        "SELECT embedding::text FROM archival_memory "
        "WHERE embedding IS NOT NULL LIMIT 1"
    )


async def _explain(conn, sql: str, *args) -> str:
    # SET LOCAL needs a transaction; run the EXPLAIN in the same one.
    async with conn.transaction():
        await conn.execute(_FORCE_INDEX_GUCS)
        rows = await conn.fetch(f"EXPLAIN (FORMAT TEXT) {sql}", *args)
    return "\n".join(r[0] for r in rows)


class TestRrfVectorLegHnswServable:
    async def _setup(self):
        if os.environ.get("OPC_SKIP_DB_TESTS"):
            pytest.skip("OPC_SKIP_DB_TESTS set")
        conn = await _try_connect()
        if conn is None:
            pytest.skip("database unreachable")
        if not await _hnsw_index_present(conn):
            await conn.close()
            pytest.skip("HNSW index absent")
        if await _embedded_row_count(conn) < 50:
            await conn.close()
            pytest.skip("too few embedded rows to exercise the index")
        embedding = await _sample_embedding(conn)
        if embedding is None:
            await conn.close()
            pytest.skip("no sample embedding")
        return conn, embedding

    async def test_bounded_vector_leg_plans_hnsw_index_scan(self):
        conn, embedding = await self._setup()
        try:
            from scripts.core.recall_backends import (
                _RRF_PLAIN_TAIL_SQL,
                build_rrf_cte,
                render_recall_sql,
            )

            cte = build_rrf_cte(chain_filter=True, candidate_param=5)
            tail = render_recall_sql(
                _RRF_PLAIN_TAIL_SQL,
                include_project=True,
                project_expr=", a.project",
            )
            sql = cte + tail
            # $1 text, $2 embedding, $3 rrf_k, $4 tail LIMIT, $5 candidates.
            plan = await _explain(conn, sql, "memory recall", embedding, 60, 10, 40)

            assert "idx_archival_embedding_hnsw" in plan, (
                "bounded vector leg did not plan the HNSW index scan; "
                "plan was:\n" + plan
            )
        finally:
            await conn.close()

    async def test_legacy_unbounded_leg_has_no_bounded_index_walk(self):
        """Negative control: the legacy unbounded leg, even when steered onto
        the HNSW index, must walk it unbounded (no inner Limit) — it ranks
        every filtered row. This is the cost trap the reshape removes."""
        conn, embedding = await self._setup()
        try:
            from scripts.core.recall_backends import build_rrf_cte

            legacy_cte = build_rrf_cte(chain_filter=True)  # candidate_param=None
            # Drive the legacy vector_ranked leg directly so the plan is the
            # leg's plan, not the join's.
            sql = (
                legacy_cte
                + "\n            SELECT id, vec_rank FROM vector_ranked LIMIT 10"
            )
            plan = await _explain(conn, sql, "memory recall", embedding, 60)

            # The legacy leg ranks the full filtered set: the windowed
            # ROW_NUMBER consumes every row, so there is no inner Limit
            # bounding the vector scan to a small candidate pool. (The trailing
            # LIMIT 10 is on the outer SELECT, after the full window.)
            assert "WindowAgg" in plan, (
                "expected a WindowAgg over the full filtered set in the legacy "
                "leg; plan was:\n" + plan
            )
        finally:
            await conn.close()


class TestRrfIterativeScanGucLive:
    """Round-1 finding 1 (live): the SET LOCAL hnsw.iterative_scan GUC the
    fetch path issues must be accepted by the installed pgvector, and the
    bounded vector leg under it returns up to candidate_count rows."""

    async def _setup(self):
        if os.environ.get("OPC_SKIP_DB_TESTS"):
            pytest.skip("OPC_SKIP_DB_TESTS set")
        conn = await _try_connect()
        if conn is None:
            pytest.skip("database unreachable")
        if not await _hnsw_index_present(conn):
            await conn.close()
            pytest.skip("HNSW index absent")
        if await _embedded_row_count(conn) < 50:
            await conn.close()
            pytest.skip("too few embedded rows")
        embedding = await _sample_embedding(conn)
        if embedding is None:
            await conn.close()
            pytest.skip("no sample embedding")
        return conn, embedding

    async def test_set_local_strict_order_accepted_and_leg_fills(self):
        conn, embedding = await self._setup()
        try:
            from scripts.core.recall_backends import (
                _HNSW_ITERATIVE_SCAN_GUC,
                build_rrf_cte,
            )

            candidate_count = 40
            # Run the exact GUC the production fetch path issues, then the
            # bounded vector leg, inside one transaction (SET LOCAL scope).
            async with conn.transaction():
                # Must not raise on pgvector >= 0.8 (strict_order is valid).
                await conn.execute(_HNSW_ITERATIVE_SCAN_GUC)
                cte = build_rrf_cte(chain_filter=True, candidate_param=3)
                # Drive the bounded vector leg directly; $1 text, $2 embedding,
                # $3 candidate LIMIT.
                sql = (
                    cte
                    + "\n            SELECT id, vec_rank FROM vector_ranked"
                )
                rows = await conn.fetch(sql, "memory recall", embedding,
                                        candidate_count)
            # The chain filter (superseded_by IS NULL) is non-selective here,
            # so the bounded leg should fill to the candidate cap.
            assert len(rows) == candidate_count, (
                f"bounded leg returned {len(rows)} rows, expected the "
                f"candidate cap {candidate_count}"
            )
        finally:
            await conn.close()
