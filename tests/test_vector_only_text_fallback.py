"""Round 2 FIX 4: --vector-only honors text_fallback on a zero-match filter.

With has_embeddings true, a model-filtered vector query that matches zero rows
(split-corpus: the query's space has no rows yet) must NOT return empty when
text_fallback is true — it must run the existing _PG_TEXT_FALLBACK_SQL path.
The inverse (text_fallback=False) stays empty.
"""

from __future__ import annotations

from typing import Any

import pytest


class _SplitCorpusDb:
    """Vector queries (embedding_model filter) return nothing; the text
    fallback (ILIKE) returns one row. Embeddings are reported present."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def make_pool(self):
        db = self

        class FakeConn:
            async def fetch(self, sql: str, *args: Any) -> list[Any]:
                db.executed.append(sql)
                # Text fallback path uses ILIKE; return a hit there only.
                if "ILIKE" in sql:
                    return [
                        {
                            "id": "11111111-1111-1111-1111-111111111111",
                            "session_id": "s0",
                            "content": "fallback hit",
                            "metadata": {"type": "session_learning"},
                            "created_at": None,
                            "similarity": 0.5,
                        }
                    ]
                # Vector / recency path: model filter matches nothing.
                return []

            async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
                return {"cnt": 5}  # embeddings present

        class FakeAcquire:
            async def __aenter__(self) -> FakeConn:
                return FakeConn()

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        class FakePool:
            def acquire(self) -> FakeAcquire:
                return FakeAcquire()

        return FakePool()


def _patch(monkeypatch, db: _SplitCorpusDb) -> None:
    async def fake_get_pool():
        return db.make_pool()

    async def fake_init_pgvector(_conn: Any) -> None:
        return None

    import scripts.core.db.embedding_service as emb_mod
    import scripts.core.db.postgres_pool as pool_mod

    class FakeEmbedder:
        def __init__(self, *a: Any, **kw: Any) -> None: ...

        @property
        def model_label(self) -> str:
            return "voyage-code-3"

        async def embed(self, *_a: Any, **_kw: Any) -> list[float]:
            return [0.1] * 8

        async def aclose(self) -> None: ...

    monkeypatch.setattr(pool_mod, "get_pool", fake_get_pool)
    monkeypatch.setattr(pool_mod, "init_pgvector", fake_init_pgvector)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbedder)


class TestVectorOnlyZeroMatchFallback:
    async def test_zero_match_with_text_fallback_returns_text(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        rb.reset_embedding_model_column_cache()
        db = _SplitCorpusDb()
        _patch(monkeypatch, db)

        results = await rb.search_learnings_postgres(
            "fallback", k=3, text_fallback=True,
        )
        assert len(results) == 1
        assert results[0]["content"] == "fallback hit"
        # The model-filtered vector query ran first (and matched nothing),
        # then the ILIKE text fallback ran.
        assert any("embedding_model" in s for s in db.executed)
        assert any("ILIKE" in s for s in db.executed)

    async def test_zero_match_without_text_fallback_stays_empty(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        rb.reset_embedding_model_column_cache()
        db = _SplitCorpusDb()
        _patch(monkeypatch, db)

        results = await rb.search_learnings_postgres(
            "fallback", k=3, text_fallback=False,
        )
        assert results == []
        # Vector ran; the ILIKE fallback did NOT (flag respected).
        assert any("embedding_model" in s for s in db.executed)
        assert not any("ILIKE" in s for s in db.executed)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
