"""Tests for issue #54 — SearchCapture surfaces the query embedding.

``search_learnings_hybrid_rrf`` already computes a query embedding internally
but returns only the results list. To wire the type-affinity signal end to end,
the caller needs that embedding (and its model-space label) without paying for a
second embed call. ``SearchCapture`` is a small mutable out-param: the hybrid
backend fills it after a successful embed; the degraded text-only path (missing
key / embed failure) and the text-only/sqlite backends leave it untouched.

Mocking conventions mirror ``tests/test_recall_backends_fallback.py``: a fake
embedder + a fake pool so no real DB or network is required.
"""

from __future__ import annotations

from typing import Any

import pytest


class _OkEmbedder:
    """EmbeddingService stand-in whose ``embed`` succeeds and exposes a label."""

    model_label = "voyage-code-3"

    def __init__(self, *a: Any, **kw: Any) -> None: ...

    async def embed(self, *_a: Any, **_kw: Any) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    async def aclose(self) -> None: ...


class _RaisingEmbedder:
    """EmbeddingService stand-in whose ``embed`` always raises (degrade path)."""

    model_label = "voyage-code-3"

    def __init__(self, *a: Any, **kw: Any) -> None: ...

    async def embed(self, *_a: Any, **_kw: Any) -> list[float]:
        raise RuntimeError("connection refused")

    async def aclose(self) -> None: ...


def _patch_embedder(monkeypatch, embedder_cls: type) -> None:
    import scripts.core.db.embedding_service as emb_mod

    monkeypatch.setattr(emb_mod, "EmbeddingService", embedder_cls)


def _patch_pool(monkeypatch) -> None:
    """Patch get_pool/init_pgvector so the hybrid path needs no real DB.

    embedding_model_column_available is patched to False so the model-filter
    probe doesn't require a real connection and the path stays simple.
    """
    import scripts.core.db.postgres_pool as pool_mod
    from scripts.core import recall_backends as rb

    class FakeConn:
        async def fetch(self, _sql: str, *_args: Any) -> list[Any]:
            return []

    class FakeAcquire:
        async def __aenter__(self) -> FakeConn:
            return FakeConn()

        async def __aexit__(self, *_exc: Any) -> bool:
            return False

    class FakePool:
        def acquire(self) -> FakeAcquire:
            return FakeAcquire()

    async def fake_get_pool():
        return FakePool()

    async def fake_init_pgvector(_conn: Any) -> None:
        return None

    async def fake_col_available(_conn: Any) -> bool:
        return False

    async def fake_project_col(_conn: Any) -> bool:
        return False

    monkeypatch.setattr(pool_mod, "get_pool", fake_get_pool)
    monkeypatch.setattr(pool_mod, "init_pgvector", fake_init_pgvector)
    monkeypatch.setattr(rb, "embedding_model_column_available", fake_col_available)
    monkeypatch.setattr(rb, "project_column_available", fake_project_col)


@pytest.fixture(autouse=True)
def _reset_degrade_latch():
    from scripts.core import recall_backends as rb

    rb._EMBED_DEGRADE_WARNED = False
    yield
    rb._EMBED_DEGRADE_WARNED = False


class TestSearchCaptureDataclass:
    def test_defaults_are_none(self):
        from scripts.core.recall_backends import SearchCapture

        cap = SearchCapture()
        assert cap.query_embedding is None
        assert cap.model_label is None

    def test_is_mutable(self):
        from scripts.core.recall_backends import SearchCapture

        cap = SearchCapture()
        cap.query_embedding = [1.0, 2.0]
        cap.model_label = "voyage-code-3"
        assert cap.query_embedding == [1.0, 2.0]
        assert cap.model_label == "voyage-code-3"


class TestHybridFillsCapture:
    async def test_capture_filled_on_embed_success(self, monkeypatch):
        from scripts.core import recall_backends as rb
        from scripts.core.recall_backends import SearchCapture

        _patch_embedder(monkeypatch, _OkEmbedder)
        _patch_pool(monkeypatch)

        # Embedding_model column present -> the model-space label survives the
        # #151 probe and is surfaced for model-filtered centroids.
        async def col_present(_conn: Any) -> bool:
            return True

        monkeypatch.setattr(rb, "embedding_model_column_available", col_present)

        cap = SearchCapture()
        await rb.search_learnings_hybrid_rrf(
            "what time is it", k=3, expand=False, capture=cap,
        )

        # The backend surfaced the embedding it already computed.
        assert cap.query_embedding == [0.1, 0.2, 0.3, 0.4]
        assert cap.model_label == "voyage-code-3"

    async def test_label_nulled_when_embedding_model_column_missing(self, monkeypatch):
        """Pre-#151 DB (no embedding_model column): the embedding is still
        surfaced, but model_label is None so the caller skips model-filtered
        centroids rather than cosine across embedding spaces."""
        from scripts.core import recall_backends as rb
        from scripts.core.recall_backends import SearchCapture

        _patch_embedder(monkeypatch, _OkEmbedder)
        _patch_pool(monkeypatch)  # patches col-available -> False

        cap = SearchCapture()
        await rb.search_learnings_hybrid_rrf(
            "what time is it", k=3, expand=False, capture=cap,
        )

        assert cap.query_embedding == [0.1, 0.2, 0.3, 0.4]
        assert cap.model_label is None

    async def test_no_capture_arg_is_byte_identical(self, monkeypatch):
        """Omitting capture must not change behavior (no caller breakage)."""
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _OkEmbedder)
        _patch_pool(monkeypatch)

        results = await rb.search_learnings_hybrid_rrf(
            "query terms", k=3, expand=False,
        )
        assert results == []


class TestDegradedPathLeavesCaptureEmpty:
    async def test_capture_untouched_on_embed_failure(self, monkeypatch):
        from scripts.core import recall_backends as rb
        from scripts.core.recall_backends import SearchCapture

        _patch_embedder(monkeypatch, _RaisingEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return []

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        cap = SearchCapture()
        await rb.search_learnings_hybrid_rrf(
            "what time is it", k=3, capture=cap,
        )

        # Degraded to text-only: the capture is never populated, so the caller
        # leaves type affinity disabled (neutral reranking).
        assert cap.query_embedding is None
        assert cap.model_label is None
