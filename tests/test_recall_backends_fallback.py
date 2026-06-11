"""Tests for issue #53 — hybrid RRF degrades to text-only when query-embed fails.

The memory-awareness hook will call hybrid recall (dropping ``--text-only``).
Hybrid must never error out solely because the embedding service is
unavailable: it wraps the query-embed call in try/except and, on failure,
emits a redacted warning and returns the ``search_learnings_text_only_postgres``
results for that search (same conn pool, same ``k``/``project`` semantics).
Result shape in the degraded case is identical to the text-only path.

Per-pass degradation / redaction conventions mirror issue #139
(``recall_learnings._dispatch_search_project_first``): full traceback only in
the debug log, a sanitized one-line warning reaches hook-captured stderr, and
the query text is never included in any warning.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest


class _RaisingEmbedder:
    """EmbeddingService stand-in whose ``embed`` always raises."""

    def __init__(self, *a: Any, **kw: Any) -> None: ...

    async def embed(self, *_a: Any, **_kw: Any) -> list[float]:
        raise RuntimeError(
            "connection to postgresql://user:secret@dbhost:5432/memdb refused"
        )

    async def aclose(self) -> None: ...


class _OkEmbedder:
    """EmbeddingService stand-in whose ``embed`` succeeds."""

    def __init__(self, *a: Any, **kw: Any) -> None: ...

    async def embed(self, *_a: Any, **_kw: Any) -> list[float]:
        return [0.1] * 8

    async def aclose(self) -> None: ...


class _ConstructRaisingEmbedder:
    """EmbeddingService stand-in whose __init__ raises (e.g. Voyage with no
    VOYAGE_API_KEY, or a local model load error during construction)."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        raise ValueError(
            "VOYAGE_API_KEY environment variable required "
            "(postgresql://user:secret@dbhost:5432/memdb)"
        )

    async def aclose(self) -> None:  # pragma: no cover - never constructed
        raise AssertionError("aclose must not run when __init__ failed")


def _patch_embedder(monkeypatch, embedder_cls: type) -> None:
    import scripts.core.db.embedding_service as emb_mod

    monkeypatch.setattr(emb_mod, "EmbeddingService", embedder_cls)


class _RaiseEmbedAndCloseEmbedder:
    """embed() raises AND aclose() raises — cleanup failure must not abort
    the pending text-only fallback (review round 2, FIX A)."""

    def __init__(self, *a: Any, **kw: Any) -> None: ...

    async def embed(self, *_a: Any, **_kw: Any) -> list[float]:
        raise RuntimeError(
            "connection to postgresql://user:secret@dbhost:5432/memdb refused"
        )

    async def aclose(self) -> None:
        raise RuntimeError("aclose blew up")


class _SlowEmbedder:
    """embed() stalls (simulating a network hang past the recall deadline);
    aclose records that cleanup ran so we can assert cancellation safety."""

    SLEEP = 5.0

    def __init__(self, *a: Any, **kw: Any) -> None:
        self.closed = False

    async def embed(self, *_a: Any, **_kw: Any) -> list[float]:
        await asyncio.sleep(self.SLEEP)
        return [0.1] * 8

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_degrade_latch():
    """The once-per-process stderr warning latch must not leak across tests."""
    from scripts.core import recall_backends as rb

    rb._EMBED_DEGRADE_WARNED = False
    yield
    rb._EMBED_DEGRADE_WARNED = False


def _patch_pool(monkeypatch) -> None:
    """Patch get_pool/init_pgvector so the hybrid path needs no real DB."""
    import scripts.core.db.postgres_pool as pool_mod

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

    monkeypatch.setattr(pool_mod, "get_pool", fake_get_pool)
    monkeypatch.setattr(pool_mod, "init_pgvector", fake_init_pgvector)


# Representative text-only result (format_text_result shape).
_TEXT_RESULT = {
    "id": "pref-1",
    "session_id": "sess-1",
    "content": "user prefers terse responses",
    "metadata": {"learning_type": "USER_PREFERENCE"},
    "created_at": None,
    "similarity": 0.5,
}


class TestHybridEmbedFallback:
    """Embedding failure degrades hybrid to the text-only backend."""

    async def test_embed_failure_returns_text_only_results(self, monkeypatch):
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _RaisingEmbedder)
        _patch_pool(monkeypatch)

        captured: dict[str, Any] = {}

        async def fake_text_only(query, k=10, *, project=None):
            captured["query"] = query
            captured["k"] = k
            captured["project"] = project
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        results = await rb.search_learnings_hybrid_rrf(
            "what time is it", k=3, project="opc",
        )

        # The text-only backend produced the degraded results, unchanged.
        assert results == [dict(_TEXT_RESULT)]
        # Same k / project semantics are forwarded to the fallback.
        assert captured == {"query": "what time is it", "k": 3, "project": "opc"}

    async def test_embed_failure_emits_redacted_warning(self, monkeypatch, capsys):
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _RaisingEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        await rb.search_learnings_hybrid_rrf("secret query phrase", k=3)

        err = capsys.readouterr().err
        # A warning surfaced to hook-captured stderr.
        assert "warning" in err.lower()
        # It names the degradation (embedding -> text-only).
        assert "text-only" in err.lower()
        # DSN credentials in the exception are redacted (aegis MEDIUM-2 style).
        assert "secret" not in err
        assert "user:secret@" not in err
        # The query text is never echoed into the warning.
        assert "secret query phrase" not in err

    async def test_embed_success_does_not_invoke_fallback(self, monkeypatch):
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _OkEmbedder)
        _patch_pool(monkeypatch)

        def boom(*_a: Any, **_kw: Any):
            raise AssertionError("text-only fallback must not run on embed success")

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", boom)

        # FakeConn returns no rows, so the hybrid path yields []. The point is
        # that the fallback is never invoked when embedding succeeds.
        results = await rb.search_learnings_hybrid_rrf(
            "query terms", k=3, expand=False,
        )
        assert results == []

    async def test_embed_and_text_failure_propagates(self, monkeypatch):
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _RaisingEmbedder)
        _patch_pool(monkeypatch)

        class _FallbackError(RuntimeError):
            pass

        async def fake_text_only(query, k=10, *, project=None):
            raise _FallbackError("text-only backend also down")

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        # Both passes failing surfaces an error like the default path.
        with pytest.raises(_FallbackError):
            await rb.search_learnings_hybrid_rrf("query terms", k=3)

    async def test_degraded_shape_matches_text_only_shape(self, monkeypatch):
        """Degraded hybrid result keys are identical to the text-only path."""
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _RaisingEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        degraded = await rb.search_learnings_hybrid_rrf("query terms", k=3)

        assert degraded, "expected degraded results"
        # Key parity with format_text_result output — no RRF-only keys leak in.
        assert set(degraded[0].keys()) == set(_TEXT_RESULT.keys())
        for rrf_only in ("fts_rank", "vec_rank", "raw_rrf_score"):
            assert rrf_only not in degraded[0]

    async def test_constructor_failure_returns_text_only_results(self, monkeypatch):
        """Round 1 fix: EmbeddingService(...) construction failure (e.g. Voyage
        with no key) is guarded too — degrade to text-only, no aclose crash."""
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _ConstructRaisingEmbedder)
        _patch_pool(monkeypatch)

        captured: dict[str, Any] = {}

        async def fake_text_only(query, k=10, *, project=None):
            captured["query"] = query
            captured["k"] = k
            captured["project"] = project
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        results = await rb.search_learnings_hybrid_rrf(
            "what time is it", k=3, project="opc",
        )

        assert results == [dict(_TEXT_RESULT)]
        assert captured == {"query": "what time is it", "k": 3, "project": "opc"}

    async def test_constructor_failure_emits_redacted_warning(self, monkeypatch, capsys):
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _ConstructRaisingEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        await rb.search_learnings_hybrid_rrf("secret query phrase", k=3)

        err = capsys.readouterr().err
        assert "warning" in err.lower()
        assert "text-only" in err.lower()
        # DSN credentials embedded in the constructor error are redacted.
        assert "secret" not in err
        assert "user:secret@" not in err
        # The query text is never echoed into the warning.
        assert "secret query phrase" not in err

    async def test_real_voyage_construction_without_key_degrades(
        self, monkeypatch,
    ):
        """Hermetic constructor-path test against the REAL EmbeddingService:
        provider='voyage' with VOYAGE_API_KEY unset raises ValueError in
        __init__ (pure env check, no network/model deps), which the guard
        must catch and degrade to text-only.

        VoyageEmbeddingProvider.__init__ only reads os.environ and constructs
        an httpx client lazily after the key check — so this needs no API call
        and no model download. The text-only backend is patched, so no DB.
        """
        from scripts.core import recall_backends as rb

        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        results = await rb.search_learnings_hybrid_rrf(
            "query terms", k=3, provider="voyage",
        )
        assert results == [dict(_TEXT_RESULT)]

    async def test_aclose_failure_does_not_abort_fallback(self, monkeypatch):
        """FIX A: a raise from aclose() in finally must not override the pending
        text-only fallback return — results still come back, warning still fires."""
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _RaiseEmbedAndCloseEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        # No exception leaks even though aclose() raises.
        results = await rb.search_learnings_hybrid_rrf("query terms", k=3)
        assert results == [dict(_TEXT_RESULT)]

    async def test_aclose_failure_still_emits_warning(self, monkeypatch, capsys):
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _RaiseEmbedAndCloseEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        await rb.search_learnings_hybrid_rrf("query terms", k=3)
        err = capsys.readouterr().err
        assert "text-only" in err.lower()

    async def test_warning_latched_once_per_process(self, monkeypatch, capsys):
        """FIX B(1): two consecutive degraded searches in one process emit
        exactly one stderr warning (logger.debug still fires every time)."""
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _RaisingEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        await rb.search_learnings_hybrid_rrf("first query", k=3)
        await rb.search_learnings_hybrid_rrf("second query", k=3)

        err = capsys.readouterr().err
        assert err.lower().count("warning") == 1, err

    async def test_warning_includes_provider_name_and_is_redacted(
        self, monkeypatch, capsys,
    ):
        """FIX B(2): warning names the provider and stays redacted."""
        from scripts.core import recall_backends as rb

        _patch_embedder(monkeypatch, _RaisingEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        await rb.search_learnings_hybrid_rrf(
            "secret query phrase", k=3, provider="voyage",
        )
        err = capsys.readouterr().err
        assert "voyage" in err.lower()
        # Redaction intact: DSN password gone, query text never echoed.
        assert "secret" not in err
        assert "user:secret@" not in err
        assert "secret query phrase" not in err

    async def test_embed_timeout_degrades_to_text_only_fast(self, monkeypatch):
        """FIX (round 3): a stalled embed must hit the recall-specific deadline
        and degrade to text-only well before the hook's 5s budget — not block on
        the provider's own 30s+retry timeout."""
        from scripts.core import recall_backends as rb

        # Tiny deadline so the test is fast; the embed sleeps far longer.
        monkeypatch.setattr(rb, "QUERY_EMBED_TIMEOUT", 0.05)
        _patch_embedder(monkeypatch, _SlowEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        start = time.monotonic()
        results = await rb.search_learnings_hybrid_rrf("query terms", k=3)
        elapsed = time.monotonic() - start

        assert results == [dict(_TEXT_RESULT)]
        # Generous, loaded-machine-tolerant bound: well under the 5s embed sleep
        # (and under the hook's 5s budget). See the #140 timing-flake lesson.
        assert elapsed < 3.0, f"degrade took {elapsed:.2f}s; deadline not enforced"

    async def test_embed_timeout_emits_latched_provider_warning(
        self, monkeypatch, capsys,
    ):
        """Timeout flows into the same latched, redacted, provider-named warning,
        and the message signals a timeout."""
        from scripts.core import recall_backends as rb

        monkeypatch.setattr(rb, "QUERY_EMBED_TIMEOUT", 0.05)
        _patch_embedder(monkeypatch, _SlowEmbedder)
        _patch_pool(monkeypatch)

        async def fake_text_only(query, k=10, *, project=None):
            return [dict(_TEXT_RESULT)]

        monkeypatch.setattr(rb, "search_learnings_text_only_postgres", fake_text_only)

        # Two stalled searches → exactly one warning (latch still applies).
        await rb.search_learnings_hybrid_rrf("first query", k=3, provider="voyage")
        await rb.search_learnings_hybrid_rrf("second query", k=3, provider="voyage")

        err = capsys.readouterr().err
        assert err.lower().count("warning") == 1, err
        assert "voyage" in err.lower()
        assert "timed out" in err.lower()
        # Redaction + no query text still hold.
        assert "first query" not in err
        assert "second query" not in err
