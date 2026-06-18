"""Tests for issue #152 — process-level caching of the local sentence-transformers model.

Constructing ``LocalEmbeddingProvider`` loads the BGE model (~14s cold). Before
#152 that load ran on every ``EmbeddingService(provider="local")``, so each
hybrid recall paid the full cost. The loaded ``SentenceTransformer`` is now
cached process-wide keyed by ``(model, device)`` so the cost is paid at most
once — which is also what lets the deadline-bounded (off-thread) construction
in ``recall_backends`` pay off: the abandoned cold load populates the cache so
the next recall constructs instantly.
"""

from __future__ import annotations

from typing import Any

import pytest


class _FakeSentenceTransformer:
    """Counts instantiations so we can assert the model loads at most once."""

    instances = 0

    def __init__(self, model: str, device: str | None = None) -> None:
        type(self).instances += 1
        self._model = model

    def get_sentence_embedding_dimension(self) -> int:
        return 1024

    def encode(self, *_a: Any, **_kw: Any):  # pragma: no cover - not exercised here
        return [0.0] * 1024


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch):
    """Reset the process-level model cache and patch in the counting fake."""
    import sentence_transformers

    from scripts.core.db import embedding_providers as ep

    _FakeSentenceTransformer.instances = 0
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    ep.reset_local_model_cache()
    yield
    ep.reset_local_model_cache()


def test_repeated_construction_loads_model_once():
    from scripts.core.db.embedding_providers import LocalEmbeddingProvider

    first = LocalEmbeddingProvider(model="BAAI/bge-large-en-v1.5")
    second = LocalEmbeddingProvider(model="BAAI/bge-large-en-v1.5")

    assert _FakeSentenceTransformer.instances == 1
    # Both providers share the one cached underlying model object.
    assert first._model is second._model
    assert first.dimension == 1024


def test_distinct_models_cache_separately():
    from scripts.core.db.embedding_providers import LocalEmbeddingProvider

    LocalEmbeddingProvider(model="BAAI/bge-large-en-v1.5")
    LocalEmbeddingProvider(model="all-MiniLM-L6-v2")

    assert _FakeSentenceTransformer.instances == 2


def test_distinct_devices_cache_separately():
    from scripts.core.db.embedding_providers import LocalEmbeddingProvider

    LocalEmbeddingProvider(model="BAAI/bge-large-en-v1.5", device="cpu")
    LocalEmbeddingProvider(model="BAAI/bge-large-en-v1.5", device="mps")

    assert _FakeSentenceTransformer.instances == 2


def test_reset_clears_cache():
    from scripts.core.db import embedding_providers as ep
    from scripts.core.db.embedding_providers import LocalEmbeddingProvider

    LocalEmbeddingProvider(model="BAAI/bge-large-en-v1.5")
    assert _FakeSentenceTransformer.instances == 1

    ep.reset_local_model_cache()
    LocalEmbeddingProvider(model="BAAI/bge-large-en-v1.5")
    assert _FakeSentenceTransformer.instances == 2
