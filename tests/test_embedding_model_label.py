"""Tests for issue #151: embedding-space model_label property.

Each provider must expose a stable ``model_label`` matching the value written
to ``archival_memory.embedding_model`` so query-time SQL can filter to a single
embedding space (canonical = voyage-code-3).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestProviderModelLabel:
    """Each provider exposes the DB label for its embedding space."""

    def test_mock_provider_label(self):
        from scripts.core.db.embedding_service import MockEmbeddingProvider

        assert MockEmbeddingProvider().model_label == "mock"

    def test_local_provider_label_is_bge(self):
        # LocalEmbeddingProvider historically wrote the column default 'bge'
        # via .model_name; model_label must canonicalize that to 'bge'.
        from scripts.core.db.embedding_providers import LocalEmbeddingProvider

        # Build without loading sentence-transformers: bypass __init__.
        provider = LocalEmbeddingProvider.__new__(LocalEmbeddingProvider)
        provider.model_name = "BAAI/bge-large-en-v1.5"
        assert provider.model_label == "bge"

    def test_local_provider_label_other_bge_variant(self):
        from scripts.core.db.embedding_providers import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider.__new__(LocalEmbeddingProvider)
        provider.model_name = "BAAI/bge-base-en-v1.5"
        assert provider.model_label == "bge"

    def test_local_provider_label_non_bge_falls_back_to_model_name(self):
        from scripts.core.db.embedding_providers import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider.__new__(LocalEmbeddingProvider)
        provider.model_name = "all-MiniLM-L6-v2"
        assert provider.model_label == "all-MiniLM-L6-v2"

    def test_voyage_provider_label_is_model_name(self):
        from scripts.core.db.embedding_providers import VoyageEmbeddingProvider

        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            provider = VoyageEmbeddingProvider(model="voyage-code-3")
        assert provider.model_label == "voyage-code-3"

    def test_voyage_provider_label_honors_model_override(self):
        from scripts.core.db.embedding_providers import VoyageEmbeddingProvider

        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            provider = VoyageEmbeddingProvider(model="voyage-3-large")
        assert provider.model_label == "voyage-3-large"

    def test_openai_provider_label_is_model_constant(self):
        from scripts.core.db.embedding_providers import OpenAIEmbeddingProvider

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddingProvider()
        assert provider.model_label == "text-embedding-3-small"

    def test_ollama_provider_label_is_model_name(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(model="nomic-embed-text")
        assert provider.model_label == "nomic-embed-text"

    def test_provider_abc_declares_model_label(self):
        from scripts.core.db.embedding_service import EmbeddingProvider

        assert hasattr(EmbeddingProvider, "model_label")


class TestEmbeddingServiceModelLabel:
    """EmbeddingService surfaces the underlying provider's model_label."""

    def test_service_exposes_provider_label(self):
        from scripts.core.db.embedding_service import EmbeddingService

        service = EmbeddingService(provider="mock")
        assert service.model_label == "mock"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
