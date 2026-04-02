"""Tests for embedding service (TDD+FP refactor).

Tests cover:
- Pure functions: cache key generation, mock embedding generation, batch chunking
- Provider factory: creating providers from string names
- EmbeddingService: caching, batch with cache hits, delegation to provider
- EmbeddingProvider protocol compliance
- MockEmbeddingProvider determinism
- Error handling
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestCacheKey:
    """Test cache_key pure function."""

    def test_returns_sha256_hex(self):
        from scripts.core.db.embedding_service import cache_key

        result = cache_key("hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert result == expected

    def test_deterministic(self):
        from scripts.core.db.embedding_service import cache_key

        assert cache_key("test") == cache_key("test")

    def test_different_inputs_different_keys(self):
        from scripts.core.db.embedding_service import cache_key

        assert cache_key("a") != cache_key("b")

    def test_kwargs_affect_key(self):
        from scripts.core.db.embedding_service import cache_key

        key_doc = cache_key("text", input_type="document")
        key_query = cache_key("text", input_type="query")
        key_plain = cache_key("text")
        assert key_doc != key_query
        assert key_doc != key_plain

    def test_empty_string(self):
        from scripts.core.db.embedding_service import cache_key

        result = cache_key("")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected


class TestGenerateMockEmbedding:
    """Test generate_mock_embedding pure function."""

    def test_returns_list_of_floats(self):
        from scripts.core.db.embedding_service import generate_mock_embedding

        result = generate_mock_embedding("hello", dimension=10)
        assert isinstance(result, list)
        assert len(result) == 10
        assert all(isinstance(v, float) for v in result)

    def test_deterministic(self):
        from scripts.core.db.embedding_service import generate_mock_embedding

        a = generate_mock_embedding("test", dimension=100)
        b = generate_mock_embedding("test", dimension=100)
        assert a == b

    def test_different_texts_different_embeddings(self):
        from scripts.core.db.embedding_service import generate_mock_embedding

        a = generate_mock_embedding("hello", dimension=100)
        b = generate_mock_embedding("world", dimension=100)
        assert a != b

    def test_values_in_range(self):
        from scripts.core.db.embedding_service import generate_mock_embedding

        result = generate_mock_embedding("test", dimension=1536)
        assert all(-1.0 <= v <= 1.0 for v in result)

    def test_zero_dimension(self):
        from scripts.core.db.embedding_service import generate_mock_embedding

        result = generate_mock_embedding("test", dimension=0)
        assert result == []


class TestChunkTexts:
    """Test chunk_texts pure function for batch splitting."""

    def test_single_chunk(self):
        from scripts.core.db.embedding_service import chunk_texts

        texts = ["a", "b", "c"]
        result = list(chunk_texts(texts, max_size=10))
        assert result == [["a", "b", "c"]]

    def test_multiple_chunks(self):
        from scripts.core.db.embedding_service import chunk_texts

        texts = ["a", "b", "c", "d", "e"]
        result = list(chunk_texts(texts, max_size=2))
        assert result == [["a", "b"], ["c", "d"], ["e"]]

    def test_empty_list(self):
        from scripts.core.db.embedding_service import chunk_texts

        result = list(chunk_texts([], max_size=10))
        assert result == []

    def test_exact_chunk_size(self):
        from scripts.core.db.embedding_service import chunk_texts

        texts = ["a", "b", "c", "d"]
        result = list(chunk_texts(texts, max_size=2))
        assert result == [["a", "b"], ["c", "d"]]

    def test_zero_max_size_raises(self):
        from scripts.core.db.embedding_service import chunk_texts

        with pytest.raises(ValueError, match="max_size must be greater than 0"):
            list(chunk_texts(["a"], max_size=0))

    def test_negative_max_size_raises(self):
        from scripts.core.db.embedding_service import chunk_texts

        with pytest.raises(ValueError, match="max_size must be greater than 0"):
            list(chunk_texts(["a"], max_size=-1))


# ---------------------------------------------------------------------------
# Provider factory tests
# ---------------------------------------------------------------------------


class TestCreateProvider:
    """Test create_provider factory function."""

    def test_mock_provider(self):
        from scripts.core.db.embedding_service import create_provider

        provider = create_provider("mock")
        assert provider.dimension == 1536

    def test_mock_with_dimension(self):
        from scripts.core.db.embedding_service import create_provider

        provider = create_provider("mock", dimension=512)
        assert provider.dimension == 512

    def test_unknown_provider_returns_error(self):
        from scripts.core.db.embedding_service import create_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent")

    def test_openai_requires_api_key(self):
        from scripts.core.db.embedding_service import create_provider

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                create_provider("openai")

    def test_voyage_requires_api_key(self):
        from scripts.core.db.embedding_service import create_provider

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
                create_provider("voyage")

    def test_voyage_shorthand(self):
        """Provider name 'voyage-3' should create VoyageEmbeddingProvider."""
        from scripts.core.db.embedding_service import create_provider

        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            provider = create_provider("voyage-3")
            assert provider.dimension == 1024

    def test_openai_explicit_api_key(self):
        """Explicit api_key should be forwarded, not env var."""
        from scripts.core.db.embedding_service import create_provider

        with patch.dict("os.environ", {}, clear=True):
            provider = create_provider("openai", api_key="explicit-key")
            assert provider.api_key == "explicit-key"

    def test_voyage_explicit_api_key(self):
        """Explicit api_key should be forwarded to Voyage provider."""
        from scripts.core.db.embedding_service import create_provider

        with patch.dict("os.environ", {}, clear=True):
            provider = create_provider("voyage", api_key="explicit-key")
            assert provider.api_key == "explicit-key"

    def test_ollama_verify_tls_passthrough(self):
        """verify_tls should be forwarded to Ollama provider."""
        from scripts.core.db.embedding_service import create_provider

        provider = create_provider("ollama", verify_tls=False)
        assert provider._client is not None


# ---------------------------------------------------------------------------
# MockEmbeddingProvider tests
# ---------------------------------------------------------------------------


class TestMockEmbeddingProvider:
    """Test MockEmbeddingProvider behavior."""

    async def test_embed_returns_correct_dimension(self):
        from scripts.core.db.embedding_service import MockEmbeddingProvider

        provider = MockEmbeddingProvider(dimension=128)
        result = await provider.embed("test")
        assert len(result) == 128

    async def test_embed_deterministic(self):
        from scripts.core.db.embedding_service import MockEmbeddingProvider

        provider = MockEmbeddingProvider()
        a = await provider.embed("hello")
        b = await provider.embed("hello")
        assert a == b

    async def test_embed_batch_empty(self):
        from scripts.core.db.embedding_service import MockEmbeddingProvider

        provider = MockEmbeddingProvider()
        result = await provider.embed_batch([])
        assert result == []

    async def test_embed_batch_preserves_order(self):
        from scripts.core.db.embedding_service import MockEmbeddingProvider

        provider = MockEmbeddingProvider()
        texts = ["alpha", "beta", "gamma"]
        batch_results = await provider.embed_batch(texts)
        individual_results = [await provider.embed(t) for t in texts]
        assert batch_results == individual_results


# ---------------------------------------------------------------------------
# EmbeddingService tests
# ---------------------------------------------------------------------------


class TestEmbeddingService:
    """Test EmbeddingService caching and delegation."""

    async def test_embed_delegates_to_provider(self):
        from scripts.core.db.embedding_service import EmbeddingService

        service = EmbeddingService(provider="mock")
        result = await service.embed("test")
        assert isinstance(result, list)
        assert len(result) == 1536

    async def test_embed_caching(self):
        from scripts.core.db.embedding_service import EmbeddingService

        service = EmbeddingService(provider="mock", cache_enabled=True)
        a = await service.embed("cached text")
        b = await service.embed("cached text")
        assert a == b
        assert service.cache_size() == 1

    async def test_embed_no_cache(self):
        from scripts.core.db.embedding_service import EmbeddingService

        service = EmbeddingService(provider="mock", cache_enabled=False)
        await service.embed("text")
        assert service.cache_size() == 0

    async def test_embed_batch_empty(self):
        from scripts.core.db.embedding_service import EmbeddingService

        service = EmbeddingService(provider="mock")
        result = await service.embed_batch([])
        assert result == []

    async def test_embed_batch_uses_cache(self):
        """Texts already cached should not be re-embedded."""
        from scripts.core.db.embedding_service import EmbeddingService

        service = EmbeddingService(provider="mock", cache_enabled=True)
        # Pre-cache one text
        await service.embed("alpha")
        assert service.cache_size() == 1

        # Batch with one cached + one new
        results = await service.embed_batch(["alpha", "beta"])
        assert len(results) == 2
        assert service.cache_size() == 2

    async def test_cache_isolates_by_kwargs(self):
        """Different input_type kwargs should produce separate cache entries."""
        from scripts.core.db.embedding_service import EmbeddingService

        service = EmbeddingService(provider="mock", cache_enabled=True)
        doc = await service.embed("same text", input_type="document")
        await service.embed("same text", input_type="query")
        # Mock provider ignores input_type, but cache keys should differ
        assert service.cache_size() == 2
        # Verify batch also respects kwargs
        batch = await service.embed_batch(["same text"], input_type="document")
        assert service.cache_size() == 2  # hit existing cache
        assert batch[0] == doc

    async def test_clear_cache(self):
        from scripts.core.db.embedding_service import EmbeddingService

        service = EmbeddingService(provider="mock", cache_enabled=True)
        await service.embed("text")
        assert service.cache_size() == 1
        service.clear_cache()
        assert service.cache_size() == 0

    async def test_dimension_delegates_to_provider(self):
        from scripts.core.db.embedding_service import EmbeddingService

        service = EmbeddingService(provider="mock", dimension=256)
        assert service.dimension == 256

    async def test_context_manager(self):
        from scripts.core.db.embedding_service import EmbeddingService

        async with EmbeddingService(provider="mock") as service:
            result = await service.embed("test")
            assert len(result) == 1536

    async def test_embed_batch_cardinality_mismatch_raises(self):
        """Provider returning wrong number of embeddings should raise."""
        from scripts.core.db.embedding_service import EmbeddingError, EmbeddingService

        service = EmbeddingService(provider="mock", cache_enabled=False)
        # Monkey-patch provider to return fewer embeddings than requested
        original = service._provider.embed_batch

        async def short_batch(texts, **kwargs):
            result = await original(texts, **kwargs)
            return result[:-1]  # Drop last embedding

        service._provider.embed_batch = short_batch
        with pytest.raises(EmbeddingError, match="returned 1 embeddings for 2 texts"):
            await service.embed_batch(["a", "b"])


# ---------------------------------------------------------------------------
# Backwards compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    """Ensure all public names remain importable from embedding_service."""

    def test_import_embedding_error(self):
        from scripts.core.db.embedding_service import EmbeddingError

        assert issubclass(EmbeddingError, Exception)

    def test_import_embedding_provider(self):
        from scripts.core.db.embedding_service import EmbeddingProvider

        assert EmbeddingProvider is not None

    def test_import_embedding_service(self):
        from scripts.core.db.embedding_service import EmbeddingService

        assert EmbeddingService is not None

    def test_import_mock_provider(self):
        from scripts.core.db.embedding_service import MockEmbeddingProvider

        assert MockEmbeddingProvider is not None

    def test_import_openai_provider(self):
        from scripts.core.db.embedding_service import OpenAIEmbeddingProvider

        assert OpenAIEmbeddingProvider is not None

    def test_import_voyage_provider(self):
        from scripts.core.db.embedding_service import VoyageEmbeddingProvider

        assert VoyageEmbeddingProvider is not None

    def test_import_local_provider(self):
        from scripts.core.db.embedding_service import LocalEmbeddingProvider

        assert LocalEmbeddingProvider is not None

    def test_import_ollama_provider(self):
        from scripts.core.db.embedding_service import OllamaEmbeddingProvider

        assert OllamaEmbeddingProvider is not None


# ---------------------------------------------------------------------------
# Provider implementations (from embedding_providers module)
# ---------------------------------------------------------------------------


class TestEmbeddingProviders:
    """Test provider implementations in embedding_providers module."""

    def test_openai_provider_init_with_key(self):
        from scripts.core.db.embedding_providers import OpenAIEmbeddingProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddingProvider()
            assert provider.dimension == 1536

    def test_openai_provider_missing_key(self):
        from scripts.core.db.embedding_providers import OpenAIEmbeddingProvider

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIEmbeddingProvider()

    def test_voyage_provider_valid_model(self):
        from scripts.core.db.embedding_providers import VoyageEmbeddingProvider

        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            provider = VoyageEmbeddingProvider(model="voyage-3")
            assert provider.dimension == 1024

    def test_voyage_provider_invalid_model(self):
        from scripts.core.db.embedding_providers import VoyageEmbeddingProvider

        with pytest.raises(ValueError, match="Unknown Voyage model"):
            VoyageEmbeddingProvider(model="invalid-model")

    def test_voyage_provider_missing_key(self):
        from scripts.core.db.embedding_providers import VoyageEmbeddingProvider

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
                VoyageEmbeddingProvider(model="voyage-3")

    def test_ollama_provider_defaults(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider()
        assert provider.model == "nomic-embed-text"
        assert provider.dimension == 768

    def test_ollama_provider_custom_model(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(model="mxbai-embed-large")
        assert provider.dimension == 1024

    def test_ollama_tls_enabled_by_default(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider()
        assert provider.verify_tls is True

    def test_ollama_tls_can_be_disabled(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(verify_tls=False)
        # When verify=False, httpx still creates the client successfully
        assert provider._client is not None

    def test_ollama_rejects_invalid_scheme(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        with pytest.raises(ValueError, match="http:// or https://"):
            OllamaEmbeddingProvider(host="ftp://evil.example.com")

    def test_ollama_rejects_no_scheme(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        with pytest.raises(ValueError, match="http:// or https://"):
            OllamaEmbeddingProvider(host="169.254.169.254")

    def test_ollama_rejects_http_non_loopback(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        with pytest.raises(ValueError, match="only allowed for loopback"):
            OllamaEmbeddingProvider(host="http://192.168.1.100:11434")

    def test_ollama_allows_http_localhost(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(host="http://localhost:11434")
        assert provider.host == "http://localhost:11434"

    def test_ollama_allows_https_remote(self):
        from scripts.core.db.embedding_providers import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(host="https://remote.example.com:11434")
        assert provider.host == "https://remote.example.com:11434"


# ---------------------------------------------------------------------------
# EmbeddingError tests
# ---------------------------------------------------------------------------


class TestEmbeddingError:
    """Test EmbeddingError exception."""

    def test_is_exception(self):
        from scripts.core.db.embedding_service import EmbeddingError

        assert issubclass(EmbeddingError, Exception)

    def test_message(self):
        from scripts.core.db.embedding_service import EmbeddingError

        err = EmbeddingError("test error")
        assert str(err) == "test error"
