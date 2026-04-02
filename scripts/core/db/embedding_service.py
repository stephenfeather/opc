"""Embedding generation service.

Provides embedding generation for archival memory with:
- Multiple provider support (OpenAI, Voyage, Local, Ollama, Mock)
- Content-based caching to avoid duplicate API calls
- Batch processing for efficiency
- Configurable dimensions

Pure functions:
- cache_key(text) -> str: SHA-256 hash for cache lookup
- generate_mock_embedding(text, dimension) -> list[float]: Deterministic embeddings
- chunk_texts(texts, max_size) -> Iterator: Split texts into batches
- create_provider(name, **kwargs) -> EmbeddingProvider: Factory function

Usage:
    service = EmbeddingService(provider="openai")
    embedding = await service.embed("Some text")
    embeddings = await service.embed_batch(["Text 1", "Text 2"])
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def cache_key(text: str) -> str:
    """Generate cache key from text content hash.

    Args:
        text: Text to hash

    Returns:
        SHA-256 hex digest of text
    """
    return hashlib.sha256(text.encode()).hexdigest()


def generate_mock_embedding(text: str, dimension: int) -> list[float]:
    """Generate a deterministic embedding from text hash.

    Same text + dimension always produces the same embedding.
    Values are normalized to [-1, 1] range.

    Args:
        text: Text to embed
        dimension: Number of dimensions in the output vector

    Returns:
        Deterministic embedding vector
    """
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    return [
        ((int(text_hash[(i % 32) * 2 : (i % 32) * 2 + 2], 16) + i) % 256) / 255.0 * 2 - 1
        for i in range(dimension)
    ]


def chunk_texts(texts: list[str], max_size: int) -> Iterator[list[str]]:
    """Split a list of texts into chunks of at most max_size.

    Args:
        texts: Texts to chunk
        max_size: Maximum number of texts per chunk

    Yields:
        Lists of texts, each with at most max_size elements
    """
    for i in range(0, len(texts), max_size):
        yield texts[i : i + max_size]


# ---------------------------------------------------------------------------
# Abstract base / protocol
# ---------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Error raised when embedding generation fails."""

    pass


class EmbeddingProvider(ABC):
    """Abstract embedding provider protocol.

    All embedding providers must implement:
    - embed(text) -> list[float]: Generate embedding for text
    - embed_batch(texts) -> list[list[float]]: Batch embedding
    - dimension: int: Embedding dimension
    """

    @abstractmethod
    async def embed(self, text: str, **kwargs) -> list[float]:
        """Generate embedding for text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


# ---------------------------------------------------------------------------
# Mock provider (kept here — no external dependencies, useful for testing)
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing.

    Generates deterministic embeddings based on text hash.
    No API calls required.
    """

    DEFAULT_DIMENSION = 1536

    def __init__(self, dimension: int = DEFAULT_DIMENSION):
        self._dimension = dimension

    async def embed(self, text: str, **kwargs) -> list[float]:
        return generate_mock_embedding(text, self._dimension)

    async def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        return [generate_mock_embedding(t, self._dimension) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def create_provider(
    name: str,
    *,
    dimension: int | None = None,
    max_batch_size: int = 100,
    max_retries: int = 3,
    model: str | None = None,
    **kwargs,
) -> EmbeddingProvider:
    """Create an embedding provider by name.

    Args:
        name: Provider name ("openai", "mock", "voyage", "local", "ollama",
              or a model shorthand like "voyage-3")
        dimension: Override dimension (only for mock provider)
        max_batch_size: Maximum texts per API call
        max_retries: Maximum retry attempts
        model: Model name for provider
        **kwargs: Additional provider-specific arguments (e.g., device, host)

    Returns:
        An EmbeddingProvider instance

    Raises:
        ValueError: If provider name is unknown or required config is missing
    """
    import os

    from scripts.core.db.embedding_providers import (
        LocalEmbeddingProvider,
        OllamaEmbeddingProvider,
        OpenAIEmbeddingProvider,
        VoyageEmbeddingProvider,
    )

    if name == "mock":
        dim = dimension if dimension is not None else MockEmbeddingProvider.DEFAULT_DIMENSION
        return MockEmbeddingProvider(dimension=dim)

    if name == "openai":
        return OpenAIEmbeddingProvider(
            max_batch_size=max_batch_size,
            max_retries=max_retries,
        )

    if name == "voyage":
        voyage_model = (
            model if model is not None
            else os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3")
        )
        return VoyageEmbeddingProvider(
            model=voyage_model,
            max_batch_size=max_batch_size,
            max_retries=max_retries,
        )

    if name.startswith("voyage-"):
        return VoyageEmbeddingProvider(
            model=name,
            max_batch_size=max_batch_size,
            max_retries=max_retries,
        )

    if name == "local":
        local_model = model if model is not None else "BAAI/bge-large-en-v1.5"
        device = kwargs.get("device", None)
        return LocalEmbeddingProvider(model=local_model, device=device)

    if name == "ollama":
        ollama_model = model if model is not None else None
        ollama_host = kwargs.get("host", None)
        return OllamaEmbeddingProvider(model=ollama_model, host=ollama_host)

    raise ValueError(f"Unknown provider: {name}")


# ---------------------------------------------------------------------------
# Embedding service (orchestrator with caching)
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Embedding service with caching and provider abstraction.

    Main entry point for generating embeddings. Supports:
    - Multiple providers (OpenAI, Voyage, Local, Ollama, Mock)
    - Content-based caching
    - Batch processing with cache-aware splitting

    Implements EmbeddingProvider-compatible interface for use with MemoryServicePG.

    Usage:
        service = EmbeddingService(provider="openai")
        embedding = await service.embed("Some text")
    """

    def __init__(
        self,
        provider: str = "openai",
        cache_enabled: bool = True,
        dimension: int | None = None,
        max_batch_size: int = 100,
        max_retries: int = 3,
        model: str | None = None,
        **kwargs,
    ):
        self.cache_enabled = cache_enabled
        self._cache: dict[str, list[float]] = {}
        self._cache_lock = asyncio.Lock()
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self._provider = create_provider(
            name=provider,
            dimension=dimension,
            max_batch_size=max_batch_size,
            max_retries=max_retries,
            model=model,
            **kwargs,
        )

    async def embed(self, text: str, **kwargs) -> list[float]:
        """Generate embedding for text with optional caching."""
        if self.cache_enabled:
            key = cache_key(text)
            async with self._cache_lock:
                if key in self._cache:
                    return self._cache[key]

        embedding = await self._provider.embed(text, **kwargs)

        if self.cache_enabled:
            async with self._cache_lock:
                self._cache[cache_key(text)] = embedding

        return embedding

    async def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings for multiple texts with cache-aware batching."""
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []

        async with self._cache_lock:
            for i, text in enumerate(texts):
                if self.cache_enabled:
                    key = cache_key(text)
                    if key in self._cache:
                        results[i] = self._cache[key]
                        continue
                texts_to_embed.append((i, text))

        if texts_to_embed:
            indices, remaining_texts = zip(*texts_to_embed)
            remaining_list = list(remaining_texts)
            new_embeddings = await self._provider.embed_batch(remaining_list, **kwargs)
            if len(new_embeddings) != len(remaining_list):
                raise EmbeddingError(
                    f"Provider returned {len(new_embeddings)} embeddings "
                    f"for {len(remaining_list)} texts"
                )
            async with self._cache_lock:
                for idx, text, embedding in zip(indices, remaining_list, new_embeddings):
                    results[idx] = embedding
                    if self.cache_enabled:
                        self._cache[cache_key(text)] = embedding

        return [r for r in results if r is not None]

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)

    async def aclose(self) -> None:
        if hasattr(self._provider, "aclose"):
            await self._provider.aclose()

    async def __aenter__(self) -> EmbeddingService:
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------
# Backwards-compatible lazy re-exports from embedding_providers
# ---------------------------------------------------------------------------
# These ensure existing imports like:
#   from scripts.core.db.embedding_service import VoyageEmbeddingProvider
# continue to work without creating a circular import at module load time.

_PROVIDER_REEXPORTS = {
    "OpenAIEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "LocalEmbeddingProvider",
    "OllamaEmbeddingProvider",
}


def __getattr__(name: str):  # noqa: N807
    if name in _PROVIDER_REEXPORTS:
        from scripts.core.db import embedding_providers

        return getattr(embedding_providers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [  # noqa: F822
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingService",
    "MockEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "LocalEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "cache_key",
    "generate_mock_embedding",
    "chunk_texts",
    "create_provider",
]
