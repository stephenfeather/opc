"""Embedding provider implementations.

Concrete providers for embedding generation via external APIs and local models.
Each provider implements the EmbeddingProvider ABC from embedding_service.py.

Providers:
- OpenAIEmbeddingProvider: OpenAI text-embedding-3-small (1536 dim)
- VoyageEmbeddingProvider: Voyage AI models (512-1024 dim)
- LocalEmbeddingProvider: sentence-transformers local models
- OllamaEmbeddingProvider: Ollama server-based models
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

import httpx

from scripts.core.db.embedding_service import EmbeddingError, EmbeddingProvider

# ---------------------------------------------------------------------------
# Host validation
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTNAMES = {"localhost"}


def _is_loopback_address(hostname: str) -> bool:
    """Check if hostname is a loopback address using proper IP parsing."""
    import ipaddress

    if hostname in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_ollama_host(host: str) -> None:
    """Validate Ollama host URL for scheme and SSRF safety.

    Allows http:// and https:// schemes only. Non-loopback hosts
    over plain http:// are rejected to prevent accidental SSRF
    against internal services.

    Raises:
        ValueError: If scheme is invalid or non-loopback http target
    """
    from urllib.parse import urlparse

    parsed = urlparse(host)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Ollama host must use http:// or https:// scheme, got: {host}"
        )
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and not _is_loopback_address(hostname):
        raise ValueError(
            f"Ollama host over plain http:// is only allowed for loopback addresses. "
            f"Got: {host}. Use https:// for remote hosts."
        )


# ---------------------------------------------------------------------------
# API response validation
# ---------------------------------------------------------------------------


def _validate_embedding_response(
    data: object, expected_count: int, provider_name: str
) -> list[list[float]]:
    """Validate and extract embeddings from an API response.

    Checks that:
    - data is a dict with a "data" key containing a list
    - Each item has an integer "index" and a list "embedding"
    - Indices cover exactly 0..expected_count-1 with no duplicates

    Returns:
        Embeddings sorted by index

    Raises:
        EmbeddingError: If response is malformed
    """
    if not isinstance(data, dict) or "data" not in data:
        raise EmbeddingError(
            f"{provider_name} response missing 'data' key or not a dict"
        )
    items = data["data"]
    if not isinstance(items, list):
        raise EmbeddingError(
            f"{provider_name} response 'data' is not a list"
        )
    if len(items) != expected_count:
        raise EmbeddingError(
            f"{provider_name} returned {len(items)} items, expected {expected_count}"
        )
    seen_indices: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            raise EmbeddingError(
                f"{provider_name} response item is not a dict"
            )
        idx = item.get("index")
        if not isinstance(idx, int):
            raise EmbeddingError(
                f"{provider_name} response item missing integer 'index'"
            )
        if idx in seen_indices:
            raise EmbeddingError(
                f"{provider_name} response has duplicate index {idx}"
            )
        seen_indices.add(idx)
        emb = item.get("embedding")
        if not isinstance(emb, list):
            raise EmbeddingError(
                f"{provider_name} response item missing list 'embedding'"
            )
    expected_indices = set(range(expected_count))
    if seen_indices != expected_indices:
        raise EmbeddingError(
            f"{provider_name} response indices {sorted(seen_indices)} "
            f"do not match expected {sorted(expected_indices)}"
        )
    sorted_items = sorted(items, key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_items]


# Module-level crash log file handle, kept open for faulthandler's lifetime.
_crash_log_file = None


def _enable_faulthandler(faulthandler_mod) -> None:
    """Enable faulthandler for native crash dumps without leaking fds."""
    global _crash_log_file  # noqa: PLW0603
    if _crash_log_file is not None:
        return  # already enabled
    crash_log_path = os.path.expanduser("~/.claude/logs/opc_crash.log")
    os.makedirs(os.path.dirname(crash_log_path), exist_ok=True)
    fd = os.open(crash_log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    _crash_log_file = os.fdopen(fd, "a")
    faulthandler_mod.enable(file=_crash_log_file, all_threads=True)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small provider.

    Requires OPENAI_API_KEY environment variable.
    Dimension: 1536 (text-embedding-3-small)
    """

    DIMENSION = 1536
    MODEL = "text-embedding-3-small"
    API_URL = "https://api.openai.com/v1/embeddings"
    DEFAULT_MAX_BATCH_SIZE = 100
    DEFAULT_MAX_RETRIES = 3
    RETRY_DELAY = 0.5

    def __init__(
        self,
        api_key: str | None = None,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenAIEmbeddingProvider:
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    async def embed(self, text: str, **kwargs) -> list[float]:
        embeddings = await self._call_api([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        if not texts:
            return []
        from scripts.core.db.embedding_service import chunk_texts

        all_embeddings: list[list[float]] = []
        for chunk in chunk_texts(texts, max_size=self.max_batch_size):
            chunk_embeddings = await self._call_api(chunk)
            all_embeddings.extend(chunk_embeddings)
        return all_embeddings

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    self.API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.MODEL,
                        "input": texts if len(texts) > 1 else texts[0],
                    },
                )
                response.raise_for_status()
                data = response.json()
                return _validate_embedding_response(data, len(texts), "OpenAI")
            except EmbeddingError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
        status = ""
        if isinstance(last_error, httpx.HTTPStatusError):
            status = f" (HTTP {last_error.response.status_code})"
        raise EmbeddingError(
            f"OpenAI API call failed after {self.max_retries} attempts: "
            f"{type(last_error).__name__}{status}"
        ) from None

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def model_label(self) -> str:
        return self.MODEL


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI embedding provider.

    Supports: voyage-3 (1024), voyage-3-large (1024), voyage-code-3 (1024),
    voyage-3-lite (512). Requires VOYAGE_API_KEY environment variable.
    """

    MODELS = {
        "voyage-3": 1024,
        "voyage-3-large": 1024,
        "voyage-code-3": 1024,
        "voyage-3-lite": 512,
    }
    API_URL = "https://api.voyageai.com/v1/embeddings"
    DEFAULT_MAX_BATCH_SIZE = 128
    DEFAULT_MAX_RETRIES = 3
    RETRY_DELAY = 0.5

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        if model not in self.MODELS:
            raise ValueError(
                f"Unknown Voyage model: {model}. Available: {list(self.MODELS.keys())}"
            )
        self.model = model
        self._dimension = self.MODELS[model]
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY environment variable required")
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> VoyageEmbeddingProvider:
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    async def embed(self, text: str, input_type: str = "document", **kwargs) -> list[float]:
        embeddings = await self._call_api([text], input_type=input_type)
        return embeddings[0]

    async def embed_batch(
        self,
        texts: list[str],
        input_type: str = "document",
        **kwargs,
    ) -> list[list[float]]:
        if not texts:
            return []
        from scripts.core.db.embedding_service import chunk_texts

        all_embeddings: list[list[float]] = []
        for chunk in chunk_texts(texts, max_size=self.max_batch_size):
            chunk_embeddings = await self._call_api(chunk, input_type=input_type)
            all_embeddings.extend(chunk_embeddings)
        return all_embeddings

    async def _call_api(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": texts,
                        "input_type": input_type,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return _validate_embedding_response(data, len(texts), "Voyage")
            except EmbeddingError:
                raise
            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
        status = ""
        if isinstance(last_error, httpx.HTTPStatusError):
            status = f" (HTTP {last_error.response.status_code})"
        raise EmbeddingError(
            f"Voyage API call failed after {self.max_retries} attempts: "
            f"{type(last_error).__name__}{status}"
        ) from None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_label(self) -> str:
        # Honors --model overrides (voyage-3, voyage-3-large, voyage-code-3).
        return self.model


# ---------------------------------------------------------------------------
# Local model cache (issue #152)
# ---------------------------------------------------------------------------

# Process-level cache of loaded sentence-transformers models, keyed by
# (model, device). Loading the BGE model is ~14s cold; before #152 it ran on
# every EmbeddingService(provider="local"), so each hybrid recall paid it.
# Caching the loaded object means the cost is paid at most once per process --
# which also makes recall's deadline-bounded off-thread construction pay off:
# an abandoned cold load still populates this cache, so the next recall
# constructs instantly and succeeds within QUERY_EMBED_TIMEOUT.
_LOCAL_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}
# Serialises concurrent cold loads (avoids a thundering-herd double-load) and
# safely publishes the model to other threads -- recall constructs the local
# provider on a worker thread (#152). The fast-path read below is lock-free
# (dict.get is atomic under the GIL), so a warm load never blocks behind an
# in-progress cold load.
_LOCAL_MODEL_CACHE_LOCK = threading.Lock()


def reset_local_model_cache() -> None:
    """Drop all cached local models (test isolation; issue #152)."""
    with _LOCAL_MODEL_CACHE_LOCK:
        _LOCAL_MODEL_CACHE.clear()


def _load_sentence_transformer(model: str, device: str | None) -> Any:
    """Load (or return a process-cached) SentenceTransformer (issue #152).

    The heavy load runs at most once per (model, device). Native stdout/stderr
    from the loader is redirected to /dev/null and noisy library loggers are
    quieted for the duration of the load only. Raises ImportError with install
    guidance when sentence-transformers is absent.
    """
    key = (model, device)
    # Lock-free fast path: an already-loaded model is returned without taking
    # the lock, so it never blocks behind another thread's in-flight cold load.
    cached = _LOCAL_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required for local embeddings. "
            "Install with: pip install sentence-transformers torch"
        )

    import faulthandler
    import logging as _logging

    _enable_faulthandler(faulthandler)

    with _LOCAL_MODEL_CACHE_LOCK:
        # Re-check under the lock: another thread may have loaded it while we
        # waited (double-checked locking).
        cached = _LOCAL_MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        # IMPORTANT (#152 round 1): do NOT redirect file descriptors 1/2 here.
        # This load now runs on a daemon worker thread that can outlive the
        # recall caller (deadline-bounded construction in recall_backends).
        # ``os.dup2`` on fds 1/2 is process-global, so suppressing native
        # loader output that way would also swallow the MAIN thread's degraded
        # recall warning and the CLI's result print for the entire ~14s cold
        # load — failing the fix in exactly the degraded case it exists to
        # serve. Quiet only at the Python logging / tqdm level, which filters
        # by logger and cannot hijack the parent process's stdout/stderr.
        loggers_to_quiet = [
            "sentence_transformers",
            "transformers",
            "safetensors",
            "torch",
        ]
        prev_levels = {name: _logging.getLogger(name).level for name in loggers_to_quiet}
        for name in loggers_to_quiet:
            _logging.getLogger(name).setLevel(_logging.ERROR)
        prev_env = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            loaded = SentenceTransformer(model, device=device)
        finally:
            if prev_env is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev_env
            for name in loggers_to_quiet:
                _logging.getLogger(name).setLevel(prev_levels[name])

        _LOCAL_MODEL_CACHE[key] = loaded
        return loaded


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.

    Supported models:
    - BAAI/bge-large-en-v1.5: 1024 dim (default, matches Voyage)
    - BAAI/bge-base-en-v1.5: 768 dim
    - all-MiniLM-L6-v2: 384 dim
    - all-mpnet-base-v2: 768 dim

    Requires: pip install sentence-transformers torch
    """

    MODELS = {
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
    }

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        device: str | None = None,
    ):
        self.model_name = model
        # Process-cached load (issue #152): the ~14s cold model load happens at
        # most once per (model, device) for the whole process.
        self._model = _load_sentence_transformer(model, device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str, **kwargs) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._model.encode(text, convert_to_numpy=True).tolist()
        )

    async def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._model.encode(texts, convert_to_numpy=True).tolist()
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_label(self) -> str:
        # The legacy column default and the live corpus label BGE rows 'bge'
        # (issue #151). Map every BGE variant to that canonical label; other
        # local models fall back to their full model name.
        if "bge" in self.model_name.lower():
            return "bge"
        return self.model_name


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider using local or remote Ollama server.

    Environment variables:
    - OLLAMA_HOST: Server URL (default: http://localhost:11434)
    - OLLAMA_EMBED_MODEL: Model name (default: nomic-embed-text)
    """

    MODELS = {
        "nomic-embed-text": 768,
        "nomic-embed-text-v1.5": 768,
        "nomic-embed-text-v2-moe": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        verify_tls: bool = True,
    ):
        # Defaults from opc.toml [embedding], resolved at instantiation time
        from scripts.core.config import get_config
        _emb_cfg = get_config().embedding

        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", _emb_cfg.ollama_model)
        self.host = host or os.getenv("OLLAMA_HOST", _emb_cfg.ollama_host)
        self.verify_tls = verify_tls
        _validate_ollama_host(self.host)
        self._client = httpx.AsyncClient(timeout=30.0, verify=verify_tls)
        self._dimension = self.MODELS.get(self.model, 768)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OllamaEmbeddingProvider:
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    async def embed(self, text: str, **kwargs) -> list[float]:
        url = f"{self.host.rstrip('/')}/api/embeddings"
        response = await self._client.post(
            url, json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_label(self) -> str:
        return self.model
