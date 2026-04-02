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

import httpx

from scripts.core.db.embedding_service import EmbeddingError, EmbeddingProvider

# Module-level crash log file handle, kept open for faulthandler's lifetime.
_crash_log_file = None


def _enable_faulthandler(faulthandler_mod) -> None:
    """Enable faulthandler for native crash dumps without leaking fds."""
    global _crash_log_file  # noqa: PLW0603
    if _crash_log_file is not None:
        return  # already enabled
    crash_log_path = os.path.expanduser("~/.claude/logs/opc_crash.log")
    os.makedirs(os.path.dirname(crash_log_path), exist_ok=True)
    _crash_log_file = open(crash_log_path, "a")  # noqa: SIM115
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
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in sorted_data]
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
        raise EmbeddingError(
            f"API call failed after {self.max_retries} attempts: {last_error}"
        )

    @property
    def dimension(self) -> int:
        return self.DIMENSION


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
        last_response_text: str | None = None
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
                last_response_text = response.text
                response.raise_for_status()
                data = response.json()
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in sorted_data]
            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
        error_msg = f"Voyage API call failed after {self.max_retries} attempts.\n"
        error_msg += f"Last error: {type(last_error).__name__}: {last_error}\n"
        if last_response_text:
            error_msg += f"Response body: {last_response_text[:500]}"
        raise EmbeddingError(error_msg)

    @property
    def dimension(self) -> int:
        return self._dimension


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

        self.model_name = model
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
        devnull_fd = -1
        old_stdout_fd = -1
        old_stderr_fd = -1
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            self._model = SentenceTransformer(model, device=device)
        finally:
            if old_stderr_fd >= 0:
                os.dup2(old_stderr_fd, 2)
                os.close(old_stderr_fd)
            if old_stdout_fd >= 0:
                os.dup2(old_stdout_fd, 1)
                os.close(old_stdout_fd)
            if devnull_fd >= 0:
                os.close(devnull_fd)
            if prev_env is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev_env
            for name in loggers_to_quiet:
                _logging.getLogger(name).setLevel(prev_levels[name])
        self._dimension = self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str, **kwargs) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._model.encode(text, convert_to_numpy=True).tolist()
        )

    async def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._model.encode(texts, convert_to_numpy=True).tolist()
        )

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider using local or remote Ollama server.

    Environment variables:
    - OLLAMA_HOST: Server URL (default: http://localhost:11434)
    - OLLAMA_EMBED_MODEL: Model name (default: nomic-embed-text)
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_HOST = "http://localhost:11434"

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
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", self.DEFAULT_MODEL)
        self.host = host or os.getenv("OLLAMA_HOST", self.DEFAULT_HOST)
        self._client = httpx.AsyncClient(timeout=30.0, verify=verify_tls)
        self._dimension = self.MODELS.get(self.model, 768)

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
