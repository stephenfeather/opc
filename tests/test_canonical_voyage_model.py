"""Round 3 FIX 6: one canonical voyage model across the codebase (issue #151).

The model-filtered recall corpus, re_embed_voyage.py's target, and the default
voyage provider must all agree. A default voyage query that binds 'voyage-3'
would match zero rows against a 'voyage-code-3' corpus and new stores would
mint a third space. These tests pin a single source of truth.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

CANONICAL = "voyage-code-3"


class TestCanonicalConstant:
    def test_config_default_is_canonical(self):
        from scripts.core.config.models import EmbeddingConfig

        assert EmbeddingConfig().voyage_model == CANONICAL

    def test_re_embed_target_matches_config_default(self):
        from scripts.core.config.models import EmbeddingConfig
        from scripts.core.re_embed_voyage import TARGET_MODEL

        assert TARGET_MODEL == EmbeddingConfig().voyage_model


class TestDefaultVoyageProviderLabel:
    def test_default_voyage_provider_label_is_canonical(self):
        """EmbeddingService(provider='voyage') with no model arg and no env
        override must produce the canonical label — the same one
        re_embed_voyage.py writes."""
        from scripts.core.config import reset_config
        from scripts.core.config.models import EmbeddingConfig
        from scripts.core.db.embedding_service import EmbeddingService

        # Ensure no env override leaks in from the test environment.
        env = dict(os.environ)
        env.pop("VOYAGE_EMBEDDING_MODEL", None)
        env["VOYAGE_API_KEY"] = "test-key"
        with patch.dict(os.environ, env, clear=True):
            reset_config()
            try:
                svc = EmbeddingService(provider="voyage")
            finally:
                reset_config()
        assert svc.model_label == EmbeddingConfig().voyage_model
        assert svc.model_label == CANONICAL

    def test_env_override_still_respected(self):
        from scripts.core.config import reset_config
        from scripts.core.db.embedding_service import EmbeddingService

        env = dict(os.environ)
        env["VOYAGE_EMBEDDING_MODEL"] = "voyage-3-large"
        env["VOYAGE_API_KEY"] = "test-key"
        with patch.dict(os.environ, env, clear=True):
            # get_config() caches; clear it so the env override is re-read.
            reset_config()
            try:
                svc = EmbeddingService(provider="voyage")
            finally:
                reset_config()  # restore clean cache for other tests
        assert svc.model_label == "voyage-3-large"

    def test_explicit_model_arg_wins(self):
        from scripts.core.db.embedding_service import EmbeddingService

        env = dict(os.environ)
        env["VOYAGE_API_KEY"] = "test-key"
        with patch.dict(os.environ, env, clear=True):
            svc = EmbeddingService(provider="voyage", model="voyage-3")
        assert svc.model_label == "voyage-3"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
