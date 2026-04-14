"""Tests for config models (frozen dataclasses with defaults)."""

from scripts.core.config.models import (
    ArchivalConfig,
    DatabaseConfig,
    DaemonConfig,
    DedupConfig,
    EmbeddingConfig,
    OPCConfig,
    PatternsConfig,
    QueryExpansionConfig,
    RerankerConfig,
    RecallConfig,
)


class TestDedupConfig:
    def test_default_threshold(self):
        cfg = DedupConfig()
        assert cfg.threshold == 0.85

    def test_custom_threshold(self):
        cfg = DedupConfig(threshold=0.90)
        assert cfg.threshold == 0.90

    def test_frozen(self):
        cfg = DedupConfig()
        try:
            cfg.threshold = 0.50  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass


class TestDaemonConfig:
    def test_defaults(self):
        cfg = DaemonConfig()
        assert cfg.poll_interval == 60
        assert cfg.stale_threshold == 900
        assert cfg.max_concurrent_extractions == 4
        assert cfg.max_retries == 5
        assert cfg.extraction_timeout == 1800
        assert cfg.harvest_grace_period == 300
        assert cfg.pattern_detection_interval_hours == 6
        assert cfg.extraction_model == "sonnet"
        assert cfg.extraction_max_turns == 15
        assert cfg.log_rotation_days == 7
        assert cfg.log_backup_count == 4


class TestRerankerConfig:
    def test_defaults(self):
        cfg = RerankerConfig()
        assert cfg.project_weight == 0.15
        assert cfg.recency_half_life_days == 45.0
        assert cfg.rrf_scale_factor == 60.0

    def test_total_signal_weight(self):
        cfg = RerankerConfig()
        expected = 0.15 + 0.05 * 7  # project + 7 others (incl. kg_weight) at 0.05
        assert abs(cfg.total_signal_weight - expected) < 1e-9

    def test_rejects_overweight_config(self):
        """Finding F2 fix: operators upgrading from pre-Phase-3 may have an
        opc.toml whose pre-existing weights already sum to 1.0. Adding the
        new default kg_weight=0.05 would push total above 1.0 and flip
        retrieval_weight negative. __post_init__ must reject that config."""
        import pytest
        with pytest.raises(ValueError, match="sum to <= 1.0"):
            # Pre-Phase-3 weights summed to 1.0 plus new kg_weight=0.05.
            RerankerConfig(
                project_weight=0.15,
                recency_weight=0.10,
                confidence_weight=0.10,
                recall_weight=0.10,
                type_affinity_weight=0.15,
                tag_overlap_weight=0.20,
                pattern_weight=0.20,
                # kg_weight defaults to 0.05 -> total = 1.05.
            )

    def test_rejects_negative_total(self):
        import pytest
        with pytest.raises(ValueError, match="sum to <= 1.0"):
            RerankerConfig(project_weight=-1.0)

    def test_accepts_exactly_one(self):
        # Boundary: total_signal_weight == 1.0 is allowed (retrieval_weight=0).
        cfg = RerankerConfig(
            project_weight=0.20,
            recency_weight=0.15,
            confidence_weight=0.10,
            recall_weight=0.10,
            type_affinity_weight=0.10,
            tag_overlap_weight=0.15,
            pattern_weight=0.15,
            kg_weight=0.05,
        )
        assert abs(cfg.total_signal_weight - 1.0) < 1e-9

    def test_effective_signal_weight_redistributes_kg_when_inactive(self):
        cfg = RerankerConfig()
        assert cfg.effective_signal_weight(kg_active=True) == cfg.total_signal_weight
        assert (
            cfg.effective_signal_weight(kg_active=False)
            == cfg.total_signal_weight - cfg.kg_weight
        )


class TestRecallConfig:
    def test_defaults(self):
        cfg = RecallConfig()
        assert cfg.default_k == 5
        assert cfg.rrf_k == 60
        assert cfg.max_expansion_terms == 5


class TestPatternsConfig:
    def test_defaults(self):
        cfg = PatternsConfig()
        assert cfg.min_cluster_size == 5
        assert cfg.min_samples == 3
        assert cfg.overlap_threshold == 0.3


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.ollama_model == "nomic-embed-text"
        assert cfg.ollama_host == "http://localhost:11434"


class TestQueryExpansionConfig:
    def test_defaults(self):
        cfg = QueryExpansionConfig()
        assert cfg.idf_max_age_hours == 24
        assert cfg.idf_drift_threshold == 0.10


class TestArchivalConfig:
    def test_defaults(self):
        cfg = ArchivalConfig()
        assert cfg.compress_timeout == 300
        assert cfg.skip_recent_minutes == 10


class TestDatabaseConfig:
    def test_defaults(self):
        cfg = DatabaseConfig()
        assert cfg.max_pool_size == 10
        assert cfg.max_archival_context == 10


class TestOPCConfig:
    def test_all_sections_have_defaults(self):
        cfg = OPCConfig()
        assert isinstance(cfg.dedup, DedupConfig)
        assert isinstance(cfg.daemon, DaemonConfig)
        assert isinstance(cfg.reranker, RerankerConfig)
        assert isinstance(cfg.patterns, PatternsConfig)
        assert isinstance(cfg.recall, RecallConfig)
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert isinstance(cfg.query_expansion, QueryExpansionConfig)
        assert isinstance(cfg.archival, ArchivalConfig)
        assert isinstance(cfg.database, DatabaseConfig)

    def test_frozen(self):
        cfg = OPCConfig()
        try:
            cfg.dedup = DedupConfig(threshold=0.5)  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass
