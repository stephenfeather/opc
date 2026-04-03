"""Configuration data models — frozen dataclasses with defaults.

No I/O, no logic beyond computed properties. Pure data definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DedupConfig:
    threshold: float = 0.85


@dataclass(frozen=True)
class DaemonConfig:
    poll_interval: int = 60
    stale_threshold: int = 900
    max_concurrent_extractions: int = 4
    max_retries: int = 5
    extraction_timeout: int = 1800
    pattern_detection_interval_hours: int = 6
    extraction_model: str = "sonnet"
    extraction_max_turns: int = 15


@dataclass(frozen=True)
class RerankerConfig:
    project_weight: float = 0.15
    recency_weight: float = 0.05
    confidence_weight: float = 0.05
    recall_weight: float = 0.05
    type_affinity_weight: float = 0.05
    tag_overlap_weight: float = 0.05
    pattern_weight: float = 0.05
    recency_half_life_days: float = 45.0
    recall_log2_normalizer: int = 4
    rrf_scale_factor: float = 60.0

    @property
    def total_signal_weight(self) -> float:
        return (
            self.project_weight
            + self.recency_weight
            + self.confidence_weight
            + self.recall_weight
            + self.type_affinity_weight
            + self.tag_overlap_weight
            + self.pattern_weight
        )


@dataclass(frozen=True)
class PatternsConfig:
    min_cluster_size: int = 5
    min_samples: int = 3
    min_confidence: float = 0.3
    tag_noise_percentile: int = 10
    min_cooccurrence: float = 1.0
    min_component_size: int = 5
    overlap_threshold: float = 0.3
    max_cluster_size: int = 20
    anti_pattern_threshold: float = 0.6
    problem_solution_threshold: float = 0.4
    expertise_threshold: float = 0.6
    tool_cluster_threshold: float = 0.7
    cross_project_days: int = 30
    cross_project_contexts: int = 4
    cross_project_sessions: int = 3


@dataclass(frozen=True)
class RecallConfig:
    default_k: int = 5
    rrf_k: int = 60
    max_expansion_terms: int = 5
    recall_boost_multiplier: float = 0.002
    bm25_normalization_divisor: float = 25.0


@dataclass(frozen=True)
class EmbeddingConfig:
    ollama_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    re_embed_batch_size: int = 64


@dataclass(frozen=True)
class QueryExpansionConfig:
    idf_max_age_hours: int = 24
    idf_drift_threshold: float = 0.10


@dataclass(frozen=True)
class ArchivalConfig:
    compress_timeout: int = 300
    upload_timeout: int = 300
    skip_recent_minutes: int = 10


@dataclass(frozen=True)
class DatabaseConfig:
    max_pool_size: int = 10
    max_archival_context: int = 10


@dataclass(frozen=True)
class OPCConfig:
    dedup: DedupConfig = field(default_factory=DedupConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    patterns: PatternsConfig = field(default_factory=PatternsConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    query_expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)
    archival: ArchivalConfig = field(default_factory=ArchivalConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
