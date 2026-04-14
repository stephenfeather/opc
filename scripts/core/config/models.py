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
    harvest_grace_period: int = 300
    extraction_model: str = "sonnet"
    extraction_max_turns: int = 15
    log_rotation_days: int = 7
    log_backup_count: int = 4


@dataclass(frozen=True)
class RerankerConfig:
    project_weight: float = 0.15
    recency_weight: float = 0.05
    confidence_weight: float = 0.05
    recall_weight: float = 0.05
    type_affinity_weight: float = 0.05
    tag_overlap_weight: float = 0.05
    pattern_weight: float = 0.05
    # kg_weight is conditionally redistributed to the retrieval term when KG
    # data is inactive for a given rerank call (no query entities, sqlite
    # backend, or no result carries kg_context). That preserves pre-Phase-3
    # ranking math exactly on KG-absent paths. See rerank() and
    # effective_signal_weight() for the redistribution logic. Changing or
    # removing that redirection requires fresh adversarial review.
    kg_weight: float = 0.05
    recency_half_life_days: float = 45.0
    recall_log2_normalizer: float = 4.0
    rrf_scale_factor: float = 60.0

    def __post_init__(self) -> None:
        """Enforce the ranking-math invariant: retrieval_weight >= 0.

        The reranker derives retrieval_weight as 1 - effective_signal_weight.
        If operators tune signal weights whose sum exceeds 1.0 (e.g. upgrading
        a pre-Phase-3 opc.toml that summed to 1.0, then adding kg_weight=0.05),
        retrieval_weight goes negative and high-similarity hits get actively
        penalized. Validate at construction time so bad configs fail loudly
        instead of silently mis-ranking. See adversarial-review finding F2.
        """
        total = self.total_signal_weight
        if total < 0.0 or total > 1.0:
            raise ValueError(
                f"RerankerConfig signal weights must sum to <= 1.0, "
                f"got total_signal_weight={total:.4f}. Individual weights: "
                f"project={self.project_weight}, recency={self.recency_weight}, "
                f"confidence={self.confidence_weight}, recall={self.recall_weight}, "
                f"type_affinity={self.type_affinity_weight}, "
                f"tag_overlap={self.tag_overlap_weight}, "
                f"pattern={self.pattern_weight}, kg={self.kg_weight}. "
                f"Lower one or more weights, or reduce kg_weight to restore "
                f"a non-negative retrieval_weight."
            )

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
            + self.kg_weight
        )

    def effective_signal_weight(self, *, kg_active: bool) -> float:
        """Signal weight actually applied in this rerank call.

        Invariant: retrieval_weight + effective_signal_weight(kg_active) == 1.0

        When kg_active is False (no query entities, sqlite backend, or no
        result carries kg_context), kg_weight is redirected to retrieval
        so the scoring reduces exactly to the pre-Phase-3 math. Deterministic
        -- same inputs always yield the same redistribution.
        """
        if kg_active:
            return self.total_signal_weight
        return self.total_signal_weight - self.kg_weight


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
    default_search_limit: int = 10
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
