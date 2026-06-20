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
    # Issue #146: automated recall_log retention. Rows older than
    # recall_log_retention_days are pruned by the daemon every
    # recall_log_prune_interval_hours. Set retention to 0 to disable pruning.
    recall_log_retention_days: int = 90
    recall_log_prune_interval_hours: int = 24


@dataclass(frozen=True)
class RerankerConfig:
    # Issue #54: lowered from 0.15. With RRF calibration de-saturated (see
    # rrf_scale_factor), a *substring* project match (project_match=0.5) was
    # still enough to bury a strictly superior retrieval hit from a sibling
    # project (e.g. "opc" caller vs an "opc-memory-mcp" result). 0.09 keeps an
    # exact project match a strong signal without letting a partial match
    # override clear retrieval evidence.
    project_weight: float = 0.09
    recency_weight: float = 0.05
    confidence_weight: float = 0.05
    # Issue #54: recall-count bias damping. Halved from 0.05 so a heavily
    # recalled row (count 600+) no longer wins the tie-break over a strong
    # retrieval hit by a full 0.05-weight margin. The freed 0.03 flows to the
    # retrieval term implicitly (retrieval_weight = 1 - total_signal_weight).
    recall_weight: float = 0.02
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
    # Issue #54: raised from 4.0 to 10.0 so recall_score grows slowly and does
    # not saturate at count~=15. count 621 -> 0.93, count 15 -> ~0.4, count 4
    # -> ~0.23, leaving headroom between the heavy tail and light new rows.
    recall_log2_normalizer: float = 10.0
    # Issue #54: lowered from 60.0. calibrate_score multiplies a hybrid_rrf raw
    # score (~0.02-0.04 for top hits) by this factor, then clamps to [0, 1]. At
    # 60 every top hit mapped to >=1.0, flattening the retrieval ranking so the
    # contextual signals decided everything (and the heavily-recalled tail
    # won). 25 keeps the top band inside (0, 1) so retrieval rank — the
    # strongest evidence — is preserved through calibration.
    rrf_scale_factor: float = 25.0
    # Issue #54: softmax temperature for infer_query_type. Raw cosine sims
    # across 7 type centroids cluster tightly; a low temperature sharpens the
    # distribution so type_match differentiates instead of returning a
    # near-uniform ~0.14 per type. Lower -> peakier. Tuned empirically against
    # the acceptance + regression queries. None/<=0 disables sharpening.
    type_softmax_temperature: float = 0.05
    # Round 3 finding 3: type_match maps a softmax probability p to a [0,1]
    # signal centered on the neutral 0.5 via clamp01(0.5 + alpha*(p - 1/N)),
    # where N = len(distribution). Without this, a softmax prob (which sums to 1
    # across ~7 types, so a legit best type is often < 0.5) was used as the raw
    # score, letting an unknown type's 0.5 fallback outrank an evidenced type.
    # alpha scales the deviation from neutral; tuned empirically (benchmark A/B).
    type_signal_alpha: float = 1.5

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
    # Issue #153: vector-leg ANN candidate pool size = default_k * this.
    # The bounded inner ORDER BY ... LIMIT lets the HNSW index accelerate
    # the hybrid RRF vector leg. Keep generous — RRF fuses ranks and
    # truncating the candidate set shifts scores for rows outside it.
    vector_candidate_multiplier: int = 8


@dataclass(frozen=True)
class EmbeddingConfig:
    ollama_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    re_embed_batch_size: int = 64
    # Single source of truth for the canonical Voyage embedding space
    # (issue #151). The model-filtered recall corpus, re_embed_voyage.py's
    # TARGET_MODEL, and the default voyage provider all read this so they
    # never disagree (a default of 'voyage-3' would query a third space the
    # 'voyage-code-3' corpus has no rows in). Override via VOYAGE_EMBEDDING_MODEL.
    voyage_model: str = "voyage-code-3"


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
