"""Stable data contracts for session-audit parsing and reporting.

This module contains data definitions only. Parsing, detection, timing, judging,
and rendering belong to their respective modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Any

SCHEMA_VERSION = 1
RULESET_VERSION = 1

MAX_INPUT_BYTES = 67_108_864
MAX_INPUT_LINE_BYTES = 8_388_608
MAX_INPUT_RECORDS = 125_000
MAX_NORMALIZED_EVENTS = 25_000
MAX_LOCAL_CANDIDATES = 500
MAX_JUDGE_CANDIDATES = 12
MAX_JUDGE_WINDOW_CHARS = 4_000
MAX_JUDGE_TOTAL_CHARS = 48_000
MAX_JUDGE_CALLS = 1
MAX_JUDGE_OUTPUT_TOKENS = 2_048
JUDGE_DEADLINE_SECONDS = 60.0


class AnalysisStatus(StrEnum):
    """Completeness state serialized at the top level of an audit report."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    REFUSED = "refused"


class Classification(StrEnum):
    """Evidence-backed confidence assigned to a mistake candidate."""

    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    UNCONFIRMED = "unconfirmed"


class EvidenceSourceKind(StrEnum):
    """Origin of evidence cited by a candidate or episode."""

    ASSISTANT_TEXT = "assistant_text"
    ASSISTANT_THINKING = "assistant_thinking"
    USER_PROMPT = "user_prompt"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    DERIVED = "derived"


class EventActor(StrEnum):
    """Actor responsible for a normalized event."""

    HUMAN = "human"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class ContentKind(StrEnum):
    """Normalized content-block kind."""

    VISIBLE_TEXT = "visible_text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    METADATA = "metadata"


class OutputFormat(StrEnum):
    """User-selectable report representation."""

    MARKDOWN = "markdown"
    JSON = "json"


class AuditExitCode(IntEnum):
    """Process exit codes promised by the session-audit CLI."""

    COMPLETE = 0
    OPERATIONAL_FAILURE = 1
    USAGE_ERROR = 2
    JUDGE_FAILURE = 3
    REFUSED = 4
    CANDIDATE_OVERFLOW = 5


@dataclass(frozen=True)
class AuditLimits:
    """Effective hard resource and remote-call limits for one invocation."""

    max_input_bytes: int = MAX_INPUT_BYTES
    max_input_line_bytes: int = MAX_INPUT_LINE_BYTES
    max_input_records: int = MAX_INPUT_RECORDS
    max_normalized_events: int = MAX_NORMALIZED_EVENTS
    max_local_candidates: int = MAX_LOCAL_CANDIDATES
    max_judge_candidates: int = MAX_JUDGE_CANDIDATES
    max_judge_window_chars: int = MAX_JUDGE_WINDOW_CHARS
    max_judge_total_chars: int = MAX_JUDGE_TOTAL_CHARS
    max_judge_calls: int = MAX_JUDGE_CALLS
    max_judge_output_tokens: int = MAX_JUDGE_OUTPUT_TOKENS
    judge_deadline_seconds: float = JUDGE_DEADLINE_SECONDS


@dataclass(frozen=True)
class RunProvenance:
    """Configuration and versions that make an audit result reproducible."""

    tool_version: str
    ruleset_version: int = RULESET_VERSION
    output_format: OutputFormat = OutputFormat.MARKDOWN
    thinking_included: bool = True
    judge_requested: bool = False
    judge_model: str | None = None
    judge_timeout_seconds: float = JUDGE_DEADLINE_SECONDS
    effective_limits: AuditLimits = field(default_factory=AuditLimits)


@dataclass(frozen=True)
class SessionMetadata:
    """Input identity recorded without retaining the transcript itself."""

    input_bytes: int
    sha256: str


@dataclass(frozen=True)
class NormalizedEvent:
    """One reportable content block in observed file order."""

    event_id: str
    chronological_index: int
    source_line: int
    content_block_index: int
    timestamp: datetime | None
    actor: EventActor
    kind: ContentKind
    source_kind: EvidenceSourceKind
    text: str
    text_truncated: bool = False
    record_uuid: str | None = None
    parent_uuid: str | None = None
    is_sidechain: bool = False
    source_tool_assistant_uuid: str | None = None
    lineage: tuple[str, ...] = ()
    lineage_root_uuid: str | None = None
    lineage_depth: int = 0
    lineage_truncated: bool = False
    branch_path: tuple[str, ...] = ()
    branch_id: str | None = None
    branch_depth: int = 0
    branch_path_truncated: bool = False
    ancestry_start: int | None = None
    ancestry_end: int | None = None
    lineage_valid: bool = True
    is_abandoned_branch: bool = False
    warnings: tuple[str, ...] = ()
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: tuple[tuple[str, str], ...] = ()
    tool_input_truncated_fields: tuple[str, ...] = ()
    tool_result_is_error: bool | None = None
    correlated_event_id: str | None = None


@dataclass(frozen=True)
class ParserDiagnostics:
    """Counters collected without retaining unknown transcript records."""

    nonblank_lines: int = 0
    valid_json_objects: int = 0
    non_object_json_values: int = 0
    message_like_records: int = 0
    recognized_message_records: int = 0
    normalized_events: int = 0
    unknown_record_types: int = 0
    malformed_lines: int = 0
    trailing_partial_lines: int = 0
    failed_message_records: int = 0
    empty_message_records: int = 0
    internal_metadata_records: int = 0
    invalid_field_types: int = 0
    unknown_content_blocks: int = 0
    forks: int = 0
    structural_forks: int = 0
    dangling_parents: int = 0
    cyclic_uuids: int = 0
    duplicate_uuids: int = 0
    ambiguous_parent_uuids: int = 0
    sidechain_only: bool = False
    unmatched_tool_uses: int = 0
    orphan_tool_results: int = 0
    duplicate_tool_ids: int = 0
    ambiguous_tool_ids: int = 0
    invalid_tool_ids: int = 0
    negative_tool_intervals: int = 0
    incompatible_tool_pairs: int = 0
    oversized_tool_id_groups: int = 0
    correlation_pairs_examined: int = 0
    text_excerpts_truncated: int = 0
    tool_input_fields_truncated: int = 0
    lineage_summaries_truncated: int = 0
    branch_summaries_truncated: int = 0
    unicode_replacement_events: int = 0
    digest_complete: bool = True
    refusal_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedSession:
    """Bounded parser output consumed by later deterministic stages."""

    status: AnalysisStatus
    session: SessionMetadata
    events: tuple[NormalizedEvent, ...]
    diagnostics: ParserDiagnostics


class SignalKind(StrEnum):
    """Deterministic signal family that opened or corroborated a candidate."""

    AGENT_ADMISSION = "agent_admission"
    AGENT_SELF_CORRECTION = "agent_self_correction"
    USER_CORRECTION = "user_correction"
    TOOL_FAILURE = "tool_failure"
    MATERIAL_RETRY = "material_retry"
    REVERT = "revert"
    RECOVERY_VALIDATION = "recovery_validation"


class EvidenceKind(StrEnum):
    """Machine-readable semantic role of one cited event."""

    AFFECTED_WORK = "affected_work"
    VISIBLE_ADMISSION = "visible_admission"
    THINKING_ADMISSION = "thinking_admission"
    USER_CORRECTION = "user_correction"
    OBJECTIVE_CONTRADICTION = "objective_contradiction"
    CORRECTIVE_ACTION = "corrective_action"
    REVERT = "revert"
    SUCCESSFUL_RECOVERY = "successful_recovery"
    WEAK_FRICTION = "weak_friction"


@dataclass(frozen=True)
class EvidenceRef:
    """Stable evidence citation resolved through the parser event index."""

    event_id: str
    source_kind: EvidenceSourceKind
    signal_kind: SignalKind
    evidence_kind: EvidenceKind
    corroboration_group: str
    qualifies_for_promotion: bool = False


@dataclass(frozen=True)
class ObjectiveChain:
    """Locally verified contradiction, correction, and recovery relationship."""

    chain_id: str
    root_event_id: str
    contradiction_event_id: str
    correction_event_ids: tuple[str, ...]
    recovery_event_id: str


@dataclass(frozen=True)
class GapRef:
    """Detector-approved inferred gap with explicit timing exclusion flags."""

    start_event_id: str
    end_event_id: str
    same_lineage: bool
    crosses_human_boundary: bool


@dataclass(frozen=True)
class Episode:
    """One causally merged deterministic mistake candidate."""

    episode_id: str
    category: str
    local_classification: Classification
    onset_event_id: str | None
    detection_event_id: str
    recovery_end_event_id: str | None
    affected_event_ids: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    objective_chains: tuple[ObjectiveChain, ...] = ()
    context_window_event_ids: tuple[str, ...] = ()
    affected_gap_refs: tuple[GapRef, ...] = ()
    ambiguous_gap_refs: tuple[GapRef, ...] = ()
    retry_event_ids: tuple[str, ...] = ()
    reverted_edit_event_ids: tuple[str, ...] = ()

    @property
    def qualifying_evidence_event_ids(self) -> tuple[str, ...]:
        """Evidence IDs that may support a locally validated judge promotion."""

        return tuple(
            dict.fromkeys(
                evidence.event_id for evidence in self.evidence if evidence.qualifies_for_promotion
            )
        )

    @property
    def context_only_event_ids(self) -> tuple[str, ...]:
        """Window IDs retained for semantic review but not cited as local evidence."""

        evidence_ids = {evidence.event_id for evidence in self.evidence}
        return tuple(
            event_id for event_id in self.context_window_event_ids if event_id not in evidence_ids
        )


@dataclass(frozen=True)
class DetectorDiagnostics:
    """Bounded detector counters suitable for stable reporting."""

    raw_signal_candidates: int = 0
    suppressed_non_mistakes: int = 0
    eligible_candidates: int = 0
    retained_candidates: int = 0
    omitted_candidates: int = 0


@dataclass(frozen=True)
class DetectionResult:
    """Deterministic detector output separated by reportability."""

    episodes: tuple[Episode, ...] = ()
    unconfirmed_candidates: tuple[Episode, ...] = ()
    eligible_candidates: int = 0
    retained_candidates: int = 0
    omitted_candidates: int = 0
    diagnostics: DetectorDiagnostics = field(default_factory=DetectorDiagnostics)

    @property
    def overflowed(self) -> bool:
        """Return whether bounded selection omitted eligible candidates."""

        return self.omitted_candidates > 0

    @property
    def candidates_complete(self) -> bool:
        """Return whether all eligible local candidates were retained."""

        return not self.overflowed


@dataclass(frozen=True)
class AuditResult:
    """Top-level schema-versioned report envelope.

    Episode, summary, and diagnostic payloads remain generic until their
    deterministic contracts are introduced by later implementation tasks.
    """

    schema_version: int = field(default=SCHEMA_VERSION, init=False)
    run: RunProvenance
    session: SessionMetadata
    status: AnalysisStatus = AnalysisStatus.COMPLETE
    summary: dict[str, Any] = field(default_factory=dict)
    episodes: tuple[Any, ...] = ()
    unconfirmed_candidates: tuple[Any, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
