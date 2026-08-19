"""Pure construction and rendering of deterministic session-audit reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum

from scripts.core.log_safety import redact_secrets
from scripts.core.session_audit.models import (
    AnalysisStatus,
    AuditResult,
    Classification,
    ContentKind,
    DetectionResult,
    DetectorDiagnostics,
    Episode,
    EventActor,
    EvidenceKind,
    EvidenceSourceKind,
    GapRef,
    NormalizedEvent,
    ParsedSession,
    ParserDiagnostics,
    RunProvenance,
    SignalKind,
)
from scripts.core.session_audit.timing import (
    AffectedCounts,
    AffectedEventIds,
    EpisodeTimingInput,
    HeuristicEstimateRange,
    IntervalInput,
    TimingDiagnostics,
    TimingMetrics,
    calculate_time_attribution,
)

MAX_REPORT_EXCERPT_CHARS = 500


class ReportInvariantError(ValueError):
    """Raised when detector output cannot be traced to parsed session events."""


class ReportSerializationError(TypeError):
    """Raised when a report contains a value outside the stable wire contract."""


@dataclass(frozen=True)
class EpisodeBoundaries:
    """Event identifiers delimiting one local or final episode projection."""

    onset_event_id: str | None
    detection_event_id: str
    recovery_end_event_id: str | None

    @classmethod
    def from_episode(cls, episode: Episode) -> EpisodeBoundaries:
        """Copy the detector's local boundaries without reinterpreting them."""

        return cls(
            onset_event_id=episode.onset_event_id,
            detection_event_id=episode.detection_event_id,
            recovery_end_event_id=episode.recovery_end_event_id,
        )


@dataclass(frozen=True)
class EvidenceCitation:
    """Bounded, traceable evidence displayed in a deterministic report."""

    event_id: str
    source_line: int
    timestamp: datetime | None
    source_kind: EvidenceSourceKind
    signal_kind: SignalKind
    evidence_kind: EvidenceKind
    excerpt: str
    excerpt_truncated: bool


@dataclass(frozen=True)
class TimingEstimate:
    """Serializable timing projection with diagnostics kept at envelope level."""

    directly_attributed_observed_seconds: float | None
    central_avoidable_active_seconds: float | None
    inclusive_avoidable_active_seconds: float | None
    estimate_range: HeuristicEstimateRange
    observed_onset_to_recovery_seconds: float | None
    affected_counts: AffectedCounts

    @classmethod
    def from_metrics(cls, metrics: TimingMetrics) -> TimingEstimate:
        """Remove diagnostics while preserving every user-facing metric."""

        return cls(
            directly_attributed_observed_seconds=metrics.directly_attributed_observed_seconds,
            central_avoidable_active_seconds=metrics.central_avoidable_active_seconds,
            inclusive_avoidable_active_seconds=metrics.inclusive_avoidable_active_seconds,
            estimate_range=metrics.estimate_range,
            observed_onset_to_recovery_seconds=metrics.observed_onset_to_recovery_seconds,
            affected_counts=metrics.affected_counts,
        )


@dataclass(frozen=True)
class EpisodeReport:
    """Typed local/final projection for a reportable episode."""

    episode_id: str
    local_category: str
    final_category: str
    local_classification: Classification
    final_classification: Classification
    local_boundaries: EpisodeBoundaries
    final_boundaries: EpisodeBoundaries
    local_affected_event_ids: tuple[str, ...]
    final_affected_event_ids: tuple[str, ...]
    evidence: tuple[EvidenceCitation, ...]
    local_timing: TimingEstimate
    final_timing: TimingEstimate
    judge: object | None = None


@dataclass(frozen=True)
class UnconfirmedCandidateReport:
    """Typed candidate projection explicitly excluded from time totals."""

    episode_id: str
    local_category: str
    final_category: str
    local_classification: Classification
    final_classification: Classification
    local_boundaries: EpisodeBoundaries
    final_boundaries: EpisodeBoundaries
    local_affected_event_ids: tuple[str, ...]
    final_affected_event_ids: tuple[str, ...]
    evidence: tuple[EvidenceCitation, ...]
    local_timing: TimingEstimate
    final_timing: TimingEstimate
    excluded_from_totals: bool = True
    judge: object | None = None


def _event_index(parsed: ParsedSession) -> dict[str, NormalizedEvent]:
    event_by_id = {event.event_id: event for event in parsed.events}
    if len(event_by_id) != len(parsed.events):
        raise ReportInvariantError("parsed session contains duplicate event identifiers")
    return event_by_id


def _resolve_event(
    event_by_id: dict[str, NormalizedEvent],
    event_id: str,
    *,
    role: str,
) -> NormalizedEvent:
    try:
        return event_by_id[event_id]
    except KeyError as exc:
        raise ReportInvariantError(f"{role} references an unknown event identifier") from exc


def _validated_boundary_range(
    event_by_id: dict[str, NormalizedEvent],
    boundaries: EpisodeBoundaries,
) -> tuple[int | None, int, int | None]:
    onset = (
        _resolve_event(event_by_id, boundaries.onset_event_id, role="episode onset")
        if boundaries.onset_event_id is not None
        else None
    )
    detection = _resolve_event(
        event_by_id,
        boundaries.detection_event_id,
        role="episode detection",
    )
    recovery = (
        _resolve_event(event_by_id, boundaries.recovery_end_event_id, role="episode recovery")
        if boundaries.recovery_end_event_id is not None
        else None
    )
    if onset is not None and onset.chronological_index > detection.chronological_index:
        raise ReportInvariantError("episode onset occurs after detection")
    if recovery is not None and detection.chronological_index > recovery.chronological_index:
        raise ReportInvariantError("episode detection occurs after recovery")
    return (
        onset.chronological_index if onset is not None else None,
        detection.chronological_index,
        recovery.chronological_index if recovery is not None else None,
    )


def build_episode_timing_input(
    parsed: ParsedSession,
    episode: Episode,
    *,
    boundaries: EpisodeBoundaries | None = None,
) -> EpisodeTimingInput:
    """Translate detector-approved causal references into timing inputs.

    A boundary override may narrow an episode, but it cannot introduce affected work or
    causal intervals absent from the local detector result.
    """

    event_by_id = _event_index(parsed)
    selected_boundaries = boundaries or EpisodeBoundaries.from_episode(episode)
    lower, _, upper = _validated_boundary_range(event_by_id, selected_boundaries)

    def in_bounds(event: NormalizedEvent) -> bool:
        return (lower is None or lower <= event.chronological_index) and (
            upper is None or event.chronological_index <= upper
        )

    local_affected = tuple(
        _resolve_event(event_by_id, event_id, role="affected work")
        for event_id in episode.affected_event_ids
    )
    local_affected_ids = {event.event_id for event in local_affected}
    affected = tuple(event for event in local_affected if in_bounds(event))
    affected_ids = {event.event_id for event in affected}

    observed_intervals: list[IntervalInput] = []
    tool_call_ids: set[str] = set()
    failed_tool_call_ids: set[str] = set()
    for event in affected:
        if event.kind is not ContentKind.TOOL_USE or event.event_id in tool_call_ids:
            continue
        tool_call_ids.add(event.event_id)
        if event.correlated_event_id is None:
            continue
        result = _resolve_event(
            event_by_id,
            event.correlated_event_id,
            role="tool correlation",
        )
        if (
            result.kind is not ContentKind.TOOL_RESULT
            or result.correlated_event_id != event.event_id
            or (
                event.tool_use_id is not None
                and result.tool_use_id is not None
                and event.tool_use_id != result.tool_use_id
            )
        ):
            raise ReportInvariantError("invalid tool correlation in affected work")
        if not in_bounds(result):
            continue
        observed_intervals.append(IntervalInput(event.timestamp, result.timestamp))
        if result.tool_result_is_error is True:
            failed_tool_call_ids.add(event.event_id)

    def gap_intervals(gaps: tuple[GapRef, ...]) -> tuple[IntervalInput, ...]:
        intervals: list[IntervalInput] = []
        for gap in gaps:
            start_id = gap.start_event_id
            end_id = gap.end_event_id
            start = _resolve_event(event_by_id, start_id, role="timing gap")
            end = _resolve_event(event_by_id, end_id, role="timing gap")
            if start_id not in local_affected_ids or end_id not in local_affected_ids:
                raise ReportInvariantError("timing gap endpoint is outside local affected work")
            if start_id not in affected_ids or end_id not in affected_ids:
                continue
            if in_bounds(start) and in_bounds(end):
                intervals.append(
                    IntervalInput(
                        start.timestamp,
                        end.timestamp,
                        same_lineage=gap.same_lineage,
                        crosses_human_boundary=gap.crosses_human_boundary,
                    )
                )
        return tuple(intervals)

    assistant_turn_ids = frozenset(
        event.record_uuid or f"line:{event.source_line}"
        for event in affected
        if event.actor is EventActor.ASSISTANT
    )

    def filtered_local_ids(event_ids: tuple[str, ...], *, role: str) -> frozenset[str]:
        resolved = (_resolve_event(event_by_id, event_id, role=role) for event_id in event_ids)
        return frozenset(event.event_id for event in resolved if in_bounds(event))

    onset = (
        event_by_id[selected_boundaries.onset_event_id].timestamp
        if selected_boundaries.onset_event_id is not None
        else None
    )
    recovery = (
        event_by_id[selected_boundaries.recovery_end_event_id].timestamp
        if selected_boundaries.recovery_end_event_id is not None
        else None
    )
    return EpisodeTimingInput(
        observed_tool_intervals=tuple(observed_intervals),
        affected_gaps=gap_intervals(episode.affected_gap_refs),
        ambiguous_recovery_gaps=gap_intervals(episode.ambiguous_gap_refs),
        onset=onset,
        recovery_end=recovery,
        affected_event_ids=AffectedEventIds(
            assistant_turn_ids=assistant_turn_ids,
            tool_call_ids=frozenset(tool_call_ids),
            failed_tool_call_ids=frozenset(failed_tool_call_ids),
            retry_ids=filtered_local_ids(episode.retry_event_ids, role="retry"),
            reverted_edit_ids=filtered_local_ids(
                episode.reverted_edit_event_ids,
                role="reverted edit",
            ),
        ),
    )


def _evidence_citations(
    event_by_id: dict[str, NormalizedEvent],
    episode: Episode,
) -> tuple[EvidenceCitation, ...]:
    citations: list[EvidenceCitation] = []
    required_roles = {
        EvidenceKind.VISIBLE_ADMISSION: (EventActor.ASSISTANT, ContentKind.VISIBLE_TEXT),
        EvidenceKind.THINKING_ADMISSION: (EventActor.ASSISTANT, ContentKind.THINKING),
        EvidenceKind.USER_CORRECTION: (EventActor.HUMAN, ContentKind.VISIBLE_TEXT),
        EvidenceKind.OBJECTIVE_CONTRADICTION: (EventActor.TOOL, ContentKind.TOOL_RESULT),
        EvidenceKind.CORRECTIVE_ACTION: (EventActor.ASSISTANT, ContentKind.TOOL_USE),
        EvidenceKind.REVERT: (EventActor.ASSISTANT, ContentKind.TOOL_USE),
        EvidenceKind.SUCCESSFUL_RECOVERY: (EventActor.TOOL, ContentKind.TOOL_RESULT),
    }
    for evidence in episode.evidence:
        event = _resolve_event(event_by_id, evidence.event_id, role="evidence")
        if evidence.source_kind is not event.source_kind:
            raise ReportInvariantError("evidence source does not match the resolved event")
        required_role = required_roles.get(evidence.evidence_kind)
        if required_role is not None and (event.actor, event.kind) != required_role:
            raise ReportInvariantError("evidence role does not match the resolved event")
        raw_excerpt = event.text
        if not raw_excerpt and event.kind is ContentKind.TOOL_USE:
            raw_excerpt = " ".join(
                (
                    event.tool_name or "tool",
                    *(f"{key}={value}" for key, value in event.tool_input),
                )
            )
        excerpt = redact_secrets(raw_excerpt)
        truncated = (
            event.text_truncated
            or bool(event.tool_input_truncated_fields)
            or len(excerpt) > MAX_REPORT_EXCERPT_CHARS
        )
        citations.append(
            EvidenceCitation(
                event_id=event.event_id,
                source_line=event.source_line,
                timestamp=event.timestamp,
                source_kind=evidence.source_kind,
                signal_kind=evidence.signal_kind,
                evidence_kind=evidence.evidence_kind,
                excerpt=excerpt[:MAX_REPORT_EXCERPT_CHARS],
                excerpt_truncated=truncated,
            )
        )
    return tuple(citations)


def _build_episode_report(
    parsed: ParsedSession,
    event_by_id: dict[str, NormalizedEvent],
    episode: Episode,
) -> tuple[EpisodeReport, EpisodeTimingInput]:
    if episode.local_classification is Classification.UNCONFIRMED:
        raise ReportInvariantError("an unconfirmed candidate entered reportable episodes")
    boundaries = EpisodeBoundaries.from_episode(episode)
    timing_input = build_episode_timing_input(parsed, episode)
    timing = calculate_time_attribution((timing_input,))
    timing_estimate = TimingEstimate.from_metrics(timing)
    projection = EpisodeReport(
        episode_id=episode.episode_id,
        local_category=episode.category,
        final_category=episode.category,
        local_classification=episode.local_classification,
        final_classification=episode.local_classification,
        local_boundaries=boundaries,
        final_boundaries=boundaries,
        local_affected_event_ids=episode.affected_event_ids,
        final_affected_event_ids=episode.affected_event_ids,
        evidence=_evidence_citations(event_by_id, episode),
        local_timing=timing_estimate,
        final_timing=timing_estimate,
    )
    return projection, timing_input


def _build_unconfirmed_report(
    parsed: ParsedSession,
    event_by_id: dict[str, NormalizedEvent],
    episode: Episode,
) -> UnconfirmedCandidateReport:
    if episode.local_classification is not Classification.UNCONFIRMED:
        raise ReportInvariantError("a reportable episode entered unconfirmed candidates")
    for event_id in episode.affected_event_ids:
        _resolve_event(event_by_id, event_id, role="affected work")
    boundaries = EpisodeBoundaries.from_episode(episode)
    _validated_boundary_range(event_by_id, boundaries)
    timing = TimingEstimate.from_metrics(
        calculate_time_attribution((build_episode_timing_input(parsed, episode),))
    )
    return UnconfirmedCandidateReport(
        episode_id=episode.episode_id,
        local_category=episode.category,
        final_category=episode.category,
        local_classification=episode.local_classification,
        final_classification=episode.local_classification,
        local_boundaries=boundaries,
        final_boundaries=boundaries,
        local_affected_event_ids=episode.affected_event_ids,
        final_affected_event_ids=episode.affected_event_ids,
        evidence=_evidence_citations(event_by_id, episode),
        local_timing=timing,
        final_timing=timing,
    )


def build_audit_result(
    parsed: ParsedSession,
    detection: DetectionResult | None,
    *,
    run: RunProvenance,
) -> AuditResult:
    """Build the deterministic typed report envelope without rendering or I/O."""

    if parsed.status is AnalysisStatus.REFUSED:
        return AuditResult(
            run=run,
            session=parsed.session,
            status=AnalysisStatus.REFUSED,
            summary={
                "totals_complete": False,
                "confirmed_episodes": None,
                "probable_episodes": None,
                "unconfirmed_candidates": None,
                "omitted_candidates": None,
                "local_estimates": None,
                "final_estimates": None,
            },
            diagnostics={
                "parser": parsed.diagnostics,
                "detector": None,
                "timing": None,
                "judge": None,
            },
        )
    if detection is None:
        raise ReportInvariantError("a complete parse requires detector output")

    event_by_id = _event_index(parsed)
    episode_pairs = tuple(
        _build_episode_report(parsed, event_by_id, episode) for episode in detection.episodes
    )
    projections = tuple(projection for projection, _ in episode_pairs)
    timing_inputs = tuple(timing_input for _, timing_input in episode_pairs)
    unconfirmed = tuple(
        _build_unconfirmed_report(parsed, event_by_id, episode)
        for episode in detection.unconfirmed_candidates
    )
    aggregate_timing = calculate_time_attribution(timing_inputs)
    overflowed = detection.overflowed
    status = AnalysisStatus.PARTIAL if overflowed else AnalysisStatus.COMPLETE
    local_estimates: TimingEstimate | None = (
        None if overflowed else TimingEstimate.from_metrics(aggregate_timing)
    )
    return AuditResult(
        run=run,
        session=parsed.session,
        status=status,
        summary={
            "totals_complete": not overflowed,
            "confirmed_episodes": sum(
                episode.final_classification is Classification.CONFIRMED for episode in projections
            ),
            "probable_episodes": sum(
                episode.final_classification is Classification.PROBABLE for episode in projections
            ),
            "unconfirmed_candidates": len(unconfirmed),
            "omitted_candidates": detection.omitted_candidates,
            "local_estimates": local_estimates,
            "final_estimates": local_estimates,
        },
        episodes=projections,
        unconfirmed_candidates=unconfirmed,
        diagnostics={
            "parser": parsed.diagnostics,
            "detector": detection.diagnostics,
            "timing": aggregate_timing.diagnostics,
            "judge": None,
        },
    )


def _to_wire(value: object) -> object:
    if isinstance(value, UnconfirmedCandidateReport):
        return {
            "episode_id": _to_wire(value.episode_id),
            "local_category": _to_wire(value.local_category),
            "final_category": _to_wire(value.final_category),
            "local_classification": _to_wire(value.local_classification),
            "final_classification": _to_wire(value.final_classification),
            "local_boundaries": _to_wire(value.local_boundaries),
            "final_boundaries": _to_wire(value.final_boundaries),
            "local_affected_event_ids": _to_wire(value.local_affected_event_ids),
            "final_affected_event_ids": _to_wire(value.final_affected_event_ids),
            "evidence": _to_wire(value.evidence),
            "excluded_from_totals": _to_wire(value.excluded_from_totals),
            "judge": _to_wire(value.judge),
        }
    if isinstance(value, Enum):
        return _to_wire(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportSerializationError("non-finite floats are not reportable")
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_wire(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ReportSerializationError("report mapping keys must be strings")
        return {key: _to_wire(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_to_wire(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_to_wire(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    raise ReportSerializationError(f"unsupported report value type: {type(value).__name__}")


def render_json(result: AuditResult) -> str:
    """Render a stable UTF-8 JSON document with exactly one trailing newline."""

    document = {
        "schema_version": _to_wire(result.schema_version),
        "status": _to_wire(result.status),
        "run": _to_wire(result.run),
        "session": _to_wire(result.session),
        "summary": _to_wire(result.summary),
        "episodes": _to_wire(result.episodes),
        "unconfirmed_candidates": _to_wire(result.unconfirmed_candidates),
        "diagnostics": _to_wire(result.diagnostics),
    }
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _markdown_text(value: str) -> str:
    single_line = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return "".join(
        character
        for character in single_line
        if ord(character) >= 32 and not 127 <= ord(character) <= 159
    )


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "Unavailable (missing or invalid timestamps)"
    return f"{value:.1f} seconds"


def _format_count(value: object) -> str:
    return "Unavailable" if value is None else str(value)


def _markdown_estimates(lines: list[str], estimates: TimingEstimate | None) -> None:
    if estimates is None:
        lines.extend(
            (
                "- Directly attributed observed time: Unavailable",
                "- Central estimated avoidable active time: Unavailable",
                "- Inclusive estimated avoidable active time: Unavailable",
            )
        )
        return
    lines.extend(
        (
            "- Directly attributed observed time: "
            f"{_format_seconds(estimates.directly_attributed_observed_seconds)}",
            "- Central estimated avoidable active time: "
            f"{_format_seconds(estimates.central_avoidable_active_seconds)}",
            "- Inclusive estimated avoidable active time: "
            f"{_format_seconds(estimates.inclusive_avoidable_active_seconds)}",
            "- Observed onset-to-recovery span: "
            f"{_format_seconds(estimates.observed_onset_to_recovery_seconds)}",
        )
    )


def _markdown_evidence(lines: list[str], evidence: tuple[EvidenceCitation, ...]) -> None:
    if not evidence:
        lines.append("> No displayable evidence citations")
        return
    for citation in evidence:
        thinking_label = (
            " (thinking evidence)"
            if citation.source_kind is EvidenceSourceKind.ASSISTANT_THINKING
            else ""
        )
        excerpt = _markdown_text(citation.excerpt)
        if citation.excerpt_truncated:
            excerpt += " … [truncated]"
        lines.append(
            f"> [{citation.source_kind.value} / {citation.evidence_kind.value}, "
            f"line {citation.source_line}]{thinking_label} {excerpt}"
        )


def _judge_wire(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    converted = _to_wire(value)
    if not isinstance(converted, Mapping):
        raise ReportSerializationError("judge projection has an invalid type")
    return converted


def _markdown_episode_identity(
    lines: list[str],
    value: EpisodeReport | UnconfirmedCandidateReport,
) -> None:
    judge = _judge_wire(value.judge)
    if judge is None:
        lines.extend(
            (
                f"- Classification: `{value.final_classification.value}`",
                f"- Category: `{_markdown_text(value.final_category)}`",
            )
        )
        return
    requested = judge.get("judge_classification")
    decision_status = judge.get("decision_status")
    lines.extend(
        (
            f"- Local classification: `{value.local_classification.value}`",
            "- Judge requested classification: "
            + (f"`{_markdown_text(requested)}`" if isinstance(requested, str) else "Unavailable"),
            f"- Final classification: `{value.final_classification.value}`",
            f"- Local category: `{_markdown_text(value.local_category)}`",
            f"- Final category: `{_markdown_text(value.final_category)}`",
            "- Judge decision status: "
            + (
                f"`{_markdown_text(decision_status)}`"
                if isinstance(decision_status, str)
                else "Unavailable"
            ),
        )
    )
    rationale = judge.get("rationale")
    if isinstance(rationale, str) and rationale:
        lines.append(f"- Judge rationale: {_markdown_text(rationale)}")
    clamp = judge.get("clamp")
    if isinstance(clamp, Mapping):
        reason = clamp.get("reason_code")
        if isinstance(reason, str):
            lines.append(f"- Promotion clamp: `{_markdown_text(reason)}`")


def render_markdown(result: AuditResult) -> str:
    """Render a terminal-control-safe human report with explicit metric caveats."""

    lines = [
        "# Claude Session Mistake Audit",
        "",
        f"Status: `{result.status.value}`",
        "",
    ]
    if result.status is AnalysisStatus.REFUSED:
        lines.extend(
            (
                "**Warning:** Analysis was refused; findings and time totals are unavailable.",
                "",
            )
        )
    elif result.status is AnalysisStatus.PARTIAL:
        omitted = result.summary.get("omitted_candidates")
        partial_warning = (
            "Candidate limits were reached; retained findings are shown, but aggregate "
            "totals are unavailable."
            if isinstance(omitted, int) and omitted > 0
            else "Analysis is partial; retained findings are shown, but aggregate totals "
            "may be unavailable."
        )
        lines.extend(
            (
                f"**Warning:** {partial_warning}",
                "",
            )
        )
    lines.extend(
        (
            "Time values distinguish directly observed time from estimated avoidable active time. "
            "The heuristic range is policy-based and is not a confidence interval.",
            "",
            "## Summary",
            "",
            "- Confirmed episodes: " f"{_format_count(result.summary.get('confirmed_episodes'))}",
            "- Probable episodes: " f"{_format_count(result.summary.get('probable_episodes'))}",
            "- Unconfirmed candidates: "
            f"{_format_count(result.summary.get('unconfirmed_candidates'))}",
            "- Omitted candidates: " f"{_format_count(result.summary.get('omitted_candidates'))}",
        )
    )
    estimates = result.summary.get("final_estimates")
    if estimates is not None and not isinstance(estimates, TimingEstimate):
        raise ReportSerializationError("summary final estimates have an invalid type")
    _markdown_estimates(lines, estimates)

    lines.extend(("", "## Confirmed and Probable Episodes", ""))
    if not result.episodes and result.status is AnalysisStatus.COMPLETE:
        lines.append("No confirmed or probable mistakes found.")
    for value in result.episodes:
        if not isinstance(value, EpisodeReport):
            raise ReportSerializationError("report episodes have an invalid type")
        lines.extend(
            (
                f"### {_markdown_text(value.episode_id)}",
                "",
            )
        )
        _markdown_episode_identity(lines, value)
        lines.extend(
            (
                "- Directly attributed observed time: "
                f"{_format_seconds(value.final_timing.directly_attributed_observed_seconds)}",
                "- Central estimated avoidable active time: "
                f"{_format_seconds(value.final_timing.central_avoidable_active_seconds)}",
                "- Inclusive estimated avoidable active time: "
                f"{_format_seconds(value.final_timing.inclusive_avoidable_active_seconds)}",
                "",
                "Evidence:",
            )
        )
        _markdown_evidence(lines, value.evidence)
        lines.append("")

    lines.extend(("## Unconfirmed Candidates", ""))
    if not result.unconfirmed_candidates:
        lines.append("None.")
    for value in result.unconfirmed_candidates:
        if not isinstance(value, UnconfirmedCandidateReport):
            raise ReportSerializationError("unconfirmed candidates have an invalid type")
        lines.extend(
            (
                f"### {_markdown_text(value.episode_id)}",
                "",
            )
        )
        _markdown_episode_identity(lines, value)
        lines.extend(("Excluded from time totals.", "", "Evidence:"))
        _markdown_evidence(lines, value.evidence)
        lines.append("")

    lines.extend(("## Diagnostics", ""))
    parser_diagnostics = result.diagnostics.get("parser")
    if isinstance(parser_diagnostics, ParserDiagnostics):
        reasons = ", ".join(parser_diagnostics.refusal_reasons) or "none"
        lines.append(f"- Parser refusal reasons: {_markdown_text(reasons)}")
        lines.append(f"- Normalized events: {parser_diagnostics.normalized_events}")
    else:
        lines.append("- Parser diagnostics: unavailable")
    detector_diagnostics = result.diagnostics.get("detector")
    if isinstance(detector_diagnostics, DetectorDiagnostics):
        lines.extend(
            (
                f"- Detector raw signals: {detector_diagnostics.raw_signal_candidates}",
                "- Detector suppressed non-mistakes: "
                f"{detector_diagnostics.suppressed_non_mistakes}",
                f"- Detector eligible candidates: {detector_diagnostics.eligible_candidates}",
                f"- Detector retained candidates: {detector_diagnostics.retained_candidates}",
                f"- Detector omitted candidates: {detector_diagnostics.omitted_candidates}",
            )
        )
    else:
        lines.append("- Detector diagnostics: unavailable")
    timing_diagnostics = result.diagnostics.get("timing")
    if isinstance(timing_diagnostics, TimingDiagnostics):
        lines.extend(
            (
                "- Timing missing-timestamp intervals: "
                f"{timing_diagnostics.missing_timestamp_intervals}",
                "- Timing invalid-timestamp intervals: "
                f"{timing_diagnostics.invalid_timestamp_intervals}",
                "- Timing excluded unrelated-branch intervals: "
                f"{timing_diagnostics.excluded_unrelated_branch_intervals}",
                "- Timing excluded human-idle intervals: "
                f"{timing_diagnostics.excluded_human_idle_intervals}",
            )
        )
    else:
        lines.append("- Timing diagnostics: unavailable")
    judge_diagnostics = result.diagnostics.get("judge")
    judge_wire = _judge_wire(judge_diagnostics)
    if judge_wire is None:
        lines.append("- Judge: not requested")
    else:
        judge_status = judge_wire.get("status")
        failure_code = judge_wire.get("failure_code")
        lines.append(
            "- Judge status: "
            + (
                f"`{_markdown_text(judge_status)}`"
                if isinstance(judge_status, str)
                else "Unavailable"
            )
        )
        for key, label in (
            ("selected_candidates", "Judge selected candidates"),
            ("submitted_candidates", "Judge submitted candidates"),
            ("accepted", "Judge accepted decisions"),
            ("rejected", "Judge rejected decisions"),
        ):
            count = judge_wire.get(key)
            lines.append(f"- {label}: {_format_count(count)}")
        if isinstance(failure_code, str):
            lines.append(f"- Judge failure code: `{_markdown_text(failure_code)}`")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "EpisodeBoundaries",
    "EpisodeReport",
    "EvidenceCitation",
    "MAX_REPORT_EXCERPT_CHARS",
    "ReportInvariantError",
    "ReportSerializationError",
    "TimingEstimate",
    "UnconfirmedCandidateReport",
    "build_audit_result",
    "build_episode_timing_input",
    "render_json",
    "render_markdown",
]
