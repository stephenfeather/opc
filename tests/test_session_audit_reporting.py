"""Tests for deterministic session-audit report construction and rendering."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from scripts.core.session_audit.models import (
    AnalysisStatus,
    Classification,
    ContentKind,
    DetectionResult,
    DetectorDiagnostics,
    Episode,
    EventActor,
    EvidenceKind,
    EvidenceRef,
    EvidenceSourceKind,
    GapRef,
    NormalizedEvent,
    OutputFormat,
    ParsedSession,
    ParserDiagnostics,
    RunProvenance,
    SessionMetadata,
    SignalKind,
)
from scripts.core.session_audit.reporting import (
    MAX_REPORT_EXCERPT_CHARS,
    EpisodeBoundaries,
    EpisodeReport,
    ReportInvariantError,
    ReportSerializationError,
    UnconfirmedCandidateReport,
    build_audit_result,
    build_episode_timing_input,
    render_json,
    render_markdown,
)

BASE = datetime(2026, 8, 11, 16, 0, tzinfo=UTC)


def _event(
    index: int,
    *,
    kind: ContentKind = ContentKind.VISIBLE_TEXT,
    actor: EventActor = EventActor.ASSISTANT,
    record_uuid: str | None = None,
    correlated_event_id: str | None = None,
    tool_result_is_error: bool | None = None,
    tool_name: str | None = None,
    tool_input: tuple[tuple[str, str], ...] = (),
) -> NormalizedEvent:
    return NormalizedEvent(
        event_id=f"E{index}",
        chronological_index=index,
        source_line=index + 1,
        content_block_index=0,
        timestamp=BASE + timedelta(seconds=index * 10),
        actor=actor,
        kind=kind,
        source_kind=(
            EvidenceSourceKind.TOOL_USE
            if kind is ContentKind.TOOL_USE
            else (
                EvidenceSourceKind.TOOL_RESULT
                if kind is ContentKind.TOOL_RESULT
                else EvidenceSourceKind.ASSISTANT_TEXT
            )
        ),
        text="",
        record_uuid=record_uuid,
        correlated_event_id=correlated_event_id,
        tool_result_is_error=tool_result_is_error,
        tool_name=tool_name,
        tool_input=tool_input,
    )


def _parsed(*events: NormalizedEvent) -> ParsedSession:
    return ParsedSession(
        status=AnalysisStatus.COMPLETE,
        session=SessionMetadata(input_bytes=10, sha256="a" * 64),
        events=events,
        diagnostics=ParserDiagnostics(normalized_events=len(events)),
    )


def test_boundary_override_filters_only_existing_local_timing_inputs() -> None:
    before = _event(0, record_uuid="turn-before")
    kept_use = _event(1, kind=ContentKind.TOOL_USE, correlated_event_id="E2")
    kept_result = _event(
        2,
        kind=ContentKind.TOOL_RESULT,
        actor=EventActor.TOOL,
        correlated_event_id="E1",
        tool_result_is_error=True,
    )
    kept_turn = _event(3, record_uuid="turn-kept")
    recovery = _event(4, record_uuid="turn-recovery")
    after = _event(5, record_uuid="turn-after")
    episode = Episode(
        episode_id="episode-1",
        category="incorrect_change",
        local_classification=Classification.CONFIRMED,
        onset_event_id=before.event_id,
        detection_event_id=kept_result.event_id,
        recovery_end_event_id=after.event_id,
        affected_event_ids=tuple(
            event.event_id for event in (before, kept_use, kept_result, kept_turn, recovery, after)
        ),
        evidence=(),
        affected_gap_refs=(
            GapRef(before.event_id, kept_use.event_id, True, False),
            GapRef(kept_result.event_id, kept_turn.event_id, True, False),
            GapRef(recovery.event_id, after.event_id, True, False),
        ),
        ambiguous_gap_refs=(GapRef(kept_turn.event_id, recovery.event_id, True, False),),
        retry_event_ids=(before.event_id, kept_turn.event_id, after.event_id),
        reverted_edit_event_ids=(before.event_id, recovery.event_id, after.event_id),
    )

    timing_input = build_episode_timing_input(
        _parsed(before, kept_use, kept_result, kept_turn, recovery, after),
        episode,
        boundaries=EpisodeBoundaries(
            onset_event_id=kept_use.event_id,
            detection_event_id=kept_result.event_id,
            recovery_end_event_id=recovery.event_id,
        ),
    )

    assert timing_input.onset == kept_use.timestamp
    assert timing_input.recovery_end == recovery.timestamp
    assert len(timing_input.observed_tool_intervals) == 1
    assert timing_input.observed_tool_intervals[0].start == kept_use.timestamp
    assert timing_input.observed_tool_intervals[0].end == kept_result.timestamp
    assert [(gap.start, gap.end) for gap in timing_input.affected_gaps] == [
        (kept_result.timestamp, kept_turn.timestamp)
    ]
    assert [(gap.start, gap.end) for gap in timing_input.ambiguous_recovery_gaps] == [
        (kept_turn.timestamp, recovery.timestamp)
    ]
    assert timing_input.affected_event_ids.assistant_turn_ids == frozenset(
        {"line:2", "turn-kept", "turn-recovery"}
    )
    assert timing_input.affected_event_ids.tool_call_ids == frozenset({kept_use.event_id})
    assert timing_input.affected_event_ids.failed_tool_call_ids == frozenset({kept_use.event_id})
    assert timing_input.affected_event_ids.retry_ids == frozenset({kept_turn.event_id})
    assert timing_input.affected_event_ids.reverted_edit_ids == frozenset({recovery.event_id})


def test_build_complete_report_keeps_typed_local_and_final_episode_projections() -> None:
    tool_use = _event(0, kind=ContentKind.TOOL_USE, correlated_event_id="E1")
    tool_result = _event(
        1,
        kind=ContentKind.TOOL_RESULT,
        actor=EventActor.TOOL,
        correlated_event_id="E0",
        tool_result_is_error=False,
    )
    admission = _event(2, record_uuid="turn-admission")
    recovery = _event(3, record_uuid="turn-recovery")
    admission = replace(admission, text="I was wrong about that change.")
    episode = Episode(
        episode_id="episode-typed",
        category="incorrect_change",
        local_classification=Classification.CONFIRMED,
        onset_event_id=tool_use.event_id,
        detection_event_id=admission.event_id,
        recovery_end_event_id=recovery.event_id,
        affected_event_ids=(
            tool_use.event_id,
            tool_result.event_id,
            admission.event_id,
            recovery.event_id,
        ),
        evidence=(
            EvidenceRef(
                event_id=admission.event_id,
                source_kind=EvidenceSourceKind.ASSISTANT_TEXT,
                signal_kind=SignalKind.AGENT_ADMISSION,
                evidence_kind=EvidenceKind.VISIBLE_ADMISSION,
                corroboration_group="admission:E2",
                qualifies_for_promotion=True,
            ),
        ),
        affected_gap_refs=(GapRef(tool_result.event_id, admission.event_id, True, False),),
        ambiguous_gap_refs=(GapRef(admission.event_id, recovery.event_id, True, False),),
    )
    parsed = _parsed(tool_use, tool_result, admission, recovery)

    result = build_audit_result(
        parsed,
        DetectionResult(
            episodes=(episode,),
            eligible_candidates=1,
            retained_candidates=1,
        ),
        run=RunProvenance(tool_version="0.8.0"),
    )

    assert result.status is AnalysisStatus.COMPLETE
    assert len(result.episodes) == 1
    projection = result.episodes[0]
    assert isinstance(projection, EpisodeReport)
    assert projection.local_category == projection.final_category == "incorrect_change"
    assert projection.local_classification is projection.final_classification
    assert (
        projection.local_boundaries
        == projection.final_boundaries
        == EpisodeBoundaries(
            onset_event_id="E0",
            detection_event_id="E2",
            recovery_end_event_id="E3",
        )
    )
    assert (
        projection.local_affected_event_ids
        == projection.final_affected_event_ids
        == (
            "E0",
            "E1",
            "E2",
            "E3",
        )
    )
    assert projection.local_timing.directly_attributed_observed_seconds == 10.0
    assert projection.local_timing.central_avoidable_active_seconds == 20.0
    assert projection.local_timing.inclusive_avoidable_active_seconds == 30.0
    assert projection.final_timing == projection.local_timing
    assert projection.judge is None
    assert result.summary["confirmed_episodes"] == 1
    assert result.summary["local_estimates"].inclusive_avoidable_active_seconds == 30.0


def test_adapter_rejects_gap_endpoints_outside_the_local_affected_set() -> None:
    affected = _event(0, record_uuid="turn-affected")
    context_only = _event(1, record_uuid="turn-context")
    detection = _event(2, record_uuid="turn-detection")
    episode = Episode(
        episode_id="episode-invalid-gap",
        category="wrong_assumption",
        local_classification=Classification.PROBABLE,
        onset_event_id=affected.event_id,
        detection_event_id=detection.event_id,
        recovery_end_event_id=detection.event_id,
        affected_event_ids=(affected.event_id, detection.event_id),
        evidence=(),
        affected_gap_refs=(GapRef(affected.event_id, context_only.event_id, True, False),),
    )

    with pytest.raises(ReportInvariantError, match="outside local affected work"):
        build_episode_timing_input(_parsed(affected, context_only, detection), episode)


def test_adapter_keeps_detector_approved_retry_and_revert_ids_without_duplication() -> None:
    onset = _event(0, record_uuid="turn-onset")
    retry = _event(1, kind=ContentKind.TOOL_USE)
    reverted_edit = _event(2, kind=ContentKind.TOOL_USE)
    detection = _event(3, record_uuid="turn-detection")
    episode = Episode(
        episode_id="episode-causal-keys",
        category="incorrect_change",
        local_classification=Classification.CONFIRMED,
        onset_event_id=onset.event_id,
        detection_event_id=detection.event_id,
        recovery_end_event_id=detection.event_id,
        affected_event_ids=(onset.event_id, detection.event_id),
        evidence=(),
        retry_event_ids=(retry.event_id,),
        reverted_edit_event_ids=(reverted_edit.event_id,),
    )

    timing_input = build_episode_timing_input(
        _parsed(onset, retry, reverted_edit, detection),
        episode,
    )

    assert timing_input.affected_event_ids.retry_ids == frozenset({retry.event_id})
    assert timing_input.affected_event_ids.reverted_edit_ids == frozenset({reverted_edit.event_id})


def test_complete_no_findings_json_has_stable_envelope_and_numeric_zeros() -> None:
    result = build_audit_result(
        _parsed(),
        DetectionResult(),
        run=RunProvenance(tool_version="0.8.0", output_format=OutputFormat.JSON),
    )

    rendered = render_json(result)
    document = json.loads(rendered)

    assert list(document) == [
        "schema_version",
        "status",
        "run",
        "session",
        "summary",
        "episodes",
        "unconfirmed_candidates",
        "diagnostics",
    ]
    assert rendered.endswith("\n") and not rendered.endswith("\n\n")
    assert document["status"] == "complete"
    assert document["run"]["output_format"] == "json"
    assert document["run"]["effective_limits"] == {
        "max_input_bytes": 67_108_864,
        "max_input_line_bytes": 8_388_608,
        "max_input_records": 125_000,
        "max_normalized_events": 25_000,
        "max_local_candidates": 500,
        "max_judge_candidates": 12,
        "max_judge_window_chars": 4_000,
        "max_judge_total_chars": 48_000,
        "max_judge_calls": 1,
        "max_judge_output_tokens": 2_048,
        "judge_deadline_seconds": 60.0,
    }
    assert document["summary"]["totals_complete"] is True
    assert document["summary"]["confirmed_episodes"] == 0
    assert document["summary"]["probable_episodes"] == 0
    assert document["summary"]["unconfirmed_candidates"] == 0
    assert document["summary"]["omitted_candidates"] == 0
    assert document["summary"]["local_estimates"] == {
        "directly_attributed_observed_seconds": 0.0,
        "central_avoidable_active_seconds": 0.0,
        "inclusive_avoidable_active_seconds": 0.0,
        "estimate_range": {
            "low_seconds": 0.0,
            "central_seconds": 0.0,
            "high_seconds": 0.0,
        },
        "observed_onset_to_recovery_seconds": 0.0,
        "affected_counts": {
            "assistant_turns": 0,
            "tool_calls": 0,
            "failed_tool_calls": 0,
            "retries": 0,
            "reverted_edits": 0,
        },
    }
    assert document["summary"]["final_estimates"] == document["summary"]["local_estimates"]
    assert document["episodes"] == []
    assert document["unconfirmed_candidates"] == []


def test_report_rejects_evidence_with_spoofed_source_provenance() -> None:
    affected = _event(0, record_uuid="turn-affected")
    admission = replace(_event(1, record_uuid="turn-admission"), text="I was wrong.")
    episode = Episode(
        episode_id="episode-spoofed-evidence",
        category="wrong_assumption",
        local_classification=Classification.CONFIRMED,
        onset_event_id=affected.event_id,
        detection_event_id=admission.event_id,
        recovery_end_event_id=admission.event_id,
        affected_event_ids=(affected.event_id, admission.event_id),
        evidence=(
            EvidenceRef(
                event_id=admission.event_id,
                source_kind=EvidenceSourceKind.TOOL_RESULT,
                signal_kind=SignalKind.AGENT_ADMISSION,
                evidence_kind=EvidenceKind.VISIBLE_ADMISSION,
                corroboration_group="spoofed",
            ),
        ),
    )

    with pytest.raises(ReportInvariantError, match="evidence source does not match"):
        build_audit_result(
            _parsed(affected, admission),
            DetectionResult(episodes=(episode,)),
            run=RunProvenance(tool_version="0.8.0"),
        )


def test_unconfirmed_projection_retains_a_typed_promotion_path_but_claims_no_time() -> None:
    affected = _event(0, record_uuid="turn-affected")
    user_correction = replace(
        _event(1, actor=EventActor.HUMAN),
        source_kind=EvidenceSourceKind.USER_PROMPT,
        text="That assumption is wrong.",
    )
    candidate = Episode(
        episode_id="candidate-unconfirmed",
        category="user_correction",
        local_classification=Classification.UNCONFIRMED,
        onset_event_id=affected.event_id,
        detection_event_id=user_correction.event_id,
        recovery_end_event_id=user_correction.event_id,
        affected_event_ids=(affected.event_id, user_correction.event_id),
        evidence=(
            EvidenceRef(
                event_id=user_correction.event_id,
                source_kind=EvidenceSourceKind.USER_PROMPT,
                signal_kind=SignalKind.USER_CORRECTION,
                evidence_kind=EvidenceKind.USER_CORRECTION,
                corroboration_group="user:E1",
            ),
        ),
        affected_gap_refs=(GapRef(affected.event_id, user_correction.event_id, True, False),),
    )
    result = build_audit_result(
        _parsed(affected, user_correction),
        DetectionResult(
            unconfirmed_candidates=(candidate,),
            eligible_candidates=1,
            retained_candidates=1,
        ),
        run=RunProvenance(tool_version="0.8.0", output_format=OutputFormat.JSON),
    )

    projection = result.unconfirmed_candidates[0]
    assert isinstance(projection, UnconfirmedCandidateReport)
    assert projection.local_timing.central_avoidable_active_seconds == 10.0
    assert projection.final_timing == projection.local_timing
    assert projection.judge is None
    assert result.summary["local_estimates"].central_avoidable_active_seconds == 0.0
    wire_candidate = json.loads(render_json(result))["unconfirmed_candidates"][0]
    assert wire_candidate["excluded_from_totals"] is True
    assert "local_timing" not in wire_candidate
    assert "final_timing" not in wire_candidate


def test_tool_use_evidence_gets_a_bounded_allowlisted_derived_excerpt() -> None:
    tool_use = _event(
        0,
        kind=ContentKind.TOOL_USE,
        tool_name="Edit",
        tool_input=(("file_path", "/tmp/example.py"),),
    )
    detection = _event(1, record_uuid="turn-detection")
    episode = Episode(
        episode_id="episode-tool-evidence",
        category="incorrect_change",
        local_classification=Classification.PROBABLE,
        onset_event_id=tool_use.event_id,
        detection_event_id=detection.event_id,
        recovery_end_event_id=detection.event_id,
        affected_event_ids=(tool_use.event_id, detection.event_id),
        evidence=(
            EvidenceRef(
                event_id=tool_use.event_id,
                source_kind=EvidenceSourceKind.TOOL_USE,
                signal_kind=SignalKind.AGENT_SELF_CORRECTION,
                evidence_kind=EvidenceKind.AFFECTED_WORK,
                corroboration_group="affected:E0",
            ),
        ),
    )

    result = build_audit_result(
        _parsed(tool_use, detection),
        DetectionResult(episodes=(episode,)),
        run=RunProvenance(tool_version="0.8.0"),
    )

    citation = result.episodes[0].evidence[0]
    assert citation.excerpt == "Edit file_path=/tmp/example.py"
    assert citation.excerpt_truncated is False


def test_report_rejects_evidence_role_incompatible_with_resolved_event() -> None:
    tool_use = _event(0, kind=ContentKind.TOOL_USE, tool_name="Edit")
    detection = _event(1, record_uuid="turn-detection")
    episode = Episode(
        episode_id="episode-spoofed-role",
        category="incorrect_change",
        local_classification=Classification.CONFIRMED,
        onset_event_id=tool_use.event_id,
        detection_event_id=detection.event_id,
        recovery_end_event_id=detection.event_id,
        affected_event_ids=(tool_use.event_id, detection.event_id),
        evidence=(
            EvidenceRef(
                event_id=tool_use.event_id,
                source_kind=EvidenceSourceKind.TOOL_USE,
                signal_kind=SignalKind.AGENT_ADMISSION,
                evidence_kind=EvidenceKind.VISIBLE_ADMISSION,
                corroboration_group="spoofed-role",
            ),
        ),
    )

    with pytest.raises(ReportInvariantError, match="evidence role does not match"):
        build_audit_result(
            _parsed(tool_use, detection),
            DetectionResult(episodes=(episode,)),
            run=RunProvenance(tool_version="0.8.0"),
        )


def test_markdown_labels_heuristics_null_time_and_control_safe_evidence() -> None:
    affected = replace(_event(0, record_uuid="turn-affected"), timestamp=None)
    admission = replace(
        _event(1, record_uuid="turn-admission"),
        timestamp=None,
        text="I was wrong.\x1b[31m Printable café stays.",
    )
    episode = Episode(
        episode_id="episode-markdown",
        category="wrong_assumption",
        local_classification=Classification.CONFIRMED,
        onset_event_id=affected.event_id,
        detection_event_id=admission.event_id,
        recovery_end_event_id=admission.event_id,
        affected_event_ids=(affected.event_id, admission.event_id),
        evidence=(
            EvidenceRef(
                event_id=admission.event_id,
                source_kind=EvidenceSourceKind.ASSISTANT_TEXT,
                signal_kind=SignalKind.AGENT_ADMISSION,
                evidence_kind=EvidenceKind.VISIBLE_ADMISSION,
                corroboration_group="admission:E1",
            ),
        ),
    )
    result = build_audit_result(
        _parsed(affected, admission),
        DetectionResult(episodes=(episode,)),
        run=RunProvenance(tool_version="0.8.0"),
    )

    rendered = render_markdown(result)

    assert rendered.startswith("# Claude Session Mistake Audit\n\nStatus: `complete`")
    assert "observed time" in rendered
    assert "estimated avoidable active time" in rendered
    assert "not a confidence interval" in rendered
    assert "Unavailable (missing or invalid timestamps)" in rendered
    assert "## Confirmed and Probable Episodes" in rendered
    assert (
        "> [assistant_text / visible_admission, line 2] " "I was wrong.[31m Printable café stays."
    ) in rendered
    assert "\x1b" not in rendered
    assert rendered.endswith("\n") and not rendered.endswith("\n\n")


def test_refusal_report_has_null_totals_machine_reason_and_no_false_clearance() -> None:
    parsed = ParsedSession(
        status=AnalysisStatus.REFUSED,
        session=SessionMetadata(input_bytes=67_108_865, sha256=""),
        events=(),
        diagnostics=ParserDiagnostics(
            digest_complete=False,
            refusal_reasons=("input_byte_limit",),
        ),
    )
    result = build_audit_result(
        parsed,
        None,
        run=RunProvenance(tool_version="0.8.0", output_format=OutputFormat.JSON),
    )

    document = json.loads(render_json(result))
    markdown = render_markdown(result)

    assert result.status is AnalysisStatus.REFUSED
    assert document["summary"] == {
        "confirmed_episodes": None,
        "final_estimates": None,
        "local_estimates": None,
        "omitted_candidates": None,
        "probable_episodes": None,
        "totals_complete": False,
        "unconfirmed_candidates": None,
    }
    assert document["episodes"] == []
    assert document["unconfirmed_candidates"] == []
    assert document["diagnostics"]["parser"]["refusal_reasons"] == ["input_byte_limit"]
    assert "Analysis was refused" in markdown
    assert "Confirmed episodes: Unavailable" in markdown
    assert "Confirmed episodes: None" not in markdown
    assert "input_byte_limit" in markdown
    assert "No confirmed or probable mistakes found" not in markdown


def test_overflow_retains_episode_timing_but_nulls_aggregate_totals() -> None:
    affected = _event(0, record_uuid="turn-affected")
    detection_event = _event(1, record_uuid="turn-detection")
    episode = Episode(
        episode_id="episode-retained",
        category="wrong_assumption",
        local_classification=Classification.PROBABLE,
        onset_event_id=affected.event_id,
        detection_event_id=detection_event.event_id,
        recovery_end_event_id=detection_event.event_id,
        affected_event_ids=(affected.event_id, detection_event.event_id),
        evidence=(),
        affected_gap_refs=(GapRef(affected.event_id, detection_event.event_id, True, False),),
    )
    result = build_audit_result(
        _parsed(affected, detection_event),
        DetectionResult(
            episodes=(episode,),
            eligible_candidates=2,
            retained_candidates=1,
            omitted_candidates=1,
            diagnostics=DetectorDiagnostics(
                raw_signal_candidates=3,
                suppressed_non_mistakes=1,
                eligible_candidates=2,
                retained_candidates=1,
                omitted_candidates=1,
            ),
        ),
        run=RunProvenance(tool_version="0.8.0", output_format=OutputFormat.JSON),
    )

    document = json.loads(render_json(result))
    markdown = render_markdown(result)

    assert result.status is AnalysisStatus.PARTIAL
    assert document["summary"]["totals_complete"] is False
    assert document["summary"]["local_estimates"] is None
    assert document["summary"]["final_estimates"] is None
    assert document["summary"]["omitted_candidates"] == 1
    assert document["episodes"][0]["local_timing"]["central_avoidable_active_seconds"] == 10.0
    assert markdown.index("Candidate limits were reached") < markdown.index("## Summary")
    assert "Detector omitted candidates: 1" in markdown
    assert "Detector suppressed non-mistakes: 1" in markdown


def test_adapter_rejects_a_non_tool_event_as_a_correlated_result() -> None:
    tool_use = _event(0, kind=ContentKind.TOOL_USE, correlated_event_id="E1")
    spoofed_result = _event(1, record_uuid="turn-not-a-result", correlated_event_id="E0")
    episode = Episode(
        episode_id="episode-invalid-correlation",
        category="invalid_command",
        local_classification=Classification.PROBABLE,
        onset_event_id=tool_use.event_id,
        detection_event_id=spoofed_result.event_id,
        recovery_end_event_id=spoofed_result.event_id,
        affected_event_ids=(tool_use.event_id, spoofed_result.event_id),
        evidence=(),
    )

    with pytest.raises(ReportInvariantError, match="invalid tool correlation"):
        build_episode_timing_input(_parsed(tool_use, spoofed_result), episode)


def test_json_serialization_rejects_unsupported_values_and_nonfinite_floats() -> None:
    result = build_audit_result(
        _parsed(),
        DetectionResult(),
        run=RunProvenance(tool_version="0.8.0", output_format=OutputFormat.JSON),
    )

    with pytest.raises(ReportSerializationError, match="unsupported report value"):
        render_json(replace(result, summary={"unsupported": object()}))
    with pytest.raises(ReportSerializationError, match="non-finite floats"):
        render_json(replace(result, summary={"invalid_number": float("nan")}))


def test_evidence_excerpt_is_re_redacted_bounded_and_marked_truncated() -> None:
    affected = _event(0, record_uuid="turn-affected")
    admission = replace(
        _event(1, record_uuid="turn-admission"),
        text="sk-12345678 " + "x" * 600,
    )
    episode = Episode(
        episode_id="episode-bounded-evidence",
        category="wrong_assumption",
        local_classification=Classification.CONFIRMED,
        onset_event_id=affected.event_id,
        detection_event_id=admission.event_id,
        recovery_end_event_id=admission.event_id,
        affected_event_ids=(affected.event_id, admission.event_id),
        evidence=(
            EvidenceRef(
                event_id=admission.event_id,
                source_kind=EvidenceSourceKind.ASSISTANT_TEXT,
                signal_kind=SignalKind.AGENT_ADMISSION,
                evidence_kind=EvidenceKind.VISIBLE_ADMISSION,
                corroboration_group="admission:E1",
            ),
        ),
    )
    result = build_audit_result(
        _parsed(affected, admission),
        DetectionResult(episodes=(episode,)),
        run=RunProvenance(tool_version="0.8.0", output_format=OutputFormat.JSON),
    )

    rendered = render_json(result)
    citation = json.loads(rendered)["episodes"][0]["evidence"][0]

    assert "sk-12345678" not in rendered
    assert citation["excerpt"].startswith("<redacted-secret>")
    assert len(citation["excerpt"]) == MAX_REPORT_EXCERPT_CHARS
    assert citation["excerpt_truncated"] is True


def test_markdown_explicitly_labels_thinking_evidence() -> None:
    affected = _event(0, record_uuid="turn-affected")
    thinking = replace(
        _event(1, record_uuid="turn-thinking"),
        kind=ContentKind.THINKING,
        source_kind=EvidenceSourceKind.ASSISTANT_THINKING,
        text="I used the wrong premise.",
    )
    episode = Episode(
        episode_id="episode-thinking",
        category="wrong_assumption",
        local_classification=Classification.PROBABLE,
        onset_event_id=affected.event_id,
        detection_event_id=thinking.event_id,
        recovery_end_event_id=thinking.event_id,
        affected_event_ids=(affected.event_id, thinking.event_id),
        evidence=(
            EvidenceRef(
                event_id=thinking.event_id,
                source_kind=EvidenceSourceKind.ASSISTANT_THINKING,
                signal_kind=SignalKind.AGENT_SELF_CORRECTION,
                evidence_kind=EvidenceKind.THINKING_ADMISSION,
                corroboration_group="thinking:E1",
            ),
        ),
    )
    result = build_audit_result(
        _parsed(affected, thinking),
        DetectionResult(episodes=(episode,)),
        run=RunProvenance(tool_version="0.8.0"),
    )

    markdown = render_markdown(result)

    assert "[assistant_thinking / thinking_admission, line 2] (thinking evidence)" in markdown


def test_adapter_preserves_detector_gap_exclusion_flags_without_reinterpreting_them() -> None:
    onset = _event(0, record_uuid="branch-a")
    middle = _event(1, record_uuid="branch-b")
    recovery = _event(2, record_uuid="branch-a")
    episode = Episode(
        episode_id="episode-excluded-gaps",
        category="wrong_assumption",
        local_classification=Classification.PROBABLE,
        onset_event_id=onset.event_id,
        detection_event_id=middle.event_id,
        recovery_end_event_id=recovery.event_id,
        affected_event_ids=(onset.event_id, middle.event_id, recovery.event_id),
        evidence=(),
        affected_gap_refs=(GapRef(onset.event_id, middle.event_id, False, False),),
        ambiguous_gap_refs=(GapRef(middle.event_id, recovery.event_id, True, True),),
    )

    timing_input = build_episode_timing_input(_parsed(onset, middle, recovery), episode)

    assert timing_input.affected_gaps[0].same_lineage is False
    assert timing_input.affected_gaps[0].crosses_human_boundary is False
    assert timing_input.ambiguous_recovery_gaps[0].same_lineage is True
    assert timing_input.ambiguous_recovery_gaps[0].crosses_human_boundary is True
