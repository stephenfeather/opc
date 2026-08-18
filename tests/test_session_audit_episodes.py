"""Lineage-safe episode construction, merging, and overflow tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from scripts.core.session_audit import detector as detector_module
from scripts.core.session_audit.detector import DetectionPolicy, detect_mistakes
from scripts.core.session_audit.models import (
    AnalysisStatus,
    Classification,
    ContentKind,
    EventActor,
    EvidenceKind,
    EvidenceSourceKind,
    NormalizedEvent,
    ParsedSession,
    ParserDiagnostics,
    SessionMetadata,
)
from scripts.core.session_audit.parser import events_share_lineage as parser_events_share_lineage

BASE = datetime(2026, 8, 11, 13, 0, tzinfo=UTC)


def _event(
    index: int,
    *,
    text: str = "",
    actor: EventActor = EventActor.ASSISTANT,
    kind: ContentKind = ContentKind.VISIBLE_TEXT,
    source_kind: EvidenceSourceKind = EvidenceSourceKind.ASSISTANT_TEXT,
    record_uuid: str,
    ancestry_start: int,
    ancestry_end: int,
    tool_use_id: str | None = None,
    tool_name: str | None = None,
    tool_input: tuple[tuple[str, str], ...] = (),
    tool_result_is_error: bool | None = None,
    correlated_event_id: str | None = None,
) -> NormalizedEvent:
    return NormalizedEvent(
        event_id=f"E{index}",
        chronological_index=index,
        source_line=index + 1,
        content_block_index=0,
        timestamp=BASE + timedelta(seconds=index),
        actor=actor,
        kind=kind,
        source_kind=source_kind,
        text=text,
        record_uuid=record_uuid,
        lineage_root_uuid="root",
        ancestry_start=ancestry_start,
        ancestry_end=ancestry_end,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_result_is_error=tool_result_is_error,
        correlated_event_id=correlated_event_id,
    )


def _parsed(*events: NormalizedEvent) -> ParsedSession:
    return ParsedSession(
        status=AnalysisStatus.COMPLETE,
        session=SessionMetadata(input_bytes=0, sha256="b" * 64),
        events=events,
        diagnostics=ParserDiagnostics(normalized_events=len(events)),
    )


def test_siblings_do_not_enter_or_consume_a_lineage_window_budget() -> None:
    branch_edit = _event(
        0,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
    )
    sibling_one = _event(
        1,
        text="sibling context one",
        record_uuid="branch-b",
        ancestry_start=2,
        ancestry_end=3,
    )
    sibling_two = _event(
        2,
        text="sibling context two",
        record_uuid="branch-b",
        ancestry_start=2,
        ancestry_end=3,
    )
    admission = _event(
        3,
        text="I was wrong about that edit.",
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
    )

    result = detect_mistakes(
        _parsed(branch_edit, sibling_one, sibling_two, admission),
        policy=DetectionPolicy(backward_window_events=2, forward_window_events=2),
    )

    assert result.episodes[0].local_classification is Classification.CONFIRMED
    assert result.episodes[0].context_window_event_ids == (
        branch_edit.event_id,
        admission.event_id,
    )


def test_many_incompatible_siblings_cannot_hide_an_explicit_compatible_action() -> None:
    branch_edit = _event(
        0,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    siblings = tuple(
        _event(
            index,
            text=f"sibling context {index}",
            record_uuid=f"branch-b-{index}",
            ancestry_start=2,
            ancestry_end=3,
        )
        for index in range(1, 514)
    )
    admission = _event(
        514,
        text="I was wrong about /src/a.py.",
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
    )

    result = detect_mistakes(
        _parsed(branch_edit, *siblings, admission),
        policy=DetectionPolicy(backward_window_events=1, forward_window_events=1),
    )

    assert len(result.episodes) == 1
    assert result.episodes[0].onset_event_id == branch_edit.event_id
    assert result.episodes[0].context_window_event_ids == (
        branch_edit.event_id,
        admission.event_id,
    )


def test_siblings_do_not_consume_objective_chain_window_budget() -> None:
    root_edit = _event(
        0,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="wrong-edit",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E1",
    )
    root_result = _event(
        1,
        actor=EventActor.TOOL,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="wrong-edit",
        tool_result_is_error=False,
        correlated_event_id="E0",
    )
    failed_use = _event(
        2,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="failed-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E5",
    )
    sibling_before_one = _event(
        3,
        record_uuid="branch-b",
        ancestry_start=2,
        ancestry_end=3,
        text="sibling before one",
    )
    sibling_before_two = _event(
        4,
        record_uuid="branch-b",
        ancestry_start=2,
        ancestry_end=3,
        text="sibling before two",
    )
    contradiction = _event(
        5,
        text="AssertionError in /src/a.py",
        actor=EventActor.TOOL,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="failed-test",
        tool_result_is_error=True,
        correlated_event_id="E2",
    )
    sibling_after_one = _event(
        6,
        record_uuid="branch-b",
        ancestry_start=2,
        ancestry_end=3,
        text="sibling after one",
    )
    correction = _event(
        7,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="correction",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E8",
    )
    correction_result = _event(
        8,
        actor=EventActor.TOOL,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="correction",
        tool_result_is_error=False,
        correlated_event_id="E7",
    )
    sibling_after_two = _event(
        9,
        record_uuid="branch-b",
        ancestry_start=2,
        ancestry_end=3,
        text="sibling after two",
    )
    recovered_use = _event(
        10,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="recovered-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E11",
    )
    recovery = _event(
        11,
        text="1 passed",
        actor=EventActor.TOOL,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="recovered-test",
        tool_result_is_error=False,
        correlated_event_id="E10",
    )

    result = detect_mistakes(
        _parsed(
            root_edit,
            root_result,
            failed_use,
            sibling_before_one,
            sibling_before_two,
            contradiction,
            sibling_after_one,
            correction,
            correction_result,
            sibling_after_two,
            recovered_use,
            recovery,
        ),
        policy=DetectionPolicy(backward_window_events=3, forward_window_events=4),
    )

    assert len(result.episodes) == 1
    assert result.episodes[0].objective_chains[0].root_event_id == root_edit.event_id
    assert result.episodes[0].recovery_end_event_id == recovery.event_id


def test_sibling_candidates_sharing_an_ancestor_do_not_merge_or_corroborate() -> None:
    shared_root_edit = _event(
        0,
        record_uuid="root",
        ancestry_start=0,
        ancestry_end=3,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="root-edit",
        tool_name="Edit",
    )
    branch_a_claim = _event(
        1,
        text="You were wrong about the root edit.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
        record_uuid="branch-a",
        ancestry_start=1,
        ancestry_end=2,
    )
    branch_b_admission = _event(
        2,
        text="I was wrong about the root edit.",
        record_uuid="branch-b",
        ancestry_start=2,
        ancestry_end=3,
    )

    result = detect_mistakes(_parsed(shared_root_edit, branch_a_claim, branch_b_admission))

    assert result.eligible_candidates == 2
    assert len(result.episodes) == 1
    assert len(result.unconfirmed_candidates) == 1
    assert result.episodes[0].detection_event_id == branch_b_admission.event_id
    assert result.unconfirmed_candidates[0].detection_event_id == branch_a_claim.event_id
    assert branch_a_claim.event_id not in result.unconfirmed_candidates[0].affected_event_ids


def test_overlapping_windows_with_distinct_causal_artifacts_remain_separate() -> None:
    edit_a = _event(
        0,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    edit_b = _event(
        1,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-b",
        tool_name="Edit",
        tool_input=(("file_path", "/src/b.py"),),
    )
    admission_a = _event(
        2,
        text="I was wrong about the /src/a.py edit.",
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
    )
    admission_b = _event(
        3,
        text="I was wrong about the /src/b.py edit.",
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
    )

    result = detect_mistakes(_parsed(edit_a, edit_b, admission_a, admission_b))

    assert len(result.episodes) == 2
    assert {episode.onset_event_id for episode in result.episodes} == {
        edit_a.event_id,
        edit_b.event_id,
    }


def test_direct_retry_chain_and_admission_merge_by_shared_affected_event() -> None:
    failed_use = _event(
        0,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="bad-command",
        tool_name="Bash",
        tool_input=(("command", "pytest --bad-option tests/test_a.py"),),
        correlated_event_id="E1",
    )
    failure = _event(
        1,
        text="pytest: error: unrecognized arguments: --bad-option",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        tool_use_id="bad-command",
        tool_result_is_error=True,
        correlated_event_id="E0",
    )
    retry = _event(
        2,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="good-command",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py -q"),),
        correlated_event_id="E3",
    )
    success = _event(
        3,
        text="1 passed",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        tool_use_id="good-command",
        tool_result_is_error=False,
        correlated_event_id="E2",
    )
    admission = _event(
        4,
        text="I should have checked pytest tests/test_a.py -q.",
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
    )

    retry_only = detect_mistakes(_parsed(failed_use, failure, retry, success))

    assert retry_only.episodes == ()
    assert len(retry_only.unconfirmed_candidates) == 1
    assert retry_only.unconfirmed_candidates[0].local_classification is Classification.UNCONFIRMED

    result = detect_mistakes(_parsed(failed_use, failure, retry, success, admission))

    assert result.eligible_candidates == 2
    assert len(result.episodes) == 1
    assert result.unconfirmed_candidates == ()
    episode = result.episodes[0]
    assert episode.local_classification is Classification.CONFIRMED
    assert episode.onset_event_id == failed_use.event_id
    assert episode.retry_event_ids == (retry.event_id,)


def test_wrong_command_admission_merges_with_unique_completed_retry_chain() -> None:
    failed_use = _event(
        0,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="bad-overlay-command",
        tool_name="Bash",
        tool_input=(("command", "overlay base --restore"),),
        correlated_event_id="E1",
    )
    failure = _event(
        1,
        text="error: pilot profile required",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        tool_use_id="bad-overlay-command",
        tool_result_is_error=True,
        correlated_event_id="E0",
    )
    retry = _event(
        2,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="good-overlay-command",
        tool_name="Bash",
        tool_input=(("command", "overlay base --pilot pilot --restore"),),
        correlated_event_id="E3",
    )
    success = _event(
        3,
        text="overlay restored",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        tool_use_id="good-overlay-command",
        tool_result_is_error=False,
        correlated_event_id="E2",
    )
    admission = _event(
        4,
        text=(
            "Correction first: the overlay isn't broken — I gave you the wrong command. "
            "It's designed for base + pilot + restore, and I dropped the pilot flag."
        ),
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
    )

    result = detect_mistakes(_parsed(failed_use, failure, retry, success, admission))

    assert result.eligible_candidates == 2
    assert len(result.episodes) == 1
    assert result.unconfirmed_candidates == ()
    episode = result.episodes[0]
    assert episode.local_classification is Classification.CONFIRMED
    assert episode.onset_event_id == failed_use.event_id
    assert episode.retry_event_ids == (retry.event_id,)
    assert any(
        evidence.event_id == admission.event_id
        and evidence.evidence_kind is EvidenceKind.VISIBLE_ADMISSION
        for evidence in episode.evidence
    )


def test_retained_bridge_candidate_merges_all_causal_components() -> None:
    root_edit = _event(
        0,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="root-edit",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E1",
    )
    root_result = _event(
        1,
        actor=EventActor.TOOL,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="root-edit",
        tool_result_is_error=False,
        correlated_event_id="E0",
    )
    first_correction = _event(
        2,
        text="You were wrong about the /src/a.py edit.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
    )
    failed_use = _event(
        3,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="failed-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E5",
    )
    second_correction = _event(
        4,
        text="You were wrong about pytest tests/test_a.py.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
    )
    contradiction = _event(
        5,
        text="AssertionError in /src/a.py",
        actor=EventActor.TOOL,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="failed-test",
        tool_result_is_error=True,
        correlated_event_id="E3",
    )
    corrective_edit = _event(
        6,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="corrective-edit",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E7",
    )
    corrective_result = _event(
        7,
        actor=EventActor.TOOL,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="corrective-edit",
        tool_result_is_error=False,
        correlated_event_id="E6",
    )
    recovered_use = _event(
        8,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="recovered-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E9",
    )
    recovery = _event(
        9,
        actor=EventActor.TOOL,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="recovered-test",
        tool_result_is_error=False,
        correlated_event_id="E8",
    )

    result = detect_mistakes(
        _parsed(
            root_edit,
            root_result,
            first_correction,
            failed_use,
            second_correction,
            contradiction,
            corrective_edit,
            corrective_result,
            recovered_use,
            recovery,
        )
    )

    assert result.eligible_candidates == 3
    assert len(result.episodes) == 1
    assert result.unconfirmed_candidates == ()
    assert {item.event_id for item in result.episodes[0].evidence} >= {
        first_correction.event_id,
        second_correction.event_id,
        contradiction.event_id,
    }


def test_overflow_retains_strongest_candidates_and_reports_exact_counts() -> None:
    events: list[NormalizedEvent] = []
    for pair_index, path in enumerate(("/src/a.py", "/src/b.py", "/src/c.py", "/src/d.py")):
        edit_index = pair_index * 2
        events.append(
            _event(
                edit_index,
                record_uuid="main",
                ancestry_start=0,
                ancestry_end=1,
                kind=ContentKind.TOOL_USE,
                source_kind=EvidenceSourceKind.TOOL_USE,
                tool_use_id=f"edit-{pair_index}",
                tool_name="Edit",
                tool_input=(("file_path", path),),
            )
        )
        if pair_index < 2:
            events.append(
                _event(
                    edit_index + 1,
                    text=f"You were wrong about the {path} edit.",
                    actor=EventActor.HUMAN,
                    source_kind=EvidenceSourceKind.USER_PROMPT,
                    record_uuid="main",
                    ancestry_start=0,
                    ancestry_end=1,
                )
            )
        else:
            events.append(
                _event(
                    edit_index + 1,
                    text=f"I was wrong about the {path} edit.",
                    record_uuid="main",
                    ancestry_start=0,
                    ancestry_end=1,
                )
            )

    result = detect_mistakes(
        _parsed(*events),
        policy=DetectionPolicy(max_candidates=2),
    )

    assert result.eligible_candidates == 4
    assert result.retained_candidates == 2
    assert result.omitted_candidates == 2
    assert result.overflowed is True
    assert result.candidates_complete is False
    assert len(result.episodes) == 2
    assert result.unconfirmed_candidates == ()
    assert {episode.detection_event_id for episode in result.episodes} == {"E5", "E7"}


def test_overflow_prefers_confirmed_objective_chain_to_unlinked_admission() -> None:
    unlinked_admission = _event(
        0,
        text="I was wrong about an unavailable earlier detail.",
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
    )
    wrong_edit = _event(
        1,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="wrong-edit",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E2",
    )
    root_result = _event(
        2,
        actor=EventActor.TOOL,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="wrong-edit",
        tool_result_is_error=False,
        correlated_event_id="E1",
    )
    failed_use = _event(
        3,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="failed-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E4",
    )
    contradiction = _event(
        4,
        text="AssertionError in /src/a.py",
        actor=EventActor.TOOL,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="failed-test",
        tool_result_is_error=True,
        correlated_event_id="E3",
    )
    correction = _event(
        5,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="correction",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E6",
    )
    correction_result = _event(
        6,
        actor=EventActor.TOOL,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="correction",
        tool_result_is_error=False,
        correlated_event_id="E5",
    )
    recovered_use = _event(
        7,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="recovered-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E8",
    )
    recovery = _event(
        8,
        actor=EventActor.TOOL,
        record_uuid="main",
        ancestry_start=0,
        ancestry_end=1,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="recovered-test",
        tool_result_is_error=False,
        correlated_event_id="E7",
    )

    result = detect_mistakes(
        _parsed(
            unlinked_admission,
            wrong_edit,
            root_result,
            failed_use,
            contradiction,
            correction,
            correction_result,
            recovered_use,
            recovery,
        ),
        policy=DetectionPolicy(max_candidates=1),
    )

    assert result.eligible_candidates == 2
    assert result.retained_candidates == 1
    assert result.omitted_candidates == 1
    assert len(result.episodes) == 1
    assert result.episodes[0].detection_event_id == contradiction.event_id
    assert result.unconfirmed_candidates == ()


def test_overflow_ties_use_stable_event_ids_independent_of_input_order() -> None:
    claims = tuple(
        replace(
            _event(
                10,
                text="You were wrong about that edit.",
                actor=EventActor.HUMAN,
                source_kind=EvidenceSourceKind.USER_PROMPT,
                record_uuid="main",
                ancestry_start=0,
                ancestry_end=1,
            ),
            event_id=event_id,
            chronological_index=10,
        )
        for event_id in ("candidate-c", "candidate-a", "candidate-b")
    )

    forward = detect_mistakes(_parsed(*claims), policy=DetectionPolicy(max_candidates=2))
    reversed_input = detect_mistakes(
        _parsed(*reversed(claims)), policy=DetectionPolicy(max_candidates=2)
    )

    forward_ids = tuple(
        candidate.detection_event_id for candidate in forward.unconfirmed_candidates
    )
    reversed_ids = tuple(
        candidate.detection_event_id for candidate in reversed_input.unconfirmed_candidates
    )
    assert forward_ids == reversed_ids == ("candidate-a", "candidate-b")


def test_suppressed_non_mistakes_do_not_count_toward_overflow() -> None:
    eligible_claims = (
        _event(
            0,
            text="You were wrong about that edit.",
            actor=EventActor.HUMAN,
            source_kind=EvidenceSourceKind.USER_PROMPT,
            record_uuid="main",
            ancestry_start=0,
            ancestry_end=1,
        ),
        _event(
            1,
            text="That is incorrect.",
            actor=EventActor.HUMAN,
            source_kind=EvidenceSourceKind.USER_PROMPT,
            record_uuid="main",
            ancestry_start=0,
            ancestry_end=1,
        ),
    )
    suppressed = (
        _event(
            2,
            text="Sorry about the delay.",
            record_uuid="main",
            ancestry_start=0,
            ancestry_end=1,
        ),
        _event(
            3,
            text="If I was wrong about it, I would correct it.",
            record_uuid="main",
            ancestry_start=0,
            ancestry_end=1,
        ),
        _event(
            4,
            text='The fixture contains "I was wrong about it".',
            record_uuid="main",
            ancestry_start=0,
            ancestry_end=1,
        ),
    )

    result = detect_mistakes(
        _parsed(*eligible_claims, *suppressed),
        policy=DetectionPolicy(max_candidates=2),
    )

    assert result.eligible_candidates == 2
    assert result.retained_candidates == 2
    assert result.omitted_candidates == 0
    assert result.overflowed is False
    assert result.diagnostics.suppressed_non_mistakes == 3
    assert result.diagnostics.raw_signal_candidates == 5


def test_sparse_anchor_detection_reuses_exact_lineage_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage_comparisons = 0

    def counted_events_share_lineage(
        first: NormalizedEvent,
        second: NormalizedEvent,
    ) -> bool:
        nonlocal lineage_comparisons
        lineage_comparisons += 1
        return parser_events_share_lineage(first, second)

    monkeypatch.setattr(
        detector_module,
        "events_share_lineage",
        counted_events_share_lineage,
    )
    events: list[NormalizedEvent] = []
    anchor_by_position = {anchor * 50: anchor for anchor in range(500)}
    for index in range(25_000):
        anchor = anchor_by_position.get(index)
        preceding_anchor = anchor_by_position.get(index - 1)
        if anchor is not None:
            events.append(
                _event(
                    index,
                    record_uuid=f"anchor-{anchor}",
                    ancestry_start=anchor * 2,
                    ancestry_end=anchor * 2 + 1,
                    kind=ContentKind.TOOL_USE,
                    source_kind=EvidenceSourceKind.TOOL_USE,
                    tool_use_id=f"edit-{anchor}",
                    tool_name="Edit",
                    tool_input=(("file_path", f"/src/file-{anchor}.py"),),
                )
            )
        elif preceding_anchor is not None:
            events.append(
                _event(
                    index,
                    text=f"I was wrong about /src/file-{preceding_anchor}.py.",
                    record_uuid=f"anchor-{preceding_anchor}",
                    ancestry_start=preceding_anchor * 2,
                    ancestry_end=preceding_anchor * 2 + 1,
                )
            )
        else:
            filler_start = 1_000_000 + index * 2
            events.append(
                _event(
                    index,
                    text=f"unrelated sibling {index}",
                    record_uuid=f"filler-{index}",
                    ancestry_start=filler_start,
                    ancestry_end=filler_start + 1,
                )
            )

    result = detect_mistakes(
        _parsed(*events),
        policy=DetectionPolicy(max_candidates=500),
    )

    assert result.eligible_candidates == 500
    assert len(result.episodes) == 500
    assert result.unconfirmed_candidates == ()
    # One public-lineage check constructs each episode gap; window lookup adds none.
    assert lineage_comparisons == 500
