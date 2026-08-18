"""High-precision deterministic session mistake detection tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

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

BASE = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def _event(
    index: int,
    *,
    text: str = "",
    actor: EventActor = EventActor.ASSISTANT,
    kind: ContentKind = ContentKind.VISIBLE_TEXT,
    source_kind: EvidenceSourceKind = EvidenceSourceKind.ASSISTANT_TEXT,
    record_uuid: str = "main",
    lineage_root_uuid: str = "main",
    ancestry_start: int = 0,
    ancestry_end: int = 1,
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
        lineage_root_uuid=lineage_root_uuid,
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
        session=SessionMetadata(input_bytes=0, sha256="a" * 64),
        events=events,
        diagnostics=ParserDiagnostics(normalized_events=len(events)),
    )


def _nonrevert_objective_events(
    *,
    include_root_result: bool = True,
    root_result_is_error: bool = False,
    failure_text: str = "AssertionError in /src/a.py",
) -> tuple[NormalizedEvent, ...]:
    root = _event(
        0,
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
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="root-edit",
        tool_result_is_error=root_result_is_error,
        correlated_event_id="E0",
    )
    failed_use = _event(
        2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="failed-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E3",
    )
    contradiction = _event(
        3,
        text=failure_text,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="failed-test",
        tool_result_is_error=True,
        correlated_event_id="E2",
    )
    correction = _event(
        4,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="correction",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E5",
    )
    correction_result = _event(
        5,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="correction",
        tool_result_is_error=False,
        correlated_event_id="E4",
    )
    recovered_use = _event(
        6,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="recovered-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E7",
    )
    recovery = _event(
        7,
        text="1 passed",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="recovered-test",
        tool_result_is_error=False,
        correlated_event_id="E6",
    )
    root_events = (root, root_result) if include_root_result else (root,)
    return (
        *root_events,
        failed_use,
        contradiction,
        correction,
        correction_result,
        recovered_use,
        recovery,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_candidates": 0},
        {"backward_window_events": -1},
        {"forward_window_events": -1},
    ],
)
def test_detection_policy_rejects_nonpositive_or_negative_bounds(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        DetectionPolicy(**kwargs)


def test_objective_chain_requires_a_correlated_root_completion() -> None:
    results = (
        detect_mistakes(_parsed(*_nonrevert_objective_events(include_root_result=False))),
        detect_mistakes(_parsed(*_nonrevert_objective_events(root_result_is_error=True))),
    )

    for result in results:
        assert result.episodes == ()
        assert not any(candidate.objective_chains for candidate in result.unconfirmed_candidates)


def test_objective_root_and_completion_must_precede_validation_use() -> None:
    failed_use = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="failed-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E3",
    )
    late_mutation = _event(
        1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="late-edit",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E2",
    )
    late_mutation_result = _event(
        2,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="late-edit",
        tool_result_is_error=False,
        correlated_event_id="E1",
    )
    contradiction = _event(
        3,
        text="AssertionError in /src/a.py",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="failed-test",
        tool_result_is_error=True,
        correlated_event_id="E0",
    )
    correction = _event(
        4,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="correction",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E5",
    )
    correction_result = _event(
        5,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="correction",
        tool_result_is_error=False,
        correlated_event_id="E4",
    )
    recovered_use = _event(
        6,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="recovered-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E7",
    )
    recovery = _event(
        7,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="recovered-test",
        tool_result_is_error=False,
        correlated_event_id="E6",
    )

    result = detect_mistakes(
        _parsed(
            failed_use,
            late_mutation,
            late_mutation_result,
            contradiction,
            correction,
            correction_result,
            recovered_use,
            recovery,
        )
    )

    assert result.episodes == ()
    assert not any(candidate.objective_chains for candidate in result.unconfirmed_candidates)


def test_nonrevert_objective_failure_must_name_the_root_artifact() -> None:
    result = detect_mistakes(
        _parsed(*_nonrevert_objective_events(failure_text="AssertionError in tests/test_b.py"))
    )

    assert result.episodes == ()
    assert not any(candidate.objective_chains for candidate in result.unconfirmed_candidates)


@pytest.mark.parametrize(
    ("failure_reference", "expected_confirmed"),
    [
        ("/src/a.py", True),
        ("/src/a.py:42", True),
        ("(/src/a.py)", True),
        ("a.py", True),
        ("a.py:42", True),
        ("/src/a.py.bak", False),
        ("/src/a.py-old", False),
        ("/src/a.py.tmp", False),
    ],
)
def test_objective_failure_selector_requires_a_whole_artifact_token(
    failure_reference: str,
    expected_confirmed: bool,
) -> None:
    result = detect_mistakes(
        _parsed(
            *_nonrevert_objective_events(failure_text=f"AssertionError in {failure_reference}.")
        )
    )

    if expected_confirmed:
        assert len(result.episodes) == 1
        assert len(result.episodes[0].objective_chains) == 1
    else:
        assert result.episodes == ()
        assert not any(candidate.objective_chains for candidate in result.unconfirmed_candidates)


def test_correction_completion_must_precede_recovery_validation_use() -> None:
    root = _event(
        0,
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
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="root-edit",
        tool_result_is_error=False,
        correlated_event_id="E0",
    )
    failed_use = _event(
        2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="failed-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E3",
    )
    contradiction = _event(
        3,
        text="AssertionError in /src/a.py",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="failed-test",
        tool_result_is_error=True,
        correlated_event_id="E2",
    )
    correction = _event(
        4,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="correction",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
        correlated_event_id="E6",
    )
    premature_recovery_use = _event(
        5,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="recovered-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E7",
    )
    late_correction_result = _event(
        6,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="correction",
        tool_result_is_error=False,
        correlated_event_id="E4",
    )
    recovery = _event(
        7,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="recovered-test",
        tool_result_is_error=False,
        correlated_event_id="E5",
    )

    result = detect_mistakes(
        _parsed(
            root,
            root_result,
            failed_use,
            contradiction,
            correction,
            premature_recovery_use,
            late_correction_result,
            recovery,
        )
    )

    assert result.episodes == ()
    assert not any(candidate.objective_chains for candidate in result.unconfirmed_candidates)


def test_objective_chain_rejects_incompatible_correlated_completion_lineage() -> None:
    events = _nonrevert_objective_events()
    correction_result = events[5]
    incompatible_result = replace(
        correction_result,
        record_uuid="sibling",
        ancestry_start=2,
        ancestry_end=3,
    )

    result = detect_mistakes(_parsed(*events[:5], incompatible_result, *events[6:]))

    assert result.episodes == ()
    assert not any(candidate.objective_chains for candidate in result.unconfirmed_candidates)


def test_linked_visible_admission_is_confirmed_with_traceable_boundaries() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
        tool_input=(("file_path", "/src/parser.py"),),
    )
    admission = _event(1, text="I was wrong about that parser edit.")

    result = detect_mistakes(_parsed(prior_edit, admission))

    assert result.eligible_candidates == 1
    assert len(result.episodes) == 1
    assert result.unconfirmed_candidates == ()
    episode = result.episodes[0]
    assert episode.local_classification is Classification.CONFIRMED
    assert episode.onset_event_id == prior_edit.event_id
    assert episode.detection_event_id == admission.event_id
    assert episode.recovery_end_event_id is None
    assert episode.affected_event_ids == (prior_edit.event_id, admission.event_id)
    assert any(
        evidence.event_id == admission.event_id
        and evidence.evidence_kind is EvidenceKind.VISIBLE_ADMISSION
        for evidence in episode.evidence
    )


@pytest.mark.parametrize(
    ("actor", "kind", "source_kind"),
    [
        (EventActor.HUMAN, ContentKind.VISIBLE_TEXT, EvidenceSourceKind.USER_PROMPT),
        (EventActor.SYSTEM, ContentKind.VISIBLE_TEXT, EvidenceSourceKind.DERIVED),
        (
            EventActor.ASSISTANT,
            ContentKind.THINKING,
            EvidenceSourceKind.ASSISTANT_TEXT,
        ),
        (
            EventActor.ASSISTANT,
            ContentKind.VISIBLE_TEXT,
            EvidenceSourceKind.ASSISTANT_THINKING,
        ),
    ],
)
def test_admission_signal_requires_exact_actor_kind_and_source_role(
    actor: EventActor,
    kind: ContentKind,
    source_kind: EvidenceSourceKind,
) -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    invalid_role = _event(
        1,
        text="I was wrong about /src/a.py.",
        actor=actor,
        kind=kind,
        source_kind=source_kind,
    )

    result = detect_mistakes(_parsed(prior_edit, invalid_role))

    assert result.eligible_candidates == 0
    assert result.episodes == ()
    assert result.unconfirmed_candidates == ()


def test_apology_does_not_suppress_a_specific_linked_admission() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )
    admission = _event(1, text="Sorry about that. I was wrong about the parser edit.")

    result = detect_mistakes(_parsed(prior_edit, admission))

    assert len(result.episodes) == 1
    assert result.episodes[0].local_classification is Classification.CONFIRMED


def test_same_clause_apology_can_prefix_a_specific_admission() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )
    admission = _event(1, text="Sorry, I was wrong about that edit.")

    result = detect_mistakes(_parsed(prior_edit, admission))

    assert len(result.episodes) == 1
    assert result.episodes[0].local_classification is Classification.CONFIRMED


def test_em_dash_discourse_prefix_preserves_a_specific_admission() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )
    admission = _event(1, text="You're right — I was wrong about that edit.")

    result = detect_mistakes(_parsed(prior_edit, admission))

    assert len(result.episodes) == 1
    assert result.episodes[0].local_classification is Classification.CONFIRMED


@pytest.mark.parametrize(
    ("text", "expected_confirmed"),
    [
        ("You're right—I was wrong about that edit.", True),
        ("You're right–I was wrong about that edit.", True),
        ("Sorry—I was wrong about that edit.", True),
        ("You're rightI was wrong about that edit.", False),
    ],
)
def test_discourse_prefix_accepts_tight_unicode_dashes_but_not_concatenated_words(
    text: str,
    expected_confirmed: bool,
) -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )

    result = detect_mistakes(_parsed(prior_edit, _event(1, text=text)))

    if expected_confirmed:
        assert len(result.episodes) == 1
        assert result.episodes[0].onset_event_id == prior_edit.event_id
    else:
        assert result.eligible_candidates == 0
        assert result.episodes == ()
        assert result.unconfirmed_candidates == ()


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("Edit", (("file_path", "/src/a.py"),)),
        ("Bash", (("command", "pwd"),)),
    ],
)
def test_bare_admission_does_not_infer_an_arbitrary_prior_action(
    tool_name: str,
    tool_input: tuple[tuple[str, str], ...],
) -> None:
    prior_action = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="action-1",
        tool_name=tool_name,
        tool_input=tool_input,
    )
    admission = _event(1, text="I was wrong.")

    result = detect_mistakes(_parsed(prior_action, admission))

    assert result.episodes == ()
    assert len(result.unconfirmed_candidates) == 1
    candidate = result.unconfirmed_candidates[0]
    assert candidate.onset_event_id is None
    assert candidate.affected_event_ids == (admission.event_id,)


def test_unique_normalized_artifact_reference_links_the_named_action() -> None:
    edit_a = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    edit_b = _event(
        1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-b",
        tool_name="Edit",
        tool_input=(("file_path", "/src/b.py"),),
    )
    admission = _event(2, text="I was wrong about a.py.")

    result = detect_mistakes(_parsed(edit_a, edit_b, admission))

    assert len(result.episodes) == 1
    assert result.episodes[0].onset_event_id == edit_a.event_id
    assert edit_b.event_id not in result.episodes[0].affected_event_ids


@pytest.mark.parametrize(
    ("reference", "expected_confirmed"),
    [
        ("/src/a.py", True),
        ("/src/a.py:42", True),
        ("(/src/a.py)", True),
        ("/src/a.py during validation", True),
        ("a.py", True),
        ("a.py:42", True),
        ("a.py)", True),
        ("/src/a.py.bak", False),
        ("/src/a.py-old", False),
        ("/src/a.py.tmp", False),
    ],
)
def test_admission_selector_matching_requires_a_whole_artifact_token(
    reference: str,
    expected_confirmed: bool,
) -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    admission = _event(1, text=f"I was wrong about {reference}.")

    result = detect_mistakes(_parsed(prior_edit, admission))

    if expected_confirmed:
        assert len(result.episodes) == 1
        assert result.episodes[0].onset_event_id == prior_edit.event_id
        assert result.unconfirmed_candidates == ()
    else:
        assert result.episodes == ()
        assert len(result.unconfirmed_candidates) == 1
        assert result.unconfirmed_candidates[0].onset_event_id is None


def test_deictic_admission_with_two_recent_actions_remains_unconfirmed() -> None:
    edit_a = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    edit_b = _event(
        1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-b",
        tool_name="Edit",
        tool_input=(("file_path", "/src/b.py"),),
    )
    admission = _event(2, text="I was wrong about that edit.")

    result = detect_mistakes(_parsed(edit_a, edit_b, admission))

    assert result.episodes == ()
    assert len(result.unconfirmed_candidates) == 1
    assert result.unconfirmed_candidates[0].onset_event_id is None


@pytest.mark.parametrize(
    ("noun", "tool_name", "tool_input", "expected_confirmed"),
    [
        ("edit", "Edit", (("file_path", "/src/a.py"),), True),
        ("write", "Write", (("file_path", "/src/a.py"),), True),
        ("patch", "MultiEdit", (("file_path", "/src/a.py"),), True),
        ("implementation", "NotebookEdit", (("notebook_path", "/src/a.ipynb"),), True),
        ("change", "Edit", (("file_path", "/src/a.py"),), True),
        ("command", "Bash", (("command", "pwd"),), True),
        ("revert", "Bash", (("command", "git restore /src/a.py"),), True),
        ("edit", "Bash", (("command", "pwd"),), False),
        ("command", "Edit", (("file_path", "/src/a.py"),), False),
        ("revert", "Bash", (("command", "pwd"),), False),
    ],
)
def test_deictic_action_noun_must_match_the_prior_tool_kind(
    noun: str,
    tool_name: str,
    tool_input: tuple[tuple[str, str], ...],
    expected_confirmed: bool,
) -> None:
    prior_action = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="action-a",
        tool_name=tool_name,
        tool_input=tool_input,
    )
    admission = _event(1, text=f"I was wrong about that {noun}.")

    result = detect_mistakes(_parsed(prior_action, admission))

    if expected_confirmed:
        assert len(result.episodes) == 1
        assert result.episodes[0].onset_event_id == prior_action.event_id
        assert result.unconfirmed_candidates == ()
    else:
        assert result.episodes == ()
        assert len(result.unconfirmed_candidates) == 1
        assert result.unconfirmed_candidates[0].onset_event_id is None


def test_live_wrong_command_acknowledgement_links_proximal_bash_action() -> None:
    prior_command = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="overlay-command",
        tool_name="Bash",
        tool_input=(("command", "overlay base --restore"),),
    )
    admission = _event(
        1,
        text=(
            "Correction first: the overlay isn't broken — I gave you the wrong command. "
            "It's designed for base + pilot + restore, and I dropped the pilot flag."
        ),
    )

    result = detect_mistakes(_parsed(prior_command, admission))

    assert len(result.episodes) == 1
    assert result.unconfirmed_candidates == ()
    episode = result.episodes[0]
    assert episode.local_classification is Classification.CONFIRMED
    assert episode.category == "wrong_assumption"
    assert episode.onset_event_id == prior_command.event_id
    assert any(
        evidence.event_id == admission.event_id
        and evidence.evidence_kind is EvidenceKind.VISIBLE_ADMISSION
        and evidence.qualifies_for_promotion
        for evidence in episode.evidence
    )


def test_wrong_command_admission_links_bash_but_rejects_wrappers() -> None:
    prior_command = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="command-a",
        tool_name="Bash",
        tool_input=(("command", "overlay base --restore"),),
    )
    admission = _event(1, text="I gave you the wrong command.")

    result = detect_mistakes(_parsed(prior_command, admission))

    assert len(result.episodes) == 1
    assert result.unconfirmed_candidates == ()
    episode = result.episodes[0]
    assert episode.local_classification is Classification.CONFIRMED
    assert episode.onset_event_id == prior_command.event_id
    assert any(
        evidence.event_id == admission.event_id
        and evidence.evidence_kind is EvidenceKind.VISIBLE_ADMISSION
        and evidence.qualifies_for_promotion
        for evidence in episode.evidence
    )

    wrapped_texts = (
        'The detector should match "I gave you the wrong command."',
        "```text\nI gave you the wrong command.\n```",
        "If I gave you the wrong command, I would correct it.",
    )
    for text in wrapped_texts:
        wrapped_result = detect_mistakes(_parsed(prior_command, _event(1, text=text)))
        assert wrapped_result.eligible_candidates == 0
        assert wrapped_result.episodes == ()
        assert wrapped_result.unconfirmed_candidates == ()


def test_omission_admissions_link_bash_but_reject_intent_and_conditionals() -> None:
    positive_texts = (
        "Correction: I dropped the pilot flag.",
        "I accidentally omitted the --pilot argument.",
        "I forgot to include the pilot option.",
        "I should have passed the pilot profile.",
    )
    for case_index, text in enumerate(positive_texts):
        prior_command = _event(
            0,
            kind=ContentKind.TOOL_USE,
            source_kind=EvidenceSourceKind.TOOL_USE,
            tool_use_id=f"omission-command-{case_index}",
            tool_name="Bash",
            tool_input=(("command", "overlay base --restore"),),
        )
        admission = _event(1, text=text)

        result = detect_mistakes(_parsed(prior_command, admission))

        assert len(result.episodes) == 1, text
        assert result.unconfirmed_candidates == (), text
        episode = result.episodes[0]
        assert episode.local_classification is Classification.CONFIRMED, text
        assert episode.onset_event_id == prior_command.event_id, text
        assert any(
            evidence.event_id == admission.event_id
            and evidence.evidence_kind is EvidenceKind.VISIBLE_ADMISSION
            and evidence.qualifies_for_promotion
            for evidence in episode.evidence
        ), text

    negative_texts = (
        "I intentionally dropped the pilot flag.",
        "I dropped the pilot flag as requested.",
        "Correction: I dropped the pilot flag as requested.",
        "Correction: I dropped the pilot flag on purpose for the test.",
        "If I omitted the pilot flag, I would correct it.",
    )
    for case_index, text in enumerate(negative_texts):
        prior_command = _event(
            0,
            kind=ContentKind.TOOL_USE,
            source_kind=EvidenceSourceKind.TOOL_USE,
            tool_use_id=f"non-omission-command-{case_index}",
            tool_name="Bash",
            tool_input=(("command", "overlay base --restore"),),
        )

        result = detect_mistakes(_parsed(prior_command, _event(1, text=text)))

        assert result.eligible_candidates == 0, text
        assert result.episodes == (), text
        assert result.unconfirmed_candidates == (), text


def test_wrong_action_admissions_link_bash_but_reject_intent_and_conditionals() -> None:
    positive_texts = (
        "I sent you the incorrect command.",
        "I provided the wrong command.",
        "I ran the wrong command.",
        "I used the incorrect profile.",
        "I passed the wrong argument.",
        "The command I gave you was wrong.",
    )
    for case_index, text in enumerate(positive_texts):
        prior_command = _event(
            0,
            kind=ContentKind.TOOL_USE,
            source_kind=EvidenceSourceKind.TOOL_USE,
            tool_use_id=f"wrong-command-{case_index}",
            tool_name="Bash",
            tool_input=(("command", "overlay base --restore"),),
        )
        admission = _event(1, text=text)

        result = detect_mistakes(_parsed(prior_command, admission))

        assert len(result.episodes) == 1, text
        assert result.unconfirmed_candidates == (), text
        episode = result.episodes[0]
        assert episode.local_classification is Classification.CONFIRMED, text
        assert episode.onset_event_id == prior_command.event_id, text
        assert any(
            evidence.event_id == admission.event_id
            and evidence.evidence_kind is EvidenceKind.VISIBLE_ADMISSION
            and evidence.qualifies_for_promotion
            for evidence in episode.evidence
        ), text

    negative_texts = (
        "I intentionally ran the wrong command to test failure handling.",
        "I used the wrong command on purpose to reproduce the bug.",
        "I ran the wrong command to test failure handling.",
        "The command I gave you was intentionally wrong for the negative test.",
        "If I used the wrong profile, I would correct it.",
    )
    for case_index, text in enumerate(negative_texts):
        prior_command = _event(
            0,
            kind=ContentKind.TOOL_USE,
            source_kind=EvidenceSourceKind.TOOL_USE,
            tool_use_id=f"intentional-command-{case_index}",
            tool_name="Bash",
            tool_input=(("command", "overlay base --restore"),),
        )

        result = detect_mistakes(_parsed(prior_command, _event(1, text=text)))

        assert result.eligible_candidates == 0, text
        assert result.episodes == (), text
        assert result.unconfirmed_candidates == (), text


def test_deictic_admission_does_not_cross_a_human_task_boundary() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    new_task = _event(
        1,
        text="Please start a new task now.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
    )
    admission = _event(2, text="I was wrong about that edit.")

    result = detect_mistakes(_parsed(prior_edit, new_task, admission))

    assert result.episodes == ()
    assert len(result.unconfirmed_candidates) == 1
    assert result.unconfirmed_candidates[0].onset_event_id is None


def test_deictic_admission_requires_a_proximal_action() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    intervening = tuple(
        _event(index, text=f"ordinary progress update {index}") for index in range(1, 10)
    )
    admission = _event(10, text="I was wrong about that edit.")

    result = detect_mistakes(_parsed(prior_edit, *intervening, admission))

    assert result.episodes == ()
    assert len(result.unconfirmed_candidates) == 1
    assert result.unconfirmed_candidates[0].onset_event_id is None


def test_user_error_claim_is_unconfirmed_and_excluded_from_reportable_episodes() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
        tool_input=(("file_path", "/src/parser.py"),),
    )
    user_claim = _event(
        1,
        text="You were wrong about that parser edit.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
    )

    result = detect_mistakes(_parsed(prior_edit, user_claim))

    assert result.episodes == ()
    assert len(result.unconfirmed_candidates) == 1
    candidate = result.unconfirmed_candidates[0]
    assert candidate.local_classification is Classification.UNCONFIRMED
    assert candidate.onset_event_id == prior_edit.event_id
    assert candidate.evidence[0].evidence_kind is EvidenceKind.USER_CORRECTION


@pytest.mark.parametrize(
    "text",
    [
        'The fixture contains "You were wrong about the parser edit."',
        "```text\nYou were wrong about the parser edit.\n```",
        "> You were wrong about the parser edit.",
    ],
)
def test_quoted_user_corrections_are_not_candidate_signals(text: str) -> None:
    user_example = _event(
        0,
        text=text,
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
    )

    result = detect_mistakes(_parsed(user_example))

    assert result.eligible_candidates == 0
    assert result.unconfirmed_candidates == ()


@pytest.mark.parametrize(
    "text",
    [
        "Sorry about that.",
        "If I was wrong about the parser edit, I would fix it.",
        "I was not wrong about the parser edit.",
        "I may be wrong about the parser edit.",
        'The detector should match "I was wrong about X".',
        "```text\nI was wrong about the parser edit.\n```",
        "> I was wrong about the parser edit.",
        "In a hypothetical example, I was wrong about the parser edit.",
    ],
)
def test_non_admission_language_and_quoted_examples_are_suppressed(text: str) -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )

    result = detect_mistakes(_parsed(prior_edit, _event(1, text=text)))

    assert result.eligible_candidates == 0
    assert result.episodes == ()
    assert result.unconfirmed_candidates == ()


@pytest.mark.parametrize(
    "text",
    [
        "I don't think I was wrong about /src/a.py.",
        "The user said I was wrong about /src/a.py.",
        "Do not say I was wrong about /src/a.py.",
        "I was wrong about /src/a.py?",
        "```text\nI was wrong about /src/a.py.",
        "~~~text\nI was wrong about /src/a.py.\n~~~",
        "~~~text\nI was wrong about /src/a.py.",
    ],
)
def test_clause_scanner_rejects_reported_imperative_question_and_fenced_text(
    text: str,
) -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )

    result = detect_mistakes(_parsed(prior_edit, _event(1, text=text)))

    assert result.eligible_candidates == 0
    assert result.episodes == ()
    assert result.unconfirmed_candidates == ()


@pytest.mark.parametrize(
    ("text", "expected_confirmed"),
    [
        ("I was wrong about /src/a.py, or maybe not.", False),
        ("I was wrong about /src/a.py, but perhaps not.", False),
        ("I was wrong about /src/a.py unless the fixture changed.", False),
        ("I was wrong about /src/a.py; I'm not sure.", False),
        ("I was wrong about /src/a.py. I still need to verify.", False),
        ("I was wrong about /src/a.py, but I fixed it.", True),
    ],
)
def test_admission_retraction_tails_are_rejected_without_blocking_a_fix_statement(
    text: str,
    expected_confirmed: bool,
) -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )

    result = detect_mistakes(_parsed(prior_edit, _event(1, text=text)))

    if expected_confirmed:
        assert len(result.episodes) == 1
        assert result.episodes[0].onset_event_id == prior_edit.event_id
    else:
        assert result.eligible_candidates == 0
        assert result.episodes == ()
        assert result.unconfirmed_candidates == ()


@pytest.mark.parametrize("correction_tool_name", ["Edit", "Write"])
def test_thinking_admission_requires_corrective_action_and_caps_at_probable(
    correction_tool_name: str,
) -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
        tool_input=(("file_path", "/src/parser.py"),),
    )
    thinking = _event(
        1,
        text="I incorrectly assumed /src/parser.py used that schema.",
        kind=ContentKind.THINKING,
        source_kind=EvidenceSourceKind.ASSISTANT_THINKING,
    )

    alone = detect_mistakes(_parsed(prior_edit, thinking))

    assert alone.episodes == ()
    assert alone.unconfirmed_candidates[0].local_classification is Classification.UNCONFIRMED

    correction = _event(
        2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-2",
        tool_name=correction_tool_name,
        tool_input=(("file_path", "/src/parser.py"),),
        correlated_event_id="E3",
    )
    recovery = _event(
        3,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="edit-2",
        tool_result_is_error=False,
        correlated_event_id="E2",
    )

    corroborated = detect_mistakes(_parsed(prior_edit, thinking, correction, recovery))

    assert len(corroborated.episodes) == 1
    episode = corroborated.episodes[0]
    assert episode.local_classification is Classification.PROBABLE
    assert episode.recovery_end_event_id == recovery.event_id
    assert {item.evidence_kind for item in episode.evidence} >= {
        EvidenceKind.THINKING_ADMISSION,
        EvidenceKind.CORRECTIVE_ACTION,
        EvidenceKind.SUCCESSFUL_RECOVERY,
    }


def test_multiple_admission_phrases_in_one_event_are_one_independent_signal() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )
    admission = _event(
        1,
        text=(
            "I was wrong about that edit; that was my mistake; "
            "I incorrectly assumed the old schema."
        ),
    )

    result = detect_mistakes(_parsed(prior_edit, admission))

    evidence = result.episodes[0].evidence
    assert [item.evidence_kind for item in evidence].count(EvidenceKind.VISIBLE_ADMISSION) == 1
    assert [item.evidence_kind for item in evidence].count(EvidenceKind.AFFECTED_WORK) == 1
    qualifying_groups = {
        item.corroboration_group for item in evidence if item.qualifies_for_promotion
    }
    assert qualifying_groups == {f"admission:{admission.event_id}"}


@pytest.mark.parametrize(
    "revert_command",
    ["git restore -- /src/wrong.py", "git checkout -- /src/wrong.py"],
)
def test_complete_objective_correction_chain_is_confirmed(revert_command: str) -> None:
    wrong_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-wrong",
        tool_name="Edit",
        tool_input=(("file_path", "/src/wrong.py"),),
        correlated_event_id="E1",
    )
    edit_result = _event(
        1,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="edit-wrong",
        tool_result_is_error=False,
        correlated_event_id="E0",
    )
    failed_validation = _event(
        2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="test-red",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_parser.py"),),
        correlated_event_id="E3",
    )
    contradiction = _event(
        3,
        text="AssertionError: expected parsed value",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="test-red",
        tool_result_is_error=True,
        correlated_event_id="E2",
    )
    revert = _event(
        4,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="revert",
        tool_name="Bash",
        tool_input=(("command", revert_command),),
        correlated_event_id="E5",
    )
    revert_result = _event(
        5,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="revert",
        tool_result_is_error=False,
        correlated_event_id="E4",
    )
    recovered_validation = _event(
        6,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="test-green",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_parser.py"),),
        correlated_event_id="E7",
    )
    recovery = _event(
        7,
        text="1 passed",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="test-green",
        tool_result_is_error=False,
        correlated_event_id="E6",
    )

    result = detect_mistakes(
        _parsed(
            wrong_edit,
            edit_result,
            failed_validation,
            contradiction,
            revert,
            revert_result,
            recovered_validation,
            recovery,
        )
    )

    assert len(result.episodes) == 1
    episode = result.episodes[0]
    assert episode.local_classification is Classification.CONFIRMED
    assert episode.onset_event_id == wrong_edit.event_id
    assert episode.detection_event_id == contradiction.event_id
    assert episode.recovery_end_event_id == recovery.event_id
    assert episode.objective_chains[0].correction_event_ids == (
        revert.event_id,
        revert_result.event_id,
    )
    assert episode.retry_event_ids == (recovered_validation.event_id,)
    assert episode.reverted_edit_event_ids == (revert.event_id,)
    known_ids = {
        item.event_id
        for item in (
            wrong_edit,
            edit_result,
            failed_validation,
            contradiction,
            revert,
            revert_result,
            recovered_validation,
            recovery,
        )
    }
    referenced_ids = {
        episode.onset_event_id,
        episode.detection_event_id,
        episode.recovery_end_event_id,
        *episode.affected_event_ids,
        *episode.context_window_event_ids,
        *episode.retry_event_ids,
        *episode.reverted_edit_event_ids,
        *(item.event_id for item in episode.evidence),
        *(item.start_event_id for item in episode.affected_gap_refs),
        *(item.end_event_id for item in episode.affected_gap_refs),
    }
    assert referenced_ids <= known_ids
    assert all(gap.same_lineage for gap in episode.affected_gap_refs)
    assert not any(gap.crosses_human_boundary for gap in episode.affected_gap_refs)
    assert episode.ambiguous_gap_refs == ()


def test_unrelated_successful_edit_does_not_complete_an_objective_chain() -> None:
    wrong_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="wrong-edit",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    failed_use = _event(
        1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="failed-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E2",
    )
    contradiction = _event(
        2,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="failed-test",
        tool_result_is_error=True,
        correlated_event_id="E1",
    )
    unrelated_edit = _event(
        3,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="unrelated-edit",
        tool_name="Edit",
        tool_input=(("file_path", "/src/b.py"),),
        correlated_event_id="E4",
    )
    unrelated_result = _event(
        4,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="unrelated-edit",
        tool_result_is_error=False,
        correlated_event_id="E3",
    )
    later_validation = _event(
        5,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="later-test",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_a.py"),),
        correlated_event_id="E6",
    )
    later_success = _event(
        6,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="later-test",
        tool_result_is_error=False,
        correlated_event_id="E5",
    )

    result = detect_mistakes(
        _parsed(
            wrong_edit,
            failed_use,
            contradiction,
            unrelated_edit,
            unrelated_result,
            later_validation,
            later_success,
        )
    )

    assert result.episodes == ()
    assert result.unconfirmed_candidates == ()


def test_expected_tdd_red_then_green_is_not_a_mistake_candidate() -> None:
    expected_red = _event(0, text="RED: this test should fail before implementation.")
    test_edit = _event(
        1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="test-edit",
        tool_name="Edit",
        tool_input=(("file_path", "/tests/test_parser.py"),),
        correlated_event_id="E2",
    )
    test_edit_result = _event(
        2,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="test-edit",
        tool_result_is_error=False,
        correlated_event_id="E1",
    )
    red_use = _event(
        3,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="red",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_parser.py"),),
        correlated_event_id="E4",
    )
    red_result = _event(
        4,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="red",
        tool_result_is_error=True,
        correlated_event_id="E3",
    )
    implementation = _event(
        5,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="implementation",
        tool_name="Edit",
        tool_input=(("file_path", "/src/parser.py"),),
        correlated_event_id="E6",
    )
    implementation_result = _event(
        6,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="implementation",
        tool_result_is_error=False,
        correlated_event_id="E5",
    )
    green_use = _event(
        7,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="green",
        tool_name="Bash",
        tool_input=(("command", "pytest tests/test_parser.py"),),
        correlated_event_id="E8",
    )
    green_result = _event(
        8,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="green",
        tool_result_is_error=False,
        correlated_event_id="E7",
    )

    result = detect_mistakes(
        _parsed(
            expected_red,
            test_edit,
            test_edit_result,
            red_use,
            red_result,
            implementation,
            implementation_result,
            green_use,
            green_result,
        )
    )

    assert result.eligible_candidates == 0
    assert result.diagnostics.suppressed_non_mistakes >= 1


def test_transient_failure_with_unchanged_successful_retry_is_suppressed() -> None:
    failed_use = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="request-1",
        tool_name="Bash",
        tool_input=(("command", "curl https://service.example/status"),),
        correlated_event_id="E1",
    )
    transient_failure = _event(
        1,
        text="connection timed out; please retry",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="request-1",
        tool_result_is_error=True,
        correlated_event_id="E0",
    )
    unchanged_retry = _event(
        2,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="request-2",
        tool_name="Bash",
        tool_input=(("command", "curl https://service.example/status"),),
        correlated_event_id="E3",
    )
    success = _event(
        3,
        text="ok",
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="request-2",
        tool_result_is_error=False,
        correlated_event_id="E2",
    )

    result = detect_mistakes(_parsed(failed_use, transient_failure, unchanged_retry, success))

    assert result.eligible_candidates == 0
    assert result.diagnostics.suppressed_non_mistakes == 1


def test_user_requirement_or_tradeoff_change_is_not_an_error_claim() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )
    requirement_change = _event(
        1,
        text="Actually, I changed my mind. Let's use PostgreSQL instead.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
    )

    result = detect_mistakes(_parsed(prior_edit, requirement_change))

    assert result.eligible_candidates == 0
    assert result.diagnostics.suppressed_non_mistakes == 1


def test_user_correction_and_linked_agent_admission_merge_into_one_episode() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )
    user_correction = _event(
        1,
        text="You were wrong about that parser edit.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
    )
    admission = _event(2, text="You're right. I was wrong about that parser edit.")

    result = detect_mistakes(_parsed(prior_edit, user_correction, admission))

    assert result.eligible_candidates == 2
    assert len(result.episodes) == 1
    assert result.unconfirmed_candidates == ()
    episode = result.episodes[0]
    assert episode.local_classification is Classification.CONFIRMED
    assert episode.onset_event_id == prior_edit.event_id
    assert episode.detection_event_id == user_correction.event_id
    assert {item.evidence_kind for item in episode.evidence} >= {
        EvidenceKind.USER_CORRECTION,
        EvidenceKind.VISIBLE_ADMISSION,
    }
    assert any(gap.crosses_human_boundary for gap in episode.affected_gap_refs)


def test_user_bridge_rejects_an_explicit_artifact_conflicting_with_its_root() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    user_correction = _event(
        1,
        text="You were wrong about /src/a.py.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
    )
    conflicting_admission = _event(2, text="You're right. I was wrong about /src/b.py.")

    result = detect_mistakes(_parsed(prior_edit, user_correction, conflicting_admission))

    assert result.episodes == ()
    admission_candidate = next(
        candidate
        for candidate in result.unconfirmed_candidates
        if candidate.detection_event_id == conflicting_admission.event_id
    )
    assert admission_candidate.onset_event_id is None
    assert prior_edit.event_id not in admission_candidate.affected_event_ids


@pytest.mark.parametrize(
    ("admission_text", "expected_confirmed"),
    [
        ("I was wrong.", False),
        ("I was wrong about that edit.", True),
        ("You're right. I was wrong.", True),
        ("I was wrong about /src/a.py.", True),
    ],
)
def test_user_bridge_requires_a_matching_reference_or_explicit_acknowledgment(
    admission_text: str,
    expected_confirmed: bool,
) -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    user_correction = _event(
        1,
        text="You were wrong about that edit.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
    )
    admission = _event(2, text=admission_text)

    result = detect_mistakes(_parsed(prior_edit, user_correction, admission))

    if expected_confirmed:
        assert len(result.episodes) == 1
        assert result.episodes[0].onset_event_id == prior_edit.event_id
        assert admission.event_id in result.episodes[0].affected_event_ids
    else:
        assert result.episodes == ()
        admission_candidate = next(
            candidate
            for candidate in result.unconfirmed_candidates
            if candidate.detection_event_id == admission.event_id
        )
        assert admission_candidate.onset_event_id is None


@pytest.mark.parametrize(
    ("intervening_kind", "expected_confirmed"),
    [
        ("same_response_thinking_and_metadata", True),
        ("different_response_thinking", False),
        ("tool", False),
        ("visible_assistant", False),
        ("system", False),
        ("human", False),
    ],
)
def test_user_bridge_only_crosses_ignorable_same_response_assistant_context(
    intervening_kind: str,
    expected_confirmed: bool,
) -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        record_uuid="edit-record",
        tool_use_id="edit-a",
        tool_name="Edit",
        tool_input=(("file_path", "/src/a.py"),),
    )
    user_correction = _event(
        1,
        text="You were wrong about that edit.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
        record_uuid="user-record",
    )
    if intervening_kind == "same_response_thinking_and_metadata":
        intervening = (
            _event(
                2,
                text="private response reasoning",
                kind=ContentKind.THINKING,
                source_kind=EvidenceSourceKind.ASSISTANT_THINKING,
                record_uuid="assistant-response",
            ),
            _event(
                3,
                text="assistant response metadata",
                kind=ContentKind.METADATA,
                source_kind=EvidenceSourceKind.DERIVED,
                record_uuid="assistant-response",
            ),
        )
    elif intervening_kind == "different_response_thinking":
        intervening = (
            _event(
                2,
                text="earlier response reasoning",
                kind=ContentKind.THINKING,
                source_kind=EvidenceSourceKind.ASSISTANT_THINKING,
                record_uuid="different-response",
            ),
        )
    elif intervening_kind == "tool":
        intervening = (
            _event(
                2,
                kind=ContentKind.TOOL_USE,
                source_kind=EvidenceSourceKind.TOOL_USE,
                record_uuid="assistant-response",
                tool_use_id="intervening-tool",
                tool_name="Bash",
                tool_input=(("command", "pwd"),),
            ),
        )
    elif intervening_kind == "visible_assistant":
        intervening = (_event(2, text="visible preface", record_uuid="assistant-response"),)
    elif intervening_kind == "system":
        intervening = (
            _event(
                2,
                text="system metadata",
                actor=EventActor.SYSTEM,
                kind=ContentKind.METADATA,
                source_kind=EvidenceSourceKind.DERIVED,
                record_uuid="assistant-response",
            ),
        )
    else:
        intervening = (
            _event(
                2,
                text="one more user message",
                actor=EventActor.HUMAN,
                source_kind=EvidenceSourceKind.USER_PROMPT,
                record_uuid="second-user-record",
            ),
        )
    admission_index = 2 + len(intervening)
    admission = _event(
        admission_index,
        text="You're right. I was wrong.",
        record_uuid="assistant-response",
    )

    result = detect_mistakes(_parsed(prior_edit, user_correction, *intervening, admission))

    if expected_confirmed:
        assert len(result.episodes) == 1
        assert result.episodes[0].onset_event_id == prior_edit.event_id
        assert admission.event_id in result.episodes[0].affected_event_ids
    else:
        assert result.episodes == ()
        admission_candidate = next(
            candidate
            for candidate in result.unconfirmed_candidates
            if candidate.detection_event_id == admission.event_id
        )
        assert admission_candidate.onset_event_id is None


def test_unrelated_later_success_is_not_used_as_recovery() -> None:
    thinking = _event(
        0,
        text="I was wrong about an earlier assumption.",
        kind=ContentKind.THINKING,
        source_kind=EvidenceSourceKind.ASSISTANT_THINKING,
    )
    unrelated_edit = _event(
        1,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="unrelated",
        tool_name="Edit",
        tool_input=(("file_path", "/src/unrelated.py"),),
        correlated_event_id="E2",
    )
    unrelated_success = _event(
        2,
        actor=EventActor.TOOL,
        kind=ContentKind.TOOL_RESULT,
        source_kind=EvidenceSourceKind.TOOL_RESULT,
        tool_use_id="unrelated",
        tool_result_is_error=False,
        correlated_event_id="E1",
    )

    result = detect_mistakes(_parsed(thinking, unrelated_edit, unrelated_success))

    candidate = result.unconfirmed_candidates[0]
    assert candidate.recovery_end_event_id is None
    assert candidate.affected_event_ids == (thinking.event_id,)
    assert EvidenceKind.SUCCESSFUL_RECOVERY not in {
        evidence.evidence_kind for evidence in candidate.evidence
    }


def test_no_thinking_removes_thinking_from_detection_and_context_windows() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )
    thinking = _event(
        1,
        text="I was wrong about a private discarded hypothesis.",
        kind=ContentKind.THINKING,
        source_kind=EvidenceSourceKind.ASSISTANT_THINKING,
    )
    visible_admission = _event(2, text="I was wrong about that edit.")

    result = detect_mistakes(
        _parsed(prior_edit, thinking, visible_admission), include_thinking=False
    )

    assert len(result.episodes) == 1
    assert result.unconfirmed_candidates == ()
    episode = result.episodes[0]
    assert thinking.event_id not in episode.context_window_event_ids
    assert thinking.event_id not in {evidence.event_id for evidence in episode.evidence}


def test_qualifying_evidence_ids_are_machine_distinct_from_user_and_context() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )
    user_claim = _event(
        1,
        text="You were wrong about that edit.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
    )
    admission = _event(2, text="I was wrong about that edit.")

    result = detect_mistakes(_parsed(prior_edit, user_claim, admission))

    episode = result.episodes[0]
    assert episode.qualifying_evidence_event_ids == (admission.event_id,)
    assert user_claim.event_id not in episode.qualifying_evidence_event_ids
    assert prior_edit.event_id not in episode.qualifying_evidence_event_ids


def test_semantic_paraphrase_stays_context_without_becoming_local_evidence() -> None:
    prior_edit = _event(
        0,
        kind=ContentKind.TOOL_USE,
        source_kind=EvidenceSourceKind.TOOL_USE,
        tool_use_id="edit-1",
        tool_name="Edit",
    )
    paraphrase = _event(
        1,
        text="The premise behind that change doesn't hold, so I'll take a different route.",
    )
    user_claim = _event(
        2,
        text="That is incorrect.",
        actor=EventActor.HUMAN,
        source_kind=EvidenceSourceKind.USER_PROMPT,
    )

    result = detect_mistakes(_parsed(prior_edit, paraphrase, user_claim))

    assert result.episodes == ()
    assert len(result.unconfirmed_candidates) == 1
    candidate = result.unconfirmed_candidates[0]
    assert paraphrase.event_id in candidate.context_only_event_ids
    assert paraphrase.event_id not in {item.event_id for item in candidate.evidence}
    assert candidate.qualifying_evidence_event_ids == ()
