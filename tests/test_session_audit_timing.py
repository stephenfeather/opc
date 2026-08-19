"""Tests for deterministic session-audit time attribution."""

from datetime import UTC, datetime, timedelta, timezone

from scripts.core.session_audit.timing import (
    AffectedEventIds,
    EpisodeTimingInput,
    IntervalInput,
    calculate_time_attribution,
    normalize_interval,
    total_interval_seconds,
    union_intervals,
)

BASE = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def at(seconds: int) -> datetime:
    """Return a stable test timestamp offset from ``BASE``."""

    return BASE + timedelta(seconds=seconds)


def test_normalize_and_union_intervals_across_timezones() -> None:
    eastern = timezone(timedelta(hours=-4))
    normalized = [
        normalize_interval(IntervalInput(at(20), at(30))),
        normalize_interval(IntervalInput(at(5).astimezone(eastern), at(25).astimezone(eastern))),
        normalize_interval(IntervalInput(at(40), at(50))),
        normalize_interval(IntervalInput(at(50), at(60))),
    ]

    assert all(interval is not None for interval in normalized)
    merged = union_intervals(interval for interval in normalized if interval is not None)

    assert [(interval.start, interval.end) for interval in merged] == [
        (at(5), at(30)),
        (at(40), at(60)),
    ]
    assert total_interval_seconds(merged) == 45.0


def test_estimates_cap_only_inferred_gaps() -> None:
    result = calculate_time_attribution(
        (
            EpisodeTimingInput(
                observed_tool_intervals=(IntervalInput(at(0), at(1_200)),),
                affected_gaps=(IntervalInput(at(1_200), at(1_800)),),
                ambiguous_recovery_gaps=(IntervalInput(at(1_800), at(3_000)),),
                onset=at(0),
                recovery_end=at(3_000),
            ),
        )
    )

    assert result.directly_attributed_observed_seconds == 1_200.0
    assert result.central_avoidable_active_seconds == 1_500.0
    assert result.inclusive_avoidable_active_seconds == 2_400.0
    assert result.estimate_range.low_seconds == 1_200.0
    assert result.estimate_range.central_seconds == 1_500.0
    assert result.estimate_range.high_seconds == 2_400.0
    assert result.observed_onset_to_recovery_seconds == 3_000.0


def test_aggregation_unions_episode_time_and_affected_identifiers() -> None:
    first = EpisodeTimingInput(
        observed_tool_intervals=(
            IntervalInput(at(0), at(120)),
            IntervalInput(at(300), at(500), same_lineage=False),
        ),
        affected_gaps=(
            IntervalInput(at(120), at(600), crosses_human_boundary=True),
            IntervalInput(at(120), at(240)),
        ),
        onset=at(0),
        recovery_end=at(240),
        affected_event_ids=AffectedEventIds(
            assistant_turn_ids=frozenset({"a1", "a2"}),
            tool_call_ids=frozenset({"t1", "t2"}),
            failed_tool_call_ids=frozenset({"t2"}),
            retry_ids=frozenset({"r1"}),
            reverted_edit_ids=frozenset({"e1"}),
        ),
    )
    second = EpisodeTimingInput(
        observed_tool_intervals=(IntervalInput(at(60), at(180)),),
        affected_gaps=(IntervalInput(at(180), at(300)),),
        ambiguous_recovery_gaps=(IntervalInput(at(300), at(400)),),
        onset=at(60),
        recovery_end=at(400),
        affected_event_ids=AffectedEventIds(
            assistant_turn_ids=frozenset({"a2", "a3"}),
            tool_call_ids=frozenset({"t2", "t3"}),
            failed_tool_call_ids=frozenset({"t2", "t3"}),
            retry_ids=frozenset({"r1", "r2"}),
            reverted_edit_ids=frozenset({"e1", "e2"}),
        ),
    )

    result = calculate_time_attribution((first, second))

    assert result.directly_attributed_observed_seconds == 180.0
    assert result.central_avoidable_active_seconds == 300.0
    assert result.inclusive_avoidable_active_seconds == 400.0
    assert result.observed_onset_to_recovery_seconds == 400.0
    assert result.affected_counts.assistant_turns == 3
    assert result.affected_counts.tool_calls == 3
    assert result.affected_counts.failed_tool_calls == 2
    assert result.affected_counts.retries == 2
    assert result.affected_counts.reverted_edits == 2
    assert result.diagnostics.excluded_unrelated_branch_intervals == 1
    assert result.diagnostics.excluded_human_idle_intervals == 1


def test_missing_and_invalid_timestamps_null_only_dependent_metrics() -> None:
    missing_observed = calculate_time_attribution(
        (
            EpisodeTimingInput(
                observed_tool_intervals=(IntervalInput(at(0), None),),
                affected_gaps=(IntervalInput(at(10), at(20)),),
                ambiguous_recovery_gaps=(IntervalInput(at(20), at(30)),),
                onset=at(0),
                recovery_end=at(30),
            ),
        )
    )
    assert missing_observed.directly_attributed_observed_seconds is None
    assert missing_observed.central_avoidable_active_seconds is None
    assert missing_observed.inclusive_avoidable_active_seconds is None
    assert missing_observed.observed_onset_to_recovery_seconds == 30.0
    assert missing_observed.diagnostics.missing_timestamp_intervals == 1

    missing_affected = calculate_time_attribution(
        (
            EpisodeTimingInput(
                observed_tool_intervals=(IntervalInput(at(0), at(10)),),
                affected_gaps=(IntervalInput(None, at(20)),),
                ambiguous_recovery_gaps=(IntervalInput(at(20), at(30)),),
                onset=at(0),
                recovery_end=at(30),
            ),
        )
    )
    assert missing_affected.directly_attributed_observed_seconds == 10.0
    assert missing_affected.central_avoidable_active_seconds is None
    assert missing_affected.inclusive_avoidable_active_seconds is None

    missing_ambiguous = calculate_time_attribution(
        (
            EpisodeTimingInput(
                observed_tool_intervals=(IntervalInput(at(0), at(10)),),
                affected_gaps=(IntervalInput(at(10), at(20)),),
                ambiguous_recovery_gaps=(IntervalInput(at(20), None),),
                onset=at(0),
                recovery_end=at(30),
            ),
        )
    )
    assert missing_ambiguous.directly_attributed_observed_seconds == 10.0
    assert missing_ambiguous.central_avoidable_active_seconds == 20.0
    assert missing_ambiguous.inclusive_avoidable_active_seconds is None

    invalid_span = calculate_time_attribution(
        (EpisodeTimingInput(onset=at(30), recovery_end=at(0)),)
    )
    assert invalid_span.observed_onset_to_recovery_seconds is None
    assert invalid_span.diagnostics.invalid_timestamp_intervals == 1


def test_naive_timestamps_are_invalid_instead_of_assuming_local_timezone() -> None:
    naive_start = at(0).replace(tzinfo=None)
    naive_end = at(10).replace(tzinfo=None)

    assert normalize_interval(IntervalInput(naive_start, naive_end)) is None


def test_empty_attribution_returns_float_zeros_and_empty_counts() -> None:
    result = calculate_time_attribution(())

    assert type(result.directly_attributed_observed_seconds) is float
    assert result.directly_attributed_observed_seconds == 0.0
    assert result.central_avoidable_active_seconds == 0.0
    assert result.inclusive_avoidable_active_seconds == 0.0
    assert result.observed_onset_to_recovery_seconds == 0.0
    assert result.affected_counts.assistant_turns == 0
    assert result.affected_counts.tool_calls == 0


def test_short_inferred_gap_at_datetime_max_does_not_overflow_cap_math() -> None:
    maximum = datetime.max.replace(tzinfo=UTC)

    result = calculate_time_attribution(
        (
            EpisodeTimingInput(
                affected_gaps=(IntervalInput(maximum, maximum),),
                onset=maximum,
                recovery_end=maximum,
            ),
        )
    )

    assert result.central_avoidable_active_seconds == 0.0
    assert result.observed_onset_to_recovery_seconds == 0.0
