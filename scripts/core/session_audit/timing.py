"""Pure interval math for session-audit time attribution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

CENTRAL_INFERRED_GAP_CAP_SECONDS = 300.0
INCLUSIVE_INFERRED_GAP_CAP_SECONDS = 900.0


@dataclass(frozen=True)
class IntervalInput:
    """Possibly incomplete interval with caller-validated causality flags.

    ``same_lineage`` is relative to the owning episode; work on an abandoned branch can
    therefore contribute through its own episode while sibling-branch activity is excluded.
    ``crosses_human_boundary`` marks inferred gaps that would otherwise bridge user idle time.
    """

    start: datetime | None
    end: datetime | None
    same_lineage: bool = True
    crosses_human_boundary: bool = False


@dataclass(frozen=True)
class NormalizedInterval:
    """Validated UTC interval used by deterministic duration calculations."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class AffectedEventIds:
    """Prevalidated affected identifiers used to de-duplicate counts across episodes."""

    assistant_turn_ids: frozenset[str] = field(default_factory=frozenset)
    tool_call_ids: frozenset[str] = field(default_factory=frozenset)
    failed_tool_call_ids: frozenset[str] = field(default_factory=frozenset)
    retry_ids: frozenset[str] = field(default_factory=frozenset)
    reverted_edit_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class AffectedCounts:
    """De-duplicated affected work counts."""

    assistant_turns: int = 0
    tool_calls: int = 0
    failed_tool_calls: int = 0
    retries: int = 0
    reverted_edits: int = 0


@dataclass(frozen=True)
class TimingDiagnostics:
    """Interval-quality counters retained alongside nullable metrics."""

    missing_timestamp_intervals: int = 0
    invalid_timestamp_intervals: int = 0
    excluded_unrelated_branch_intervals: int = 0
    excluded_human_idle_intervals: int = 0


@dataclass(frozen=True)
class EpisodeTimingInput:
    """Observed work, inferred gaps, and causal boundaries for one mistake episode."""

    observed_tool_intervals: tuple[IntervalInput, ...] = ()
    affected_gaps: tuple[IntervalInput, ...] = ()
    ambiguous_recovery_gaps: tuple[IntervalInput, ...] = ()
    onset: datetime | None = None
    recovery_end: datetime | None = None
    affected_event_ids: AffectedEventIds = field(default_factory=AffectedEventIds)


@dataclass(frozen=True)
class HeuristicEstimateRange:
    """Low/central/high policy estimates, not statistical confidence bounds."""

    low_seconds: float | None
    central_seconds: float | None
    high_seconds: float | None


@dataclass(frozen=True)
class TimingMetrics:
    """Deterministic observed and inferred time metrics.

    Null values mean at least one required included interval was unmeasurable. The heuristic
    range is policy output rather than a confidence interval or a mathematical bound.
    """

    directly_attributed_observed_seconds: float | None
    central_avoidable_active_seconds: float | None
    inclusive_avoidable_active_seconds: float | None
    estimate_range: HeuristicEstimateRange
    observed_onset_to_recovery_seconds: float | None
    affected_counts: AffectedCounts = field(default_factory=AffectedCounts)
    diagnostics: TimingDiagnostics = field(default_factory=TimingDiagnostics)


@dataclass(frozen=True)
class _EvaluatedInterval:
    interval: NormalizedInterval | None
    disposition: Literal["missing", "invalid", "unrelated", "human_idle"] | None = None


@dataclass(frozen=True)
class _IntervalBatch:
    intervals: tuple[NormalizedInterval, ...]
    missing: int = 0
    invalid: int = 0
    excluded_unrelated: int = 0
    excluded_human_idle: int = 0

    @property
    def complete(self) -> bool:
        return self.missing == 0 and self.invalid == 0


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _is_timezone_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def normalize_interval(interval: IntervalInput) -> NormalizedInterval | None:
    """Normalize an eligible, complete, timezone-aware, nonnegative interval to UTC."""

    return _evaluate_interval(interval).interval


def _evaluate_interval(
    interval: IntervalInput, *, cap_seconds: float | None = None
) -> _EvaluatedInterval:
    if not interval.same_lineage or interval.crosses_human_boundary:
        disposition: Literal["unrelated", "human_idle"] = (
            "unrelated" if not interval.same_lineage else "human_idle"
        )
        return _EvaluatedInterval(None, disposition)
    if interval.start is None or interval.end is None:
        return _EvaluatedInterval(None, "missing")
    if not _is_timezone_aware(interval.start) or not _is_timezone_aware(interval.end):
        return _EvaluatedInterval(None, "invalid")
    start = _as_utc(interval.start)
    end = _as_utc(interval.end)
    if end < start:
        return _EvaluatedInterval(None, "invalid")
    if cap_seconds is not None:
        cap = timedelta(seconds=cap_seconds)
        if end - start > cap:
            end = start + cap
    return _EvaluatedInterval(NormalizedInterval(start=start, end=end))


def union_intervals(intervals: Iterable[NormalizedInterval]) -> tuple[NormalizedInterval, ...]:
    """Return sorted, non-overlapping intervals, merging overlaps and adjacency."""

    ordered = sorted(intervals, key=lambda interval: (interval.start, interval.end))
    if not ordered:
        return ()

    merged = [ordered[0]]
    for interval in ordered[1:]:
        previous = merged[-1]
        if interval.start <= previous.end:
            merged[-1] = NormalizedInterval(previous.start, max(previous.end, interval.end))
        else:
            merged.append(interval)
    return tuple(merged)


def total_interval_seconds(intervals: Iterable[NormalizedInterval]) -> float:
    """Sum interval durations without rounding presentation values."""

    return sum(
        ((interval.end - interval.start).total_seconds() for interval in intervals),
        start=0.0,
    )


def _normalize_many(
    intervals: Iterable[IntervalInput], *, cap_seconds: float | None = None
) -> _IntervalBatch:
    evaluated = tuple(
        _evaluate_interval(interval, cap_seconds=cap_seconds) for interval in intervals
    )
    return _IntervalBatch(
        intervals=tuple(item.interval for item in evaluated if item.interval is not None),
        missing=sum(item.disposition == "missing" for item in evaluated),
        invalid=sum(item.disposition == "invalid" for item in evaluated),
        excluded_unrelated=sum(item.disposition == "unrelated" for item in evaluated),
        excluded_human_idle=sum(item.disposition == "human_idle" for item in evaluated),
    )


def calculate_time_attribution(episodes: Iterable[EpisodeTimingInput]) -> TimingMetrics:
    """Calculate overlap-safe metrics for one or more mistake episodes.

    Observed tool intervals retain their full duration. Affected and ambiguous inferred gaps
    are capped independently at five and fifteen minutes before all intervals are unioned.
    Missing or invalid intervals null only their dependent metrics; explicit lineage and human
    exclusions do not make a metric incomplete. Episode spans and affected identifiers are also
    unioned so overlapping episodes cannot inflate session totals.
    """

    episode_items = tuple(episodes)
    observed = _normalize_many(
        interval for episode in episode_items for interval in episode.observed_tool_intervals
    )
    affected = _normalize_many(
        (interval for episode in episode_items for interval in episode.affected_gaps),
        cap_seconds=CENTRAL_INFERRED_GAP_CAP_SECONDS,
    )
    ambiguous = _normalize_many(
        (interval for episode in episode_items for interval in episode.ambiguous_recovery_gaps),
        cap_seconds=INCLUSIVE_INFERRED_GAP_CAP_SECONDS,
    )
    spans = _normalize_many(
        IntervalInput(episode.onset, episode.recovery_end) for episode in episode_items
    )

    low = total_interval_seconds(union_intervals(observed.intervals)) if observed.complete else None
    central = (
        total_interval_seconds(union_intervals((*observed.intervals, *affected.intervals)))
        if observed.complete and affected.complete
        else None
    )
    high = (
        total_interval_seconds(
            union_intervals((*observed.intervals, *affected.intervals, *ambiguous.intervals))
        )
        if observed.complete and affected.complete and ambiguous.complete
        else None
    )
    span = total_interval_seconds(union_intervals(spans.intervals)) if spans.complete else None

    assistant_turn_ids = frozenset().union(
        *(episode.affected_event_ids.assistant_turn_ids for episode in episode_items)
    )
    tool_call_ids = frozenset().union(
        *(episode.affected_event_ids.tool_call_ids for episode in episode_items)
    )
    failed_tool_call_ids = frozenset().union(
        *(episode.affected_event_ids.failed_tool_call_ids for episode in episode_items)
    )
    retry_ids = frozenset().union(
        *(episode.affected_event_ids.retry_ids for episode in episode_items)
    )
    reverted_edit_ids = frozenset().union(
        *(episode.affected_event_ids.reverted_edit_ids for episode in episode_items)
    )
    affected_counts = AffectedCounts(
        assistant_turns=len(assistant_turn_ids),
        tool_calls=len(tool_call_ids),
        failed_tool_calls=len(failed_tool_call_ids),
        retries=len(retry_ids),
        reverted_edits=len(reverted_edit_ids),
    )
    batches = (observed, affected, ambiguous, spans)
    diagnostics = TimingDiagnostics(
        missing_timestamp_intervals=sum(batch.missing for batch in batches),
        invalid_timestamp_intervals=sum(batch.invalid for batch in batches),
        excluded_unrelated_branch_intervals=sum(batch.excluded_unrelated for batch in batches),
        excluded_human_idle_intervals=sum(batch.excluded_human_idle for batch in batches),
    )
    return TimingMetrics(
        directly_attributed_observed_seconds=low,
        central_avoidable_active_seconds=central,
        inclusive_avoidable_active_seconds=high,
        estimate_range=HeuristicEstimateRange(low, central, high),
        observed_onset_to_recovery_seconds=span,
        affected_counts=affected_counts,
        diagnostics=diagnostics,
    )
