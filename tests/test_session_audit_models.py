"""Contract tests for the session-audit result model."""

from dataclasses import FrozenInstanceError

import pytest

from scripts.core.session_audit.models import (
    RULESET_VERSION,
    SCHEMA_VERSION,
    AnalysisStatus,
    AuditExitCode,
    AuditLimits,
    AuditResult,
    Classification,
    ContentKind,
    EventActor,
    EvidenceSourceKind,
    OutputFormat,
    RunProvenance,
    SessionMetadata,
)


def test_schema_and_ruleset_versions_start_at_one() -> None:
    assert SCHEMA_VERSION == 1
    assert RULESET_VERSION == 1


def test_result_enums_have_stable_wire_values() -> None:
    assert {status.value for status in AnalysisStatus} == {"complete", "partial", "refused"}
    assert {classification.value for classification in Classification} == {
        "confirmed",
        "probable",
        "unconfirmed",
    }
    assert {source.value for source in EvidenceSourceKind} == {
        "assistant_text",
        "assistant_thinking",
        "user_prompt",
        "tool_use",
        "tool_result",
        "derived",
    }
    assert {actor.value for actor in EventActor} == {"human", "assistant", "tool", "system"}
    assert {kind.value for kind in ContentKind} == {
        "visible_text",
        "thinking",
        "tool_use",
        "tool_result",
        "metadata",
    }
    assert {output_format.value for output_format in OutputFormat} == {"markdown", "json"}


def test_exit_codes_match_the_public_cli_contract() -> None:
    assert AuditExitCode.COMPLETE == 0
    assert AuditExitCode.OPERATIONAL_FAILURE == 1
    assert AuditExitCode.USAGE_ERROR == 2
    assert AuditExitCode.JUDGE_FAILURE == 3
    assert AuditExitCode.REFUSED == 4
    assert AuditExitCode.CANDIDATE_OVERFLOW == 5


def test_default_limits_match_the_approved_safety_envelope() -> None:
    limits = AuditLimits()

    assert limits.max_input_bytes == 67_108_864
    assert limits.max_input_line_bytes == 8_388_608
    assert limits.max_input_records == 125_000
    assert limits.max_normalized_events == 25_000
    assert limits.max_local_candidates == 500
    assert limits.max_judge_candidates == 12
    assert limits.max_judge_window_chars == 4_000
    assert limits.max_judge_total_chars == 48_000
    assert limits.max_judge_calls == 1
    assert limits.max_judge_output_tokens == 2_048
    assert limits.judge_deadline_seconds == 60.0


def test_run_provenance_records_effective_flags_and_limits() -> None:
    run = RunProvenance(
        tool_version="0.8.0",
        output_format=OutputFormat.JSON,
        thinking_included=False,
        judge_requested=True,
        judge_model="claude-sonnet-5",
    )

    assert run.ruleset_version == RULESET_VERSION
    assert run.output_format is OutputFormat.JSON
    assert run.thinking_included is False
    assert run.judge_requested is True
    assert run.judge_model == "claude-sonnet-5"
    assert run.judge_timeout_seconds == 60.0
    assert run.effective_limits == AuditLimits()


def test_result_envelope_has_stable_defaults_and_is_frozen() -> None:
    result = AuditResult(
        run=RunProvenance(tool_version="0.8.0"),
        session=SessionMetadata(input_bytes=123, sha256="a" * 64),
    )

    assert result.schema_version == SCHEMA_VERSION
    assert result.status is AnalysisStatus.COMPLETE
    assert result.summary == {}
    assert result.episodes == ()
    assert result.unconfirmed_candidates == ()
    assert result.diagnostics == {}

    with pytest.raises(FrozenInstanceError):
        result.status = AnalysisStatus.PARTIAL  # type: ignore[misc]
