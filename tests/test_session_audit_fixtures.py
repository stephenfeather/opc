"""End-to-end checks for the reusable anonymized session-audit corpus."""

from __future__ import annotations

from pathlib import Path

from scripts.core.session_audit.cli import AuditExecution, execute_audit
from scripts.core.session_audit.models import (
    AnalysisStatus,
    AuditExitCode,
    OutputFormat,
)
from scripts.core.session_audit.parser import parse_session
from scripts.core.session_audit.reporting import TimingEstimate, render_json

FIXTURES = Path(__file__).parent / "fixtures" / "session_audit"


def _execute(name: str) -> AuditExecution:
    return execute_audit(FIXTURES / name, output_format=OutputFormat.JSON)


def test_positive_admission_fixture_is_confirmed_end_to_end() -> None:
    execution = _execute("positive-admission.jsonl")

    assert execution.exit_code is AuditExitCode.COMPLETE
    assert execution.result.status is AnalysisStatus.COMPLETE
    assert execution.result.summary["confirmed_episodes"] == 1
    assert execution.result.summary["probable_episodes"] == 0


def test_expected_tdd_fixture_has_zero_reportable_mistakes() -> None:
    execution = _execute("negative-expected-tdd.jsonl")

    assert execution.exit_code is AuditExitCode.COMPLETE
    assert execution.result.summary["confirmed_episodes"] == 0
    assert execution.result.summary["probable_episodes"] == 0
    assert execution.result.summary["unconfirmed_candidates"] == 0


def test_unsupported_schema_fixture_refuses_without_a_prefix_report() -> None:
    execution = _execute("schema-unsupported.jsonl")

    assert execution.exit_code is AuditExitCode.REFUSED
    assert execution.result.status is AnalysisStatus.REFUSED
    assert execution.result.summary["totals_complete"] is False


def test_safety_fixture_masks_command_flag_values_at_normalization() -> None:
    parsed = parse_session(FIXTURES / "safety-redaction.jsonl")
    execution = _execute("safety-redaction.jsonl")
    retained_inputs = repr(tuple(event.tool_input for event in parsed.events))
    rendered = render_json(execution.result)

    assert "fixture-sensitive-value" not in retained_inputs
    assert "fixture-password" not in retained_inputs
    assert "<redacted-secret>" in retained_inputs
    assert "fixture-sensitive-value" not in rendered
    assert "fixture-password" not in rendered


def test_time_fixture_reproduces_observed_tool_duration() -> None:
    execution = _execute("time-attribution.jsonl")
    estimates = execution.result.summary["final_estimates"]

    assert execution.exit_code is AuditExitCode.COMPLETE
    assert execution.result.summary["confirmed_episodes"] == 1
    assert isinstance(estimates, TimingEstimate)
    assert estimates.directly_attributed_observed_seconds == 10.0
