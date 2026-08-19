"""Tests for the deterministic session-audit command surface."""

from __future__ import annotations

import importlib.metadata
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.core.session_audit import cli
from scripts.core.session_audit.cli import build_parser, execute_audit, main
from scripts.core.session_audit.judge import JudgeRunStatus
from scripts.core.session_audit.models import (
    AnalysisStatus,
    AuditExitCode,
    AuditLimits,
    OutputFormat,
)


def test_parser_exposes_deterministic_and_opt_in_judge_flags_with_safe_defaults() -> None:
    parser = build_parser()

    defaults = parser.parse_args(["--jsonl", "session.jsonl"])
    explicit = parser.parse_args(
        [
            "--jsonl",
            "session.jsonl",
            "--format",
            "json",
            "--output",
            "report.json",
            "--no-thinking",
            "--judge",
            "--judge-model",
            "claude-test-model",
        ]
    )

    assert defaults.jsonl == Path("session.jsonl")
    assert defaults.format is OutputFormat.MARKDOWN
    assert defaults.output is None
    assert defaults.include_thinking is True
    assert defaults.judge is False
    assert defaults.judge_model is None
    assert explicit.format is OutputFormat.JSON
    assert explicit.output == Path("report.json")
    assert explicit.include_thinking is False
    assert explicit.judge is True
    assert explicit.judge_model == "claude-test-model"
    assert {action.dest for action in parser._actions} == {
        "help",
        "jsonl",
        "format",
        "output",
        "include_thinking",
        "judge",
        "judge_model",
    }
    normalized_help = " ".join(parser.format_help().split()).casefold()
    assert "project or personal data may remain" in normalized_help
    assert "raw jsonl and unselected records are not uploaded" in normalized_help
    assert "short session's normalized content" in normalized_help


def test_real_tiny_jsonl_runs_network_free_to_markdown_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "timestamp": "2026-08-11T12:00:00Z",
                "message": {"content": "The task is complete."},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(["--jsonl", str(transcript)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("# Claude Session Mistake Audit")
    assert "Status: `complete`" in captured.out
    assert "No confirmed or probable mistakes found" in captured.out
    assert captured.err == ""


def test_judge_with_no_candidates_does_not_read_key_or_touch_transport(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {"content": "No local mistake signal."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    key_reads = 0

    def forbidden_key_read() -> str | None:
        nonlocal key_reads
        key_reads += 1
        raise AssertionError("key provider must stay lazy when no candidate is submitted")

    limits = replace(AuditLimits(), judge_deadline_seconds=7.5)
    execution = execute_audit(
        transcript,
        output_format=OutputFormat.JSON,
        limits=limits,
        judge_requested=True,
        judge_model="claude-test-model",
        api_key_provider=forbidden_key_read,
        judge_transport=cast(Any, object()),
    )

    assert key_reads == 0
    assert execution.exit_code is AuditExitCode.COMPLETE
    assert execution.result.run.judge_requested is True
    assert execution.result.run.judge_model == "claude-test-model"
    assert execution.result.run.judge_timeout_seconds == 7.5
    assert execution.result.run.effective_limits.judge_deadline_seconds == 7.5
    assert execution.result.diagnostics["judge"].status is JudgeRunStatus.NOT_NEEDED


def test_no_thinking_excludes_thinking_from_the_opt_in_judge_payload(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "no-thinking-judge.jsonl"
    records = [
        {
            "type": "assistant",
            "uuid": "assistant-edit",
            "timestamp": "2026-08-11T12:00:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "edit-1",
                        "name": "Edit",
                        "input": {"file_path": "/src/parser.py"},
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "uuid": "assistant-admission",
            "parentUuid": "assistant-edit",
            "timestamp": "2026-08-11T12:00:10Z",
            "message": {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "PRIVATE_THINKING_SENTINEL",
                    },
                    {
                        "type": "text",
                        "text": "I was wrong about that parser.py edit.",
                    },
                ]
            },
        },
    ]
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    class CaptureThenFailTransport:
        def __init__(self) -> None:
            self.request_body: object | None = None

        async def create_message(
            self,
            request: object,
            *,
            api_key: str,
            timeout_seconds: float,
        ) -> dict[str, object]:
            del api_key, timeout_seconds
            self.request_body = cast(Any, request).body
            raise RuntimeError("fixed transport failure")

    transport = CaptureThenFailTransport()
    execution = execute_audit(
        transcript,
        output_format=OutputFormat.JSON,
        include_thinking=False,
        judge_requested=True,
        judge_model="claude-test-model",
        api_key_provider=lambda: "test-key-never-logged",
        judge_transport=cast(Any, transport),
    )

    assert execution.exit_code is AuditExitCode.JUDGE_FAILURE
    assert transport.request_body is not None
    payload_text = cast(Any, transport.request_body)["messages"][0]["content"]
    assert "PRIVATE_THINKING_SENTINEL" not in payload_text
    assert "assistant_thinking" not in payload_text
    assert "I was wrong about that parser.py edit." in payload_text


def test_successful_opt_in_judge_round_trip_updates_the_cli_result(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "successful-judge.jsonl"
    records = [
        {
            "type": "assistant",
            "uuid": "assistant-edit",
            "timestamp": "2026-08-11T12:00:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "edit-1",
                        "name": "Edit",
                        "input": {"file_path": "/src/parser.py"},
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "uuid": "assistant-admission",
            "parentUuid": "assistant-edit",
            "timestamp": "2026-08-11T12:00:10Z",
            "message": {"content": "I was wrong about that parser.py edit."},
        },
    ]
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    class EchoValidJudgeTransport:
        def __init__(self) -> None:
            self.calls = 0

        async def create_message(
            self,
            request: object,
            *,
            api_key: str,
            timeout_seconds: float,
        ) -> dict[str, object]:
            del timeout_seconds
            self.calls += 1
            assert api_key == "test-key-never-logged"
            request_body = cast(Any, request).body
            assert "test-key-never-logged" not in json.dumps(request_body)
            payload = json.loads(request_body["messages"][0]["content"])
            candidate = payload["candidates"][0]
            boundaries = candidate["local_boundaries"]
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "classify_session_mistakes",
                        "input": {
                            "results": [
                                {
                                    "episode_id": candidate["episode_id"],
                                    "classification": "confirmed",
                                    "category": "incorrect_change",
                                    "boundaries": boundaries,
                                    "evidence": [
                                        {
                                            "event_id": boundaries["onset_event_id"],
                                            "role": "affected_work",
                                        },
                                        {
                                            "event_id": boundaries["detection_event_id"],
                                            "role": "visible_admission",
                                        },
                                    ],
                                    "rationale": "The admission identifies the earlier edit.",
                                }
                            ]
                        },
                    }
                ],
            }

    transport = EchoValidJudgeTransport()
    execution = execute_audit(
        transcript,
        output_format=OutputFormat.JSON,
        judge_requested=True,
        judge_model="claude-test-model",
        api_key_provider=lambda: "test-key-never-logged",
        judge_transport=cast(Any, transport),
    )

    assert transport.calls == 1
    assert execution.exit_code is AuditExitCode.COMPLETE
    assert execution.result.status is AnalysisStatus.COMPLETE
    assert execution.result.diagnostics["judge"].status is JudgeRunStatus.COMPLETE
    judge_projection = cast(Any, execution.result.episodes[0].judge)
    assert judge_projection.decision_status == "accepted"
    assert execution.result.episodes[0].final_classification.value == "confirmed"


def test_requested_judge_missing_key_emits_partial_local_report_and_exit_three(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "mistake.jsonl"
    records = [
        {
            "type": "assistant",
            "uuid": "assistant-edit",
            "timestamp": "2026-08-11T12:00:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "edit-1",
                        "name": "Edit",
                        "input": {"file_path": "/src/parser.py"},
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "uuid": "assistant-admission",
            "parentUuid": "assistant-edit",
            "timestamp": "2026-08-11T12:00:10Z",
            "message": {"content": "I was wrong about that parser.py edit."},
        },
    ]
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    exit_code = main(["--jsonl", str(transcript), "--format", "json", "--judge"])

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert exit_code == 3
    assert document["status"] == "partial"
    assert document["run"]["judge_requested"] is True
    assert document["diagnostics"]["judge"]["status"] == "failed"
    assert document["diagnostics"]["judge"]["failure_code"] == "missing_api_key"
    assert document["episodes"][0]["local_classification"] == "confirmed"
    assert document["episodes"][0]["final_classification"] == "confirmed"
    assert captured.err == (
        "session-audit warning: judge_status=failed\n"
        "session-audit warning: judge_failure=missing_api_key\n"
    )


def test_output_atomically_receives_exact_report_and_keeps_stdout_empty(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {"content": "No local mistake signal."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"

    exit_code = main(
        [
            "--jsonl",
            str(transcript),
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert list(tmp_path.glob("audit.json.*.tmp")) == []


def test_output_hardlink_alias_is_rejected_before_the_transcript_can_be_replaced(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript = tmp_path / "session.jsonl"
    original = (
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {"content": "Preserve this transcript."},
            }
        )
        + "\n"
    ).encode()
    transcript.write_bytes(original)
    output_alias = tmp_path / "report.json"
    os.link(transcript, output_alias)

    exit_code = main(
        ["--jsonl", str(transcript), "--output", str(output_alias), "--format", "json"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "must not alias" in captured.err
    assert transcript.read_bytes() == original
    assert output_alias.read_bytes() == original


def test_replace_failure_preserves_old_output_cleans_temp_and_hides_error_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {"content": "A valid session."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    output.write_text("old report\n", encoding="utf-8")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("publish failed with sk-12345678")

    monkeypatch.setattr(os, "replace", fail_replace)

    exit_code = main(["--jsonl", str(transcript), "--output", str(output), "--format", "json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert output.read_text(encoding="utf-8") == "old report\n"
    assert list(tmp_path.glob("report.json.*.tmp")) == []
    assert captured.err == "session-audit error: unexpected operational failure\n"


def test_output_failure_overrides_requested_judge_failure_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "mistake-output.jsonl"
    records = [
        {
            "type": "assistant",
            "uuid": "assistant-edit",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "edit-1",
                        "name": "Edit",
                        "input": {"file_path": "/src/a.py"},
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "uuid": "assistant-admission",
            "parentUuid": "assistant-edit",
            "message": {"content": "I was wrong about that a.py edit."},
        },
    ]
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    output.write_text("old report\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("publish failed with sk-12345678")

    monkeypatch.setattr(os, "replace", fail_replace)

    exit_code = main(
        [
            "--jsonl",
            str(transcript),
            "--format",
            "json",
            "--output",
            str(output),
            "--judge",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert output.read_text(encoding="utf-8") == "old report\n"
    assert list(tmp_path.glob("report.json.*.tmp")) == []
    assert captured.err == "session-audit error: unexpected operational failure\n"


def test_fifo_input_is_rejected_before_parser_open_can_block(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fifo = tmp_path / "session.jsonl"
    os.mkfifo(fifo)

    exit_code = main(["--jsonl", str(fifo)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "input is not a regular file" in captured.err


def test_json_stdout_stays_parseable_with_warnings_and_no_thinking_provenance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "{not-json}\n"
        + json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "I was wrong privately."},
                        {"type": "text", "text": "Visible answer."},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(["--jsonl", str(transcript), "--format", "json", "--no-thinking"])

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert exit_code == 0
    assert document["run"]["thinking_included"] is False
    assert document["diagnostics"]["parser"]["normalized_events"] == 1
    assert captured.err == "session-audit warning: parser_malformed_lines=1\n"


def test_unsupported_schema_emits_refusal_report_and_exit_four(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript = tmp_path / "unsupported.jsonl"
    transcript.write_text(
        json.dumps({"type": "attachment", "payload": "not a message schema"}) + "\n",
        encoding="utf-8",
    )

    exit_code = main(["--jsonl", str(transcript), "--format", "json"])

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert exit_code == 4
    assert document["status"] == "refused"
    assert document["summary"]["local_estimates"] is None
    assert document["diagnostics"]["parser"]["refusal_reasons"] == ["unsupported_schema"]
    assert captured.err == "session-audit warning: parser_unknown_record_types=1\n"


def test_candidate_overflow_keeps_partial_status_separate_from_exit_five(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "overflow.jsonl"
    records = [
        {
            "type": "user",
            "uuid": "user-1",
            "message": {"content": "You were wrong about the first edit."},
        },
        {
            "type": "user",
            "uuid": "user-2",
            "parentUuid": "user-1",
            "message": {"content": "You were wrong about the second edit."},
        },
    ]
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    limits = replace(AuditLimits(), max_local_candidates=1)

    execution = execute_audit(
        transcript,
        output_format=OutputFormat.JSON,
        limits=limits,
    )

    assert execution.exit_code is AuditExitCode.CANDIDATE_OVERFLOW
    assert execution.result.status is AnalysisStatus.PARTIAL
    assert execution.result.run.effective_limits is limits
    assert execution.result.summary["totals_complete"] is False
    assert execution.result.summary["omitted_candidates"] == 1
    assert execution.result.summary["local_estimates"] is None
    assert execution.warnings == ("candidate_overflow_omitted=1",)


def test_candidate_overflow_precedes_requested_judge_failure_exit(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "overflow-judge.jsonl"
    records = [
        {
            "type": "user",
            "uuid": "user-1",
            "message": {"content": "You were wrong about the first edit."},
        },
        {
            "type": "user",
            "uuid": "user-2",
            "parentUuid": "user-1",
            "message": {"content": "You were wrong about the second edit."},
        },
    ]
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    execution = execute_audit(
        transcript,
        output_format=OutputFormat.JSON,
        limits=replace(AuditLimits(), max_local_candidates=1),
        judge_requested=True,
        judge_model="claude-test-model",
        api_key_provider=lambda: None,
    )

    assert execution.exit_code is AuditExitCode.CANDIDATE_OVERFLOW
    assert execution.result.status is AnalysisStatus.PARTIAL
    assert execution.result.diagnostics["judge"].status is JudgeRunStatus.FAILED
    assert execution.warnings == (
        "judge_status=failed",
        "judge_failure=missing_api_key",
        "candidate_overflow_omitted=1",
    )


def test_parser_refusal_with_judge_never_reads_key_and_keeps_exit_four(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "unsupported-judge.jsonl"
    transcript.write_text(
        json.dumps({"type": "attachment", "payload": "not a message schema"}) + "\n",
        encoding="utf-8",
    )

    def forbidden_key_read() -> str | None:
        raise AssertionError("a refused parse must never read the judge key")

    execution = execute_audit(
        transcript,
        output_format=OutputFormat.JSON,
        judge_requested=True,
        judge_model="claude-test-model",
        api_key_provider=forbidden_key_read,
    )

    assert execution.exit_code is AuditExitCode.REFUSED
    assert execution.result.status is AnalysisStatus.REFUSED
    assert execution.result.run.judge_requested is True
    assert execution.result.diagnostics["judge"] is None


def test_tool_version_falls_back_to_checkout_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def package_missing(distribution_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(importlib.metadata, "version", package_missing)

    assert cli.resolve_tool_version() == "0.8.0"


def test_judge_model_precedence_is_cli_then_session_audit_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_config",
        lambda: SimpleNamespace(session_audit=SimpleNamespace(judge_model="configured-model")),
    )

    assert cli.resolve_judge_model("cli-model") == "cli-model"
    assert cli.resolve_judge_model(None) == "configured-model"


def test_unexpected_internal_failure_maps_to_safe_exit_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    def fail_execution(*args: object, **kwargs: object) -> object:
        raise RuntimeError("internal failure sk-12345678 private customer transcript detail")

    monkeypatch.setattr(cli, "execute_audit", fail_execution)

    exit_code = main(["--jsonl", str(transcript)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "session-audit error: unexpected operational failure\n"


def test_real_detected_admission_flows_through_typed_json_episode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript = tmp_path / "mistake.jsonl"
    records = [
        {
            "type": "assistant",
            "uuid": "assistant-edit",
            "timestamp": "2026-08-11T12:00:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "edit-1",
                        "name": "Edit",
                        "input": {"file_path": "/src/parser.py"},
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "uuid": "assistant-admission",
            "parentUuid": "assistant-edit",
            "timestamp": "2026-08-11T12:00:10Z",
            "message": {"content": "I was wrong about that parser.py edit."},
        },
    ]
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    exit_code = main(["--jsonl", str(transcript), "--format", "json"])

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert document["summary"]["confirmed_episodes"] == 1
    assert document["episodes"][0]["local_classification"] == "confirmed"
    assert document["episodes"][0]["final_classification"] == "confirmed"
    assert document["episodes"][0]["local_affected_event_ids"] == ["L1:B0", "L2:B0"]
    assert document["episodes"][0]["judge"] is None
