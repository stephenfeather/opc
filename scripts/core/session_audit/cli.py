"""Argument interface for the session-audit command."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
import stat
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tomllib import TOMLDecodeError, load
from typing import cast

from scripts.core.config import get_config
from scripts.core.session_audit.detector import DetectionPolicy, detect_mistakes
from scripts.core.session_audit.judge import (
    MessagesTransport,
    apply_judge_outcome_to_result,
    judge_detection,
)
from scripts.core.session_audit.models import (
    AnalysisStatus,
    AuditExitCode,
    AuditLimits,
    AuditResult,
    OutputFormat,
    RunProvenance,
)
from scripts.core.session_audit.parser import parse_session
from scripts.core.session_audit.reporting import (
    build_audit_result,
    render_json,
    render_markdown,
)


@dataclass(frozen=True)
class AuditExecution:
    """Report plus process semantics, kept separate from analysis status."""

    result: AuditResult
    exit_code: AuditExitCode
    warnings: tuple[str, ...] = ()


class AuditOperationalError(RuntimeError):
    """Safe, expected command failure that maps to exit code 1."""


class _InputOpenError(AuditOperationalError):
    """The requested transcript cannot be opened."""


class _NonRegularInputError(AuditOperationalError):
    """The requested transcript is not a regular file."""


class _InputOutputIdentityError(AuditOperationalError):
    """Input/output identity could not be validated."""


class _OutputAliasError(AuditOperationalError):
    """The requested output aliases the input transcript."""


def build_parser() -> argparse.ArgumentParser:
    """Build the stable command-specific argument parser."""
    parser = argparse.ArgumentParser(
        prog="opc session audit",
        description="Find mistakes and estimate avoidable time in a Claude session.",
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        type=Path,
        metavar="PATH",
        help="Claude Code session JSONL file to analyze",
    )
    parser.add_argument(
        "--format",
        type=OutputFormat,
        choices=tuple(OutputFormat),
        default=OutputFormat.MARKDOWN,
        help="report representation (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="atomically write the report to PATH instead of stdout",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_false",
        dest="include_thinking",
        help="exclude assistant thinking blocks from analysis",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "send up to 12 bounded, best-effort credential-redacted excerpts to the "
            "configured Anthropic model; project or personal data may remain, raw JSONL and "
            "unselected records are not uploaded, but a short session's normalized content "
            "can fit inside a candidate window; omitting this flag keeps analysis local"
        ),
    )
    parser.add_argument(
        "--judge-model",
        metavar="MODEL",
        help="override [session_audit].judge_model for an opt-in --judge run",
    )
    return parser


def _fallback_project_version() -> str:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            project = load(handle).get("project")
    except (OSError, TOMLDecodeError):
        return "unknown"
    if isinstance(project, dict):
        version = project.get("version")
        if isinstance(version, str) and version:
            return version
    return "unknown"


def resolve_tool_version() -> str:
    """Resolve installed package metadata, then the checkout version, honestly."""

    try:
        return importlib.metadata.version("mcp-execution")
    except importlib.metadata.PackageNotFoundError:
        return _fallback_project_version()


def resolve_judge_model(override: str | None) -> str:
    """Resolve CLI override before the dedicated session-audit config default."""

    return override if override is not None else get_config().session_audit.judge_model


def _read_anthropic_api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


def execute_audit(
    jsonl_path: Path,
    *,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    include_thinking: bool = True,
    limits: AuditLimits | None = None,
    judge_requested: bool = False,
    judge_model: str | None = None,
    api_key_provider: Callable[[], str | None] = _read_anthropic_api_key,
    judge_transport: MessagesTransport | None = None,
) -> AuditExecution:
    """Run the bounded local pipeline and optional one-call judge stage."""

    effective_limits = limits or AuditLimits()
    effective_judge_model = resolve_judge_model(judge_model) if judge_requested else None
    run = RunProvenance(
        tool_version=resolve_tool_version(),
        output_format=output_format,
        thinking_included=include_thinking,
        judge_requested=judge_requested,
        judge_model=effective_judge_model,
        judge_timeout_seconds=effective_limits.judge_deadline_seconds,
        effective_limits=effective_limits,
    )
    parsed = parse_session(
        jsonl_path,
        include_thinking=include_thinking,
        limits=effective_limits,
    )
    warnings: list[str] = []
    if parsed.diagnostics.malformed_lines:
        warnings.append(f"parser_malformed_lines={parsed.diagnostics.malformed_lines}")
    if parsed.diagnostics.unknown_record_types:
        warnings.append(f"parser_unknown_record_types={parsed.diagnostics.unknown_record_types}")
    if parsed.status is AnalysisStatus.REFUSED:
        return AuditExecution(
            result=build_audit_result(parsed, None, run=run),
            exit_code=AuditExitCode.REFUSED,
            warnings=tuple(warnings),
        )

    detection = detect_mistakes(
        parsed,
        include_thinking=include_thinking,
        policy=DetectionPolicy(max_candidates=effective_limits.max_local_candidates),
    )
    judge_outcome = None
    if judge_requested:
        judge_outcome = asyncio.run(
            judge_detection(
                parsed,
                detection,
                model=effective_judge_model or "",
                api_key=None,
                api_key_provider=api_key_provider,
                limits=effective_limits,
                transport=judge_transport,
            )
        )
        result = apply_judge_outcome_to_result(
            parsed,
            detection,
            run=run,
            outcome=judge_outcome,
        )
        if judge_outcome.diagnostics.status.value in {"partial", "failed"}:
            warnings.append(f"judge_status={judge_outcome.diagnostics.status.value}")
            if judge_outcome.diagnostics.failure_code is not None:
                warnings.append(f"judge_failure={judge_outcome.diagnostics.failure_code}")
    else:
        result = build_audit_result(parsed, detection, run=run)

    if detection.overflowed:
        warnings.append(f"candidate_overflow_omitted={detection.omitted_candidates}")
        exit_code = AuditExitCode.CANDIDATE_OVERFLOW
    elif judge_outcome is not None and judge_outcome.requires_exit_3:
        exit_code = AuditExitCode.JUDGE_FAILURE
    else:
        exit_code = AuditExitCode.COMPLETE
    return AuditExecution(result=result, exit_code=exit_code, warnings=tuple(warnings))


def _regular_input(path: Path) -> None:
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise _InputOpenError from exc
    if not stat.S_ISREG(mode):
        raise _NonRegularInputError


def _reject_output_alias(input_path: Path, output_path: Path | None) -> None:
    if output_path is None:
        return
    try:
        resolved_alias = input_path.resolve(strict=True) == output_path.resolve(strict=False)
        same_file = output_path.exists() and os.path.samefile(input_path, output_path)
    except OSError as exc:
        raise _InputOutputIdentityError from exc
    if resolved_alias or same_file:
        raise _OutputAliasError


def _render(result: AuditResult, output_format: OutputFormat) -> str:
    rendered = (
        render_json(result) if output_format is OutputFormat.JSON else render_markdown(result)
    )
    return cast(str, rendered)


def _atomic_write(path: Path, content: str) -> None:
    """Publish fully rendered UTF-8 content with a same-directory atomic rename."""

    temporary_name: str | None = None
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded deterministic audit and emit only its selected report."""

    args = build_parser().parse_args(argv)
    try:
        _regular_input(args.jsonl)
        _reject_output_alias(args.jsonl, args.output)
        execution = execute_audit(
            args.jsonl,
            output_format=args.format,
            include_thinking=args.include_thinking,
            judge_requested=args.judge,
            judge_model=args.judge_model,
        )
        rendered = _render(execution.result, args.format)
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            _atomic_write(args.output, rendered)
        for warning in execution.warnings:
            print(f"session-audit warning: {warning}", file=sys.stderr)
        return int(execution.exit_code)
    except _InputOpenError:
        print("session-audit error: input cannot be opened", file=sys.stderr)
        return int(AuditExitCode.OPERATIONAL_FAILURE)
    except _NonRegularInputError:
        print("session-audit error: input is not a regular file", file=sys.stderr)
        return int(AuditExitCode.OPERATIONAL_FAILURE)
    except _InputOutputIdentityError:
        print("session-audit error: input/output identity cannot be validated", file=sys.stderr)
        return int(AuditExitCode.OPERATIONAL_FAILURE)
    except _OutputAliasError:
        print("session-audit error: output must not alias the input transcript", file=sys.stderr)
        return int(AuditExitCode.OPERATIONAL_FAILURE)
    except Exception:  # noqa: BLE001 - top-level boundary emits no exception details
        print("session-audit error: unexpected operational failure", file=sys.stderr)
        return int(AuditExitCode.OPERATIONAL_FAILURE)


__all__ = [
    "AuditExecution",
    "AuditOperationalError",
    "build_parser",
    "execute_audit",
    "main",
    "resolve_judge_model",
    "resolve_tool_version",
]
