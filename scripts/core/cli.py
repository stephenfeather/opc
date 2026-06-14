"""Human-facing OPC command dispatcher.

The MCP server is the agent-facing entrypoint for these tools. This module is
the human-facing twin: a small registry that discovers commands via ``opc
--help`` and shells out to the existing argparse scripts without changing them.
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

_SCRIPTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Command:
    """A command exposed by the ``opc`` dispatcher."""

    script: str
    description: str
    group: str


CommandKey = tuple[str, ...]


COMMANDS: dict[CommandKey, Command] = {
    ("recall",): Command(
        "recall_learnings.py",
        "Search learnings (semantic/hybrid)",
        "Recall & Store",
    ),
    ("store",): Command(
        "store_learning.py",
        "Store a learning",
        "Recall & Store",
    ),
    ("feedback",): Command(
        "memory_feedback.py",
        "Submit or inspect recall feedback",
        "Recall & Store",
    ),
    ("push",): Command(
        "push_learnings.py",
        "Surface high-value learnings proactively",
        "Recall & Store",
    ),
    ("artifact", "query"): Command(
        "artifact_query.py",
        "Search indexed context artifacts",
        "Artifacts",
    ),
    ("artifact", "mark"): Command(
        "artifact_mark.py",
        "Mark handoff outcome",
        "Artifacts",
    ),
    ("artifact", "index"): Command(
        "artifact_index.py",
        "Index context graph artifacts",
        "Artifacts",
    ),
    ("pattern", "detect"): Command(
        "pattern_batch.py",
        "Run cross-session pattern detection",
        "Patterns & Metrics",
    ),
    ("pattern", "batch"): Command(
        "pattern_batch.py",
        "Run cross-session pattern detection batch job",
        "Patterns & Metrics",
    ),
    ("pattern", "report"): Command(
        "pattern_report.py",
        "Show pattern detection reports",
        "Patterns & Metrics",
    ),
    ("metrics",): Command(
        "memory_metrics.py",
        "Memory system metrics",
        "Patterns & Metrics",
    ),
    ("duplicate-density",): Command(
        "duplicate_density.py",
        "Analyze near-duplicate learning density",
        "Patterns & Metrics",
    ),
    ("confidence", "calibrate"): Command(
        "confidence_calibrator.py",
        "Calibrate learning confidence scores",
        "Patterns & Metrics",
    ),
    ("type-affinity",): Command(
        "type_affinity.py",
        "Refresh type-affinity centroid cache",
        "Patterns & Metrics",
    ),
    ("daemon",): Command(
        "memory_daemon.py",
        "Start, stop, or inspect the memory daemon",
        "Daemon",
    ),
    ("extract", "session"): Command(
        "extract_session.py",
        "Extract learnings from one session for testing",
        "Extraction",
    ),
    ("extract", "thinking"): Command(
        "extract_thinking_blocks.py",
        "Extract thinking blocks from session JSONL",
        "Extraction",
    ),
    ("extract", "workflow"): Command(
        "extract_workflow_patterns.py",
        "Extract workflow patterns from session JSONL",
        "Extraction",
    ),
    ("handoff",): Command(
        "generate_mini_handoff.py",
        "Generate a mini-handoff from session JSONL",
        "Extraction",
    ),
    ("backfill", "sessions"): Command(
        "backfill_sessions.py",
        "Backfill unregistered sessions",
        "Maintenance",
    ),
    ("backfill", "learnings"): Command(
        "backfill_learnings.py",
        "Backfill learnings from archived sessions",
        "Maintenance",
    ),
    ("backfill", "kg"): Command(
        "backfill_kg.py",
        "Backfill knowledge graph rows",
        "Maintenance",
    ),
    ("backfill", "archive"): Command(
        "backfill_archive.py",
        "Archive JSONL files to S3",
        "Maintenance",
    ),
    ("re-embed", "voyage"): Command(
        "re_embed_voyage.py",
        "Re-embed learnings with Voyage",
        "Maintenance",
    ),
}

# Files with argparse CLIs that are intentionally not listed in the drift test.
# Keep this set explicit so new user-runnable scripts either join COMMANDS or
# document why they remain path-only.
EXCLUDED_CORE_SCRIPTS: set[str] = set()
_MAX_COMMAND_LEN = max(len(k) for k in COMMANDS)


def _format_command_name(key: CommandKey) -> str:
    return " ".join(key)


def _matching_commands(prefix: CommandKey) -> dict[CommandKey, Command]:
    return {
        key: command
        for key, command in COMMANDS.items()
        if len(key) > len(prefix) and key[: len(prefix)] == prefix
    }


def format_help(prefix: CommandKey = ()) -> str:
    """Render grouped help for all commands or a command prefix."""
    if prefix:
        commands = _matching_commands(prefix)
        title = f"opc {_format_command_name(prefix)}"
        description = "Available subcommands"
    else:
        commands = COMMANDS
        title = "opc - Opinionated Persistent Context"
        description = "Usage: opc <command> [args...]"

    lines = [title, "", description, ""]

    if not commands:
        lines.append("No subcommands are registered for this prefix.")
        return "\n".join(lines)

    grouped: dict[str, list[tuple[CommandKey, Command]]] = defaultdict(list)
    for key, command in commands.items():
        grouped[command.group].append((key, command))

    width = max(len(_format_command_name(key)) for key in commands)
    for group in sorted(grouped):
        lines.append(f"{group}:")
        for key, command in sorted(grouped[group], key=lambda item: item[0]):
            lines.append(f"  {_format_command_name(key):<{width}}  {command.description}")
        lines.append("")

    lines.append("Run `opc <command> --help` to see the wrapped script's help.")
    return "\n".join(lines)


def resolve_command(argv: list[str]) -> tuple[CommandKey | None, Command | None, list[str]]:
    """Resolve the longest registered command prefix from argv."""
    for length in range(min(len(argv), _MAX_COMMAND_LEN), 0, -1):
        key = tuple(argv[:length])
        command = COMMANDS.get(key)
        if command is not None:
            return key, command, argv[length:]
    return None, None, argv


def _print_unknown(argv: list[str], stream: TextIO) -> None:
    attempted = " ".join(argv) if argv else "<empty>"
    print(f"Unknown opc command: {attempted}", file=stream)
    print("", file=stream)
    print(format_help(), file=stream)


def execute_command(command: Command, args: list[str]) -> int:
    """Run a registered command script under the current Python interpreter."""
    script_path = _SCRIPTS_DIR / command.script
    if not script_path.exists():
        print(f"Registered script is missing: {script_path}", file=sys.stderr)
        return 127

    completed = subprocess.run([sys.executable, str(script_path), *args], check=False)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    """Console entrypoint for ``opc``."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args in (["-h"], ["--help"]):
        print(format_help())
        return 0

    key, command, remainder = resolve_command(args)
    if command is not None:
        return execute_command(command, remainder)

    prefix = tuple(args[:-1]) if args[-1] in {"-h", "--help"} else tuple(args)
    if prefix and _matching_commands(prefix):
        print(format_help(prefix))
        return 0

    _print_unknown(args, sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
