#!/usr/bin/env python3
"""
Generate mini-handoff YAML from session JSONL files.

Mechanical parsing -- no LLM calls. Produces a structured session summary
with files touched, commands run, git state, and tool usage statistics.

Supports two data sources for Phase 3 readiness:
  - JSONL session transcript (default, Phase 2)
  - Hook-collected state file (Phase 3, when available)

Usage:
    python generate_mini_handoff.py --jsonl path/to/session.jsonl \\
        --session-id s-abc123 --project-dir /path/to/project
    python generate_mini_handoff.py --jsonl path/to/session.jsonl \\
        --session-id s-abc123 --project-dir /path/to/project --format json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure parsers: lines -> structured data
# ---------------------------------------------------------------------------


def parse_jsonl_entries(lines: Iterable[str]) -> list[dict]:
    """Parse JSONL lines into a list of entry dicts, skipping non-dict/malformed lines."""
    entries: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


def parse_state_events(lines: Iterable[str]) -> list[dict]:
    """Parse Phase 3 state-file JSONL lines into event dicts."""
    return parse_jsonl_entries(lines)


def _iter_parsed_lines(lines: Iterable[str]) -> Iterable[dict]:
    """Yield parsed JSON dicts from lines, skipping non-dict/malformed ones. Lazy."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            yield parsed


# ---------------------------------------------------------------------------
# Pure extractors: entries -> derived data
# ---------------------------------------------------------------------------


def _iter_tool_blocks(entries: list[dict]):
    """Yield (tool_name, tool_input, timestamp) from assistant tool_use blocks."""
    for entry in entries:
        if entry.get("type") != "assistant":
            continue
        content = entry.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        timestamp = entry.get("timestamp", "")
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name") or "unknown"
            tool_input = block.get("input")
            if not isinstance(name, str):
                name = "unknown"
            if not isinstance(tool_input, dict):
                tool_input = {}
            yield name, tool_input, timestamp


def extract_files_touched(entries: list[dict]) -> dict[str, list[str]]:
    """Extract file operations from parsed entries, grouped by operation type.

    Returns dict with keys: read, modified, created.
    'created' = files that appear in Write but not in any prior Read.
    """
    read_files: list[str] = []
    edit_files: list[str] = []
    write_files: list[str] = []
    seen_reads: set[str] = set()

    for tool_name, tool_input, _ in _iter_tool_blocks(entries):
        file_path = tool_input.get("file_path", "")
        if not file_path:
            continue

        if tool_name == "Read":
            if file_path not in seen_reads:
                read_files.append(file_path)
                seen_reads.add(file_path)
        elif tool_name in ("Edit", "MultiEdit"):
            if file_path not in edit_files:
                edit_files.append(file_path)
        elif tool_name == "Write":
            if file_path not in write_files:
                write_files.append(file_path)

    created = [f for f in write_files if f not in seen_reads]
    modified = list(dict.fromkeys(edit_files + [f for f in write_files if f in seen_reads]))

    return {
        "read": read_files,
        "modified": modified,
        "created": created,
    }


def extract_commands_run(entries: list[dict]) -> list[dict]:
    """Extract Bash commands from parsed entries.

    Returns list of dicts with: command, timestamp.
    """
    return [
        {"command": tool_input.get("command", "")[:500], "timestamp": timestamp}
        for tool_name, tool_input, timestamp in _iter_tool_blocks(entries)
        if tool_name == "Bash" and tool_input.get("command", "")
    ]


def extract_git_state(commands: list[dict]) -> dict | None:
    """Find the last git command from the commands list.

    Returns dict with last_command and timestamp, or None.
    """
    last_git = None
    for cmd in commands:
        if "git" in cmd.get("command", ""):
            last_git = cmd

    if last_git:
        return {
            "last_command": last_git["command"],
            "timestamp": last_git["timestamp"],
        }
    return None


def extract_timestamps(entries: list[dict]) -> dict[str, str]:
    """Extract first and last timestamps from parsed entries."""
    first_ts = ""
    last_ts = ""

    for entry in entries:
        ts = entry.get("timestamp", "")
        if ts:
            if not first_ts:
                first_ts = ts
            last_ts = ts

    return {"first_timestamp": first_ts, "last_timestamp": last_ts}


def extract_tool_counts(entries: list[dict]) -> dict[str, int]:
    """Count tool usage by tool name from parsed entries."""
    counts: dict[str, int] = {}

    for tool_name, _, _ in _iter_tool_blocks(entries):
        counts[tool_name] = counts.get(tool_name, 0) + 1

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _extract_date(timestamp_str: str) -> str:
    """Extract date string from ISO timestamp, falling back to today's date."""
    if not timestamp_str:
        return datetime.now(UTC).strftime("%Y-%m-%d")
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return datetime.now(UTC).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Single-pass streaming accumulator
# ---------------------------------------------------------------------------


def _accumulate_from_entries(entries: Iterable[dict]) -> dict:
    """Single-pass accumulation of all extracted data from entry stream.

    Processes each entry exactly once, discarding it after extraction.
    Uses bounded storage: deque(maxlen=50) for commands, incremental last_git.
    Returns a raw accumulation dict (not a handoff — use _assemble_handoff).
    """
    read_files: list[str] = []
    edit_files: list[str] = []
    write_files: list[str] = []
    seen_reads: set[str] = set()
    recent_commands: deque[dict] = deque(maxlen=50)
    last_git: dict | None = None
    tool_counts: dict[str, int] = {}
    first_ts = ""
    last_ts = ""

    for entry in entries:
        ts = entry.get("timestamp", "")
        if ts:
            if not first_ts:
                first_ts = ts
            last_ts = ts

        if entry.get("type") != "assistant":
            continue

        content = entry.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue

        timestamp = entry.get("timestamp", "")
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            tool_name = block.get("name") or "unknown"
            if not isinstance(tool_name, str):
                tool_name = "unknown"
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                tool_input = {}
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

            file_path = tool_input.get("file_path", "")
            if file_path:
                if tool_name == "Read":
                    if file_path not in seen_reads:
                        read_files.append(file_path)
                        seen_reads.add(file_path)
                elif tool_name in ("Edit", "MultiEdit"):
                    if file_path not in edit_files:
                        edit_files.append(file_path)
                elif tool_name == "Write":
                    if file_path not in write_files:
                        write_files.append(file_path)

            if tool_name == "Bash":
                command = tool_input.get("command", "")
                if command:
                    cmd_entry = {"command": command[:500], "timestamp": timestamp}
                    recent_commands.append(cmd_entry)
                    if "git" in command:
                        last_git = cmd_entry

    created = [f for f in write_files if f not in seen_reads]
    modified = list(
        dict.fromkeys(edit_files + [f for f in write_files if f in seen_reads])
    )

    return {
        "read_files": read_files,
        "modified": modified,
        "created": created,
        "recent_commands": list(recent_commands),
        "last_git": last_git,
        "tool_counts": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def _assemble_handoff(acc: dict, session_id: str) -> dict:
    """Assemble a handoff dict from accumulation results."""
    git_state = None
    if acc["last_git"]:
        git_state = {
            "last_command": acc["last_git"]["command"],
            "timestamp": acc["last_git"]["timestamp"],
        }
    date_str = _extract_date(acc["first_ts"])
    recent_commands = [c["command"] for c in acc["recent_commands"]]

    return {
        "session": session_id,
        "date": date_str,
        "status": "complete",
        "outcome": "auto-extracted",
        "goal": "Auto-extracted session summary",
        "files": {
            "read": acc["read_files"],
            "modified": acc["modified"],
            "created": acc["created"],
        },
        "commands_run": recent_commands,
        "git_state": git_state,
        "tool_usage": acc["tool_counts"],
        "duration": {
            "first_timestamp": acc["first_ts"],
            "last_timestamp": acc["last_ts"],
        },
    }


# ---------------------------------------------------------------------------
# Pure assembly: extractors -> handoff dict (list-based, for testing)
# ---------------------------------------------------------------------------


def build_handoff_from_entries(
    entries: list[dict], session_id: str, _project_dir: str = ""
) -> dict:
    """Build handoff dict from parsed JSONL entries (pure function)."""
    acc = _accumulate_from_entries(entries)
    return _assemble_handoff(acc, session_id)


def _accumulate_from_state_events(events: Iterable[dict]) -> dict:
    """Single-pass accumulation from Phase 3 state events. Bounded storage."""
    read_files: list[str] = []
    edited_files: list[str] = []
    created_files: list[str] = []
    recent_commands: deque[dict] = deque(maxlen=50)
    last_git: dict | None = None
    tool_counts: dict[str, int] = {}
    first_ts = ""
    last_ts = ""

    for event in events:
        ts = event.get("timestamp", "")
        if ts:
            if not first_ts:
                first_ts = ts
            last_ts = ts

        tool = event.get("tool", "")
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

        file_path = event.get("file", "")
        if tool == "Read" and file_path and file_path not in read_files:
            read_files.append(file_path)
        elif tool in ("Edit", "MultiEdit") and file_path and file_path not in edited_files:
            edited_files.append(file_path)
        elif tool == "Write" and file_path:
            if file_path in read_files or file_path in edited_files:
                if file_path not in edited_files:
                    edited_files.append(file_path)
            elif file_path not in created_files:
                created_files.append(file_path)

        command = event.get("command", "")
        if tool == "Bash" and command:
            cmd_entry = {"command": command[:500], "timestamp": ts}
            recent_commands.append(cmd_entry)
            if "git" in command:
                last_git = cmd_entry

    return {
        "read_files": read_files,
        "modified": edited_files,
        "created": created_files,
        "recent_commands": list(recent_commands),
        "last_git": last_git,
        "tool_counts": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def build_handoff_from_state_events(
    events: Iterable[dict], session_id: str, _project_dir: str = ""
) -> dict:
    """Build handoff dict from Phase 3 state events (pure, streaming)."""
    acc = _accumulate_from_state_events(events)
    return _assemble_handoff(acc, session_id)


# ---------------------------------------------------------------------------
# Pure formatters
# ---------------------------------------------------------------------------


def _format_yaml_value(value, indent=0):
    """Format a value as YAML without external dependencies."""
    prefix = "  " * indent

    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        if any(c in value for c in ":{}\n[]#&*!|>',\"@`") or value.startswith("- "):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'"{escaped}"'
        return value
    elif isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            if isinstance(item, str):
                formatted = _format_yaml_value(item)
            else:
                formatted = _format_yaml_value(item, indent + 1)
            lines.append(f"\n{prefix}- {formatted}")
        return "".join(lines)
    elif isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for k, v in value.items():
            formatted = _format_yaml_value(v, indent + 1)
            lines.append(f"\n{prefix}{k}: {formatted}")
        return "".join(lines)
    return str(value)


def format_as_yaml(handoff: dict) -> str:
    """Format handoff dict as YAML string without pyyaml dependency."""
    lines = ["---"]
    for key in ("session", "date", "status", "outcome"):
        val = handoff.get(key, "")
        lines.append(f"{key}: {_format_yaml_value(val)}")
    lines.append("---")
    lines.append("")

    for key in ("goal", "files", "commands_run", "git_state", "tool_usage", "duration"):
        val = handoff.get(key)
        if val is None:
            continue
        formatted = _format_yaml_value(val, 1)
        if isinstance(val, (dict, list)) and val:
            lines.append(f"{key}:{formatted}")
        else:
            lines.append(f"{key}: {formatted}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# I/O boundary: file reading, writing, CLI
# ---------------------------------------------------------------------------


def generate_handoff(
    session_id: str,
    project_dir: str,
    jsonl_path: Path | None = None,
    state_file: Path | None = None,
) -> dict:
    """Generate a mini-handoff dict from available data source.

    Prefers state_file (Phase 3, real-time) over jsonl_path (Phase 2, post-session).
    """
    if state_file and state_file.exists():
        with open(state_file) as f:
            acc = _accumulate_from_state_events(_iter_parsed_lines(f))
        return _assemble_handoff(acc, session_id)
    elif jsonl_path and jsonl_path.exists():
        with open(jsonl_path) as f:
            acc = _accumulate_from_entries(_iter_parsed_lines(f))
        return _assemble_handoff(acc, session_id)
    else:
        raise ValueError("No data source provided (need jsonl_path or state_file)")


_SAFE_SESSION_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _sanitize_session_id(session_id: str) -> str:
    """Validate session_id contains only safe filename characters."""
    if not session_id or not _SAFE_SESSION_ID.match(session_id):
        raise ValueError(
            f"Invalid session_id: must contain only letters, digits, '_' or '-'"
            f" (pattern: ^[a-zA-Z0-9_-]+$), got {session_id!r}"
        )
    return session_id


def write_handoff(handoff: dict, output_dir: Path, session_id: str) -> Path:
    """Write handoff YAML to the auto-handoffs directory.

    Output path: <output_dir>/thoughts/shared/handoffs/auto/<session_id>.yaml
    Returns the written path.
    """
    safe_id = _sanitize_session_id(session_id)
    handoff_dir = output_dir / "thoughts" / "shared" / "handoffs" / "auto"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    output_path = handoff_dir / f"{safe_id}.yaml"
    output_path.write_text(format_as_yaml(handoff))
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate mini-handoff from session JSONL"
    )
    parser.add_argument(
        "--jsonl", help="Path to session JSONL file"
    )
    parser.add_argument(
        "--state-file", help="Path to hook-collected state file (Phase 3)"
    )
    parser.add_argument(
        "--session-id", required=True, help="Session identifier"
    )
    parser.add_argument(
        "--project-dir", required=True, help="Project directory for handoff output"
    )
    parser.add_argument(
        "--output", help="Override output path (default: auto directory)"
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format",
    )

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl) if args.jsonl else None
    state_file = Path(args.state_file) if args.state_file else None

    if not jsonl_path and not state_file:
        print("Error: Must provide --jsonl or --state-file", file=sys.stderr)
        sys.exit(1)

    if jsonl_path and not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    if state_file and not state_file.exists():
        print(f"Error: File not found: {state_file}", file=sys.stderr)
        sys.exit(1)

    handoff = generate_handoff(
        session_id=args.session_id,
        project_dir=args.project_dir,
        jsonl_path=jsonl_path,
        state_file=state_file,
    )

    if args.format == "json":
        output = json.dumps(handoff, indent=2)
    else:
        output = format_as_yaml(handoff)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"Written to {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
