#!/usr/bin/env python3
"""
Generate mini-handoff YAML from session JSONL files.

Mechanical parsing -- no LLM calls. Produces a structured session summary
with files touched, commands run, git state, and tool usage statistics.

Supports two data sources for Phase 3 readiness:
  - JSONL session transcript (default, Phase 2)
  - Hook-collected state file (Phase 3, when available)

Usage:
    python generate_mini_handoff.py --jsonl path/to/session.jsonl --session-id s-abc123 --project-dir /path/to/project
    python generate_mini_handoff.py --jsonl path/to/session.jsonl --session-id s-abc123 --project-dir /path/to/project --format json
"""

import argparse
import faulthandler
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

faulthandler.enable(
    file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"),
    all_threads=True,
)


def extract_files_touched(jsonl_path: Path) -> dict[str, list[str]]:
    """Scan JSONL for file operations and group by operation type.

    Returns dict with keys: read, modified, created.
    'created' = files that appear in Write but not in any prior Read.
    """
    read_files: list[str] = []
    edit_files: list[str] = []
    write_files: list[str] = []
    seen_reads: set[str] = set()

    with open(jsonl_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            if entry.get("type") != "assistant":
                continue

            content = entry.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue

                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
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

    # Classify: Write to a file never Read before = created
    created = [f for f in write_files if f not in seen_reads]
    modified = list(dict.fromkeys(edit_files + [f for f in write_files if f in seen_reads]))

    return {
        "read": read_files,
        "modified": modified,
        "created": created,
    }


def extract_commands_run(jsonl_path: Path) -> list[dict]:
    """Scan JSONL for Bash tool_use blocks and extract commands.

    Returns list of dicts with: command, timestamp.
    """
    commands = []

    with open(jsonl_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            if entry.get("type") != "assistant":
                continue

            timestamp = entry.get("timestamp", "")
            content = entry.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                if block.get("name") != "Bash":
                    continue

                command = block.get("input", {}).get("command", "")
                if command:
                    commands.append({
                        "command": command[:500],
                        "timestamp": timestamp,
                    })

    return commands


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


def extract_timestamps(jsonl_path: Path) -> dict[str, str]:
    """Extract first and last timestamps from JSONL."""
    first_ts = ""
    last_ts = ""

    with open(jsonl_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            ts = entry.get("timestamp", "")
            if ts:
                if not first_ts:
                    first_ts = ts
                last_ts = ts

    return {"first_timestamp": first_ts, "last_timestamp": last_ts}


def extract_tool_counts(jsonl_path: Path) -> dict[str, int]:
    """Count tool usage by tool name."""
    counts: dict[str, int] = {}

    with open(jsonl_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            if entry.get("type") != "assistant":
                continue

            content = entry.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_name = block.get("name", "unknown")
                counts[tool_name] = counts.get(tool_name, 0) + 1

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _handoff_from_jsonl(jsonl_path: Path, session_id: str, project_dir: str) -> dict:
    """Build handoff dict from JSONL session transcript."""
    files = extract_files_touched(jsonl_path)
    commands = extract_commands_run(jsonl_path)
    git_state = extract_git_state(commands)
    timestamps = extract_timestamps(jsonl_path)
    tool_usage = extract_tool_counts(jsonl_path)

    # Extract date from first timestamp
    date_str = ""
    if timestamps["first_timestamp"]:
        try:
            dt = datetime.fromisoformat(timestamps["first_timestamp"].replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Limit commands to last 50
    recent_commands = [c["command"] for c in commands[-50:]]

    return {
        "session": session_id,
        "date": date_str,
        "status": "complete",
        "outcome": "auto-extracted",
        "goal": "Auto-extracted session summary",
        "files": {
            "read": files["read"],
            "modified": files["modified"],
            "created": files["created"],
        },
        "commands_run": recent_commands,
        "git_state": git_state,
        "tool_usage": tool_usage,
        "duration": timestamps,
    }


def _handoff_from_state_file(state_file: Path, session_id: str, project_dir: str) -> dict:
    """Build handoff dict from hook-collected state file (Phase 3).

    State file is JSONL with events: {timestamp, tool, file?, command?, exit_code?}
    """
    read_files: list[str] = []
    edited_files: list[str] = []
    created_files: list[str] = []
    commands: list[dict] = []
    tool_counts: dict[str, int] = {}
    first_ts = ""
    last_ts = ""

    with open(state_file) as f:
        for line in f:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

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
            elif tool == "Write" and file_path and file_path not in created_files:
                created_files.append(file_path)

            command = event.get("command", "")
            if tool == "Bash" and command:
                commands.append({"command": command, "timestamp": ts})

    git_state = extract_git_state(commands)

    date_str = ""
    if first_ts:
        try:
            dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    recent_commands = [c["command"] for c in commands[-50:]]

    return {
        "session": session_id,
        "date": date_str,
        "status": "complete",
        "outcome": "auto-extracted",
        "goal": "Auto-extracted session summary",
        "files": {
            "read": read_files,
            "modified": edited_files,
            "created": created_files,
        },
        "commands_run": recent_commands,
        "git_state": git_state,
        "tool_usage": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
        "duration": {"first_timestamp": first_ts, "last_timestamp": last_ts},
    }


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
        return _handoff_from_state_file(state_file, session_id, project_dir)
    elif jsonl_path and jsonl_path.exists():
        return _handoff_from_jsonl(jsonl_path, session_id, project_dir)
    else:
        raise ValueError("No data source provided (need jsonl_path or state_file)")


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
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value
    elif isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            if isinstance(item, str):
                formatted = _format_yaml_value(item)
                lines.append(f"\n{prefix}- {formatted}")
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
            if isinstance(v, (dict, list)) and v:
                lines.append(f"\n{prefix}{k}: {formatted}")
            else:
                lines.append(f"\n{prefix}{k}: {formatted}")
        return "".join(lines)
    return str(value)


def format_as_yaml(handoff: dict) -> str:
    """Format handoff dict as YAML string without pyyaml dependency."""
    lines = ["---"]
    # Frontmatter fields
    for key in ("session", "date", "status", "outcome"):
        val = handoff.get(key, "")
        lines.append(f"{key}: {_format_yaml_value(val)}")
    lines.append("---")
    lines.append("")

    # Body fields
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


def write_handoff(handoff: dict, output_dir: Path, session_id: str) -> Path:
    """Write handoff YAML to the auto-handoffs directory.

    Output path: <output_dir>/thoughts/shared/handoffs/auto/<session_id>.yaml
    Returns the written path.
    """
    handoff_dir = output_dir / "thoughts" / "shared" / "handoffs" / "auto"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    output_path = handoff_dir / f"{session_id}.yaml"
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
