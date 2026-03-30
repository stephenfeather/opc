#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Session state collector - PostToolUse hook for mini-handoff generation.

Collects file edits, writes, and bash commands into a per-session JSONL state
file. The daemon's generate_mini_handoff.py reads this state file to produce
handoff YAML without needing the full session JSONL transcript.

State file location: <project>/.claude/cache/session-state/<session-id>.jsonl

Each line is a JSON object:
  {timestamp, tool, file?, command?, exit_code?}
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path


def get_state_file(session_id: str, project_dir: str) -> Path:
    """Get the state file path for this session."""
    state_dir = Path(project_dir) / ".claude" / "cache" / "session-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{session_id}.jsonl"


def process_event(tool_info: dict) -> dict | None:
    """Extract a state event from PostToolUse hook input.

    Returns a JSONL-ready dict, or None if this event should be skipped.
    """
    tool_name = tool_info.get("tool_name", "")
    tool_input = tool_info.get("tool_input", {}) or {}
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    if tool_name in ("Edit", "MultiEdit"):
        file_path = tool_input.get("file_path", "")
        if not file_path:
            return None
        return {"timestamp": timestamp, "tool": tool_name, "file": file_path}

    if tool_name == "Write":
        file_path = tool_input.get("file_path", "")
        if not file_path:
            return None
        return {"timestamp": timestamp, "tool": tool_name, "file": file_path}

    if tool_name == "Read":
        file_path = tool_input.get("file_path", "")
        if not file_path:
            return None
        return {"timestamp": timestamp, "tool": tool_name, "file": file_path}

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if not command:
            return None
        # Extract exit code from tool response
        exit_code = None
        for key in ("tool_output", "tool_response", "tool_result"):
            sub = tool_info.get(key, {})
            if isinstance(sub, dict):
                ec = sub.get("exit") or sub.get("exit_code")
                if ec is not None:
                    try:
                        exit_code = int(ec)
                    except (ValueError, TypeError):
                        pass
                    break
        event: dict = {"timestamp": timestamp, "tool": tool_name, "command": command}
        if exit_code is not None:
            event["exit_code"] = exit_code
        return event

    return None


def main() -> None:
    """Read stdin, append event to state file, exit 0."""
    try:
        stdin_data = sys.stdin.read()
        tool_info = json.loads(stdin_data) if stdin_data.strip() else {}
    except json.JSONDecodeError:
        sys.exit(0)

    event = process_event(tool_info)
    if event is None:
        sys.exit(0)

    session_id = tool_info.get("session_id", "") or os.environ.get("CLAUDE_SESSION_ID", "")
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())

    if not session_id:
        sys.exit(0)

    state_file = get_state_file(session_id, project_dir)

    try:
        with open(state_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass  # Non-fatal — don't break the hook chain

    sys.exit(0)


if __name__ == "__main__":
    main()
