#!/usr/bin/env python3
"""
Extract workflow patterns from session JSONL files.

Parses tool_use calls to capture command/tool sequences and detect
common workflow patterns (test-edit-test, search-read-edit, etc.).

Usage:
    python extract_workflow_patterns.py --jsonl path/to/session.jsonl
    python extract_workflow_patterns.py --jsonl path/to/session.jsonl --patterns-only
    python extract_workflow_patterns.py --jsonl path/to/session.jsonl --output /tmp/workflows.json
"""

import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path

faulthandler.enable(
    file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"),
    all_threads=True,
)


def extract_tool_uses(jsonl_path: Path) -> list[dict]:
    """Stream through JSONL and extract tool_use blocks with their results.

    Returns ordered list of dicts with:
        tool_name, input, timestamp, line_num, tool_use_id,
        result_content (str|None), is_error (bool)
    """
    tool_uses = []
    # First pass: collect all tool_use entries
    pending_ids: dict[str, int] = {}  # tool_use_id -> index in tool_uses

    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            message = data.get("message", {})
            content = message.get("content")
            if not isinstance(content, list):
                continue

            if data.get("type") == "assistant":
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "tool_use":
                        continue

                    tool_name = item.get("name", "")
                    tool_input = item.get("input", {})
                    tool_id = item.get("id", "")

                    entry = {
                        "tool_name": tool_name,
                        "input": _extract_input(tool_name, tool_input),
                        "timestamp": data.get("timestamp"),
                        "line_num": line_num,
                        "tool_use_id": tool_id,
                        "result_error": None,
                    }
                    pending_ids[tool_id] = len(tool_uses)
                    tool_uses.append(entry)

            elif data.get("type") == "user":
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "tool_result":
                        continue

                    tid = item.get("tool_use_id", "")
                    if tid in pending_ids:
                        idx = pending_ids[tid]
                        tool_uses[idx]["result_error"] = (
                            item.get("is_error", False)
                        )

    return tool_uses


def _extract_input(tool_name: str, tool_input: dict) -> dict:
    """Extract the relevant fields from tool input based on tool type."""
    if tool_name in ("Read", "Edit", "Write", "MultiEdit"):
        return {"file_path": tool_input.get("file_path", "")}
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return {"command": cmd[:500]}
    elif tool_name in ("Grep", "Glob"):
        return {
            "pattern": tool_input.get("pattern", ""),
            "path": tool_input.get("path", ""),
        }
    elif tool_name == "Agent":
        return {
            "subagent_type": tool_input.get("subagent_type", ""),
            "description": tool_input.get("description", ""),
        }
    else:
        return {}


# Pattern definitions: (name, sequence of tool names, min_length)
WORKFLOW_PATTERNS = [
    ("test-edit-test", ["Bash", ("Edit", "Write"), "Bash"], 3),
    ("search-read-edit", [("Grep", "Glob"), "Read", ("Edit", "Write")], 3),
    ("read-edit", ["Read", ("Edit", "Write")], 2),
]


def _matches_step(tool_name: str, step) -> bool:
    """Check if a tool name matches a pattern step (string or tuple)."""
    if isinstance(step, tuple):
        return tool_name in step
    return tool_name == step


def detect_workflow_sequences(tool_uses: list[dict]) -> list[dict]:
    """Identify common workflow patterns in the tool use sequence.

    Returns list of dicts:
        pattern_type, tools (list of tool entries), files (list),
        success (bool|None)
    """
    patterns_found = []

    for pat_name, pat_steps, min_len in WORKFLOW_PATTERNS:
        i = 0
        while i < len(tool_uses):
            # Try to match the pattern starting at position i
            matched = []
            step_idx = 0
            j = i

            while j < len(tool_uses) and step_idx < len(pat_steps):
                if _matches_step(
                    tool_uses[j]["tool_name"], pat_steps[step_idx]
                ):
                    matched.append(tool_uses[j])
                    step_idx += 1
                elif step_idx > 0:
                    # Pattern broken, restart
                    break
                j += 1

            if step_idx == len(pat_steps) and len(matched) >= min_len:
                # Determine success: check the last Bash result
                success = None
                for m in reversed(matched):
                    if m["tool_name"] == "Bash":
                        if m["result_error"] is not None:
                            success = not m["result_error"]
                        break

                files = list({
                    m["input"].get("file_path", "")
                    for m in matched
                    if m["input"].get("file_path")
                })

                patterns_found.append({
                    "pattern_type": pat_name,
                    "tools": [
                        {
                            "tool_name": m["tool_name"],
                            "input": m["input"],
                        }
                        for m in matched
                    ],
                    "files": files,
                    "success": success,
                })
                i = j  # skip past matched sequence
            else:
                i += 1

    return patterns_found


def summarize_tool_usage(tool_uses: list[dict]) -> dict:
    """Summarize tool usage frequency and files touched."""
    tool_counts: dict[str, int] = {}
    files_by_tool: dict[str, set] = {}
    commands: list[str] = []

    for tu in tool_uses:
        name = tu["tool_name"]
        tool_counts[name] = tool_counts.get(name, 0) + 1

        fp = tu["input"].get("file_path", "")
        if fp:
            files_by_tool.setdefault(name, set()).add(fp)

        cmd = tu["input"].get("command", "")
        if cmd:
            commands.append(cmd)

    return {
        "tool_counts": tool_counts,
        "files_by_tool": {
            k: sorted(v) for k, v in files_by_tool.items()
        },
        "unique_commands": len(set(commands)),
        "total_tool_calls": len(tool_uses),
    }


def format_pattern_as_learning(pattern: dict) -> str:
    """Format a detected workflow pattern as a learning string."""
    pat_type = pattern["pattern_type"]
    tools = pattern["tools"]
    files = pattern["files"]
    success = pattern["success"]

    tool_seq = " -> ".join(t["tool_name"] for t in tools)
    file_list = ", ".join(os.path.basename(f) for f in files[:5])
    status = "succeeded" if success else "failed" if success is False else "unknown"

    parts = [f"Workflow pattern ({pat_type}): {tool_seq}"]
    if file_list:
        parts.append(f"Files: {file_list}")
    parts.append(f"Outcome: {status}")

    # Add command details for test-edit-test
    if pat_type == "test-edit-test":
        for t in tools:
            if t["tool_name"] == "Bash":
                cmd = t["input"].get("command", "")
                if cmd:
                    parts.append(f"Command: {cmd[:200]}")
                    break

    return ". ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Extract workflow patterns from session JSONL"
    )
    parser.add_argument(
        "--jsonl", required=True, help="Path to session JSONL file"
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format",
    )
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--patterns-only",
        action="store_true",
        help="Only output detected workflow patterns",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show tool usage statistics only",
    )

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    tool_uses = extract_tool_uses(jsonl_path)

    if args.stats:
        summary = summarize_tool_usage(tool_uses)
        print(json.dumps(summary, indent=2))
        return

    patterns = detect_workflow_sequences(tool_uses)

    if args.patterns_only:
        result = patterns
    else:
        result = {
            "tool_uses": [
                {
                    "tool_name": t["tool_name"],
                    "input": t["input"],
                    "timestamp": t["timestamp"],
                    "result_error": t["result_error"],
                }
                for t in tool_uses
            ],
            "patterns": patterns,
            "summary": summarize_tool_usage(tool_uses),
        }

    if args.format == "json":
        output = json.dumps(result, indent=2)
    else:
        if args.patterns_only:
            lines = []
            for p in patterns:
                lines.append(format_pattern_as_learning(p))
            output = "\n\n".join(lines) if lines else "No patterns detected."
        else:
            output = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(output)
        print(
            f"Wrote {len(patterns)} patterns from "
            f"{len(tool_uses)} tool uses to {args.output}",
            file=sys.stderr,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
