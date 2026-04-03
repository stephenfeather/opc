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

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PatternStep = str | tuple[str, ...]


# ---------------------------------------------------------------------------
# Pure functions — input extraction
# ---------------------------------------------------------------------------

INPUT_EXTRACTORS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "Edit": ("file_path",),
    "Write": ("file_path",),
    "MultiEdit": ("file_path",),
    "Bash": ("command",),
    "Grep": ("pattern", "path"),
    "Glob": ("pattern", "path"),
    "Agent": ("subagent_type", "description"),
}


def _extract_input(tool_name: str, tool_input: dict) -> dict:
    """Extract the relevant fields from tool input based on tool type."""
    fields = INPUT_EXTRACTORS.get(tool_name)
    if fields is None:
        return {}
    extracted = {f: tool_input.get(f, "") for f in fields}
    if tool_name == "Bash":
        extracted["command"] = extracted["command"][:500]
    return extracted


# ---------------------------------------------------------------------------
# Pure functions — pattern matching
# ---------------------------------------------------------------------------

# Pattern definitions: (name, sequence of tool names, min_length)
WORKFLOW_PATTERNS: list[tuple[str, list[PatternStep], int]] = [
    ("test-edit-test", ["Bash", ("Edit", "Write"), "Bash"], 3),
    ("search-read-edit", [("Grep", "Glob"), "Read", ("Edit", "Write")], 3),
    ("read-edit", ["Read", ("Edit", "Write")], 2),
]


def _matches_step(tool_name: str, step: PatternStep) -> bool:
    """Check if a tool name matches a pattern step (string or tuple)."""
    if isinstance(step, tuple):
        return tool_name in step
    return tool_name == step


def _match_pattern_at(
    tool_uses: list[dict],
    pat_steps: list[PatternStep],
    start: int,
) -> list[dict] | None:
    """Try to match a pattern starting at position start.

    Returns matched tool entries if the full pattern is found, None otherwise.
    """
    matched: list[dict] = []
    step_idx = 0
    j = start

    while j < len(tool_uses) and step_idx < len(pat_steps):
        if _matches_step(tool_uses[j]["tool_name"], pat_steps[step_idx]):
            matched.append(tool_uses[j])
            step_idx += 1
        elif step_idx > 0:
            return None
        j += 1

    if step_idx == len(pat_steps):
        return matched
    return None


def _determine_success(matched: list[dict]) -> bool | None:
    """Determine success from matched tool entries.

    Checks last Bash result first, then last Edit/Write result.
    Returns None if no result_error information available.
    """
    for m in reversed(matched):
        if m["tool_name"] == "Bash" and m["result_error"] is not None:
            return not m["result_error"]

    for m in reversed(matched):
        if m["tool_name"] in ("Edit", "Write") and m["result_error"] is not None:
            return not m["result_error"]

    return None


def _collect_files(matched: list[dict]) -> list[str]:
    """Extract unique file paths from matched tool entries."""
    return list({
        m["input"].get("file_path", "")
        for m in matched
        if m["input"].get("file_path")
    })


def _build_pattern_result(pat_name: str, matched: list[dict]) -> dict:
    """Build a pattern result dict from matched tool entries."""
    return {
        "pattern_type": pat_name,
        "tools": [
            {"tool_name": m["tool_name"], "input": m["input"]}
            for m in matched
        ],
        "files": _collect_files(matched),
        "success": _determine_success(matched),
    }


def detect_workflow_sequences(tool_uses: list[dict]) -> list[dict]:
    """Identify common workflow patterns in the tool use sequence.

    Returns list of dicts:
        pattern_type, tools (list of tool entries), files (list),
        success (bool|None)
    """
    patterns_found: list[dict] = []

    for pat_name, pat_steps, min_len in WORKFLOW_PATTERNS:
        i = 0
        while i < len(tool_uses):
            matched = _match_pattern_at(tool_uses, pat_steps, i)
            if matched is not None and len(matched) >= min_len:
                patterns_found.append(
                    _build_pattern_result(pat_name, matched)
                )
                i += len(matched)
            else:
                i += 1

    return patterns_found


# ---------------------------------------------------------------------------
# Pure functions — summarization
# ---------------------------------------------------------------------------


def summarize_tool_usage(tool_uses: list[dict]) -> dict:
    """Summarize tool usage frequency and files touched."""
    tool_counts = dict(Counter(tu["tool_name"] for tu in tool_uses))

    files_by_tool: dict[str, set[str]] = {}
    for tu in tool_uses:
        fp = tu["input"].get("file_path", "")
        if fp:
            files_by_tool.setdefault(tu["tool_name"], set()).add(fp)

    commands = {
        tu["input"].get("command", "")
        for tu in tool_uses
        if tu["input"].get("command")
    }

    return {
        "tool_counts": tool_counts,
        "files_by_tool": {k: sorted(v) for k, v in files_by_tool.items()},
        "unique_commands": len(commands),
        "total_tool_calls": len(tool_uses),
    }


# ---------------------------------------------------------------------------
# Pure functions — formatting
# ---------------------------------------------------------------------------


def format_pattern_as_learning(pattern: dict) -> str:
    """Format a detected workflow pattern as a learning string."""
    pat_type = pattern["pattern_type"]
    tools = pattern["tools"]
    files = pattern["files"]
    success = pattern["success"]

    tool_seq = " -> ".join(t["tool_name"] for t in tools)
    file_list = ", ".join(os.path.basename(f) for f in files[:5])
    status = (
        "succeeded" if success
        else "failed" if success is False
        else "unknown"
    )

    parts = [f"Workflow pattern ({pat_type}): {tool_seq}"]
    if file_list:
        parts.append(f"Files: {file_list}")
    parts.append(f"Outcome: {status}")

    if pat_type == "test-edit-test":
        for t in tools:
            cmd = t["input"].get("command", "")
            if t["tool_name"] == "Bash" and cmd:
                parts.append(f"Command: {cmd[:200]}")
                break

    return ". ".join(parts)


# ---------------------------------------------------------------------------
# I/O — JSONL parsing
# ---------------------------------------------------------------------------


def _parse_tool_use_entry(
    item: dict, data: dict, line_num: int,
) -> dict | None:
    """Parse a single tool_use content item into a tool entry dict."""
    if not isinstance(item, dict) or item.get("type") != "tool_use":
        return None
    tool_name = item.get("name", "")
    return {
        "tool_name": tool_name,
        "input": _extract_input(tool_name, item.get("input", {})),
        "timestamp": data.get("timestamp"),
        "line_num": line_num,
        "tool_use_id": item.get("id", ""),
        "result_error": None,
    }


def _parse_tool_result(item: dict) -> tuple[str, bool] | None:
    """Parse a tool_result content item. Returns (tool_use_id, is_error)."""
    if not isinstance(item, dict) or item.get("type") != "tool_result":
        return None
    return (item.get("tool_use_id", ""), item.get("is_error", False))


def extract_tool_uses(jsonl_path: Path) -> list[dict]:
    """Stream through JSONL and extract tool_use blocks with their results.

    Returns ordered list of dicts with:
        tool_name, input, timestamp, line_num, tool_use_id,
        result_error (bool|None)
    """
    tool_uses: list[dict] = []
    pending_ids: dict[str, int] = {}

    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            content = data.get("message", {}).get("content")
            if not isinstance(content, list):
                continue

            msg_type = data.get("type")

            if msg_type == "assistant":
                for item in content:
                    entry = _parse_tool_use_entry(item, data, line_num)
                    if entry is not None:
                        pending_ids[entry["tool_use_id"]] = len(tool_uses)
                        tool_uses.append(entry)

            elif msg_type == "user":
                for item in content:
                    result = _parse_tool_result(item)
                    if result is not None:
                        tid, is_error = result
                        if tid in pending_ids:
                            tool_uses[pending_ids[tid]]["result_error"] = (
                                is_error
                            )

    return tool_uses


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------


def _format_output(
    patterns: list[dict],
    tool_uses: list[dict],
    fmt: str,
    patterns_only: bool,
) -> str:
    """Format results for output."""
    if patterns_only:
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

    if fmt == "json":
        return json.dumps(result, indent=2)

    if patterns_only:
        lines = [format_pattern_as_learning(p) for p in patterns]
        return "\n\n".join(lines) if lines else "No patterns detected."

    return json.dumps(result, indent=2)


def _write_output(
    output: str,
    output_path: str | None,
    pattern_count: int,
    tool_count: int,
) -> None:
    """Write output to file or stdout."""
    if output_path:
        Path(output_path).write_text(output)
        print(
            f"Wrote {pattern_count} patterns from "
            f"{tool_count} tool uses to {output_path}",
            file=sys.stderr,
        )
    else:
        print(output)


def main() -> None:
    faulthandler.enable(
        file=open(
            os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"
        ),
        all_threads=True,
    )

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
        print(json.dumps(summarize_tool_usage(tool_uses), indent=2))
        return

    patterns = detect_workflow_sequences(tool_uses)
    output = _format_output(
        patterns, tool_uses, args.format, args.patterns_only
    )
    _write_output(output, args.output, len(patterns), len(tool_uses))


if __name__ == "__main__":
    main()
