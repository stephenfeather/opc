#!/usr/bin/env python3
"""
Extract thinking blocks from session JSONL files.

Two-phase extraction:
1. Deterministic: Extract all thinking blocks (grep-like)
2. Filter: Keep only blocks with perception change signals

Usage:
    python extract_thinking_blocks.py --jsonl path/to/session.jsonl
    python extract_thinking_blocks.py --jsonl path/to/session.jsonl --filter
    python extract_thinking_blocks.py --jsonl path/to/session.jsonl --output /tmp/blocks.txt
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import re
import sys
from pathlib import Path

# Perception change signal patterns (Alan Kay "point of view" moments)
PERCEPTION_SIGNALS = [
    # Corrections and realizations
    r"\bactually\b",
    r"\brealized?\b",
    r"\bthe issue\b",
    r"\bthat'?s why\b",
    r"\bturns out\b",
    r"\bI was wrong\b",
    r"\bworks because\b",
    r"\bthe problem is\b",
    r"\bOh,",
    r"\bAha\b",
    r"\bnow I see\b",
    r"\bnow I understand\b",
    r"\bI see now\b",
    r"\bmisunderstood\b",
    r"\bwait,?\b",
    r"\bhmm\b",
    r"\binteresting\b",
    r"\bunexpected\b",
    r"\bsurpris",  # surprising, surprised
    r"\bdifferent than\b",
    r"\bdifferent from\b",
    r"\bnot what I\b",
    r"\bwasn'?t\b.*\bexpect",
    # Validation and success signals
    r"\bthis works\b",
    r"\bthat works\b",
    r"\bworked well\b",
    r"\bgood approach\b",
    r"\bright approach\b",
    r"\bas expected\b",
    r"\bconfirm(?:s|ed)?\b.*\bapproach\b",
    r"\bthis pattern\b.*\b(?:works|effective|reliable)\b",
    r"\bkeep doing\b",
    r"\bcorrect approach\b",
    r"\bsolid approach\b",
    r"\bclean(?:er)? (?:solution|approach|pattern)\b",
    r"\beffective(?:ly)?\b.*\b(?:handles?|solves?|addresses)\b",
    r"\brobust\b.*\b(?:solution|approach|pattern)\b",
    r"\belegant\b.*\b(?:solution|approach|pattern)\b",
    r"\bthis is the right\b",
    r"\bvalidated\b",
]

PERCEPTION_PATTERN = re.compile("|".join(PERCEPTION_SIGNALS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def has_perception_signal(text: str) -> bool:
    """Check whether text contains any perception change signal."""
    return bool(PERCEPTION_PATTERN.search(text))


def parse_jsonl_entry(line: str, *, line_num: int = 0) -> list[dict]:
    """Parse a single JSONL line and return extracted thinking blocks.

    Returns an empty list when the line is invalid JSON, a non-message type,
    or contains no thinking blocks.
    """
    try:
        data = json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return []

    if data.get("type") not in ("assistant", "user"):
        return []

    message = data.get("message")
    if not isinstance(message, dict):
        return []

    content = message.get("content")
    if not isinstance(content, list):
        return []

    blocks: list[dict] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "thinking":
            continue
        thinking_text = item.get("thinking", "")
        if not isinstance(thinking_text, str) or not thinking_text:
            continue
        blocks.append(
            {
                "thinking": thinking_text,
                "timestamp": data.get("timestamp"),
                "line_num": line_num,
                "has_perception_signal": has_perception_signal(thinking_text),
            }
        )
    return blocks


def compute_stats(blocks: list[dict]) -> dict:
    """Compute summary statistics for a list of thinking blocks."""
    total = len(blocks)
    with_signal = sum(1 for b in blocks if b["has_perception_signal"])
    return {
        "total": total,
        "with_signal": with_signal,
        "ratio": (with_signal / total * 100) if total > 0 else None,
    }


def format_blocks_text(blocks: list[dict]) -> str:
    """Format thinking blocks as human-readable text."""
    if not blocks:
        return ""
    return "\n\n---\n\n".join(
        f"[Line {b['line_num']}] {'*' if b['has_perception_signal'] else ''}\n{b['thinking']}"
        for b in blocks
    )


def format_blocks_json(blocks: list[dict]) -> str:
    """Format thinking blocks as indented JSON."""
    return json.dumps(blocks, indent=2)


# ---------------------------------------------------------------------------
# I/O boundary
# ---------------------------------------------------------------------------


def extract_thinking_blocks(
    jsonl_path: Path, *, filter_perception: bool = False
) -> list[dict]:
    """Read a JSONL file and extract thinking blocks.

    Args:
        jsonl_path: Path to session JSONL file.
        filter_perception: If True, only return blocks with perception signals.

    Returns:
        List of dicts with 'thinking', 'timestamp', 'line_num',
        'has_perception_signal'.
    """
    blocks: list[dict] = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            parsed = parse_jsonl_entry(line, line_num=line_num)
            if filter_perception:
                blocks.extend(b for b in parsed if b["has_perception_signal"])
            else:
                blocks.extend(parsed)
    return blocks


_faulthandler_file = None


def _enable_faulthandler() -> None:
    """Best-effort crash logging — idempotent, does not raise if unavailable."""
    global _faulthandler_file  # noqa: PLW0603
    if _faulthandler_file is not None:
        return
    try:
        log_path = os.path.expanduser("~/.claude/logs/opc_crash.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        _faulthandler_file = open(log_path, "a")  # noqa: SIM115
        faulthandler.enable(file=_faulthandler_file)
    except OSError:
        pass


def main() -> None:
    """CLI entry point — handles arg parsing, file I/O, and output."""
    _enable_faulthandler()
    parser = argparse.ArgumentParser(description="Extract thinking blocks from session JSONL")
    parser.add_argument("--jsonl", required=True, help="Path to session JSONL file")
    parser.add_argument(
        "--filter", action="store_true", help="Only extract blocks with perception signals"
    )
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics only")

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    blocks = extract_thinking_blocks(jsonl_path, filter_perception=args.filter)

    if args.stats:
        stats = compute_stats(blocks)
        print(f"Total thinking blocks: {stats['total']}")
        print(f"With perception signals: {stats['with_signal']}")
        print(f"Ratio: {stats['ratio']:.1f}%" if stats["ratio"] is not None else "Ratio: N/A")
        return

    output = format_blocks_json(blocks) if args.format == "json" else format_blocks_text(blocks)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote {len(blocks)} blocks to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
