"""Output formatting for recall learnings.

Contains presentation logic:
- format_result_preview: truncate content for display
- format_json_output: serialize results as JSON
- format_human_output: human-readable text output
- group_by_type: group results by learning_type (for --structured)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def format_result_preview(content: str, max_length: int = 200) -> str:
    """Format content for display, truncating if needed.

    Args:
        content: Full content string
        max_length: Maximum characters before truncation

    Returns:
        Content string, truncated with ... if over max_length
    """
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def _format_created_at(created_at: Any) -> str:
    """Convert created_at to ISO string for JSON output."""
    if isinstance(created_at, datetime):
        return created_at.isoformat()
    return str(created_at)


def _build_json_result(result: dict[str, Any]) -> dict[str, Any]:
    """Build a single JSON result dict from a raw result."""
    json_result = {
        "id": result.get("id", ""),
        "score": result.get("final_score", result["similarity"]),
        "raw_score": result["similarity"],
        "learning_type": result.get("metadata", {}).get("learning_type", "UNKNOWN"),
        "session_id": result["session_id"],
        "content": result["content"],
        "created_at": _format_created_at(result["created_at"]),
    }
    if "rerank_details" in result:
        json_result["rerank_details"] = result["rerank_details"]
    return json_result


def format_json_output(results: list[dict[str, Any]], structured: bool = False) -> str:
    """Format results as JSON string.

    Args:
        results: List of result dicts from search/rerank pipeline
        structured: If True, group by learning_type

    Returns:
        JSON string ready for print()
    """
    if structured:
        grouped = group_by_type(results)
        structured_output: dict[str, list[dict]] = {}
        for type_name, type_results in grouped.items():
            structured_output[type_name] = [
                _build_json_result(r) for r in type_results
            ]
        return json.dumps({
            "structured": True,
            "groups": structured_output,
            "total": len(results),
        })

    json_results = [_build_json_result(r) for r in results]
    return json.dumps({"results": json_results})


def format_json_full_output(results: list[dict[str, Any]]) -> str:
    """Format results with full metadata as JSON (for benchmarking).

    Extends the standard JSON output with metadata, recall_count,
    pattern_strength, and pattern_tags needed by the reranker sweep.
    """
    json_results = []
    for result in results:
        jr = _build_json_result(result)
        metadata = result.get("metadata", {})
        jr["metadata"] = metadata
        jr["recall_count"] = result.get("recall_count", 0)
        jr["pattern_strength"] = result.get("pattern_strength", 0.0)
        jr["pattern_tags"] = result.get("pattern_tags", [])
        json_results.append(jr)
    return json.dumps({"results": json_results})


def format_human_output(results: list[dict[str, Any]], structured: bool = False) -> str:
    """Format results as human-readable text.

    Args:
        results: List of result dicts from search/rerank pipeline
        structured: If True, group by learning_type with headers

    Returns:
        Multi-line string ready for print()
    """
    if not results:
        return "No matching learnings found."

    lines: list[str] = []

    if structured:
        grouped = group_by_type(results)
        lines.append(f"Found {len(results)} matching learnings in {len(grouped)} types:")
        lines.append("")
        idx = 1
        for type_name, type_results in grouped.items():
            lines.append(f"## {type_name} ({len(type_results)})")
            for result in type_results:
                score = result.get("final_score", result["similarity"])
                content_preview = format_result_preview(result["content"], max_length=300)
                session_id = result["session_id"]
                created_at = result["created_at"]
                if isinstance(created_at, datetime):
                    created_str = created_at.strftime("%Y-%m-%d %H:%M")
                else:
                    created_str = str(created_at)[:16]
                lines.append(f"  {idx}. [{score:.3f}] Session: {session_id} ({created_str})")
                lines.append(f"     {content_preview}")
                idx += 1
            lines.append("")
    else:
        lines.append(f"Found {len(results)} matching learnings:")
        lines.append("")
        for i, result in enumerate(results, 1):
            score = result.get("final_score", result["similarity"])
            content_preview = format_result_preview(result["content"], max_length=300)
            session_id = result["session_id"]
            created_at = result["created_at"]
            if isinstance(created_at, datetime):
                created_str = created_at.strftime("%Y-%m-%d %H:%M")
            else:
                created_str = str(created_at)[:16]
            lines.append(f"{i}. [{score:.3f}] Session: {session_id} ({created_str})")
            lines.append(f"   {content_preview}")
            lines.append("")

    return "\n".join(lines)


# Canonical display order for learning types
LEARNING_TYPE_ORDER = [
    "FAILED_APPROACH",
    "ERROR_FIX",
    "WORKING_SOLUTION",
    "ARCHITECTURAL_DECISION",
    "CODEBASE_PATTERN",
    "USER_PREFERENCE",
    "OPEN_THREAD",
]


def group_by_type(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group results by learning_type, preserving relevance order within each group.

    Returns a dict keyed by learning_type. Types appear in LEARNING_TYPE_ORDER;
    any unexpected types are appended at the end alphabetically.
    Results within each group retain their original relevance ordering.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        lt = result.get("metadata", {}).get("learning_type", "UNKNOWN")
        groups.setdefault(lt, []).append(result)

    # Reorder by canonical order
    ordered: dict[str, list[dict[str, Any]]] = {}
    for lt in LEARNING_TYPE_ORDER:
        if lt in groups:
            ordered[lt] = groups.pop(lt)
    # Append any remaining (unknown types)
    for lt in sorted(groups.keys()):
        ordered[lt] = groups[lt]

    return ordered
