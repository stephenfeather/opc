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


def get_api_version() -> str:
    """Return the OPC package version for API response envelopes."""
    try:
        from importlib.metadata import version

        return version("mcp-execution")
    except Exception:
        return "0.7.3"


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


def _format_created_at_human(created_at: Any) -> str:
    """Convert created_at to short human-readable string (up to 16 chars)."""
    if isinstance(created_at, datetime):
        return created_at.strftime("%Y-%m-%d %H:%M")
    return str(created_at)[:16]


def _extract_learning_type(result: dict[str, Any]) -> str:
    """Extract learning_type from a result dict, defaulting to UNKNOWN."""
    metadata = result.get("metadata") or {}
    return metadata.get("learning_type") or "UNKNOWN"


def _extract_score(result: dict[str, Any]) -> float:
    """Extract the display score from a result, preferring final_score over similarity."""
    score = result.get("final_score")
    if score is None:
        score = result["similarity"]
    return float(score)


def _build_json_result(result: dict[str, Any]) -> dict[str, Any]:
    """Build a single JSON result dict from a raw result."""
    json_result: dict[str, Any] = {
        "id": result.get("id", ""),
        "score": _extract_score(result),
        "raw_score": result["similarity"],
        "learning_type": _extract_learning_type(result),
        "session_id": result["session_id"],
        "content": result["content"],
        "created_at": _format_created_at(result["created_at"]),
    }
    if "rerank_details" in result:
        json_result["rerank_details"] = result["rerank_details"]
    if "kg_context" in result:
        # kg_context carries display names and edge metadata sourced from the
        # DB (via _fetch_kg_rows jsonb_build_object). Current consumers are
        # CLI/JSON only, but treat the contents as untrusted strings at any
        # future rendering boundary (HTML, markdown with nested templates).
        # See aegis audit finding LOW-3.
        json_result["kg_context"] = result["kg_context"]
    return json_result


def format_json_output(results: list[dict[str, Any]], structured: bool = False) -> str:
    """Format results as JSON string.

    Args:
        results: List of result dicts from search/rerank pipeline
        structured: If True, add groups key organized by learning_type

    Returns:
        JSON string ready for print(). Envelope always has "results" and "total".
        When structured=True, "groups" key is added alongside.
    """
    json_results = [_build_json_result(r) for r in results]
    output: dict[str, Any] = {
        "version": get_api_version(),
        "results": json_results,
        "total": len(json_results),
    }

    if structured:
        grouped = group_by_type(results)
        output["structured"] = True
        output["groups"] = {
            type_name: [_build_json_result(r) for r in type_results]
            for type_name, type_results in grouped.items()
        }

    return json.dumps(output)


def format_json_full_output(results: list[dict[str, Any]]) -> str:
    """Format results with full metadata as JSON (for benchmarking).

    Extends the standard JSON output with metadata, recall_count,
    pattern_strength, and pattern_tags needed by the reranker sweep.
    """
    json_results = [
        {
            **_build_json_result(result),
            "metadata": result.get("metadata") or {},
            "recall_count": result.get("recall_count", 0),
            "pattern_strength": result.get("pattern_strength", 0.0),
            "pattern_tags": result.get("pattern_tags", []),
        }
        for result in results
    ]
    return json.dumps({"version": get_api_version(), "results": json_results})


def _format_result_line(
    index: int, result: dict[str, Any], indent: str = "", content_indent: str = "   "
) -> tuple[str, str]:
    """Format a single result as a numbered score/session line and content line.

    Args:
        index: 1-based result number
        result: Result dict from search/rerank pipeline
        indent: Prefix for the header line (e.g. "  " for structured)
        content_indent: Prefix for the content line (independent of indent)

    Returns:
        Tuple of (header_line, content_line)
    """
    score = _extract_score(result)
    session_id = result["session_id"]
    created_str = _format_created_at_human(result["created_at"])
    content_preview = format_result_preview(result["content"], max_length=300)
    header = f"{indent}{index}. [{score:.3f}] Session: {session_id} ({created_str})"
    indented_lines = (f"{content_indent}{line}" for line in content_preview.split("\n"))
    content = "\n".join(indented_lines)
    return header, content


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
                header, content = _format_result_line(
                    idx, result, indent="  ", content_indent="     "
                )
                lines.append(header)
                lines.append(content)
                idx += 1
            lines.append("")
    else:
        lines.append(f"Found {len(results)} matching learnings:")
        lines.append("")
        for i, result in enumerate(results, 1):
            header, content = _format_result_line(i, result)
            lines.append(header)
            lines.append(content)
            lines.append("")

    return "\n".join(lines)


# Canonical display order for learning types (immutable tuple)
LEARNING_TYPE_ORDER: tuple[str, ...] = (
    "FAILED_APPROACH",
    "ERROR_FIX",
    "WORKING_SOLUTION",
    "ARCHITECTURAL_DECISION",
    "CODEBASE_PATTERN",
    "USER_PREFERENCE",
    "OPEN_THREAD",
)


def group_by_type(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group results by learning_type, preserving relevance order within each group.

    Returns a dict keyed by learning_type. Types appear in LEARNING_TYPE_ORDER;
    any unexpected types are appended at the end alphabetically.
    Results within each group retain their original relevance ordering.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        lt = _extract_learning_type(result)
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
