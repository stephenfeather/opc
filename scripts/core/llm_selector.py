"""LLM-as-selector recall stage (issue #228 item 3).

An optional selector that runs after candidate retrieval, exclusion filtering,
and enrichment. It asks the Anthropic Messages API (via a single forced
tool-use call) to pick and reorder the best candidates, then maps the returned
ids back onto the candidate pool.

Design contract:
  * The orchestrator ``llm_select`` returns ``None`` on EVERY failure mode and
    NEVER raises to the caller. ``None`` is the sentinel telling the recall call
    site to fall back to the pure ``rerank()`` in ``reranker.py``.
  * The pure functions (``build_manifest``, ``parse_selection``,
    ``apply_selection``) do no I/O; the single I/O edge is ``call_anthropic``.
  * Selected records carry the reranker's output record shape (``final_score`` +
    ``rerank_details``) so downstream output/telemetry are unaffected.
  * Never log the api key, request headers, or the full response body.

Built on ``httpx`` (a transitive, locked dependency already imported in
``scripts/core/db/embedding_providers.py``). The ``anthropic`` SDK is NOT a
core dependency and is deliberately not used here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

# Bounded so the call cannot blow the memory-awareness hook's 5s spawn budget;
# on timeout we fall back to the sub-millisecond pure rerank(). Mirrors the
# module-level timeout-constant pattern in recall_learnings.py.
LLM_SELECTOR_TIMEOUT: float = 2.5

# Truncation bound for each candidate's description in the manifest.
MANIFEST_DESC_MAXLEN: int = 200

# Hard upper bound on the raw content slice processed per candidate BEFORE
# whitespace normalization. Bounds CPU/memory on the synchronous pre-network hot
# path: an oversized (multi-MB) archival memory must not be split/rejoined in
# full just to render a 200-char desc. Generous (8x the desc cap) so normal text
# still collapses to a full MANIFEST_DESC_MAXLEN line; the slice makes the work
# O(cap) rather than O(len(content)).
MANIFEST_RAW_SLICE: int = MANIFEST_DESC_MAXLEN * 8

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_TOOL_NAME = "select_memories"
_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_manifest(candidates: list[dict]) -> str:
    """Render candidates as a compact one-line-per-row manifest (PURE).

    Line shape: ``[<learning_type or UNKNOWN>] <id> (<created_at iso or ?>): <desc>``
    where ``desc`` is the content truncated to ``MANIFEST_DESC_MAXLEN``.

    Trust boundary: archival memory ``content`` is model-visible data that crosses
    a trust boundary. A poisoned memory whose content embeds newlines (or fake
    ``[TYPE] id (ts):`` row prefixes / instruction-like text) could otherwise
    forge extra apparent candidate rows or inject instructions into the forced
    tool call. To guarantee exactly one line per candidate, all whitespace
    (newlines, carriage returns, tabs, runs of spaces) in ``content`` is
    collapsed to single spaces BEFORE truncation — so an embedded prefix can
    never start a new row. ``apply_selection`` additionally drops any id absent
    from the real pool, so even a forged in-desc id cannot be selected.
    """
    lines: list[str] = []
    for c in candidates:
        meta = c.get("metadata") or {}
        ltype = meta.get("learning_type") or "UNKNOWN"
        cid = c.get("id", "?")
        created = c.get("created_at")
        ts = created.isoformat() if hasattr(created, "isoformat") else "?"
        # Bound the work BEFORE normalizing (round 3): slice a generous prefix
        # so split/join is O(cap), not O(len(content)) for an oversized memory.
        # Then collapse all whitespace/control runs to single spaces so one
        # candidate is always exactly one line (manifest-injection mitigation),
        # and finally truncate to the desc cap. Collapsing a bounded prefix still
        # removes any newline in it, so no newline can survive into the line.
        raw = (c.get("content") or "")[:MANIFEST_RAW_SLICE]
        desc = " ".join(raw.split())[:MANIFEST_DESC_MAXLEN]
        lines.append(f"[{ltype}] {cid} ({ts}): {desc}")
    return "\n".join(lines)


def parse_selection(api_response: dict) -> list[str]:
    """Extract the selected id list from an Anthropic tool-use response (PURE).

    Returns ``[]`` on ANY structural miss (no tool_use block, wrong tool name,
    missing key, non-list value). Non-string items are dropped.
    """
    content = api_response.get("content")
    if not isinstance(content, list):
        return []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use" or block.get("name") != _TOOL_NAME:
            continue
        selected = (block.get("input") or {}).get("selected_memories")
        if not isinstance(selected, list):
            return []
        return [item for item in selected if isinstance(item, str)]
    return []


def apply_selection(
    ids: list[str],
    candidates: list[dict],
    k: int,
    *,
    model: str = "",
) -> list[dict] | None:
    """Map selected ids onto candidate dicts, stamping the reranker shape (PURE).

    Dedupes ids preserving first-seen order, drops ids absent from the pool, and
    returns ``None`` if nothing survives (caller then falls back to rerank()).
    Each returned record is a shallow copy stamped with ``final_score``
    (monotonically descending by selection order) and ``rerank_details`` so
    downstream consumers (output/telemetry/benchmark) see the reranker contract.
    The input pool is never mutated.
    """
    by_id = {c.get("id"): c for c in candidates}

    ordered: list[str] = []
    seen: set[str] = set()
    for cid in ids:
        if cid in seen or cid not in by_id:
            continue
        seen.add(cid)
        ordered.append(cid)

    if not ordered:
        return None

    ordered = ordered[:k]
    n = len(ordered)
    out: list[dict] = []
    for i, cid in enumerate(ordered):
        record = dict(by_id[cid])  # shallow copy — do not mutate the pool
        record["final_score"] = 1.0 - (i / n)
        record["rerank_details"] = {
            "source": "llm_selector",
            "model": model,
            "rank": i,
        }
        out.append(record)
    return out


def _anthropic_tool_schema() -> dict:
    """The forced-tool input_schema for ``{selected_memories: string[]}``."""
    return {
        "type": "object",
        "properties": {
            "selected_memories": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["selected_memories"],
    }


def _build_prompt(manifest: str, query: str) -> str:
    return (
        "You are selecting the most relevant memories for a recall query.\n"
        f"Query: {query}\n\n"
        "Candidates (one per line as [type] id (timestamp): description):\n"
        f"{manifest}\n\n"
        "Call the select_memories tool with the ids of the most relevant "
        "candidates, most relevant first. Only use ids from the list above."
    )


# ---------------------------------------------------------------------------
# I/O edge (single network boundary)
# ---------------------------------------------------------------------------


async def call_anthropic(
    manifest: str,
    query: str,
    *,
    model: str,
    api_key: str,
    timeout: float,
) -> dict:
    """POST a forced tool-use request to the Anthropic Messages API (I/O edge).

    Wrapped in ``asyncio.wait_for`` INSIDE the ``async with httpx.AsyncClient``
    block so a timeout cancellation always closes the client (T1 mitigation).
    Returns the parsed JSON response dict. Raises on HTTP/network/timeout error;
    the orchestrator catches and converts to the ``None`` sentinel.
    """
    body = {
        "model": model,
        "max_tokens": _MAX_TOKENS,
        "tools": [
            {
                "name": _TOOL_NAME,
                "description": "Select the most relevant memory ids for the query.",
                "input_schema": _anthropic_tool_schema(),
            }
        ],
        "tool_choice": {"type": "tool", "name": _TOOL_NAME},
        "messages": [{"role": "user", "content": _build_prompt(manifest, query)}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    # Explicit try/finally close (rather than `async with`) so a timeout
    # cancellation of the in-flight post() always closes the client via the
    # named aclose() — the T1 mitigation. wait_for sits INSIDE this block so
    # cancellation can never leak the connection past the timeout.
    client = httpx.AsyncClient(timeout=timeout)
    try:
        response = await asyncio.wait_for(
            client.post(_ANTHROPIC_URL, headers=headers, json=body),
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def llm_select(
    candidates: list[dict],
    *,
    query: str,
    model: str,
    k: int,
    timeout: float = LLM_SELECTOR_TIMEOUT,
) -> list[dict] | None:
    """Select and reorder candidates via the LLM; ``None`` => caller falls back.

    Returns ``None`` on EVERY failure mode (empty pool, missing key, API/network
    error, timeout, malformed output, empty/unknown-only selection) and NEVER
    raises to the caller. Logs only counts/elapsed/exception-type at DEBUG —
    never the api key, headers, or response body.
    """
    # T2: short-circuit an empty pool BEFORE reading the env var or any I/O.
    if not candidates:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.debug("llm_select: no ANTHROPIC_API_KEY; falling back")
        return None

    manifest = build_manifest(candidates)
    started = time.monotonic()
    try:
        response = await asyncio.wait_for(
            call_anthropic(manifest, query, model=model, api_key=api_key, timeout=timeout),
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to fallback on any failure
        # Covers TimeoutError, httpx.HTTPStatusError, httpx.RequestError, etc.
        # Log only the exception TYPE — never the key, headers, or body.
        logger.debug(
            "llm_select: API call failed (%s); falling back",
            type(exc).__name__,
        )
        return None

    ids = parse_selection(response)
    selected = apply_selection(ids, candidates, k, model=model)
    logger.debug(
        "llm_select: candidates=%d selected=%d elapsed=%.3fs",
        len(candidates),
        len(selected) if selected else 0,
        time.monotonic() - started,
    )
    return selected
