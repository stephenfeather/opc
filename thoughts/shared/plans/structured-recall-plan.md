# Implementation Plan: Item K -- Structured Recall (Group by learning_type)

Generated: 2026-04-01

## Goal

When recall results are returned, allow grouping them by `learning_type` (WORKING_SOLUTION, FAILED_APPROACH, ERROR_FIX, etc.) so consumers see organized output rather than a flat ranked list. This is opt-in via a `--structured` / `structured=true` flag. The flat list remains the default for backward compatibility.

## Existing Codebase Analysis

### Data Flow (verified by reading source)

```
User/Hook/MCP
    |
    v
recall_learnings.py main() -- CLI entry point, argparse
    |
    v
search_learnings_*() -- 4 search paths, all return list[dict]
    |                    Each dict has: id, session_id, content, metadata, created_at, similarity
    |                    metadata is a dict with keys: type, learning_type, confidence, tags, project, context
    v
reranker.rerank() -- adds final_score, rerank_details to each dict
    |
    v
JSON output (--json) or human-readable output
    |
    v
MCP server (main.py in opc-memory-mcp) -- calls recall_learnings.py with --json, parses stdout
    |
    v
memory-awareness.ts hook -- calls recall_learnings.py with --json --text-only, parses stdout
```

### Key Files and Current State

1. **`/Users/stephenfeather/opc/scripts/core/recall_learnings.py`** (987 lines)
   - Lines 936-956: JSON output mode -- currently emits `score`, `raw_score`, `session_id`, `content`, `created_at`, and optionally `rerank_details`. Does NOT emit `learning_type` or `id`.
   - Lines 959-981: Human-readable output -- shows score, session_id, timestamp, content preview. No type grouping.
   - Lines 296-308, 370-383, 523-558, 705-738: All four search paths return `metadata` dict which contains `learning_type` (stored in postgres as `metadata->>'learning_type'` within the JSONB metadata column).
   - Lines 778-852: argparse setup -- where `--structured` flag will be added.

2. **`/Users/stephenfeather/Tools/opc-memory-mcp/main.py`** (MCP server, external repo)
   - Lines 178-240: `recall_learnings()` tool -- calls `run_opc_script("recall_learnings.py", args)`, does NOT pass `--json` flag currently. It parses stdout as JSON. Need to add `--json` and `--structured` pass-through.
   - Lines 218-240: Result parsing -- returns `{"success": True, "learnings": parsed_json}`.

3. **`/Users/stephenfeather/opc/hooks/ts/src/memory-awareness.ts`** (237 lines)
   - Lines 25-31: `LearningResult` interface already has `type: string` field.
   - Lines 161-177: Result parsing -- already extracts `r.learning_type || r.type || 'UNKNOWN'` from JSON results.
   - Lines 219-224: Output formatting -- currently flat: `"1. [TYPE] content (id: xxx)"`.
   - The hook calls `recall_learnings.py --json --text-only` but the JSON output does not currently include `learning_type`, so the hook falls back to the `type` field from metadata.

4. **`/Users/stephenfeather/opc/scripts/core/reranker.py`** (pure Python, no I/O)
   - Passes `metadata` dict through untouched -- no changes needed.

5. **`/Users/stephenfeather/opc/hooks/ts/src/shared/memory-client.ts`**
   - `MemorySearchResult` interface has `metadata: Record<string, unknown>` -- flexible enough.

### Critical Finding: learning_type Is Available but Not Emitted

The `metadata` dict in every search result contains `learning_type`, but the JSON output serializer (lines 946-956) strips it out. The first prerequisite is to include `learning_type` in the JSON output, which also fixes the memory-awareness hook's fallback behavior.

## Implementation Phases

### Phase 1: Include learning_type in JSON Output (Prerequisite) -- ALREADY DONE

> **Pre-mortem finding (2026-04-01):** This phase is already implemented.
> Lines 946-954 of `recall_learnings.py` already emit `id` and `learning_type`
> in JSON output. Verified by reading the source directly.

**No work required. Skip to Phase 2.**

**Acceptance criteria (all met):**
- [x] `--json` output includes `learning_type` for every result (line 950)
- [x] `--json` output includes `id` for every result (line 947)
- [x] Existing tests still pass
- [x] memory-awareness hook can use `r.learning_type` directly (hook line 173)

### Phase 2: Add --structured Flag and Grouping Logic

**Files to modify:**
- `/Users/stephenfeather/opc/scripts/core/recall_learnings.py` -- argparse + output formatting

**Steps:**
1. Add `--structured` argument to argparse (after line 846):
   ```python
   parser.add_argument(
       "--structured",
       action="store_true",
       help="Group results by learning_type in output",
   )
   ```

2. Add a pure function `group_by_type(results)` near the formatting functions (around line 152):
   ```python
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

   def group_by_type(results: list[dict]) -> dict[str, list[dict]]:
       """Group results by learning_type, preserving relevance order within each group.

       Returns an OrderedDict-like dict keyed by learning_type. Types appear
       in LEARNING_TYPE_ORDER; any unexpected types are appended at the end.
       Results within each group retain their original relevance ordering.
       """
       groups: dict[str, list[dict]] = {}
       for result in results:
           lt = result.get("metadata", {}).get("learning_type", "UNKNOWN")
           groups.setdefault(lt, []).append(result)

       # Reorder by canonical order
       ordered: dict[str, list[dict]] = {}
       for lt in LEARNING_TYPE_ORDER:
           if lt in groups:
               ordered[lt] = groups.pop(lt)
       # Append any remaining (unknown types)
       for lt in sorted(groups.keys()):
           ordered[lt] = groups[lt]

       return ordered
   ```

3. In the JSON output block (after Phase 1 changes), add structured mode:
   ```python
   if args.structured:
       grouped = group_by_type(results)
       structured_output = {}
       for type_name, type_results in grouped.items():
           structured_output[type_name] = []
           for result in type_results:
               # ... same json_result dict as flat mode ...
               structured_output[type_name].append(json_result)
       print(json.dumps({"structured": True, "groups": structured_output, "total": len(results)}))
       return 0
   ```

4. In the human-readable output block, add structured mode:
   ```python
   if args.structured:
       grouped = group_by_type(results)
       print(f"Found {len(results)} matching learnings in {len(grouped)} types:")
       print()
       idx = 1
       for type_name, type_results in grouped.items():
           print(f"## {type_name} ({len(type_results)})")
           for result in type_results:
               score = result.get("final_score", result["similarity"])
               content_preview = format_result_preview(result["content"], max_length=300)
               print(f"  {idx}. [{score:.3f}] {content_preview}")
               idx += 1
           print()
       return 0
   ```

**Acceptance criteria:**
- [ ] `--structured --json` produces `{"structured": true, "groups": {"WORKING_SOLUTION": [...], ...}, "total": N}`
- [ ] `--structured` (human-readable) groups results under type headers
- [ ] Without `--structured`, output is identical to current behavior
- [ ] `group_by_type` preserves relevance ordering within each group
- [ ] Canonical type ordering is applied (FAILED_APPROACH first, OPEN_THREAD last)

### Phase 3: MCP Server Pass-Through

**Files to modify:**
- `/Users/stephenfeather/Tools/opc-memory-mcp/main.py` -- `recall_learnings()` tool

**Steps:**
1. Add `structured` parameter to the `recall_learnings` tool function:
   ```python
   structured: bool = Field(
       description="Group results by learning_type",
       default=False,
   ),
   ```

2. Add `--json` flag (it is currently missing -- the MCP server does not pass `--json` which means it gets human-readable output parsed as JSON, which silently fails):
   ```python
   args = [
       "--query", query,
       "--k", str(k),
       "--threshold", str(threshold),
       "--json",  # Always use JSON for MCP
   ]
   ```

3. Pass `--structured` when requested:
   ```python
   if structured:
       args.append("--structured")
   ```

4. Handle both flat and structured response shapes in the JSON parsing block.

5. **Fix existing parse bug (pre-mortem tiger):** The current parse logic at line 233 does
   `len(learnings) if isinstance(learnings, list)` but `recall_learnings.py` emits
   `{"results": [...]}` (a dict, not a list). Fix to properly extract the results array:
   ```python
   # For flat mode:
   results = learnings.get("results", [])
   return {
       "success": True,
       "learnings": results,
       "count": len(results),
   }
   # For structured mode:
   groups = learnings.get("groups", {})
   return {
       "success": True,
       "learnings": learnings,
       "count": learnings.get("total", 0),
   }
   ```

**Acceptance criteria:**
- [ ] `mcp__opc-memory__recall_learnings(query="x", structured=true)` returns grouped results
- [ ] Default (structured=false) returns flat results identical to current behavior
- [ ] MCP server always passes `--json` flag
- [ ] MCP parse correctly extracts `results` array from `{"results": [...]}` response (count is accurate)

### Phase 4: Memory-Awareness Hook Enhancement (Optional)

**Files to modify:**
- `/Users/stephenfeather/opc/hooks/ts/src/memory-awareness.ts`

**Steps:**
1. Since Phase 1 adds `learning_type` to JSON output, the hook's fallback logic (`r.learning_type || r.type || 'UNKNOWN'`) will now get the value from the top-level field. No change needed for basic functionality.

2. Optional enhancement: when the hook has multiple results across different types, format the `additionalContext` with type grouping:
   ```
   MEMORY MATCH (3 results) for "intent":
   ## WORKING_SOLUTION
   1. content... (id: abc)
   ## FAILED_APPROACH
   2. content... (id: def)
   ```
   This is low priority since the hook only returns 3 results max.

**Acceptance criteria:**
- [ ] Hook correctly displays learning_type from new JSON field
- [ ] (Optional) Grouped format in additionalContext

## Testing Strategy

### Unit Tests

**New file: `/Users/stephenfeather/opc/tests/test_structured_recall.py`**

Following the existing test pattern from `test_recall_reranking.py`:

1. **`test_group_by_type_preserves_relevance_order`** -- verify results within each group keep their original ranking
2. **`test_group_by_type_canonical_ordering`** -- verify FAILED_APPROACH appears before WORKING_SOLUTION
3. **`test_group_by_type_unknown_types`** -- verify unknown types are appended at end
4. **`test_group_by_type_empty_results`** -- verify empty input returns empty dict
5. **`test_json_output_includes_learning_type`** -- mock search results, run main() with `--json`, verify learning_type in output
6. **`test_json_structured_output_format`** -- mock search results, run main() with `--json --structured`, verify grouped format
7. **`test_structured_flag_does_not_affect_search`** -- verify the same search function is called regardless of --structured
8. **`test_human_readable_structured_output`** -- capture stdout, verify type headers appear

### Integration Test

- Run `uv run python scripts/core/recall_learnings.py --query "hook" --json --structured` against live DB
- Verify output shape matches spec

### MCP Server Tests

- Test in `/Users/stephenfeather/Tools/opc-memory-mcp/` if tests exist
- Otherwise, manual test: `mcp__opc-memory__recall_learnings(query="hooks", structured=true)`

## Migration Considerations

**None required.** This is purely a presentation-layer change:
- No database schema changes
- No new tables or columns
- No changes to search algorithms or ranking
- Fully backward compatible (opt-in flag)

The `metadata` column already stores `learning_type` for all learnings. Older learnings without `learning_type` in metadata will be grouped under `"UNKNOWN"`.

## Risks and Considerations

1. **MCP server is in a separate repo** (`/Users/stephenfeather/Tools/opc-memory-mcp`). Changes there need separate deployment. The OPC-side changes (Phases 1-2) work independently.

2. **MCP server missing --json flag**: The current MCP server calls `recall_learnings.py` without `--json`, which means it gets human-readable output and the JSON parse silently fails (returns `raw_output` instead of `learnings`). Phase 3 fixes this as a side effect. This is a pre-existing bug.

3. **Type ordering is a presentation choice**: The canonical ordering (FAILED_APPROACH first) prioritizes warnings. Users may want different ordering. The current design hardcodes the order but could be made configurable later.

4. **Hook token budget**: The memory-awareness hook injects results into system context. Structured formatting uses slightly more tokens (type headers). With only 3 results max, this is negligible.

## Estimated Complexity

**Overall: Low-Medium**

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1: learning_type in JSON | ~~30 min~~ DONE | Already implemented |
| Phase 2: --structured flag + grouping | ~1-2 hours | Low -- pure presentation logic |
| Phase 3: MCP server pass-through + parse fix | ~45 min | Medium -- includes parse bug fix |
| Phase 4: Hook enhancement | ~30 min | Very low -- optional |
| Tests | ~1-2 hours | Low |

**Total: ~2.5-4 hours including tests** (Phase 1 already complete)

## Dependency Graph

```
Phase 1 (learning_type in JSON) -- ALREADY DONE ✓
    |
    v
Phase 2 (--structured flag) -- ready to implement
    |
    v
Phase 3 (MCP pass-through + parse fix) -- depends on Phase 2
    |
Phase 4 (hook enhancement) -- no blockers (Phase 1 done)
```

Phase 1 is already complete. Implementation starts at Phase 2.

## Risk Mitigations (Pre-Mortem)

**Pre-mortem run:** 2026-04-01 | Mode: deep | Tigers: 2 HIGH, 1 MEDIUM | Elephants: 1

### Tigers Addressed:

1. **Plan stale: Phase 1 already implemented** (HIGH)
   - Mitigation: Phase 1 marked as DONE in plan. Implementer starts at Phase 2.
   - Verified: `recall_learnings.py:946-954` already includes `id` and `learning_type`.

2. **MCP parse logic miscounts results** (HIGH)
   - Mitigation: Added Step 5 to Phase 3 — fix `isinstance(learnings, list)` check to
     properly extract `results` array from the `{"results": [...]}` dict response.
   - Location: `/Users/stephenfeather/Tools/opc-memory-mcp/main.py:229-234`

3. **MCP missing --json is a prerequisite, not a side effect** (MEDIUM)
   - Mitigation: Phase 3 Step 2 already adds `--json`. Noted as first step to implement.

### Accepted Risks:

1. **File size (elephant):** `recall_learnings.py` at 987 lines will grow to ~1040+ lines.
   Consider extracting output formatting into a separate module in a future refactor.
   Not blocking for this change since the additions are localized to the output section.

### Paper Tigers (verified safe):
- Backward compatibility of JSON output (additive fields, hook already resilient)
- Structured flag affecting search behavior (purely post-processing)
- record_recall timing (runs before output branching, unaffected)
