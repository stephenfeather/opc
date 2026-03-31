# Implementation Plan: Fix Project Column NULLs in archival_memory

Generated: 2026-03-31
Status: Phase 2 COMPLETE (backfill applied), Phase 1 & 3 pending (write path fixes, validation)

## Goal

Resolve the 67% NULL rate in the `archival_memory.project` column (1,500 of 2,233 learnings). The project column powers the contextual reranker's project_match signal (weight: 0.15, the highest single signal). Without it, recall results for project-specific queries are degraded.

## Current State Analysis

### Distribution (verified via DB)

| Project | Count |
|---------|-------|
| NULL | 1,500 |
| alexa-skill | 404 |
| opc | 275 |
| fa-wpmcp | 54 |

### Root Causes

**1. MCP server does not pass `--project` to `store_learning.py`**
- File: `/Users/stephenfeather/Tools/opc-memory-mcp/main.py` (line 88-120)
- The `store_learning` tool definition has no `project` parameter
- Even though `store_learning.py` accepts `--project` and auto-detects from `CLAUDE_PROJECT_DIR`, the MCP server subprocess doesn't inherit that env var

**2. Daemon extraction subprocess lacks `CLAUDE_PROJECT_DIR`**
- File: `scripts/core/memory_daemon.py` (line 458-530)
- `extract_memories()` receives `project_dir` but does NOT set `CLAUDE_PROJECT_DIR` in the subprocess env
- The spawned `claude -p` extraction session calls `store_learning.py` which then gets no project context

**3. `_extract_and_store_workflows()` doesn't pass `project`**
- File: `scripts/core/memory_daemon.py` (line 592-637)
- Calls `store_learning_v2()` directly but omits the `project=` kwarg
- The `project` parameter is available as the `project` argument to the function

**4. Legacy `store_learning()` (v1) ignores project entirely**
- File: `scripts/core/store_learning.py` (line 172-230)
- The v1 path builds metadata and calls `memory.store()` without any project parameter

### Backfill Feasibility (verified via DB)

**Tier 1: Sessions table cross-reference (1,289 learnings, 86%)**
- 449 of 605 NULL-project session_ids match a row in `sessions` table
- The sessions table has full path in `project` column (e.g., `/Users/stephenfeather/Tools/Continuous-Claude-v3/opc`)
- Mapping those paths to short names covers 1,289 of 1,500 NULL learnings
- Breakdown by session project path:
  - `*/opc` or `*/Continuous-Claude-v3/opc` -> `opc` (1,085 learnings)
  - `*/binbrain-ios` -> `binbrain-ios` (61)
  - `*/Pharmacokinetics-Grapher` -> `pharmacokinetics-grapher` (42)
  - `*/agentic-work` -> `agentic-work` (36)
  - `*/fa-wpmcp` -> `fa-wpmcp` (26)
  - `*/binbrain` -> `binbrain` (18+3)
  - `*/sermon-browser*` -> `sermon-browser` (8)
  - `*/opc-memory-mcp` -> `opc-memory-mcp` (4)
  - Others (6)

**Tier 2: Tag/content heuristics (remaining ~211 learnings)**
- Tag signals available: `perception` (19), `wordpress` (32), `ios/swift` (24), `binbrain` (12), `3d-print-logger` (16), `fa-toolkit` (13), `pharmacokinetics` (implied)
- Content keyword analysis on full NULL set showed: vue/perception=1,207, wordpress=151, opc=118, pharma=77, alexa=84
- Many of these are already covered by Tier 1; the residual 211 need tag-based inference

**Tier 3: Unresolvable (~50-100 learnings)**
- Generic learnings about git, testing, npm that span multiple projects
- Should be flagged `_ambiguous` rather than guessed

### Metadata Available in NULL Rows

- `metadata->>'context'`: present in 1,500/1,500 (100%)
- `metadata->'tags'`: present in 1,496/1,500 (99.7%)
- `metadata->>'project'`: present in 0/1,500 (never stored in metadata)

## Implementation Phases

### Phase 1: Fix Write Paths (Prevent Future NULLs)

**1A. Fix MCP server - add project parameter**

**File to modify:** `/Users/stephenfeather/Tools/opc-memory-mcp/main.py`

**Steps:**
1. Add `project` parameter to the `store_learning` tool function signature with `default=""`
2. If empty, auto-detect from `CLAUDE_PROJECT_DIR` env var (basename only)
3. If `CLAUDE_PROJECT_DIR` is empty, try `git rev-parse --show-toplevel` and take basename
4. Pass `--project <value>` to the subprocess args

```python
project: str = Field(
    description="Project name for recall relevance (auto-detected if omitted)",
    default="",
)
```

Then in the args builder:
```python
# Auto-detect project
effective_project = project
if not effective_project:
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if project_dir:
        effective_project = Path(project_dir).name
if not effective_project:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            effective_project = Path(result.stdout.strip()).name
    except Exception:
        pass
if effective_project:
    args.extend(["--project", effective_project])
```

**Acceptance criteria:**
- [ ] MCP `store_learning` calls include `--project` in subprocess args
- [ ] Auto-detection works when `CLAUDE_PROJECT_DIR` is set
- [ ] Graceful fallback when no project detectable

**1B. Fix daemon extraction subprocess env**

**File to modify:** `scripts/core/memory_daemon.py` (line ~510)

**Steps:**
1. In `extract_memories()`, add `CLAUDE_PROJECT_DIR` to the `env` dict passed to `subprocess.Popen`
2. The `project_dir` parameter is already available

```python
env["CLAUDE_PROJECT_DIR"] = project_dir or ""
```

This single line ensures the spawned Claude extraction session has project context, so every `store_learning.py` call from within that session auto-detects the project.

**Acceptance criteria:**
- [ ] Extraction subprocess receives `CLAUDE_PROJECT_DIR` in its environment
- [ ] New extractions produce learnings with non-NULL project

**1C. Fix `_extract_and_store_workflows()` - pass project kwarg**

**File to modify:** `scripts/core/memory_daemon.py` (line ~621)

**Steps:**
1. Add `project=` to the `store_learning_v2()` call
2. Normalize the full path to basename

```python
# Normalize project path to short name
project_name = Path(project).name if project else None

result = asyncio.run(store_learning_v2(
    session_id=session_id,
    content=content,
    learning_type="WORKING_SOLUTION",
    context=project or "unknown",
    tags=["workflow", pattern["pattern_type"]],
    confidence="high",
    project=project_name,
))
```

**Acceptance criteria:**
- [ ] Workflow-extracted learnings get a project value
- [ ] Project is normalized to basename (not full path)

**1D. Fix legacy store_learning v1 path**

**File to modify:** `scripts/core/store_learning.py` (line ~172-230)

**Steps:**
1. Add project auto-detection to the v1 `store_learning()` function
2. Pass `project=` to `memory.store()`

This is low priority since v1 is legacy, but it ensures completeness.

**Acceptance criteria:**
- [ ] v1 store path includes project in the store call

**1E. Normalize project values**

**File to modify:** `scripts/core/store_learning.py` (around line 270, the auto-detect)

**Current code:**
```python
project = args.project or os.environ.get("CLAUDE_PROJECT_DIR", "").rsplit("/", 1)[-1] or None
```

**Issue:** When `CLAUDE_PROJECT_DIR` points to a worktree (e.g., `/Users/stephenfeather/opc/.worktrees/feature-x`), the basename is `feature-x`, not `opc`.

**Fix:** Add worktree-aware detection:
```python
def detect_project_name(explicit: str | None = None) -> str | None:
    """Detect project name with worktree awareness."""
    if explicit:
        return explicit

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if project_dir:
        p = Path(project_dir)
        # Worktree detection: if path contains .worktrees, use the parent project name
        parts = p.parts
        if ".worktrees" in parts:
            idx = parts.index(".worktrees")
            return parts[idx - 1] if idx > 0 else p.name
        return p.name

    # Fallback: git toplevel
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()).name
    except Exception:
        pass

    return None
```

**Acceptance criteria:**
- [ ] Worktree paths resolve to parent project name
- [ ] Git fallback works when CLAUDE_PROJECT_DIR is unset

---

### Phase 2: Backfill Existing NULLs

**File to create:** `scripts/migrations/backfill_project_column.py`

This should be a Python script (not raw SQL) for auditability, dry-run support, and complex heuristic logic.

**Steps:**

**2A. Tier 1 backfill: Sessions table join**

```sql
-- Path-to-name mapping (applied in Python with normalization)
UPDATE archival_memory am
SET project = <normalized_basename>
FROM sessions s
WHERE am.session_id = s.id
  AND am.project IS NULL
  AND s.project IS NOT NULL;
```

Path normalization rules (Python-side):
| Path Pattern | Normalized Name |
|---|---|
| `*/Continuous-Claude-v3/opc` | `opc` |
| `*/opc` | `opc` |
| `*/opc/scripts/core` | `opc` |
| `*/alexa-care-log-skill` | `alexa-skill` |
| `*/alexa-medical-logger-skill` | `alexa-skill` |
| `*/fa-wpmcp*` | `fa-wpmcp` |
| `*/binbrain-ios` | `binbrain-ios` |
| `*/binbrain/binbrain` | `binbrain` |
| `*/binbrain` | `binbrain` |
| `*/Pharmacokinetics-Grapher` | `pharmacokinetics-grapher` |
| `*/sermon-browser*` | `sermon-browser` |
| `*/.dotfiles/claude` | `opc` (rules/hooks are OPC infra) |
| Others | basename of path, lowercased |

Special cases:
- `/Users/stephenfeather` alone -> `_personal` (ambiguous top-level)
- `/Users/stephenfeather/Development` alone -> `_ambiguous`

**2B. Tier 2 backfill: Tag/content heuristics (for remaining NULLs)**

Tag-to-project rules (ordered by specificity):
| Tag(s) Present | Inferred Project |
|---|---|
| `alexa` OR `skill-builder` | `alexa-skill` |
| `binbrain` | `binbrain` |
| `pharmacokinetics` OR `pk` | `pharmacokinetics-grapher` |
| `wordpress` AND (`fa-toolkit` OR `wpmcp`) | `fa-wpmcp` |
| `3d-print-logger` | `3d-print-logger` |
| `fa-toolkit` | `fa-toolkit` |
| `ios` AND `swift` | `binbrain-ios` |
| `hooks` AND `daemon` | `opc` |
| `crash-recovery` AND `database` | `opc` |

Content keyword rules (lower priority, only if tags don't match):
| Content Pattern | Inferred Project |
|---|---|
| `CLAUDE_PROJECT_DIR` OR `memory_daemon` OR `archival_memory` | `opc` |
| `Alexa` AND `skill` | `alexa-skill` |
| `pharmacokinet` | `pharmacokinetics-grapher` |

Context field rules:
| Context Pattern | Inferred Project |
|---|---|
| Contains `Vue` or `perception` | Needs manual review (could be alexa-skill Vue frontend or other) |

**2C. Tier 3: Flag ambiguous**

Remaining NULLs after Tier 1+2 get `project = '_unresolved'` so they're distinguishable from truly unset values. These can be manually reviewed later.

**Script features:**
- `--dry-run` flag: show counts per tier without modifying
- `--tier 1|2|3|all`: run specific tiers
- `--verbose`: show per-row decisions
- Transaction-wrapped: all-or-nothing per tier
- Outputs before/after counts

**Acceptance criteria:**
- [ ] Tier 1 resolves ~1,289 learnings via sessions join
- [ ] Tier 2 resolves additional ~100-150 via tag/content heuristics
- [ ] Remaining flagged as `_unresolved` (not left NULL)
- [ ] Dry-run shows expected changes before applying
- [ ] Script is idempotent (safe to re-run)

---

### Phase 3: Validation

**3A. Before/after counts**

```sql
SELECT project, COUNT(*) FROM archival_memory GROUP BY project ORDER BY COUNT(*) DESC;
```

**Expected after backfill:**

| Project | Count (approx) |
|---------|------|
| opc | ~1,400 |
| alexa-skill | ~490 |
| binbrain-ios | ~65 |
| fa-wpmcp | ~80 |
| pharmacokinetics-grapher | ~45 |
| agentic-work | ~36 |
| binbrain | ~21 |
| sermon-browser | ~8 |
| _unresolved | ~50-100 |

**3B. Spot-check accuracy**

Sample 5-10 learnings per inferred project and verify content matches:
```sql
SELECT id, LEFT(content, 100), project
FROM archival_memory
WHERE project = '<inferred>'
ORDER BY RANDOM() LIMIT 5;
```

**3C. Reranker integration test**

After backfill, run recall queries with project context and verify `rerank_details.project_match` is non-zero for relevant results.

**Acceptance criteria:**
- [ ] Zero NULL values in project column
- [ ] Spot-check confirms >95% accuracy for Tier 1
- [ ] Spot-check confirms >80% accuracy for Tier 2
- [ ] Reranker project_match signal fires for known-project queries

---

## Testing Strategy

1. **Unit tests for `detect_project_name()`**: Test worktree paths, normal paths, git fallback, no-project scenarios
2. **Integration test for MCP store_learning**: Verify project flows through MCP -> CLI -> DB
3. **Backfill script dry-run**: Run with `--dry-run` and review output before applying
4. **Post-backfill validation**: SQL counts + spot-checks as described in Phase 3

## Risks and Considerations

| Risk | Mitigation |
|------|-----------|
| Wrong project inference in Tier 2 | Use `_unresolved` for low-confidence; keep tag rules conservative |
| Sessions table has stale/wrong project paths | Spot-check before bulk update; the 1,084 OPC matches look correct |
| Multiple projects share same session_id | Query shows no overlapping session_ids between projects |
| Path normalization misses edge cases | Map explicitly for known projects; lowercase-basename for unknowns |
| Backfill breaks reranker behavior | project_match was returning 0.0 for all NULLs already; adding data can only improve |
| MCP server is in separate repo | Changes needed in `/Users/stephenfeather/Tools/opc-memory-mcp/main.py` |

## Estimated Complexity

| Phase | Effort | Files Changed |
|-------|--------|---------------|
| 1A: MCP server project param | Small (30 min) | 1 file (external repo) |
| 1B: Daemon env fix | Trivial (5 min) | 1 line in `memory_daemon.py` |
| 1C: Workflow extraction fix | Trivial (5 min) | 2 lines in `memory_daemon.py` |
| 1D: Legacy v1 fix | Small (15 min) | `store_learning.py` |
| 1E: Worktree-aware detection | Small (20 min) | `store_learning.py` |
| 2: Backfill script | Medium (1-2 hrs) | New file `scripts/migrations/backfill_project_column.py` |
| 3: Validation | Small (30 min) | Manual SQL + review |
| **Total** | **~3-4 hours** | **4 files modified, 1 new file** |

## File Summary

| File | Action |
|------|--------|
| `/Users/stephenfeather/Tools/opc-memory-mcp/main.py` | Add project param + auto-detect |
| `/Users/stephenfeather/opc/scripts/core/memory_daemon.py` | Pass CLAUDE_PROJECT_DIR to extraction env; pass project to store_learning_v2 |
| `/Users/stephenfeather/opc/scripts/core/store_learning.py` | Add `detect_project_name()` helper; fix v1 path |
| `/Users/stephenfeather/opc/scripts/migrations/backfill_project_column.py` | New: 3-tier backfill script |
| `/Users/stephenfeather/opc/scripts/migrations/add_project_column.sql` | No changes needed (schema already correct) |
