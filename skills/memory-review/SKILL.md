---
name: memory-review
description: Review a project's archival_memory and propose promotions, near-duplicate merges, and stale cleanup for user approval. Read-only — applies nothing.
user-invocable: true
---

# Memory Review

Review the memory landscape for a project and produce a grouped report of proposed
changes. **Do NOT apply anything** — present proposals for the user to approve item by
item. This is the OPC analogue of Claude Code's `/remember` skill (issue #63), but
hybrid: hard signals from `archival_memory` (recall frequency, embedding cosine, age)
*find* the candidates, then you *judge* destinations against the live memory layers.

## Usage

```
/memory-review [project]            # default: current project (cwd basename)
/memory-review opc --promote-only   # skip the cleanup detectors
/memory-review opc --cleanup-only   # skip the promotion detector
/memory-review opc --threshold 0.92 # near-dup cosine cutoff (default 0.90)
```

## Process

### 1. Run the detector

Run the read-only candidate detector for the target project:

```bash
uv run python scripts/core/memory_review.py <project>
```

It prints a grouped report with three sections: **Promotions**, **Cleanup — merges**,
and **Cleanup — stale**. It changes nothing.

**Large projects (e.g. `opc`, ~2,700 learnings):** the merge scan runs one sequential HNSW
probe per active learning, so on a corpus this size it exceeds the connection pool's 60s
cap and the report shows a "merge scan exceeded its time budget" note (promotions and stale
still complete). `--ef-search` / `--merge-timeout` help **medium** projects finish, but do
**not** rescue opc-scale corpora — there the right tool is the offline SQL prototype
(`thoughts/shared/2026-06-20-issue-63-candidate-detection.sql`), which runs in psql with no
pool cap and completes the full merge scan. Promotion and stale detection always work
regardless of corpus size.

### 2. Judge each candidate against the live memory layers

The detector proposes destinations by `learning_type` (USER_PREFERENCE → `rules/`,
CODEBASE_PATTERN → `MEMORY.md`, ARCHITECTURAL_DECISION → `CLAUDE.md`). Before presenting,
verify each:

- Read the proposed destination layer (`MEMORY.md`, project `CLAUDE.md`, or the relevant
  `rules/` file) and check the learning is **not already captured there** — if it is, that's
  a cleanup (remove the redundant archival row), not a promotion.
- Confirm the destination fits. A high-recall fact belongs in `MEMORY.md`; a prescriptive
  "always/never" belongs in a `rules/` file. When the right layer is genuinely unclear,
  move it to **Ambiguous** and ask — do not guess.
- For **merge** pairs, read both learnings and confirm they are truly redundant (not
  complementary). Propose keeping the higher-recall row and superseding the other.
- For **stale** candidates, remember "never recalled" ≠ "worthless" — it may simply not
  have come up. Treat as review candidates, never auto-delete.

### 3. Present the grouped report

Output the detector's report, refined by your judgment, grouped by action:

1. **Promotions** — entries to move, with destination and rationale
2. **Cleanup — merges** — near-duplicate pairs to consolidate
3. **Cleanup — stale** — aged/unused entries to archive
4. **Ambiguous** — entries needing the user's call on destination
5. **No action** — brief note on what stays on-demand and why

### 4. Apply promotions on approval (Phase 2a)

Detection (steps 1–3) is read-only. **Applying** approved promotions is a separate,
explicitly-gated step via `scripts/core/memory_apply.py` (`opc memory-apply`). It is
**dry-run by default** and writes nothing without `--execute`.

After the user approves specific promotion items, apply them by their learning ids:

1. **Dry-run first (always).** Show exactly what would be written:
   ```bash
   uv run python scripts/core/memory_apply.py <project> --ids <id1,id2,...>
   ```
   This prints the resolved write targets (`MEMORY.md` dir, `CLAUDE.md` path) and a plan
   grouped by target, with skips for already-promoted or unsupported (e.g. `rules/`) items.
   **Verify the resolved paths** before going further.
2. **Confirm with the user**, then execute. A `pg_dump` DB backup runs first; the apply
   aborts if the backup fails:
   ```bash
   uv run python scripts/core/memory_apply.py <project> --ids <id1,id2,...> --execute
   ```

What apply does (Phase 2a):
- `CODEBASE_PATTERN` → a new `promoted-<slug>.md` Claude-memory file + a `MEMORY.md` pointer.
- `ARCHITECTURAL_DECISION` → appended to the project `CLAUDE.md` under `## Promoted Decisions`.
- Tags the source `archival_memory` row's metadata with `promoted_to` (idempotent: a
  re-apply is skipped; the row is **never** deleted — promotion is additive and reversible).

**Deferred (not yet in apply):** `USER_PREFERENCE` → `rules/` (separate repo), and
merge-supersede / stale-archive cleanup. Those still stop at the read-only report.

Never hard-delete an `archival_memory` row (provenance-tag only — recall must keep working).

## Rules

- Present ALL proposals before making any change.
- Do NOT modify files or the database without explicit user approval.
- Ask about ambiguous entries — don't guess the destination.
- Every detector query is project-scoped; never promote one project's learning into
  another's memory layer.
- Promotion never deletes the source archival row (recall must keep working).
- Stay-on-demand types (WORKING_SOLUTION, ERROR_FIX, FAILED_APPROACH) are intentionally
  never promoted — they flood always-loaded context.
- **Phase-1 limitation:** the promotion detector filters by `learning_type`, so a
  high-recall entry *mislabeled* as a stay-on-demand or unknown type is not surfaced for
  promotion. A type-agnostic sweep that re-judges every high-recall entry is a Phase-2
  concern; for now, fixing mislabels is the data-quality path.
- **Merge scan is single-space:** it scans only the project's dominant `embedding_model`.
  After a partial re-embed the report discloses how many rows in other spaces were
  skipped — treat a "partial scan" note as a signal the merge results are incomplete.
