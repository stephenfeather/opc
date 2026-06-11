# OPC API Reference

`scripts/core/recall_learnings.py` — Semantic recall of session learnings.

## Default JSON Response

```bash
uv run python scripts/core/recall_learnings.py --query "hook patterns" --json
```

```json
{
  "results": [
    {
      "id": "abc-123",
      "score": 0.85,
      "raw_score": 0.5,
      "learning_type": "WORKING_SOLUTION",
      "session_id": "session-1",
      "content": "TypeScript hooks require npm install...",
      "created_at": "2026-03-29T12:00:00+00:00"
    }
  ],
  "total": 1
}
```

When reranking is active (default), results may also include:

```json
{
  "results": [
    {
      ...
      "rerank_details": {
        "calibrated_score": 0.72,
        "project_match": 1.0,
        "recency": 0.85,
        "confidence": 0.5,
        "recall": 0.0,
        "type_match": 0.5,
        "tag_overlap": 0.0,
        "pattern": 0.0,
        "kg_overlap": 0.0,
        "kg_active": false
      }
    }
  ],
  "total": 1
}
```

## Flags

### `--query`, `-q` (required)

Search query for semantic matching.

### `--k` (default: 5)

Number of results to return.

### `--json`

Output as JSON instead of human-readable text. Required for programmatic consumers (MCP server, hooks).

### `--json-full`

Extended JSON for benchmarking. Adds `metadata`, `recall_count`, `pattern_strength`, and `pattern_tags` to each result:

```json
{
  "results": [
    {
      "id": "abc-123",
      "score": 0.85,
      "raw_score": 0.5,
      "learning_type": "WORKING_SOLUTION",
      "session_id": "session-1",
      "content": "...",
      "created_at": "2026-03-29T12:00:00+00:00",
      "metadata": { "learning_type": "WORKING_SOLUTION", "tags": ["hooks"], ... },
      "recall_count": 3,
      "pattern_strength": 0.72,
      "pattern_tags": ["hook-development"]
    }
  ],
  "total": 1
}
```

Note: `--json-full` skips `record_recall` (read-only mode for benchmarking).

### `--structured`

Adds a `groups` key that organizes results by `learning_type`. The `results` flat list is always present.

```json
{
  "results": [ ... ],
  "total": 3,
  "structured": true,
  "groups": {
    "FAILED_APPROACH": [
      { "id": "...", "score": 0.72, ... }
    ],
    "WORKING_SOLUTION": [
      { "id": "...", "score": 0.85, ... },
      { "id": "...", "score": 0.60, ... }
    ]
  }
}
```

Group keys appear in canonical order:

1. `FAILED_APPROACH`
2. `ERROR_FIX`
3. `WORKING_SOLUTION`
4. `ARCHITECTURAL_DECISION`
5. `CODEBASE_PATTERN`
6. `USER_PREFERENCE`
7. `OPEN_THREAD`
8. *(unknown types sorted alphabetically)*

Relevance order is preserved within each group.

### `--text-only`

Use BM25 text search only (faster, no embeddings). Skips vector search entirely.

### `--vector-only`

Use vector-only search (disables hybrid RRF). Enables SQL-level recency blending when `--no-rerank` is also set.

### `--threshold`, `-t` (default: 0.2)

Minimum similarity threshold. Filters low-quality results before reranking.

### `--recency`, `-r` (default: 0.1)

Recency weight for `--vector-only` mode (0.0-1.0). Only applies when `--no-rerank` is also set (otherwise the reranker handles recency).

### `--tags`

Space-separated tags to boost in reranking. Results matching these tags score higher.

### `--tags-strict`

Hard-filter: only return results sharing at least one tag with `--tags`. Applied before reranking.

### `--no-rerank`

Bypass contextual re-ranking. Returns raw retrieval scores. When set, `score` equals `raw_score`.

### `--project`

Project context for re-ranking. Default: auto-detected from `CLAUDE_PROJECT_DIR`.
The value is canonicalized (lowercased, known aliases collapsed — see
`scripts/core/project_naming.py`) before matching, and the match itself is
case-insensitive.

`--project` alone only affects **re-ranking** (the `project_match` signal),
not the fetch pool. On a large project, the global fetch can fill its entire
candidate pool with foreign rows before the reranker runs, so own-project
rows are never surfaced. Use `--project-first` to change the fetch composition.

### `--project-first`

Opt-in fetch-time project scoping (issue #139). Runs a **two-pass fetch**:
pass 1 is scoped to the resolved project via a case-insensitive SQL
`AND LOWER(project) = $N` clause, pass 2 fetches globally; the two are merged
own-project-first (with a reserved global quota of ~half the pool so the global
pass is never fully starved), deduped by id, and truncated to the rerank pool
size. This guarantees own-project rows are in the pool when they exist —
complementary to `--project`'s rerank signal, not a replacement.

**Legacy data prerequisite:** the scoped predicate is case-insensitive, so
un-migrated rows stored as `OPC`/`Opc` still match the canonical `opc` bind.
But alias and flattened-path legacy values (which the canonicalizer collapses,
e.g. `operations-digitalocean` → `digitalocean`) are *not* matched by the
scoped pass — they are only surfaced via the global fill pass and the
`project_match` rerank signal. Run `scripts/migrations/normalize_project_values.py`
to canonicalize stored project values for full `--project-first` effectiveness
on legacy data.

If one pass fails transiently, the other pass's results are still returned (a
warning naming the degraded pass is printed to stderr); recall only errors when
both passes fail.

The project is resolved from `--project` if given, otherwise auto-detected
from `CLAUDE_PROJECT_DIR` (worktree-aware), then canonicalized before the SQL
bind. If no project resolves, a warning is printed to stderr and recall
degrades to the normal global fetch (exit code unchanged). On a pre-migration
database that lacks the `project` column, the scoped pass is skipped and a
single global fetch runs — no error. The SQLite cache backend has no project
column and always degrades to a global fetch.

### `--source`

Short caller label recorded with each recall event in the `recall_log` table
(issue #140), e.g. `hook`, `mcp`, or `cli`. Optional; defaults to `NULL`
(unknown). **Label-only** — pass a fixed identifier for the call site, never
prompt-derived or user text. The label is validated at the writer against
`^[a-z][a-z0-9_-]{0,31}$`; any value that does not match (uppercase, spaces,
over 32 chars, prompt-like text) is **dropped to `NULL`** rather than stored,
so arbitrary text cannot leak into the append-only log. No query text is ever
logged (privacy). See the [Recall Event Log](#recall-event-log) section.

### `--provider` (default: local)

Embedding provider. Choices: `local` (BGE), `voyage` (Voyage AI).

## Search Modes

| Flags | Search Method | Score Range |
|-------|---------------|-------------|
| *(default)* | Hybrid RRF (text + vector) | 0.01-0.03 |
| `--text-only` | BM25 text search | 0.01-0.05 |
| `--vector-only` | Cosine similarity | 0.4-0.6 |

`--project-first` is orthogonal to the search mode: it changes the fetch-pool
composition (own-project rows first, then a global fill) and applies to every
PostgreSQL mode above. The SQLite backend ignores it and fetches globally.

## Recall Event Log

Every recall (except `--json-full` benchmarking) appends one row to the
`recall_log` table (issue #140) so cross-project recall mis-scope frequency
(issue #130) is measurable. The recalled rows' projects are captured
point-in-time via `UPDATE archival_memory ... RETURNING id, project`, so no
extra round trip is added. **Zero-result recalls are logged too** — with empty
`recalled_ids`/`recalled_projects` arrays and `result_count = 0` — because
finding nothing is the signature of over-restrictive project scoping (#130).

**Counters and the log are decoupled by design.** The counter columns
(`recall_count` / `last_recalled` on `archival_memory`) and `recall_log` are
written as **two separate autocommitted statements**, not a transaction or CTE,
so a pre-migration DB keeps working counters even when the log INSERT fails. A
consequence: a transient INSERT failure can bump counters without writing a log
row. Treat counters as a coarse popularity signal and `recall_log` as the
analysis source of truth.

### Schema

| Column | Type | Notes |
|--------|------|-------|
| `id` | `BIGINT` identity | Primary key |
| `caller_project` | `TEXT` | Canonicalized caller project; `NULL` = no project context |
| `recalled_ids` | `UUID[]` | Ids of the rows whose counters were bumped |
| `recalled_projects` | `TEXT[]` | Parallel to `recalled_ids`; `NULL` elements = unattributed memories |
| `result_count` | `INTEGER` | Number of recalled rows (length of the arrays) |
| `source` | `TEXT` | Short caller label from `--source` (`hook`/`mcp`/`cli`); validated `^[a-z][a-z0-9_-]{0,31}$` at the writer, invalid → `NULL`; `NULL` = unknown |
| `created_at` | `TIMESTAMPTZ` | Event time, defaults to `NOW()` |

### Write semantics

- **Skipped entirely** for `--json-full` (benchmark mode) and the SQLite
  backend (no project/log columns — logs a debug line and returns).
- **Zero-result vs. stale-id:** an empty `result_ids` (the recall found
  nothing) skips the counter UPDATE but still logs a `result_count = 0` row.
  This is distinct from supplying ids that match no rows (e.g. concurrently
  deleted ids) — that is a stale-id event and the INSERT is skipped, since the
  recall did surface candidates.
- **Best-effort:** the INSERT runs as a separate autocommitted statement after
  the counter UPDATE (not a CTE or transaction). On a pre-migration DB that
  lacks `recall_log`, the INSERT fails alone, is swallowed (debug log), and the
  counter UPDATE still persists.
- **Version-skew safe:** if `archival_memory.project` itself is missing
  (temporal-decay columns applied but `add_project_column.sql` not), the
  `RETURNING project` fetch raises `UndefinedColumnError`; recall falls back to
  the original counter-only `UPDATE` (no `RETURNING`) so counters never
  silently stop, and skips the `recall_log` INSERT for that event.
- **Source validated at the writer:** `source` is checked against
  `^[a-z][a-z0-9_-]{0,31}$` in Python before the INSERT; an invalid label is
  dropped to `NULL`. This is deliberately *not* a DB `CHECK` constraint — a
  CHECK violation would abort the whole INSERT and silently drop the entire log
  row, losing the recall event.
- **Latency-bounded:** the call is wrapped in `asyncio.wait_for(...,
  timeout=RECORD_RECALL_TIMEOUT)` (2.0s, well under the memory-awareness hook's
  5s `spawnSync` budget, which waits for process exit). On timeout the write is
  cancelled (cancellation-safe — the `pool.acquire()` context manager releases
  the connection) and the event is dropped with a debug log; output is already
  printed by then.
- **Never raises:** the whole path is wrapped so recall can never break.
- **No query text, ever:** only project labels, ids, a count, and a source
  label are stored (privacy — see issue #139).

### Mis-scope analysis (answers issue #130)

```sql
SELECT caller_project,
       COUNT(*) AS recalled_rows,
       COUNT(*) FILTER (WHERE rp IS NULL) AS unattributed_rows,
       COUNT(*) FILTER (WHERE rp IS NOT NULL AND rp <> caller_project) AS mis_scoped_rows,
       ROUND(100.0 * COUNT(*) FILTER (WHERE rp IS NOT NULL AND rp <> caller_project) / COUNT(*), 1) AS mis_scope_pct
FROM recall_log
CROSS JOIN LATERAL unnest(recalled_projects) AS rp
WHERE caller_project IS NOT NULL
  -- time-scope to a recent window; the created_at DESC index
  -- (idx_recall_log_created) supports this range scan
  AND created_at > NOW() - INTERVAL '30 days'
GROUP BY caller_project ORDER BY mis_scope_pct DESC;
```

The `LATERAL unnest` drops zero-result rows automatically (an empty
`recalled_projects` array yields no rows), which is correct for per-row
mis-scope analysis. Count those events separately as a scoping-pressure signal
(same recent window, same index):

```sql
SELECT caller_project,
       COUNT(*) FILTER (WHERE result_count = 0) AS empty_recalls,
       COUNT(*) AS total_recalls
FROM recall_log
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY caller_project ORDER BY empty_recalls DESC;
```

**Retention:** manual until automated pruning lands (follow-up issue). Pruning
is operator-owned for now — run the documented `DELETE` periodically, e.g.:

```sql
DELETE FROM recall_log WHERE created_at < NOW() - INTERVAL '90 days';
```

## Error Response

```json
{
  "error": "Connection refused",
  "results": []
}
```

---

# Store API Reference

`scripts/core/store_learning.py` — Store session learnings with embeddings and deduplication.

## Default JSON Response (v2)

```bash
uv run python scripts/core/store_learning.py \
  --session-id "session-1" \
  --content "Pattern X works well for Y" \
  --type WORKING_SOLUTION \
  --json
```

```json
{
  "success": true,
  "memory_id": "abc-123-def-456",
  "backend": "postgres",
  "content_length": 26,
  "embedding_dim": 1024
}
```

### `--supersedes`

When provided, adds `superseded` to the response:

```json
{
  "success": true,
  "memory_id": "new-id-789",
  "backend": "postgres",
  "content_length": 26,
  "embedding_dim": 1024,
  "superseded": "old-id-456"
}
```

### Duplicate detected (semantic)

When content is >=92% similar to an existing learning:

```json
{
  "success": true,
  "skipped": true,
  "reason": "duplicate (similarity: 0.95, session: session-2)",
  "existing_id": "existing-id-123"
}
```

### Duplicate detected (content hash)

When exact content already exists:

```json
{
  "success": true,
  "skipped": true,
  "reason": "duplicate (content_hash match)"
}
```

### Error

```json
{
  "success": false,
  "error": "No content provided"
}
```

## Flags

### `--session-id` (required)

Session identifier for the learning.

### `--content`

Direct learning content (v2 mode). When provided, uses v2 storage path with deduplication, auto-classification, and confidence calibration.

### `--type`

Learning type. Choices:

| Type | Use For |
|------|---------|
| `FAILED_APPROACH` | Things that didn't work |
| `WORKING_SOLUTION` | Successful approaches |
| `USER_PREFERENCE` | User style/preferences |
| `CODEBASE_PATTERN` | Discovered code patterns |
| `ARCHITECTURAL_DECISION` | Design choices made |
| `ERROR_FIX` | Error-to-solution pairs |
| `OPEN_THREAD` | Unfinished work/TODOs |

### `--context`

What this learning relates to (e.g., "hook development", "database migration").

### `--tags`

Comma-separated tags for categorization (e.g., "hooks,typescript,build").

### `--confidence`

Confidence level. Choices: `high`, `medium`, `low`. When omitted, auto-calibrated using heuristic scorer.

### `--project`

Project name for recall relevance. Default: auto-detected from `CLAUDE_PROJECT_DIR`.
Stored canonicalized (lowercased, known aliases collapsed — see
`scripts/core/project_naming.py`). Existing rows can be normalized with
`scripts/migrations/normalize_project_values.py` (dry-run by default).

### `--supersedes`

UUID of an older learning this one replaces. The old learning is marked with `superseded_by` pointing to the new one. Recall queries filter superseded learnings via `WHERE superseded_by IS NULL`. Atomic — both the insert and update run in a single transaction.

### `--auto-classify`

Auto-classify learning type via LLM judge. Requires `BRAINTRUST_API_KEY`. Only runs when `--type` is omitted or is `WORKING_SOLUTION` (the default).

### `--host-id`

Machine identifier for multi-system support.

### `--json`

Output as JSON instead of human-readable text.

## Legacy Mode (v1)

When `--content` is not provided, falls back to legacy mode with category-based storage:

```bash
uv run python scripts/core/store_learning.py \
  --session-id "session-1" \
  --worked "Approach X worked well" \
  --failed "Y didn't work" \
  --decisions "Chose Z because..." \
  --patterns "Reusable technique..."
```

Legacy flags: `--worked`, `--failed`, `--decisions`, `--patterns`. These are concatenated into a single learning with category metadata. No deduplication, auto-classification, or confidence calibration.

## Deduplication

Two layers, checked in order:

1. **Semantic dedup** — Cosine similarity >= 0.92 against all existing learnings (cross-session). Returns `skipped` with `existing_id`.
2. **Content hash dedup** — SHA-256 of exact content. Returns `skipped` without `existing_id`.

---

# Memory Feedback API Reference

`scripts/core/memory_feedback.py` — Track whether recalled learnings were actually useful.

## Subcommands

### `store` — Record feedback for a learning

```bash
uv run python scripts/core/memory_feedback.py store \
  --learning-id <uuid> --helpful --session-id <sid>
```

#### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--learning-id` | Yes | — | UUID of the learning to rate |
| `--helpful` | One of two (required) | — | Mark as helpful |
| `--not-helpful` | One of two (required) | — | Mark as not helpful |
| `--session-id` | No | `"cli"` | Session identifier |
| `--context` | No | `""` | Why it was/wasn't helpful |
| `--source` | No | `"manual"` | Feedback source (e.g., `manual`, `hook`, `auto`) |

`--helpful` and `--not-helpful` are mutually exclusive; exactly one is required.

Upserts on `(learning_id, session_id)` — submitting again for the same pair updates the existing feedback.

#### Success Response

```json
{
  "success": true,
  "feedback_id": "770840f8-7a35-44b6-9cb1-9b8fda3dfbf4",
  "learning_id": "820c9584-08ee-4d49-b932-a7ff5334ec0a",
  "helpful": true,
  "created_at": "2026-04-01T23:09:14.622523+00:00"
}
```

#### Error Response (learning not found)

```json
{
  "success": false,
  "error": "Learning 00000000-0000-0000-0000-000000000000 not found"
}
```

---

### `get` — Retrieve feedback for a specific learning

```bash
uv run python scripts/core/memory_feedback.py get --learning-id <uuid>
```

#### Flags

| Flag | Required | Description |
|------|----------|-------------|
| `--learning-id` | Yes | UUID of the learning |

#### Response

```json
{
  "learning_id": "820c9584-08ee-4d49-b932-a7ff5334ec0a",
  "total_feedback": 2,
  "helpful_count": 1,
  "not_helpful_count": 1,
  "feedback": [
    {
      "id": "770840f8-7a35-44b6-9cb1-9b8fda3dfbf4",
      "session_id": "session-abc",
      "helpful": true,
      "context": "prevented me from repeating a known mistake",
      "source": "manual",
      "created_at": "2026-04-01T23:09:14.622523+00:00"
    },
    {
      "id": "88812345-aaaa-bbbb-cccc-dddddddddddd",
      "session_id": "session-xyz",
      "helpful": false,
      "context": null,
      "source": "manual",
      "created_at": "2026-04-01T20:00:00.000000+00:00"
    }
  ]
}
```

---

### `summary` — Aggregate feedback statistics

```bash
uv run python scripts/core/memory_feedback.py summary
```

No flags required.

#### Response

```json
{
  "total_feedback": 10,
  "helpful_count": 7,
  "not_helpful_count": 3,
  "unique_learnings_rated": 5,
  "helpfulness_rate": 70.0,
  "top_helpful": [
    {
      "learning_id": "820c9584-08ee-4d49-b932-a7ff5334ec0a",
      "content": "Patching get_pool at postgres_pool source module level...",
      "helpful_count": 3
    }
  ],
  "top_not_helpful": [
    {
      "learning_id": "aaaabbbb-cccc-dddd-eeee-ffffffffffff",
      "content": "Some irrelevant learning that kept getting surfaced...",
      "not_helpful_count": 2
    }
  ]
}
```

`top_helpful` and `top_not_helpful` each return up to 5 entries. Content is truncated to 120 characters.

## Database Table

`memory_feedback` — stores one feedback record per `(learning_id, session_id)` pair.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `learning_id` | UUID | FK to `archival_memory.id` (CASCADE delete) |
| `session_id` | TEXT | Which session provided the feedback |
| `helpful` | BOOLEAN | Was the learning useful? |
| `context` | TEXT | Optional explanation |
| `source` | TEXT | `manual`, `hook`, `auto`, etc. |
| `created_at` | TIMESTAMPTZ | When feedback was recorded |

## Metrics Integration

Feedback stats appear in `memory_metrics.py` output under `feedback_alltime`:

```text
Feedback (all-time):  7 helpful, 3 not helpful out of 10 (70.0% helpful), 5 unique learnings rated
```
