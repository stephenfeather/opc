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

### `--llm-rerank` (issue #228 item 3)

**OFF by default.** After candidate retrieval, exclusion filtering, and
enrichment, use an LLM to select and reorder the top results instead of the
deterministic contextual reranker. The candidates are rendered as a compact
manifest (`[type] id (timestamp): description`) and sent to the Anthropic
Messages API via a single forced tool-use call that returns
`{"selected_memories": ["id", ...]}`. The returned ids are mapped back onto the
candidate pool (unknown ids dropped, duplicates removed, order preserved, then
trimmed to `--k`).

- **Fallback semantics (no regression):** the selector returns nothing and
  recall falls back to the standard contextual reranker on **every** failure
  mode — empty candidate pool, missing `ANTHROPIC_API_KEY`, API/network error,
  timeout (bounded at 2.5s), malformed output, or an empty / all-unknown
  selection. Because the fallback is the existing reranker, enabling
  `--llm-rerank` can never produce worse results than the default path.
- **`--no-rerank` interaction:** `--llm-rerank` lives *inside* the rerank gate,
  so `--no-rerank` suppresses it too. Passing both `--no-rerank --llm-rerank`
  runs neither stage (raw retrieval order is returned).
- **Ignored for `--source hook` calls** (the memory-awareness hook) until
  recall is deadline-aware. The hook kills recall at 5s via `spawnSync`, and an
  LLM call inside that budget risks dropping recall output entirely if killed
  mid-call; this flag is intended for CLI/benchmark use until a recall-wide
  pre-output deadline lands.
- **Config errors are operator-visible:** if `recall.llm_selector_model` cannot
  be resolved (e.g. a malformed `opc.toml`), recall prints a warning to stderr
  (without the config value) and falls back to the contextual reranker rather
  than silently degrading.
- **`--k` is an upper bound, not a floor:** if the LLM selects fewer than `k`
  valid ids, recall returns only those (it does not pad). Selected rows carry
  the reranker output contract (`final_score`, `rerank_details` with
  `"source": "llm_selector"`), so JSON output and telemetry are unaffected.
- **Requirements:** `ANTHROPIC_API_KEY` must be set in the environment. The
  model is the `recall.llm_selector_model` config field (default
  `claude-sonnet-4-6`); set it to a cheaper model (e.g. a Haiku) in `opc.toml`
  to reduce per-call cost.
- **Cost (issue #228 E3):** each `--llm-rerank` recall makes **one** Sonnet
  Messages API call — a manifest of roughly 50 short candidate lines in, and a
  small id list out (well under 1k output tokens). This is a per-recall cost
  only when the flag is explicitly passed; the default recall path (and the
  memory-awareness hook) make no LLM call. If the selector is ever enabled in a
  hot path, weigh both the per-call token cost and the added latency, and
  consider dropping `recall.llm_selector_model` to a cheaper model.

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

### `--exclude-ids` (default: none)

Already-surfaced filtering (issue #228 item 2). Space-separated list of full
learning UUIDs to drop from the results, e.g.
`--exclude-ids 11111111-1111-1111-1111-111111111111 2222...`. Optional;
`nargs="*"` with an empty-list default, so omitting it (or passing it with no
values) is a no-op identical to prior behavior.

**Semantics — where the filter runs:** the exclusion is applied **after** the
`pool_size` telemetry capture and **before** rerank. Two consequences follow
from that ordering:

- Excluded ids are removed from the candidate pool *before* `rerank()`, so a
  previously-surfaced learning **cannot rank back into the top-k** and then be
  trimmed away — it is gone before ranking even runs.
- `pool_size` (issue #228 item 1 selection-rate telemetry) is captured on the
  **raw backend pool**, *before* this filter, so exclusion **does not affect**
  the recorded `pool_size` / selection-rate denominator. Exclusion is a
  downstream filter, like `--tags`, and is intentionally invisible to the pool
  telemetry.

Id matching normalizes both sides to strings, so a `UUID`-typed result id and a
plain-string exclude value compare equal. The filter is a pure post-fetch Python
step, so it applies uniformly across all backends (text-only, vector, hybrid
RRF) and both the single-pass and `--project-first` two-pass dispatch paths.

**Backfill — avoiding starvation:** because the filter runs *after* the backend
returns its fixed over-fetch pool, `fetch_k` is increased by the size of the
exclude set when `--exclude-ids` is non-empty. Otherwise a session that had
already surfaced the entire top-of-pool would filter every candidate out and
return nothing, even when fresh lower-ranked candidates exist. The bump is
bounded by the hook's surfaced-id cap, so the extra fetch and rerank cost stays
small.

Most callers do not pass `--exclude-ids` directly — the `memory-awareness` hook
uses `--surfaced-session` (below), which assembles the exclusion set in-process.

### `--surfaced-session` (default: none)

Already-surfaced filtering for the `memory-awareness` UserPromptSubmit hook
(issue #228 item 2). Takes a session id (the `sessions` PK `id`, which
session-register sets equal to the Claude session id). When set, recall:

1. **reads** that session's `sessions.surfaced_learning_ids` (a primary-key
   read, no extra index) and unions them into the exclusion set alongside any
   explicit `--exclude-ids`, applied with the same before-rerank / after-pool-size
   semantics described above;
2. after output, **upserts** `surfaced_learning_ids` to the prior set unioned
   with the ids returned this run (deduped, capped at the most-recent 500),
   keyed on the PK `id` via `ON CONFLICT (id) DO UPDATE` so a missing
   SessionStart row does not silently drop the write.

Both the read and the write run **in this recall process** — the hook passes a
single session id instead of reading the column itself and passing a long
`--exclude-ids` list, so the prompt hot path pays only one Python/uv startup per
turn. Both are time-bounded and best-effort: on any DB error the read degrades
to no exclusion and the write is skipped, so recall never breaks. The read also
over-fetches by the exclude-set size (see the backfill note above) so a session
that already surfaced the top of the pool still gets fresh results.

### `--provider` (default: local)

Embedding provider. Choices: `local` (BGE), `voyage` (Voyage AI).

## Embedding-space contract (issue #151)

Vector similarity is only meaningful **within a single embedding space**.
Different providers (and different models within a provider) produce vectors
that are not comparable even when they share a dimension — a `local`/BGE
vector and a `voyage-code-3` vector are both 1024-dim, so PostgreSQL will
happily cosine-compare them, but the result is semantic noise. Mixing spaces
in one corpus ("split-brain") silently degrades recall, dedup, and query
expansion.

**Canonical space: `voyage-code-3`.** This matches the global
`EMBEDDING_PROVIDER=voyage` default and is the space the corpus is being
migrated to (`scripts/core/re_embed_voyage.py`, bge → voyage-code-3).

**How the contract is enforced**

- Each provider exposes a stable `model_label` (the value written to the
  `archival_memory.embedding_model` column):

  | Provider | `model_label` |
  |----------|---------------|
  | `local` (any BGE variant) | `bge` |
  | `local` (non-BGE model) | the model name |
  | `voyage` | the model name (e.g. `voyage-code-3`, honoring `--model`) |
  | `openai` | `text-embedding-3-small` |
  | `ollama` | the model name |
  | `mock` | `mock` |

- Store paths write `embedding_model` explicitly from `model_label`, so new
  rows are labeled by their real space (previously the column silently fell
  back to its `'bge'` default, mislabeling voyage rows).

- Every vector-distance query path filters to the query provider's space via
  `AND embedding_model = $N` (hybrid RRF vector leg, `--vector-only`,
  recency-weighted vector, and query-expansion neighbors). The filter lives on
  the **vector leg only** — never the FTS leg.

**Post-canonicalization behavior of `--provider local`**

After the corpus is canonicalized to `voyage-code-3`, a `--provider local`
(BGE) query embeds in `bge` space and its `AND embedding_model = 'bge'`
vector filter matches **zero rows** until/unless the corpus is re-embedded
back to BGE. This is deliberate, not a bug: the empty vector leg makes hybrid
RRF degrade to the text (BM25) leg (see
[Hybrid → text-only degradation](#hybrid--text-only-degradation-issue-53)),
so `--provider local` becomes **text-degraded** rather than returning
cross-space noise. Use `--provider voyage` (or rely on the
`EMBEDDING_PROVIDER=voyage` default) for full vector recall. To make `local`
fully vector-capable again, re-embed the corpus to BGE.

## Search Modes

| Flags | Search Method | Score Range |
|-------|---------------|-------------|
| *(default)* | Hybrid RRF (text + vector) | 0.01-0.03 |
| `--text-only` | BM25 text search | 0.01-0.05 |
| `--vector-only` | Cosine similarity | 0.4-0.6 |

`--project-first` is orthogonal to the search mode: it changes the fetch-pool
composition (own-project rows first, then a global fill) and applies to every
PostgreSQL mode above. The SQLite backend ignores it and fetches globally.

### Hybrid → text-only degradation (issue #53)

Hybrid RRF needs a query embedding. When that embedding cannot be produced —
a missing API key, a model load error, or a network/provider timeout — the
hybrid path does **not** error out. It degrades to the text-only backend for
that one query, with the same `k` and `--project` semantics, and returns
results whose shape is identical to `--text-only` (no `fts_rank` / `vec_rank`
/ decay keys). This lets the memory-awareness hook call hybrid (instead of
hardcoding `--text-only`) without risking a hard failure when no embedding
provider is reachable: in that case behavior is identical to today's
`--text-only`.

**Trigger:** any exception from the query-embed call inside
`search_learnings_hybrid_rrf`, including a `QUERY_EMBED_TIMEOUT` (2.0s)
deadline. The deadline exists because providers carry their own long
timeouts (e.g. Voyage: httpx 30s + retries ≈ 90s+); without it a network
stall would blow the hook's 5s budget and the fallback would never run.
2.0s leaves ~3s for the text-only query, reranking, and output. The warning
says "unavailable or timed out" since both land in this path.

**Warning:** a single redacted line is printed to stderr (where the
memory-awareness hook captures it into model context):

```
warning: hybrid recall degraded to text-only (provider 'voyage' unavailable or timed out: <redacted reason>); set the provider API key or pass --text-only to silence.
```

The reason text is passed through the `#139` credential redactor
(`sanitize_log_message`), so DSN passwords never leak, and the **query text is
never included** in the warning. The full traceback is kept in the debug log
only. If the text-only fallback itself also fails, that fallback's
exception propagates (the embedding failure stays attached as the chained
`__context__`), so recall still fails loudly when neither path works.

**This degrade is deliberate, not an error.** The contract is to fall back
quietly so the hook keeps working in the exact missing-key case the feature
targets; config problems are not made fatal. To reduce noise, the
human-readable stderr warning is **latched to once per process**
(`_EMBED_DEGRADE_WARNED`) — under `--project-first` the second (global) pass
does not reprint it, and repeated recalls in a long-lived process warn only
once. `logger.debug` still records every degrade. Because the
memory-awareness hook spawns a fresh process per prompt, the practical cap is
one warning per prompt. Embedder-cleanup (`aclose`) failures during the
degrade are swallowed to the debug log so they cannot abort the fallback.

## Recall Event Log

Every recall (except `--json-full` benchmarking) appends one row to the
`recall_log` table (issue #140) so cross-project recall mis-scope frequency
(issue #130) is measurable. The recalled rows' projects are captured
point-in-time via `UPDATE archival_memory ... RETURNING id, project`, so no
extra round trip is added. **Zero-result recalls are logged too** — with empty
`recalled_ids`/`recalled_projects` arrays and `result_count = 0` — because
finding nothing is the signature of over-restrictive project scoping (#130).
Each row also records the **candidate pool size** (`pool_size`) and the
over-fetch ceiling (`fetch_k`) so the recall **selection rate**
(`result_count / pool_size`) is measurable (issue #228) — including for
zero-result events, which now carry a non-NULL denominator.

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
| `pool_size` | `INTEGER` | Issue #228 selection-shape telemetry: the raw backend candidate pool (the over-fetch, `compute_fetch_k = max(3*k, 50)` when reranking) captured **before** enrichment/tag-filter/rerank trim it. `NULL` = pre-#228 row or legacy-INSERT fallback (rate unknown) |
| `fetch_k` | `INTEGER` | The requested over-fetch ceiling (`max(3*k, 50)` when reranking, else `k`). With `pool_size` it shows corpus saturation — `pool_size < fetch_k` means the backend had fewer candidates than requested. `NULL` = pre-#228 / legacy fallback |
| `source` | `TEXT` | Short caller label from `--source` (`hook`/`mcp`/`cli`); validated `^[a-z][a-z0-9_-]{0,31}$` at the writer, invalid → `NULL`; `NULL` = unknown |
| `created_at` | `TIMESTAMPTZ` | Event time, defaults to `NOW()` |

`pool_size` is the raw backend candidate pool, captured before client-side
`--tags` filtering and the rerank trim. With `--tags`, the returned
`result_count` reflects that extra filtering, so the selection rate over
`pool_size` is a **lower bound**; for the primary telemetry source (the
memory-awareness hook, which passes no `--tags`) it is exact. Selection rate is
**not stored** — compute it at query time (see *Selection-shape analysis*).

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
- **Column-skew safe (issue #228):** the writer attempts the 7-column INSERT
  (including `pool_size`/`fetch_k`) first; on `UndefinedColumnError` — a DB where
  `recall_log` exists but `add_recall_log_pool_size.sql` has not run — it falls
  back to the legacy 5-column INSERT so the recall event is **still logged**
  (`pool_size`/`fetch_k` simply stay absent for that row). The new telemetry
  never regresses existing #140 logging. Run the migration to populate the new
  columns.
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

### Selection-shape analysis (answers issue #228)

The recall **selection rate** — what fraction of the candidate pool was actually
returned — is computed at query time from `result_count` and `pool_size`. A low
rate across a narrow score band is the signal the LLM-selector work (issue #228
item 3) targets; this query is its measurement baseline. Only rows written after
the migration have a non-NULL `pool_size`, so filter them in:

```sql
SELECT caller_project,
       COUNT(*) AS recalls,
       ROUND(AVG(result_count::numeric / NULLIF(pool_size, 0)), 3) AS avg_selection_rate,
       ROUND(AVG(pool_size), 1) AS avg_pool_size,
       COUNT(*) FILTER (WHERE pool_size = fetch_k) AS saturated_recalls
FROM recall_log
WHERE pool_size IS NOT NULL          -- post-#228 rows only
  AND created_at > NOW() - INTERVAL '30 days'  -- idx_recall_log_created
GROUP BY caller_project
ORDER BY avg_selection_rate;
```

`NULLIF(pool_size, 0)` keeps zero-pool recalls (no candidates at all) from
dividing by zero — their rate is `NULL` and drops out of the average.
`saturated_recalls` (`pool_size = fetch_k`) counts events where the backend
filled the entire over-fetch ceiling, i.e. the corpus was larger than the pool.

**Retention:** automated (issue #146). The memory daemon prunes rows older than
`daemon.recall_log_retention_days` (default `90`) every
`daemon.recall_log_prune_interval_hours` (default `24`), deleting in bounded
batches so the append-only hot-path INSERT is never blocked. Set
`recall_log_retention_days` to `0` (or env `RECALL_LOG_RETENTION_DAYS=0`) to
disable pruning. The equivalent manual prune for ad-hoc operator use:

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

### `--source-time` (issue #52)

ISO8601 timestamp used to stamp `created_at` from the originating session time
instead of the database `NOW()` default. This exists so backfilled learnings
(extracted long after the session occurred) do not masquerade as age-zero and
inflate the reranker's recency score.

- Naive timestamps (no offset) are interpreted as **UTC**.
- Garbage, more-than-5-minutes-in-the-future, or **pre-2024-01-01** values
  (implausibility floor matching `fix_backfill_created_at.sql`) are **ignored
  with a warning** — the store still succeeds with the default `created_at`.
- When omitted, behavior is unchanged (the DB default applies).
- Falls back to the `CLAUDE_SOURCE_TIME` environment variable when the flag is
  absent — but only inside the extraction subprocess: the env value is honored
  **only when `CLAUDE_MEMORY_EXTRACTION=1`** is also set (trust boundary), so
  an ambient or user-set value cannot silently backdate live stores. The
  backfill pipeline (`backfill_learnings.py`) injects a trusted source time
  via this env var, preferring `sessions.exited_at`, then the S3 listing
  LastModified, then the local JSONL mtime, so the memory-extractor agent's
  `store_learning.py` calls stamp the correct time without needing to pass
  the flag itself.

```bash
uv run python scripts/core/store_learning.py \
  --session-id "session-1" \
  --content "Pattern X works for Y" \
  --source-time "2026-03-29T20:50:15+00:00"
```

**Retroactive repair:** existing backfilled rows are fixed by the migration
`scripts/migrations/fix_backfill_created_at.sql`, which sets
`archival_memory.created_at` from `sessions.exited_at` for sessions with
`working_on = 'backfill'`, only ever moving the timestamp *earlier*. Run the
dry-run `SELECT` in the migration header before applying.

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

### Same-space-only semantic dedup during a split corpus (issue #151)

Semantic dedup is **same-embedding-space-only by design**. The probe is pinned
to the same `embedding_model` label that will be written with the new row
(see the [Embedding-space contract](#embedding-space-contract-issue-151)),
because cosine similarity across different embedding spaces is meaningless — a
voyage vector compared against bge rows produces noise, and a spurious
cross-space "match" would silently skip a legitimate write (data loss). The
consequence during the split-corpus window is that a cross-space **paraphrase**
duplicate (same idea, different space) is temporarily **not** caught by
semantic dedup. **Exact** duplicates are still caught across spaces by the
content-hash layer (it is space-independent). This window closes once
`scripts/core/re_embed_voyage.py` canonicalizes the corpus to a single space,
after which every row shares one label and semantic dedup spans the whole
corpus again. Corpus-wide retroactive duplicate cleanup is tracked separately
in issue #58.

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

---

# Memory Curation API Reference

`scripts/core/memory_apply.py` applies **approved** curation actions to `archival_memory` and the
local memory files (`CLAUDE.md`, `MEMORY.md`). It is the apply half of the review→apply loop: the
read-only detector `scripts/core/memory_review.py` proposes candidates, you approve ids, and this
tool enacts them. Issue #63 Phases 2a/2b.

```bash
uv run python scripts/core/memory_apply.py [project] <mode flags> [--execute]
```

## Safety model (every mode)

- **Dry-run by default.** Without `--execute` the tool resolves the plan, prints it, and writes
  **nothing** — no DB rows, no files.
- **`--execute` backs up first.** Before the first write it takes a `pg_dump` of `archival_memory`
  and, for file-touching modes, a snapshot of `MEMORY.md`/`CLAUDE.md`. A backup failure aborts the
  run before any change.
- **Idempotent.** Every mutation is a guarded UPDATE / file op; re-running an already-applied action
  is a reported skip, never an error. Safe to re-run after a partial failure.
- **Per-project flock.** Concurrent runs are serialized per lock root: promote and unpromote lock
  on the memory directory, while merge and archive lock on the backup directory — so overriding
  `--backup-dir` across simultaneous same-project merge/archive runs can defeat the lock (the
  guarded, idempotent SQL still prevents corruption).
- **Project-scoped.** Writes are guarded by `LOWER(project) = LOWER($n)`; a global UUID from another
  project is skipped, not mutated.

## Modes

Exactly one mode per invocation. `--merge`, `--archive`, and `--unpromote` are mutually exclusive;
omitting all three is **promote** mode.

| Mode | Flag | Inputs | Effect |
|------|------|--------|--------|
| **Promote** (default) | *(none)* | `--ids` / `--manifest` | Append approved learnings to `CLAUDE.md` or `MEMORY.md` and stamp `metadata.promoted_to` (Phase 2a) |
| **Merge-supersede** | `--merge` | `--pair ID_A:ID_B` (repeatable) | Keep the higher-recall row of each near-duplicate pair; mark the loser `superseded_by → keeper` (tie-break: higher recall → older `created_at` → smaller id) |
| **Stale-archive** | `--archive` | `--ids` / `--manifest` | Set `archived_at = NOW()` on each stale learning (no survivor) and stamp a `superseded_via {reason:"stale"}` marker. The row is retained (`archived_at` is nullable) for manual recovery or backup restore — there is no `--unarchive` CLI yet |
| **Unpromote/repair** | `--unpromote` | `--ids` / `--manifest` | Reverse a promotion: remove the promoted file artifact(s) **then** clear the `promoted_to` tag (file-first, so a partial failure never strands the DB tag) |

## Flags

### `project` (positional, optional)
Project to scope the action to. Defaults to the current working directory (worktree-aware).

### `--ids`
Comma-separated approved learning ids. Used by promote / `--archive` / `--unpromote`.

### `--manifest`
Path to a file of approved ids (one per line). Alternative to `--ids`.

### `--pair ID_A:ID_B`
A merge pair, repeatable (`--pair a:b --pair c:d`). Used only with `--merge`. Each id is validated
as a UUID; malformed pairs and self-merges are rejected, and pairs are de-duplicated
order-insensitively.

### `--execute`
Perform the writes. Without it the run is a dry-run. A DB backup (and file snapshot for
file-touching modes) runs first.

### `--memory-dir`, `--claude-md`, `--backup-dir`
Override the Claude memory directory, the `CLAUDE.md` path, and the backup output directory. Default
to the project's standard locations.

## Examples

```bash
# Promote two approved learnings (dry-run, then apply)
uv run python scripts/core/memory_apply.py --ids 1111...,2222...
uv run python scripts/core/memory_apply.py --ids 1111...,2222... --execute

# Merge-supersede a near-duplicate pair (keep higher-recall, retire the loser)
uv run python scripts/core/memory_apply.py --merge --pair 1111...:2222... --execute

# Stale-archive a batch of approved ids from a manifest file
uv run python scripts/core/memory_apply.py --archive --manifest approved-stale.txt --execute

# Reverse a promotion (remove the file artifact, clear the promoted_to tag)
uv run python scripts/core/memory_apply.py --unpromote --ids 1111... --execute
```

## Stale-archive migration (`--archive`)

`--archive` and the `archived_at IS NULL` recall filter require the `archived_at` column on
`archival_memory`. A fresh DB gets it from `docker/init-schema.sql`; an existing DB must run the
migration once:

```bash
docker exec -i opc-postgres psql -U claude -d continuous_claude -f - \
  < scripts/migrations/add_archived_at.sql
```

The migration is idempotent (`ADD COLUMN IF NOT EXISTS` + `CREATE INDEX CONCURRENTLY IF NOT
EXISTS`). Until it runs, **recall** degrades gracefully (a capability probe omits the `archived_at`
clause rather than crashing), but **`--archive --execute` aborts** (exit 1, after taking its backup)
because `archive_row` requires the column — run the migration before archiving. `CREATE INDEX
CONCURRENTLY` cannot run inside a transaction block — apply the file with `psql -f` (autocommit),
not `psql -1`.
