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
      "rerank_details": { "recency_boost": 0.1, "tag_boost": 0.05, ... }
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

### `--provider` (default: local)

Embedding provider. Choices: `local` (BGE), `voyage` (Voyage AI).

## Search Modes

| Flags | Search Method | Score Range |
|-------|---------------|-------------|
| *(default)* | Hybrid RRF (text + vector) | 0.01-0.03 |
| `--text-only` | BM25 text search | 0.01-0.05 |
| `--vector-only` | Cosine similarity | 0.4-0.6 |

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
