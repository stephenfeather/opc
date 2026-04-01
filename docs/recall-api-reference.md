# Recall API Reference

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
