# OPC - Opinionated Persistent Context

A memory and context persistence system for Claude Code, providing semantic recall, session handoffs, artifact indexing, and embedding-backed search across sessions.

## Origin

This project began as a fork of [Continuous-Claude-v3](https://github.com/parcadei/Continuous-Claude-v3) by [@parcadei](https://github.com/parcadei). Over time, personal enhancements and structural changes diverged enough that contributing back upstream no longer made sense. It is shared here for others to incorporate into their own memory systems.

## What It Does

- **Semantic memory** - Store and recall learnings across sessions using embedding-backed search (Voyage, OpenAI, local, or Ollama)
- **Session handoffs** - Generate YAML handoff documents so new sessions can resume where previous ones left off
- **Artifact indexing** - Track and query files, plans, and other artifacts across the project
- **Memory daemon** - Background process that extracts thinking blocks, workflow patterns, and generates mini-handoffs automatically
- **Multi-provider embeddings** - Pluggable embedding service supporting Voyage AI, OpenAI, local sentence-transformers, and Ollama

## Project Structure

```
scripts/core/          Core memory system
  recall_learnings.py    Semantic search over stored learnings
  store_learning.py      Persist learnings to PostgreSQL
  memory_daemon.py       Background extraction and handoff generation
  artifact_index.py      Artifact indexing and querying
  db/                    Database layer
    embedding_service.py   Multi-provider embedding abstraction
    memory_service_pg.py   PostgreSQL memory storage
    postgres_pool.py       Connection pooling

src/runtime/           MCP execution runtime

docker/                Container sandboxing
```

## Requirements

- Python 3.12+
- PostgreSQL with pgvector extension
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# Install dependencies
uv sync

# Start PostgreSQL (via Docker)
docker compose up -d

# Run the memory daemon
uv run python scripts/core/memory_daemon.py
```

## Embedding Providers

The embedding service supports multiple backends, configured via the `provider` parameter or environment variables:

| Provider | Env Variable | Dimensions |
|----------|-------------|------------|
| Voyage AI | `VOYAGE_API_KEY` | 1024 |
| OpenAI | `OPENAI_API_KEY` | 1536 |
| Local (BGE) | None needed | 1024 |
| Ollama | `OLLAMA_HOST` | Varies |

## Database Schema Changes

The database schema has diverged from the upstream [Continuous-Claude-v3](https://github.com/parcadei/Continuous-Claude-v3) `docker/init-schema.sql`. These changes were made before this repo was tracked in git. If you are migrating from the upstream schema, the following ALTER statements capture the differences.

### `sessions` table — 10 columns and 1 index added

Support for the memory daemon's extraction pipeline, process liveness checks, transcript archival, and multi-host coordination.

```sql
ALTER TABLE sessions ADD COLUMN memory_extracted_at TIMESTAMP;
ALTER TABLE sessions ADD COLUMN claude_session_id TEXT;
ALTER TABLE sessions ADD COLUMN transcript_path TEXT;
ALTER TABLE sessions ADD COLUMN exited_at TIMESTAMP;
ALTER TABLE sessions ADD COLUMN pid INTEGER;
ALTER TABLE sessions ADD COLUMN host_id TEXT;
ALTER TABLE sessions ADD COLUMN archived_at TIMESTAMP;
ALTER TABLE sessions ADD COLUMN archive_path TEXT;
ALTER TABLE sessions ADD COLUMN extraction_status TEXT DEFAULT 'pending';
ALTER TABLE sessions ADD COLUMN extraction_attempts INTEGER DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_sessions_host ON sessions(host_id);
```

| Column | Why |
|--------|-----|
| `memory_extracted_at` | Tracks when the daemon last extracted learnings from a session |
| `claude_session_id` | Links to the Claude Code session ID (distinct from the row `id`) |
| `transcript_path` | Path to the JSONL transcript file for post-hoc extraction |
| `exited_at` | Records when the session ended, enabling reaping logic |
| `pid` | Process ID for liveness checks — prevents reaping active sessions |
| `host_id` | Identifies which machine a session runs on for multi-host setups |
| `archived_at` | When the transcript was archived (S3 or local) |
| `archive_path` | Location of the archived transcript |
| `extraction_status` | Pipeline state: `pending`, `extracting`, `done`, `failed` |
| `extraction_attempts` | Retry counter for failed extractions |

### `archival_memory` table — 3 columns and 2 indexes added

Deduplication, embedding provenance tracking, and multi-host support.

```sql
ALTER TABLE archival_memory ADD COLUMN embedding_model TEXT DEFAULT 'bge';
ALTER TABLE archival_memory ADD COLUMN host_id TEXT;
ALTER TABLE archival_memory ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_archival_content_hash ON archival_memory(content_hash);
CREATE INDEX IF NOT EXISTS idx_archival_host ON archival_memory(host_id);
```

| Column | Why |
|--------|-----|
| `embedding_model` | Tracks which model (BGE, Voyage, OpenAI) generated the embedding — needed when re-embedding with a different provider |
| `host_id` | Multi-host identification, same as sessions |
| `content_hash` | SHA-256 of content for deduplication — the unique index prevents storing the same learning twice |

### New table: `continuity`

Session continuity snapshots for the continuity ledger system. Captures structured state at session boundaries.

```sql
CREATE TABLE IF NOT EXISTS continuity (
    id TEXT PRIMARY KEY,
    session_name TEXT,
    goal TEXT,
    state_done TEXT,
    state_now TEXT,
    state_next TEXT,
    key_learnings TEXT,
    key_decisions TEXT,
    snapshot_reason TEXT,
    indexed_at TIMESTAMP DEFAULT NOW()
);
```

### New table: `plans`

Indexed implementation plans so sessions can discover and resume planned work.

```sql
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    title TEXT,
    file_path TEXT,
    overview TEXT,
    approach TEXT,
    phases TEXT,
    constraints TEXT,
    indexed_at TIMESTAMP DEFAULT NOW()
);
```

## Note on Hook Scripts

The scripts in this repo reference Claude Code hook scripts that are not yet included. Some are new, others are modified versions of hooks from the original Continuous-Claude-v3 project. They will be added to the repo in the future.

## License

MIT License. See [LICENSE](LICENSE) for details.
