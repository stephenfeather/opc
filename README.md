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
- **Contextual reranking** - Adaptive reranker that reorders recall results using recency, tag relevance, type inference, and per-mode calibration
- **Cross-session pattern detection** - Identifies recurring patterns across sessions (repeated errors, tool preferences, architectural decisions) and boosts them in recall
- **Learning chains** - Learnings can supersede previous entries, keeping the knowledge base current without duplication
- **Temporal decay tracking** - Tracks when learnings are recalled and decays stale entries that haven't been useful recently
- **Semantic deduplication** - Prevents storing near-duplicate learnings using embedding similarity checks across sessions
- **Tag-based filtering** - Store and recall learnings with tags for precise filtering (`--tags`, `--tags-strict`)

## Project Structure

```
scripts/core/              Core memory system
  recall_learnings.py        Semantic search over stored learnings
  store_learning.py          Persist learnings to PostgreSQL
  memory_daemon.py           Background extraction and handoff generation
  reranker.py                Contextual reranking with adaptive overfetch
  pattern_detector.py        Cross-session pattern detection
  pattern_batch.py           Batch pattern analysis
  pattern_report.py          Pattern reporting
  artifact_index.py          Artifact indexing and querying
  artifact_query.py          Artifact querying interface
  artifact_mark.py           Artifact marking
  extract_thinking_blocks.py Thinking block extraction from transcripts
  extract_workflow_patterns.py Workflow pattern extraction
  generate_mini_handoff.py   Automatic mini-handoff generation
  db/                        Database layer
    embedding_service.py       Multi-provider embedding abstraction
    memory_service_pg.py       PostgreSQL memory storage
    memory_protocol.py         Backend protocol definition
    memory_factory.py          Backend factory
    postgres_pool.py           Connection pooling

src/runtime/               MCP execution runtime

docker/                    PostgreSQL setup, container sandboxing

hooks/                     Claude Code hook scripts
```

## Requirements

- Python 3.12+
- PostgreSQL with pgvector extension
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# Install dependencies
uv sync

# Copy and edit environment config
cp .env.example .env

# Start PostgreSQL (via Docker)
docker compose -f docker/docker-compose.yml up -d

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

## Database Schema

The complete database schema is in [`docker/init-schema.sql`](docker/init-schema.sql). It extends the upstream [Continuous-Claude-v3](https://github.com/parcadei/Continuous-Claude-v3) schema with:

- **`sessions`** — 10 extra columns for the memory daemon extraction pipeline, process liveness, transcript archival, and multi-host coordination
- **`archival_memory`** — Extra columns for embedding provenance (`embedding_model`), deduplication (`content_hash`), multi-host support (`host_id`), learning chains (`superseded_by`), temporal decay (`recall_count`, `last_recalled_at`), and project scoping (`project`)
- **`memory_tags`** — Tag table for categorizing learnings with fast lookup
- **`cross_session_patterns`** — Detected patterns across sessions (recurring errors, tool preferences, decisions)
- **`continuity`** — Session state snapshots (continuity ledger system)
- **`plans`** — Indexed implementation plans

## Hook Scripts

The `hooks/` directory contains Claude Code hook scripts used by OPC, including originals from the Continuous-Claude-v3 project and new additions for memory awareness, modern CLI enforcement, and other integrations.

## License

MIT License. See [LICENSE](LICENSE) for details.
