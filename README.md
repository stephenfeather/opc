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

docker/                PostgreSQL setup, container sandboxing
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
- **`archival_memory`** — 3 extra columns for embedding provenance (`embedding_model`), deduplication (`content_hash`), and multi-host support (`host_id`)
- **`continuity`** — New table for session state snapshots (continuity ledger system)
- **`plans`** — New table for indexed implementation plans

## Note on Hook Scripts

The scripts in this repo reference Claude Code hook scripts that are not yet included. Some are new, others are modified versions of hooks from the original Continuous-Claude-v3 project. They will be added to the repo in the future.

## License

MIT License. See [LICENSE](LICENSE) for details.
