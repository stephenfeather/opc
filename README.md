# OPC - Opinionated Persistent Context

> **⚠️ Refactoring in Progress** — This project is currently undergoing a significant refactoring effort. Expect breaking changes, restructured modules, and evolving APIs until the refactor is complete.

A memory and context persistence system for Claude Code, providing semantic recall, session handoffs, artifact indexing, and embedding-backed search across sessions.

## Origin

This project began as a fork of [Continuous-Claude-v3](https://github.com/parcadei/Continuous-Claude-v3) by [@parcadei](https://github.com/parcadei). Over time, personal enhancements and structural changes diverged enough that contributing back upstream no longer made sense. It is shared here for others to incorporate into their own memory systems.

## What It Does

- **Semantic memory** - Store and recall learnings across sessions using embedding-backed search (Voyage, OpenAI, local, or Ollama)
- **Session handoffs** - Generate YAML handoff documents so new sessions can resume where previous ones left off
- **Artifact indexing** - Track and query files, plans, and other artifacts across the project
- **Memory daemon** - Background process that extracts thinking blocks, workflow patterns, and generates mini-handoffs automatically, with date-based log rotation
- **Multi-provider embeddings** - Pluggable embedding service supporting Voyage AI, OpenAI, local sentence-transformers, and Ollama
- **Contextual reranking** - Adaptive reranker that reorders recall results using recency, tag relevance, type inference, and per-mode calibration
- **Cross-session pattern detection** - Identifies recurring patterns across sessions (repeated errors, tool preferences, architectural decisions) and boosts them in recall
- **Knowledge graph** - A structured layer over `archival_memory` that extracts typed entities (files, tools, errors, languages, etc.) and relationships (`solves`, `uses`, `supersedes`, ...) from each learning at store time, then enriches recall results with shared entity context and a `kg_overlap` reranker signal. See [docs/knowledge-graph.md](docs/knowledge-graph.md) for details.
- **Learning chains** - Learnings can supersede previous entries, keeping the knowledge base current without duplication
- **Memory curation** - A review→apply loop to keep the corpus clean: promote high-value learnings into `CLAUDE.md`/`MEMORY.md`, merge-supersede near-duplicate pairs, stale-archive retired entries, and reverse a promotion (unpromote). All actions are dry-run by default, backed up before writing, idempotent, and project-scoped. See [Memory Curation](#memory-curation) below and the [API reference](docs/recall-api-reference.md#memory-curation-api-reference).
- **Temporal decay tracking** - Tracks when learnings are recalled and decays stale entries that haven't been useful recently
- **Semantic deduplication** - Prevents storing near-duplicate learnings using embedding similarity checks across sessions, with per-session rejection tracking
- **Tag-based filtering** - Store and recall learnings with tags for precise filtering (`--tags`, `--tags-strict`)
- **LLM learning classification** - Auto-classifies learnings by type using a tuned prompt with eval harness (84.3% accuracy), wired into pattern detection
- **Project-scoped extraction** - Daemon passes project context through both LLM and workflow extraction paths, enabling project-match reranking
- **Per-project extraction opt-out** - Drop a `.claude/no-extract` sentinel file in any project to suppress memory extraction by both the daemon and the `--learn` skill
- **Memory feedback** - Track learning usefulness with per-session feedback (helpful/not helpful), surfaced in recall hints and feedback summaries
- **Active memory push** - Proactively surfaces relevant learnings at session start and on prompt submission via hooks, reducing stale learning rates
- **Memory metrics** - CLI health dashboard reporting totals, confidence distribution, extraction stats, tag usage, and temporal trends (`--human` or `--json`)
- **TF-IDF query expansion** - Expands text queries with semantically related terms before hybrid RRF search using pseudo-relevance feedback over vector neighbors and corpus IDF scoring (`--no-expand` to disable)
- **Rerank A/B benchmarking** - Golden-set bootstrap tool and sweep framework for tuning reranker weights with measurable accuracy metrics
- **TOML-driven configuration** - All daemon, reranker, dedup, recall, embedding, and pattern settings configurable via `opc.toml` with type validation and env overrides
- **Dedup rejection tracking** - Records rejected (near-duplicate) learnings in `learning_rejections` table with similarity scores, surfaced in daemon extraction logs
- **Document-collection RAG** - Scope-aware retrieval over folders of born-digital documents (`.pdf` text layer, `.docx`, `.txt`, `.csv`, `.md`, `.html`, `.xml`). A YAML registry tracks folders, each with a `global` or `restricted` scope: default queries search only `global` collections, so `restricted` records (medical/legal/business) are retrievable only when explicitly named. Incremental, idempotent ingestion (sha256 skip-unchanged, deletion reconciliation) into pgvector via the existing embedding service, queryable with an `opc-docs` CLI and a cron rescan wrapper. See [Document-Collection RAG](#document-collection-rag) below.

## Project Structure

```
opc.toml                   Configuration (daemon, reranker, dedup, recall, etc.)
scripts/core/              Core memory system
  recall_learnings.py        Semantic search over stored learnings
  store_learning.py          Persist learnings to PostgreSQL
  memory_daemon.py           Background extraction and handoff generation
  memory_metrics.py          Memory system health and quality metrics
  reranker.py                Contextual reranking with adaptive overfetch
  query_expansion.py         TF-IDF query expansion for hybrid RRF recall
  pattern_detector.py        Cross-session pattern detection
  pattern_batch.py           Batch pattern analysis
  pattern_report.py          Pattern reporting
  push_learnings.py          Active memory push for proactive recall
  artifact_index.py          Artifact indexing and querying
  artifact_query.py          Artifact querying interface
  artifact_mark.py           Artifact marking
  extract_thinking_blocks.py Thinking block extraction from transcripts
  extract_workflow_patterns.py Workflow pattern extraction
  generate_mini_handoff.py   Automatic mini-handoff generation
  config/                     TOML config loading, validation, models
  db/                        Database layer
    embedding_service.py       Multi-provider embedding abstraction
    memory_service_pg.py       PostgreSQL memory storage
    memory_protocol.py         Backend protocol definition
    memory_factory.py          Backend factory
    postgres_pool.py           Connection pooling
  documents/                 Document-collection RAG layer
    registry.py                YAML registry of tracked folders (scope, extensions)
    extract.py                 Born-digital text extraction (pdf/docx/txt/csv/md/html/xml)
    chunk.py                   Page-aware chunking
    db.py                      Scope-gated pgvector storage and retrieval
    ingest.py                  Incremental, idempotent ingest pipeline
    query.py                   Scoped semantic query
    cli.py                     opc-docs CLI (create/scan/list/query)
    rescan_cron.sh             Cron wrapper for incremental rescans

src/runtime/               MCP execution runtime

docker/                    PostgreSQL setup, container sandboxing

hooks/                     Claude Code hook scripts
```

## Requirements

- Python 3.13+
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
uv run python scripts/core/memory_daemon.py start

# Run with verbose diagnostic logging (Issue #99)
uv run python scripts/core/memory_daemon.py start --debug
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
- **`memory_feedback`** — Per-session learning usefulness feedback with upsert on (learning_id, session_id)
- **`learning_rejections`** — Dedup rejection audit log with similarity scores, thresholds, and matched existing learning references
- **`cross_session_patterns`** — Detected patterns across sessions (recurring errors, tool preferences, decisions)
- **`continuity`** — Session state snapshots (continuity ledger system)
- **`plans`** — Indexed implementation plans
- **`documents`** — One row per ingested file (collection, path, sha256 hash, extraction status), with a unique `(collection_name, file_path)` constraint for idempotent re-ingestion
- **`document_chunks`** — Embedded text chunks for the RAG layer, carrying the retrieval `scope` (`global`/`restricted`), an HNSW vector index, and a content full-text index; cascades on `documents` delete
- **`archival_memory` HNSW index** — Approximate nearest-neighbor index on embeddings for fast vector search

## Document-Collection RAG

Ingest folders of born-digital documents into pgvector and query them semantically. The core property is **scope safety**: each collection is `global` or `restricted`, and a default query searches `global` collections only — `restricted` records (medical/legal/business) surface only when a collection is named explicitly.

```bash
# Register a folder (restricted -> retrievable only by explicit --collection)
uv run python scripts/core/documents/cli.py create caleb-records \
    --path "~/Documents/Feather, Caleb" --scope restricted --extensions .pdf,.docx

# Ingest (incremental: unchanged files skipped, deleted files purged)
uv run python scripts/core/documents/cli.py scan --all

# List collections and ingest stats
uv run python scripts/core/documents/cli.py list

# Default query: global collections only (restricted never leaks)
uv run python scripts/core/documents/cli.py query "who flagged home meds"

# Explicit collection: the only way to reach a restricted collection
uv run python scripts/core/documents/cli.py query "who flagged home meds" --collection caleb-records
```

Supported formats: `.pdf` (text layer), `.docx`, `.txt`, `.csv`, `.md`, `.html`/`.htm`, `.xml`. Scanned/image-only PDFs are recorded as `skipped_needs_ocr` for a future OCR phase. The registry path defaults to `~/.claude/opc/document_collections.yaml` (override with `OPC_DOC_REGISTRY`); ceilings are tunable via `OPC_DOC_MAX_FILE_MB` and `OPC_DOC_MAX_CHUNKS`. A cron line calling `scripts/core/documents/rescan_cron.sh` keeps collections current.

## Memory Curation

Apply **approved** curation actions to the corpus with `scripts/core/memory_apply.py`. Every mode is
**dry-run by default** — add `--execute` to write (a `pg_dump` backup runs first). Actions are
idempotent and project-scoped.

```bash
# Promote approved learnings into CLAUDE.md / MEMORY.md (dry-run, then apply)
uv run python scripts/core/memory_apply.py --ids <id1>,<id2>
uv run python scripts/core/memory_apply.py --ids <id1>,<id2> --execute

# Merge-supersede a near-duplicate pair: keep the higher-recall row, retire the loser
uv run python scripts/core/memory_apply.py --merge --pair <id_a>:<id_b> --execute

# Stale-archive retired learnings (sets archived_at; reversible)
uv run python scripts/core/memory_apply.py --archive --ids <id1>,<id2> --execute

# Unpromote: reverse a promotion (remove the file artifact, clear the promoted_to tag)
uv run python scripts/core/memory_apply.py --unpromote --ids <id1> --execute
```

`--archive` needs the `archived_at` column. A fresh DB has it from `docker/init-schema.sql`; an
existing DB runs the migration once (recall degrades gracefully until then):

```bash
docker exec -i opc-postgres psql -U claude -d continuous_claude -f - \
  < scripts/migrations/add_archived_at.sql
```

Full flag reference, the safety model, and the keeper tie-break: [docs/recall-api-reference.md](docs/recall-api-reference.md#memory-curation-api-reference).

## Hook Scripts

The `hooks/` directory contains Claude Code hook scripts used by OPC, including originals from the Continuous-Claude-v3 project and new additions for memory awareness, modern CLI enforcement, and other integrations.

## License

MIT License. See [LICENSE](LICENSE) for details.
