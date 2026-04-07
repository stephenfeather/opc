# OPC Database Reference

Complete reference for all database tables, their schemas, and the scripts/hooks that read from and write to them.

## Database Architecture

OPC uses **two database systems**:

1. **PostgreSQL** (primary) — Long-term memory, session coordination, feedback, patterns. Connection via `CONTINUOUS_CLAUDE_DB_URL` or `DATABASE_URL` env var. Runs in Docker (`continuous-claude-postgres`).
2. **SQLite** (artifact index) — Context Graph for handoffs, plans, continuity ledgers, and queries. Located at `.claude/cache/artifact-index/context.db`. Used for FTS5-powered search.

Some tables exist in **both** databases with slightly different schemas (handoffs, plans, continuity). The PostgreSQL versions are the authoritative store; the SQLite versions are a search-optimized index.

### Extensions (PostgreSQL)

- `vector` — pgvector for embedding storage and similarity search
- `pg_trgm` — trigram matching for fuzzy text search

---

## PostgreSQL Tables

### Coordination Layer

#### `sessions`

Cross-terminal session awareness. Tracks active Claude Code sessions, their heartbeats, and extraction pipeline state.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | TEXT | PK | Session identifier |
| `project` | TEXT | NOT NULL | Project name |
| `working_on` | TEXT | | Current task description |
| `started_at` | TIMESTAMP | NOW() | Session start time |
| `last_heartbeat` | TIMESTAMP | NOW() | Last activity timestamp |
| `memory_extracted_at` | TIMESTAMP | | When memory extraction completed |
| `claude_session_id` | TEXT | | Claude Code's internal session UUID |
| `transcript_path` | TEXT | | Path to session JSONL transcript |
| `exited_at` | TIMESTAMP | | Clean exit timestamp (NULL = crashed) |
| `pid` | INTEGER | | Process ID of the Claude Code session |
| `host_id` | TEXT | | Machine identifier |
| `archived_at` | TIMESTAMP | | When session was archived to S3 |
| `archive_path` | TEXT | | S3 path of archived transcript |
| `extraction_status` | TEXT | 'pending' | Pipeline status: pending/in_progress/done/failed |
| `extraction_attempts` | INTEGER | 0 | Number of extraction attempts |

**Indexes:** `idx_sessions_host` on `host_id`

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `hooks/ts/src/session-register.ts` | WRITE (upsert) | `INSERT...ON CONFLICT` on SessionStart, also runs `ALTER TABLE ADD COLUMN IF NOT EXISTS` migrations |
| `hooks/ts/src/session-register.ts` | READ | `getActiveSessions()` — sessions with heartbeat < 5 min |
| `hooks/ts/src/session-clean-exit.ts` | WRITE (update) | Sets `exited_at = NOW()` on Stop event |
| `hooks/ts/src/session-crash-recovery.ts` | READ | `getCrashedSessions()` — `exited_at IS NULL` |
| `hooks/ts/src/session-crash-recovery.ts` | WRITE (update) | `markSessionsAcknowledged()` — bulk set `exited_at` |
| `scripts/core/memory_daemon.py` | READ | Selects pending sessions for extraction |
| `scripts/core/memory_daemon.py` | WRITE (update) | Updates `extraction_status`, `extraction_attempts`, `memory_extracted_at`, `exited_at` |
| `scripts/core/backfill_sessions.py` | READ | `SELECT id FROM sessions` to find already-registered sessions |
| `scripts/core/backfill_sessions.py` | WRITE (insert) | `INSERT INTO sessions` for unregistered JSONL files |

---

#### `file_claims`

Cross-terminal file locking. Prevents concurrent edits to the same file from different Claude sessions.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `file_path` | TEXT | NOT NULL, PK | Absolute file path |
| `project` | TEXT | NOT NULL, PK | Project name |
| `session_id` | TEXT | | Claiming session ID |
| `claimed_at` | TIMESTAMP | NOW() | When the claim was made |

**Primary Key:** `(file_path, project)`

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `hooks/ts/src/file-claims.ts` | READ | `checkFileClaim()` — check if another session holds the file |
| `hooks/ts/src/file-claims.ts` | WRITE (upsert) | `claimFile()` — `INSERT...ON CONFLICT DO UPDATE` |

---

### Memory Layer

#### `archival_memory`

Long-term learnings with embeddings. The core memory store for the entire system.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | UUID | `gen_random_uuid()` | Primary key |
| `session_id` | TEXT | NOT NULL | Source session |
| `agent_id` | TEXT | | Agent that created this memory |
| `content` | TEXT | NOT NULL | Learning content (free text) |
| `metadata` | JSONB | `'{}'` | Structured metadata (type, confidence, context, tags, etc.) |
| `embedding` | vector(1024) | | BGE/Voyage embedding for similarity search |
| `created_at` | TIMESTAMPTZ | NOW() | Creation timestamp |
| `embedding_model` | TEXT | 'bge' | Which model generated the embedding |
| `host_id` | TEXT | | Machine identifier |
| `content_hash` | TEXT | | SHA hash for deduplication |
| `last_recalled` | TIMESTAMPTZ | | Last time this learning was recalled |
| `recall_count` | INTEGER | 0 | Number of times recalled |
| `superseded_by` | UUID | FK→self | Points to the learning that replaced this one |
| `superseded_at` | TIMESTAMPTZ | | When this learning was superseded |
| `project` | TEXT | | Project this learning belongs to |

**Indexes:** session, agent, created_at DESC, FTS (GIN on content), content_hash (UNIQUE), host_id, last_recalled, superseded (partial), active (partial WHERE superseded_by IS NULL), project (partial), HNSW on embedding

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/db/memory_service_pg.py` | WRITE (insert) | `store()` — INSERT with dedup via `ON CONFLICT (content_hash)` |
| `scripts/core/db/memory_service_pg.py` | WRITE (update) | `store()` — marks old learning as superseded when `supersedes` param provided |
| `scripts/core/db/memory_service_pg.py` | READ | `search_text()`, `search_vector()`, `search_hybrid()`, `search_hybrid_rrf()`, `search_vector_with_threshold()`, `search_vector_with_filter()`, `search_vector_global()`, `search_with_tags()` |
| `scripts/core/db/memory_service_pg.py` | WRITE (delete) | `delete_archival()` |
| `scripts/core/store_learning.py` | WRITE (insert) | `store_learning_v2()` — high-level store with embedding generation |
| `scripts/core/recall_learnings.py` | WRITE (update) | `record_recall()` — updates `last_recalled` and `recall_count` |
| `scripts/core/recall_backends.py` | READ | `search_learnings_text_only_postgres()`, `search_learnings_postgres()`, `search_learnings_hybrid_rrf()` — all query archival_memory |
| `scripts/core/push_learnings.py` | READ | Selects learnings for proactive push |
| `scripts/core/re_embed_voyage.py` | READ + WRITE | Reads embeddings, re-embeds with Voyage, updates `embedding` and `embedding_model` |
| `scripts/core/confidence_calibrator.py` | READ + WRITE | Reads learnings, updates confidence metadata |
| `scripts/core/memory_metrics.py` | READ | Aggregate statistics (counts, distribution, per-session stats) |
| `scripts/core/track_stale_rate.py` | READ | Counts total vs stale (recall_count=0) learnings |
| `scripts/core/duplicate_density.py` | READ | Queries for duplicate analysis |
| `scripts/core/pattern_batch.py` | READ | `load_learnings()` — loads all active learnings with embeddings for clustering |
| `scripts/core/memory_feedback.py` | READ | Verifies learning exists before storing feedback |
| `scripts/core/memory_daemon.py` | READ | `detected_patterns` last run check |

---

#### `memory_tags`

Structured tag storage for archival_memory entries. Many-to-many relationship.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `memory_id` | UUID | NOT NULL, FK→archival_memory | Learning ID |
| `tag` | TEXT | NOT NULL | Tag string |
| `session_id` | TEXT | NOT NULL | Session that added the tag |
| `created_at` | TIMESTAMPTZ | NOW() | When the tag was added |

**Primary Key:** `(memory_id, tag)`
**Indexes:** tag, session_id, memory_id

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/db/memory_service_pg.py` | WRITE (insert) | `store()` — inserts tags alongside new learnings |
| `scripts/core/db/memory_service_pg.py` | WRITE (insert) | `add_tag()` |
| `scripts/core/db/memory_service_pg.py` | WRITE (delete) | `remove_tag()` |
| `scripts/core/db/memory_service_pg.py` | READ | `get_tags()`, `get_all_session_tags()`, `search_with_tags()` |

---

### Handoffs Layer

#### `handoffs` (PostgreSQL)

Session handoffs/task completions with embeddings for semantic search. Indexed by `artifact_index.py`.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | UUID | `gen_random_uuid()` | Primary key |
| `session_name` | TEXT | NOT NULL | Session name |
| `file_path` | TEXT | UNIQUE, NOT NULL | Path to handoff markdown file |
| `format` | TEXT | 'yaml' | File format |
| `session_id` | TEXT | | Claude session ID |
| `agent_id` | TEXT | | Agent ID |
| `root_span_id` | TEXT | | Braintrust trace ID |
| `jsonl_path` | TEXT | | Path to session JSONL |
| `goal` | TEXT | | Session goal |
| `what_worked` | TEXT | | What went well |
| `what_failed` | TEXT | | What went wrong |
| `key_decisions` | TEXT | | Key decisions made |
| `outcome` | TEXT | | SUCCEEDED/PARTIAL_PLUS/PARTIAL_MINUS/FAILED/UNKNOWN |
| `outcome_notes` | TEXT | | Notes on the outcome |
| `content` | TEXT | | Full handoff content |
| `embedding` | VECTOR(1024) | | For semantic search |
| `created_at` | TIMESTAMPTZ | NOW() | |
| `indexed_at` | TIMESTAMPTZ | NOW() | |

**Indexes:** session_name, session_id, root_span_id, created_at DESC, outcome, goal FTS (GIN), HNSW on embedding

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/artifact_index.py` | WRITE (upsert) | Indexes handoff markdown files into PostgreSQL via `_adapt_for_postgres()` |

---

### OPC Extensions

#### `continuity` (PostgreSQL)

Session state snapshots for the continuity ledger system.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | TEXT | PK | Snapshot ID |
| `session_name` | TEXT | | Session name |
| `goal` | TEXT | | Session goal |
| `state_done` | TEXT | | Completed items |
| `state_now` | TEXT | | Current work |
| `state_next` | TEXT | | Next steps |
| `key_learnings` | TEXT | | Learnings captured |
| `key_decisions` | TEXT | | Decisions made |
| `snapshot_reason` | TEXT | | Why the snapshot was taken |
| `indexed_at` | TIMESTAMP | NOW() | When indexed |

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/artifact_index.py` | WRITE (upsert) | Indexes continuity ledger files |

---

#### `memory_feedback`

Tracks whether recalled learnings were actually useful. Enables feedback loop for recall quality.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | UUID | `gen_random_uuid()` | Primary key |
| `learning_id` | UUID | NOT NULL, FK→archival_memory | Which learning was rated |
| `session_id` | TEXT | NOT NULL | Session that provided feedback |
| `helpful` | BOOLEAN | NOT NULL | Was it helpful? |
| `context` | TEXT | | Why it was/wasn't helpful |
| `source` | TEXT | 'manual' | manual/hook/auto |
| `created_at` | TIMESTAMPTZ | NOW() | |

**Unique constraint:** `(learning_id, session_id)` — one rating per session per learning
**Indexes:** learning_id, session_id, helpful, created_at DESC

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/memory_feedback.py` | WRITE (upsert) | `store_feedback()` — INSERT ON CONFLICT updates |
| `scripts/core/memory_feedback.py` | READ | `get_feedback_for_learning()`, `get_feedback_summary()` |

---

#### `backfill_log`

Tracks S3 transcript extraction attempts. Keyed by S3 UUID so sessions without a DB row still get tracked.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `s3_uuid` | TEXT | PK | S3 file UUID |
| `session_id` | TEXT | | Session ID |
| `project` | TEXT | | Project name |
| `status` | TEXT | NOT NULL | ok/in_progress/failed/timed_out |
| `learnings_stored` | INTEGER | 0 | Number of learnings extracted |
| `file_size_bytes` | BIGINT | | Size of the JSONL file |
| `processed_at` | TIMESTAMPTZ | NOW() | When processed |

**Indexes:** status

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/backfill_learnings.py` | READ | `is_session_extracted()` — checks if UUID already processed |
| `scripts/core/backfill_learnings.py` | WRITE (upsert) | `claim_session()` — atomically claims UUID for extraction |
| `scripts/core/backfill_learnings.py` | WRITE (upsert) | `log_extraction_result()` — records final status |

---

#### `learning_rejections`

Dedup rejection details for extraction diagnostics. Records why a learning was rejected during store.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | SERIAL | PK | Auto-increment |
| `session_id` | TEXT | NOT NULL | Source session |
| `similarity` | REAL | | Cosine similarity with existing |
| `threshold` | REAL | | Dedup threshold used |
| `existing_id` | TEXT | | ID of the existing duplicate |
| `existing_session` | TEXT | | Session of the existing duplicate |
| `project` | TEXT | | Project name |
| `learning_type` | TEXT | | Type of the rejected learning |
| `context` | TEXT | | Context of the rejected learning |
| `tags` | TEXT[] | | Tags of the rejected learning |
| `rejected_at` | TIMESTAMP | NOW() | When rejected |

**Indexes:** session_id

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/store_learning.py` | WRITE (insert) | `_record_rejection()` — logs rejected duplicates |
| `scripts/core/store_learning.py` | READ | `get_rejection_count()` — counts rejections per session |

---

#### `plans` (PostgreSQL)

Indexed implementation plans for cross-session discovery. Defined in `init-schema.sql`.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | TEXT | PK | Plan ID |
| `title` | TEXT | | Plan title |
| `file_path` | TEXT | | Path to plan markdown file |
| `overview` | TEXT | | Plan overview |
| `approach` | TEXT | | Implementation approach |
| `phases` | TEXT | | Phases (JSON) |
| `constraints` | TEXT | | Constraints and limitations |
| `indexed_at` | TIMESTAMP | NOW() | When indexed |

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/artifact_index.py` | WRITE (upsert) | Indexes plan markdown files via `DatabaseConnection` |

---

#### `detected_patterns` (via migration)

Detected cross-session patterns from HDBSCAN clustering. Created by `scripts/migrations/add_detected_patterns.sql`.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | PK |
| `pattern_type` | TEXT | cluster/tag_cooccurrence/fusion |
| `label` | TEXT | Human-readable label |
| `description` | TEXT | Pattern description |
| `tags` | TEXT[] | Associated tags |
| `confidence` | REAL | Detection confidence |
| `member_count` | INTEGER | Number of learnings in pattern |
| `representative_id` | UUID | FK→archival_memory, representative learning |
| `run_id` | TEXT | Batch run identifier |
| `superseded_at` | TIMESTAMPTZ | NULL = active |
| `created_at` | TIMESTAMPTZ | |

**Indexes:** pattern_type, run_id, active (WHERE superseded_at IS NULL)

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/pattern_batch.py` | WRITE | Inserts detected patterns after clustering |
| `scripts/core/pattern_report.py` | READ | Generates human-readable pattern reports |
| `scripts/core/recall_learnings.py` | READ | `enrich_with_pattern_strength()` — joins with pattern_members |
| `scripts/core/push_learnings.py` | READ | Joins to get representative learnings |
| `scripts/core/memory_daemon.py` | READ | Checks last pattern detection run |

---

#### `pattern_members` (via migration)

Maps learnings to detected patterns (many-to-many).

| Column | Type | Description |
|--------|------|-------------|
| `pattern_id` | UUID | FK→detected_patterns |
| `memory_id` | UUID | FK→archival_memory |
| `distance` | REAL | Distance from cluster center |

**Primary Key:** `(pattern_id, memory_id)`
**Indexes:** memory_id

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/pattern_batch.py` | WRITE (insert) | Maps learnings to detected patterns |
| `scripts/core/pattern_report.py` | READ | Joins with detected_patterns for reports |
| `scripts/core/recall_learnings.py` | READ | Joins for pattern strength enrichment |

---

#### `core_memory`

**Not defined in `init-schema.sql`.** Referenced by `memory_service_pg.py` for Letta-compatible key/value core memory blocks. Table must be created manually or by the application. Schema inferred from queries:

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT | Session scope |
| `agent_id` | TEXT | Agent scope (nullable) |
| `key` | TEXT | Block label |
| `value` | TEXT | Block content |
| `updated_at` | TIMESTAMP | Last update |

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/db/memory_service_pg.py` | READ + WRITE | `set_core()`, `get_core()`, `list_core_keys()`, `delete_core()`, `get_all_core()` |

---

#### `findings`

**Not defined in `init-schema.sql`.** Created inline by `db-utils-pg.ts`. Used for cross-session knowledge sharing.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | PK |
| `session_id` | TEXT | Source session |
| `topic` | TEXT | Finding topic |
| `finding` | TEXT | Finding content |
| `relevant_to` | TEXT[] | Array of relevant project/file names |
| `created_at` | TIMESTAMPTZ | |

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `hooks/ts/src/shared/db-utils-pg.ts` | WRITE (insert) | `broadcastFinding()` |
| `hooks/ts/src/shared/db-utils-pg.ts` | READ | `getRelevantFindings()` — ILIKE search |

**Note:** Exported functions exist but no hook currently calls them.

---

## SQLite Tables

### Artifact Index (`context.db`)

Located at `.claude/cache/artifact-index/context.db`. Schema defined in `scripts/core/artifact_schema.sql`.

#### `handoffs` (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT | PK (MD5 hash of file path) |
| `session_name` | TEXT | Session name |
| `task_number` | INTEGER | Task number |
| `file_path` | TEXT | Path to handoff file |
| `task_summary` | TEXT | Summary of task |
| `what_worked` | TEXT | |
| `what_failed` | TEXT | |
| `key_decisions` | TEXT | |
| `files_modified` | TEXT | JSON array |
| `outcome` | TEXT | SUCCEEDED/PARTIAL_PLUS/PARTIAL_MINUS/FAILED/UNKNOWN |
| `outcome_notes` | TEXT | |
| `root_span_id` | TEXT | Braintrust trace ID |
| `turn_span_id` | TEXT | |
| `session_id` | TEXT | |
| `braintrust_session_id` | TEXT | Legacy |
| `created_at` | TIMESTAMP | |
| `indexed_at` | TIMESTAMP | |

**FTS5 index:** `handoffs_fts` on task_summary, what_worked, what_failed, key_decisions, files_modified

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/artifact_index.py` | WRITE | Indexes handoff files |
| `scripts/core/artifact_query.py` | READ | `search_handoffs()`, `get_handoff_by_span_id()` — FTS5 search |
| `scripts/braintrust_analyze.py` | READ | `search_handoffs()` for contextual critique |

---

#### `plans` (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT | PK |
| `session_name` | TEXT | |
| `title` | TEXT | NOT NULL |
| `file_path` | TEXT | NOT NULL |
| `overview` | TEXT | |
| `approach` | TEXT | |
| `phases` | TEXT | JSON array |
| `constraints` | TEXT | |
| `created_at` | TIMESTAMP | |
| `indexed_at` | TIMESTAMP | |

**FTS5 index:** `plans_fts` on title, overview, approach, phases, constraints

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/artifact_index.py` | WRITE | Indexes plan files |
| `scripts/core/artifact_query.py` | READ | `search_plans()` — FTS5 search |

---

#### `continuity` (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT | PK |
| `session_name` | TEXT | NOT NULL |
| `goal` | TEXT | |
| `state_done` | TEXT | JSON array |
| `state_now` | TEXT | |
| `state_next` | TEXT | |
| `key_learnings` | TEXT | |
| `key_decisions` | TEXT | |
| `snapshot_reason` | TEXT | phase_complete/session_end/milestone/manual |
| `created_at` | TIMESTAMP | |

**FTS5 index:** `continuity_fts` on goal, key_learnings, key_decisions, state_now

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/artifact_index.py` | WRITE | Indexes continuity ledger files |
| `scripts/core/artifact_query.py` | READ | `search_continuity()`, `get_ledger_for_session()` |

---

#### `queries` (SQLite)

Compound learning from Q&A — stores past questions and answers for deduplication.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT | PK |
| `question` | TEXT | NOT NULL |
| `answer` | TEXT | NOT NULL |
| `handoffs_matched` | TEXT | JSON array of matched handoff IDs |
| `plans_matched` | TEXT | JSON array of matched plan IDs |
| `continuity_matched` | TEXT | JSON array of matched continuity IDs |
| `braintrust_sessions` | TEXT | |
| `was_helpful` | BOOLEAN | |
| `created_at` | TIMESTAMP | |

**FTS5 index:** `queries_fts` on question, answer

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `scripts/core/artifact_query.py` | READ | `search_past_queries()` |

---

#### `instance_sessions` (SQLite)

Located at `.claude/cache/artifact-index/context.db`. Maps terminal PIDs to session names for handoff correlation.

| Column | Type | Description |
|--------|------|-------------|
| `terminal_pid` | TEXT | PK — process ID |
| `session_name` | TEXT | Session name |
| `updated_at` | TIMESTAMP | |

| Script/Hook | Operation | Details |
|-------------|-----------|---------|
| `hooks/ts/src/handoff-index.ts` | WRITE | `storeSessionAffinity()` — `INSERT OR REPLACE` on handoff file write |
| `hooks/python/session_start_continuity.py` | READ | Queries session name by terminal PID |

---

### Legacy SQLite (`coordination.db`)

Defined in `hooks/ts/src/shared/db-utils.ts`. **Not imported by any active hook.** Contains an `agents` table for swarm coordination.

---

## Tables Referenced in CLAUDE.md but Absent from Schema

| Table | Status | Notes |
|-------|--------|-------|
| `cross_session_patterns` | **Does not exist** | CLAUDE.md references it, but the actual tables are `detected_patterns` + `pattern_members` (via migration) |
| `core_memory` | **Not in init-schema.sql** | Used by `memory_service_pg.py` but must be created separately |
| `findings` | **Not in init-schema.sql** | Created inline by `db-utils-pg.ts` but no hook currently calls the broadcast/query functions |

---

## Migrations

All in `scripts/migrations/`:

| Migration | Tables Affected |
|-----------|----------------|
| `add_temporal_decay.sql` | `archival_memory` (last_recalled, recall_count) |
| `add_learning_chains.sql` | `archival_memory` (superseded_by, superseded_at) |
| `add_memory_tags.sql` | Creates `memory_tags` |
| `backfill_memory_tags.sql` | Backfills `memory_tags` from metadata |
| `add_detected_patterns.sql` | Creates `detected_patterns`, `pattern_members` |
| `add_project_column.sql` | `archival_memory` (project) |
| `add_confidence_calibration.sql` | `archival_memory` (calibration columns) |
| `add_memory_feedback.sql` | Creates `memory_feedback` |
| `add_archival_embedding_hnsw.sql` | `archival_memory` (HNSW index on embedding) |
| `add_learning_rejections.sql` | Creates `learning_rejections` |

---

## Connection Patterns

### Python (asyncpg — async)

Used by: `memory_service_pg.py`, `recall_backends.py`, `memory_feedback.py`, `memory_metrics.py`, `track_stale_rate.py`, `pattern_batch.py`, `recall_learnings.py`, `re_embed_voyage.py`, `confidence_calibrator.py`

```python
from scripts.core.db.postgres_pool import get_pool, get_connection, get_transaction

# Single query
async with get_connection() as conn:
    rows = await conn.fetch("SELECT ...", params)

# Transaction
async with get_transaction() as conn:
    await conn.execute("INSERT ...", params)
```

### Python (psycopg2 — sync)

Used by: `store_learning.py`, `artifact_index.py`, `backfill_learnings.py`, `backfill_sessions.py`, `memory_daemon.py`

```python
import psycopg2
conn = psycopg2.connect(url)
cur = conn.cursor()
cur.execute("INSERT ...", params)
conn.commit()
```

### TypeScript Hooks (subprocess)

Used by: All hooks via `db-utils-pg.ts`

```typescript
// Embeds asyncpg Python as template string, executes via:
// uv run python -c "<inline_python>"
// 5-second timeout per query
```

### SQLite (direct)

Used by: `artifact_index.py`, `artifact_query.py`, `handoff-index.ts` (via `better-sqlite3`)
