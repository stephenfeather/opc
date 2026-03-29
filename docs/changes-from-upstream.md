# Changes from Upstream (Continuous-Claude-v3)

Compared: upstream `/tmp/Continuous-Claude-v3` vs local `~/.claude` and `~/opc`

Date: 2026-03-23

---

## 1. Hook Scripts (`~/.claude/hooks/src/`)

### Modified Files

#### `session-register.ts`
- Skips registration for daemon-spawned extraction sessions (`CLAUDE_MEMORY_EXTRACTION` env check)
- Passes `claude_session_id`, `transcript_path`, and `process.ppid` to `registerSession()` for crash recovery traceability

#### `transcript-parser.ts`
- Rewrote JSONL parsing to handle Claude Code's actual transcript format (`{ type: "assistant", message: { role, content: ContentBlock[] } }`)
- Added `TranscriptLine` and `ContentBlock` interfaces for structured parsing
- Extracts tool calls from `tool_use` content blocks and errors from `tool_result` blocks with `is_error` flag
- Retained legacy flat-format fallback for older transcripts
- Added `TaskCreate` alongside `TodoWrite` for task state capture
- Added `NotebookEdit` alongside `Edit`/`Write` for modified file tracking

#### `smart-search-router.ts`
- Changed search routing from `permissionDecision: 'deny'` to `permissionDecision: 'allow'` with `systemMessage` — routes via system message injection instead of blocking

#### `tldr-read-enforcer.ts`
- Changed banner text from "TLDR Enhanced Read" to "TLDR Context"

### New Files (not in upstream)

| File | Purpose |
|------|---------|
| `session-clean-exit.ts` | Marks sessions as cleanly exited in DB on SessionEnd |
| `session-crash-recovery.ts` | Detects crashed sessions on startup via PID liveness + DB state |
| `phpunit-runner.ts` | PHPUnit test runner hook |
| `pytest-runner.ts` | Pytest test runner hook |
| `tldr-search-sanitizer.ts` | Sanitizes TLDR search queries |

### Shared Modules (`hooks/src/shared/`)

#### `db-utils-pg.ts`
- `registerSession()` expanded: accepts `claudeSessionId`, `transcriptPath`, `pid` parameters
- Schema migration: adds `claude_session_id`, `transcript_path`, `exited_at`, `pid` columns to sessions table
- Upsert now clears `exited_at` on re-register (resume support)
- New functions added:
  - `markSessionExited(claudeSessionId)` — marks clean exit
  - `getCrashedSessions(project)` — finds sessions with `exited_at IS NULL`
  - `markSessionsAcknowledged(sessionIds)` — bulk-marks crashed sessions as acknowledged
- New `CrashedSession` interface exported

#### `opc-path.ts`
- Added `~/.claude/opc.json` config file lookup (priority 2, between env var and project-relative)
- New `getOpcDirFromConfig()` helper reads `{ "opc_dir": "/path" }` from config file

#### `types.ts`
- Added `SessionStartInput` interface with `session_id`, `transcript_path`, and `type` fields

---

## 2. OPC Scripts (`~/opc/scripts/`)

### `scripts/core/memory_daemon.py` (major rewrite)

#### Configuration Changes
| Setting | Upstream | Local |
|---------|----------|-------|
| `STALE_THRESHOLD` | 300s (5 min) | 900s (15 min) |
| `MAX_CONCURRENT_EXTRACTIONS` | 2 | 4 |
| `MAX_RETRIES` | (not present) | 3 |

#### Extraction State Machine (new)
Added 4-state extraction lifecycle: `pending` -> `extracting` -> `extracted` | `failed`

New columns managed: `extraction_status`, `extraction_attempts`, `archived_at`, `archive_path`

New functions (PostgreSQL + SQLite mirrors):
- `pg_mark_extracting()` / `sqlite_mark_extracting()` — marks session as actively extracting, increments attempt counter
- `pg_mark_extraction_failed()` / `sqlite_mark_extraction_failed()` — retry if under MAX_RETRIES, else permanent fail
- `mark_extracting()` / `mark_extraction_failed()` — dispatch wrappers

#### JSONL File Resolution 
- **Upstream**: Glob-based search with "most recent modified in last 10 minutes" fallback
- **Local**: Uses `transcript_path` from DB exclusively; no glob fallback
- Removed `find_session_jsonl()` function entirely (caused wrong-file matching and orphaned extractions)

#### Process Management 
- Removed `start_new_session=True` from `subprocess.Popen` — was breaking `os.waitpid()` parent-child relationship
- Changed from `os.kill(pid, 0)` to `os.waitpid(pid, WNOHANG)` for proper zombie reaping
- Added `ChildProcessError` handling for orphaned processes
- Exit code now checked: `exit=0` -> extracted + archived; non-zero -> mark failed

#### S3 Archival 
- `archive_session_jsonl()` — compresses with zstd, uploads to S3, deletes local copy
- `pg_mark_archived()` — stamps `archived_at` + `archive_path` on session, stamps learnings with source traceability
- Timeout handling with automatic decompression on upload failure

#### Other Changes
- `active_extractions` dict expanded: `pid -> (session_id, jsonl_path, project)` (was `pid -> session_id`)
- `pending_queue` tuples expanded: `(session_id, project, transcript_path)` (was `(session_id, project)`)
- `extract_memories()` now accepts `transcript_path` parameter, returns `bool`
- `get_jsonl_client()` — reads entrypoint/version from JSONL for logging
- `CLAUDE_MEMORY_EXTRACTION=1` env var set on extraction subprocesses
- `daemon_loop()` no longer calls `mark_extracted()` at queue time (was premature)
- `status` command shows extraction status counts
- SQLite schema migration adds `transcript_path` column

### `scripts/core/recall_learnings.py`
- Embedding provider default changed from hardcoded `"local"` to `os.getenv("EMBEDDING_PROVIDER", "local")`

### `scripts/core/store_learning.py`
- Added `content_hash` (SHA-256) for deduplication
- Added `host_id` parameter for multi-system support
- Added `embedding_model` metadata field
- New CLI arg: `--host-id`

### `scripts/core/artifact_index.py`
- YAML frontmatter splitting: changed from line-anchored regex to simple `split("---", 2)`
- Removed filter that skipped files not in `handoffs` directories

### `scripts/core/db/memory_service_pg.py`
- `store_memory()` accepts `content_hash` and `host_id` parameters
- INSERT uses `ON CONFLICT (content_hash) DO NOTHING` for dedup
- Both embedding and no-embedding paths updated

### `scripts/core/db/embedding_service.py`
- Voyage model default changed from hardcoded `"voyage-3"` to `os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3")`

### New Files (not in upstream)

| File | Purpose |
|------|---------|
| `scripts/core/backfill_archive.py` | Backfill S3 archival for existing sessions |
| `scripts/core/backfill_sessions.py` | Backfill session metadata from transcripts |
| `scripts/core/re_embed_voyage.py` | Re-embed learnings with Voyage provider |

---

## 3. Database Schema Changes

### `sessions` table — new columns

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `claude_session_id` | text | | Claude's internal session UUID |
| `transcript_path` | text | | Path to JSONL transcript file |
| `exited_at` | timestamp | | Clean exit timestamp (NULL = possibly crashed) |
| `pid` | integer | | Claude CLI process PID for crash detection |
| `host_id` | text | | Machine identifier for multi-system |
| `archived_at` | timestamp | | When JSONL was archived to S3 |
| `archive_path` | text | | S3 key for archived transcript |
| `extraction_status` | text | `'pending'` | State machine: pending/extracting/extracted/failed |
| `extraction_attempts` | integer | `0` | Retry counter for extraction |

New index: `idx_sessions_host` on `(host_id)`

### `archival_memory` table — new columns

| Column | Type | Purpose |
|--------|------|---------|
| `content_hash` | text | SHA-256 dedup (UNIQUE index) |
| `host_id` | text | Machine identifier |

New index: `idx_archival_content_hash` UNIQUE on `(content_hash)`

Existing index added: `idx_archival_host` on `(host_id)`

---

## Summary

The changes fall into four major themes:

1. **Crash recovery** — session-register now tracks PID + transcript path; new hooks detect crashes and mark clean exits
2. **Extraction reliability** — 4-state machine with retries, proper process reaping, transcript_path-based file resolution (no glob guessing)
3. **S3 archival** — automatic compress + upload of session transcripts after extraction
4. **Multi-system / dedup** — content_hash dedup on learnings, host_id tracking, configurable embedding providers
