# OPC Memory Daemon: Missing Code Report

**Date:** 2026-03-29
**Location:** `~/opc/scripts/core/memory_daemon.py`
**Status:** Code-DB mismatch — the daemon code is missing features, but the **database schema and the running daemon have them**

---

## What Happened

Between March 21-28, 2026, at least six Claude Code sessions worked on significant enhancements to `memory_daemon.py`. These sessions are documented in both the OPC memory system (archival_memory table) and in `~/opc/docs/changes-from-upstream.md`. However, the current `memory_daemon.py` on disk contains **none of these changes** — it appears to be at an earlier version, prior to the work.

A session that was adding faulthandler/trace dump functionality appears to have reverted the daemon file, and the revert swept away all the accumulated work. The changes were never recommitted to git. Two artifact files from this era survive as untracked files:

- `~/opc/scripts/core/backfill_archive.py` (untracked, never committed)
- `~/Development/agentic-work/add_faulthandler.py` (untracked, never committed)

### Critical Finding: The Features ARE Working Despite Missing Code

The database tells a different story from the code on disk:

- **Database schema has all columns:** `extraction_status`, `extraction_attempts`, `archived_at`, `archive_path`, `transcript_path`, `claude_session_id`, `exited_at`, `pid`, `host_id` — all 15 columns present
- **S3 archival is actively working:** 214 sessions have been archived to S3 (`s3://claude-session-archive-010093054341/...`), with the most recent on 2026-03-29 (today)
- **Extraction state machine is active:** 3,711 extracted, 2 pending, 0 failed out of 3,713 total sessions
- **The running daemon process has the enhanced code** — it was loaded from a version of the file that had all the features, and hasn't been restarted since the revert

This means: **a running daemon process in memory has the correct code, but the file on disk has been reverted.** When the daemon is next restarted, it will lose all the enhancements and revert to the basic version.

### Sessions Not in Coordination DB

The original development sessions (`memory-daemon-fix`, `daemon-archival-recovery`, etc.) do not appear in the `sessions` table. They predate the coordination DB session registration, or used a different session ID format. Only `s-mn1ixp8r` (2026-03-22) appears, and it was archived to S3.

---

## Evidence Sources

### Memory System Learnings (stored in PostgreSQL `archival_memory`)

| Session ID | Date | Key Content |
|------------|------|-------------|
| `memory-daemon-fix` | 2026-03-21 17:02 | S3 archival integrated into daemon: `archive_session_jsonl()` compresses with zstd, uploads to `s3://claude-session-archive-010093054341/sessions/{project}/{uuid}.jsonl.zst`, deletes local. S3 bucket: `CLAUDE_SESSION_ARCHIVE_BUCKET` env var. |
| `memory-daemon-fix` | 2026-03-21 17:31 | Process management bug: `start_new_session=True` caused orphaned extractions — `os.waitpid()` raises `ChildProcessError` because child is reaped before daemon's 60s poll. Orphaned extractions never get archived to S3. |
| `memory-daemon-fix` | 2026-03-21 17:50 | Query to find lost sessions: `SELECT e.id FROM sessions e LEFT JOIN (SELECT DISTINCT session_id FROM archival_memory) h ON h.session_id = e.id WHERE e.memory_extracted_at IS NOT NULL AND h.session_id IS NULL` |
| `daemon-archival-recovery` | 2026-03-21 17:59 | Added `source_session_id` and `archive_path` traceability to `pg_mark_archived()`. Updates `archival_memory.metadata` with S3 path after archival. |
| `daemon-archival-recovery` | 2026-03-21 19:30 | 4-state extraction machine: `pending` → `extracting` → `extracted` \| `failed`. `extraction_status` column replaces binary `memory_extracted_at IS NULL`. `mark_extracting()` called at queue time, `mark_extracted()` after success, `mark_extraction_failed()` on error with retry up to `MAX_RETRIES=3`. |
| `daemon-archival-recovery-2` | 2026-03-21 19:34 | Consolidation session: (1) traceability stamps, (2) 4-state machine, (3) process reaping fixes, (4) `active_extractions` expanded to `pid → (session_id, jsonl_path, project)`. |
| `s-mn1ixp8r` | 2026-03-22 09:26 | S3 archival broken for s-* sessions: `active_extractions` stored only `session_id`, but `archive_session_jsonl()` called `find_session_jsonl(session_id)` which can't match s-* IDs to UUID-named JSONL files. Fix: pass `jsonl_path` through `active_extractions`. |
| `memory-daemon-jsonl-fix` | 2026-03-23 13:18 | JSONL lookup bug: `find_session_jsonl()` searched for session ID (e.g., "s-mn36n8in") inside filenames (e.g., "6bebf1a2-...jsonl") — two different ID systems that never match. Fix: use `transcript_path` from DB exclusively. |
| `daemon-extraction-fix` | 2026-03-23 13:42 | 11 permanently failed sessions caused by glob fallback matching wrong JSONL files. Root cause chain documented. |
| `daemon-extraction-fix` | 2026-03-23 13:47 | `start_new_session=True` must NOT be used when parent needs `os.waitpid()`. Daemon already a session leader via `os.setsid()` double-fork. Fix: remove `start_new_session=True` from `subprocess.Popen`. |
| `daemon-db-retry` | 2026-03-28 15:31 | `pg_*` functions each created fresh `psycopg2.connect()` with no retry logic. When PostgreSQL enters recovery mode, all DB ops fail immediately — including `pg_mark_archived()` after S3 upload, losing archive metadata. Fix: add retry with backoff. |

### Documentation (`~/opc/docs/changes-from-upstream.md`)

This file (dated 2026-03-23) documents the **intended** state of `memory_daemon.py` compared to the upstream Continuous-Claude-v3. It describes all the features listed above as present. The document itself is committed and accurate — the code it describes is what's missing.

---

## What Is Currently in `memory_daemon.py`

The current file has these functions (verified via grep):

```
log(msg)
get_postgres_url()
use_postgres()
pg_ensure_column()
pg_get_stale_sessions()
pg_mark_extracted(session_id)
get_sqlite_path()
sqlite_ensure_table()
sqlite_get_stale_sessions()
sqlite_mark_extracted(session_id)
ensure_schema()
get_stale_sessions()
mark_extracted(session_id)
extract_memories(session_id, project_dir)
reap_completed_extractions()
process_pending_queue()
queue_or_extract(session_id, project)
daemon_loop()
is_running()
_run_as_daemon()
start_daemon()
stop_daemon()
status_daemon()
main()
```

This is the **pre-enhancement** version. It uses the binary `memory_extracted_at IS NULL` check and has no S3, no state machine, no transcript_path resolution, and no retry logic.

---

## Database Schema (Already Applied — Code Must Match)

The PostgreSQL `sessions` table already has all 15 columns. The daemon code must be updated to **use** them — no schema migrations needed.

```
sessions table columns (verified):
  id, project, working_on, started_at, last_heartbeat,
  memory_extracted_at, claude_session_id, transcript_path,
  exited_at, pid, host_id, archived_at, archive_path,
  extraction_status, extraction_attempts
```

The `archival_memory` table also has `content_hash` and `host_id` columns (per `changes-from-upstream.md`).

### Current DB State (2026-03-29)

| Metric | Value |
|--------|-------|
| Total sessions | 3,713 |
| Extracted | 3,711 |
| Pending | 2 |
| Failed | 0 |
| Archived to S3 | 214 |

### Schema Changes the Code Must Account For

The current daemon code uses `memory_extracted_at IS NULL` as the extraction check. The enhanced version must use `extraction_status = 'pending'` instead. The following columns exist in the DB but are **not referenced by the current code on disk:**

| Column | Used By (missing function) | Purpose |
|--------|---------------------------|---------|
| `extraction_status` | `pg_mark_extracting()`, `pg_mark_extraction_failed()`, `pg_get_stale_sessions()` | 4-state machine |
| `extraction_attempts` | `pg_mark_extracting()`, `pg_mark_extraction_failed()` | Retry counter |
| `archived_at` | `pg_mark_archived()` | S3 archive timestamp |
| `archive_path` | `pg_mark_archived()` | S3 key for archived transcript |
| `transcript_path` | `pg_get_stale_sessions()`, `extract_memories()` | Direct JSONL lookup (replaces glob) |
| `claude_session_id` | session-register hook (not daemon) | Claude's internal UUID |
| `exited_at` | session-clean-exit hook (not daemon) | Clean exit detection |
| `pid` | session-crash-recovery hook (not daemon) | Crash detection via PID liveness |
| `host_id` | `store_learning.py` (not daemon directly) | Multi-machine support |

The SQLite fallback schema must also be updated to mirror these columns, though SQLite is the secondary backend.

---

## What Needs to Be Restored

### 1. Extraction State Machine (Priority: HIGH)

**What:** Replace binary `memory_extracted_at IS NULL` with 4-state `extraction_status` column.

**Functions to add:**
- `pg_mark_extracting(session_id)` — sets `extraction_status='extracting'`, increments `extraction_attempts`
- `pg_mark_extraction_failed(session_id)` — sets status to `'pending'` if under `MAX_RETRIES`, else `'failed'`
- `sqlite_mark_extracting(session_id)` — SQLite mirror
- `sqlite_mark_extraction_failed(session_id)` — SQLite mirror
- `mark_extracting(session_id)` — dispatch wrapper
- `mark_extraction_failed(session_id)` — dispatch wrapper

**Schema changes:**
- Add column `extraction_status TEXT DEFAULT 'pending'` to `sessions`
- Add column `extraction_attempts INTEGER DEFAULT 0` to `sessions`

**Config:**
- `MAX_RETRIES = 3`
- `STALE_THRESHOLD = 900` (was 300)
- `MAX_CONCURRENT_EXTRACTIONS = 4` (was 2)

**Why it matters:** Without this, failed extractions are never retried and sessions stuck in "extracting" are never recovered.

### 2. Transcript Path Resolution (Priority: HIGH)

**What:** Use `transcript_path` from the DB `sessions` table instead of glob-based `find_session_jsonl()`.

**Why it matters:** The glob approach matches wrong JSONL files (session IDs like `s-mn36n8in` never match UUID filenames like `6bebf1a2-...jsonl`). This caused 11 permanently failed extractions and orphaned processes.

**Changes:**
- `pg_get_stale_sessions()` should return `transcript_path` alongside `session_id` and `project`
- `extract_memories()` should accept `transcript_path` parameter
- Remove or bypass `find_session_jsonl()` entirely
- `pending_queue` tuples become `(session_id, project, transcript_path)`
- `active_extractions` dict becomes `pid → (session_id, jsonl_path, project)`

### 3. S3 Archival (Priority: MEDIUM)

**What:** After successful extraction, compress the JSONL with zstd and upload to S3.

**Functions to add:**
- `archive_session_jsonl(session_id, jsonl_path, project)` — compress with zstd, upload via `aws s3 cp`, delete local, call `pg_mark_archived()`
- `pg_mark_archived(session_id, archive_path)` — set `archived_at=NOW()`, `archive_path=<s3_key>` on session; also stamp `archival_memory.metadata` with `source_session_id` and `archive_path`

**Schema changes:**
- Add column `archived_at TIMESTAMP` to `sessions`
- Add column `archive_path TEXT` to `sessions`

**S3 details:**
- Bucket: env var `CLAUDE_SESSION_ARCHIVE_BUCKET`
- Key format: `s3://{bucket}/sessions/{project-name}/{uuid}.jsonl.zst`
- Timeout handling: decompress on upload failure

**Integration point:** Called in `reap_completed_extractions()` when exit code is 0.

### 4. Process Management Fixes (Priority: HIGH)

**What:** Fix subprocess spawning and reaping.

**Changes:**
- Remove `start_new_session=True` from `subprocess.Popen` in `extract_memories()`
- Change from `os.kill(pid, 0)` to `os.waitpid(pid, WNOHANG)` for proper zombie reaping
- Add `ChildProcessError` handling for already-reaped processes
- Check exit code: `exit=0` → extracted + archived; non-zero → `mark_extraction_failed()`

### 5. DB Retry Logic (Priority: MEDIUM)

**What:** Add retry with backoff to all `pg_*` functions.

**Why it matters:** When PostgreSQL enters recovery mode (transient restart), all DB operations fail immediately. `pg_mark_archived()` runs *after* S3 upload — if it fails, the archive metadata is lost even though the file was uploaded.

**Implementation:** Wrap `psycopg2.connect()` calls with retry logic (e.g., 3 attempts with exponential backoff).

### 6. Missing Utility Scripts (Priority: LOW)

These files were developed but never committed:

| File | Status | Purpose |
|------|--------|---------|
| `scripts/core/backfill_archive.py` | Exists untracked in `~/opc` | One-time S3 archival of existing JSONL files |
| `scripts/core/backfill_sessions.py` | **Missing entirely** | Backfill session metadata from transcripts |
| `scripts/core/re_embed_voyage.py` | **Missing entirely** | Re-embed learnings with Voyage provider |
| `add_faulthandler.py` | Exists untracked in `~/Development/agentic-work` | Script to add faulthandler crash tracing |

### 7. Other Changes (Priority: LOW)

- `extract_memories()` should return `bool`
- `get_jsonl_client()` — read entrypoint/version from JSONL for logging
- `CLAUDE_MEMORY_EXTRACTION=1` env var set on extraction subprocesses (prevents session-register hook from registering extraction sessions)
- `daemon_loop()` should not call `mark_extracted()` at queue time (premature)
- `status` command should show extraction status counts

---

## Reconstruction Approach

### Status: Enhanced Daemon Already Stopped

The daemon process that had the enhanced code in memory has been stopped. A new daemon started from the current file on disk will be running the **stripped-down version** without S3 archival, state machine, retry logic, or transcript resolution. Any sessions that end now will be extracted but NOT archived to S3.

### Option A: Recover from S3 Session Transcripts

Session `s-mn1ixp8r` (2026-03-22) was archived to S3:
`s3://claude-session-archive-010093054341/sessions/-Users-stephenfeather-Development-agentic-work/8a418207-20e2-4a44-9240-471c4d8ae402.jsonl.zst`

Other relevant sessions may also be archived. Download, decompress, and search the JSONL transcripts for Write/Edit tool calls targeting `memory_daemon.py`.

```bash
aws s3 cp s3://claude-session-archive-010093054341/sessions/-Users-stephenfeather-Development-agentic-work/8a418207-20e2-4a44-9240-471c4d8ae402.jsonl.zst /tmp/
zstd -d /tmp/8a418207-20e2-4a44-9240-471c4d8ae402.jsonl.zst
# Search for daemon code writes
rg "memory_daemon" /tmp/8a418207-20e2-4a44-9240-471c4d8ae402.jsonl
```

The later sessions (`daemon-extraction-fix`, `daemon-db-retry`) may still have local JSONL files if they weren't archived or were in the OPC project directory.

**Pros:** Recovers exact code as written
**Cons:** Code evolved across 6+ sessions; need to find the final version

### Option C: Rebuild from Memory + Documentation

Use the memory learnings (11 entries), `changes-from-upstream.md`, and the existing `backfill_archive.py` (which contains working S3 code) as references to rebuild the missing functions into the current `memory_daemon.py`.

**Pros:** Clean implementation, can fix all known bugs from the start, comprehensive documentation exists
**Cons:** Time-intensive, may miss edge cases from the original implementation

### Option D: Hybrid (Recommended)

1. Search for local JSONL transcripts from the OPC project directory for sessions `daemon-extraction-fix` and `daemon-db-retry` (the most recent)
2. Check S3 for archived versions of those sessions
3. If found, extract the final Write/Edit calls to `memory_daemon.py`
4. If not found, rebuild from memory + documentation + `backfill_archive.py` reference code
5. Validate the restored code by comparing DB behavior (214 archived sessions prove the S3 code works)

---

## Embedding Issues (Separate from Daemon Code Loss)

### Current Embedding State (verified 2026-03-29)

| Metric | Value |
|--------|-------|
| Total learnings | 2,195 |
| With embeddings | 2,195 (100%) |
| Embedding dimensions | 1024 (all consistent) |
| Tagged `voyage-code-3` | 164 |
| No embedding_model tag | 336 |
| Unsampled | ~1,695 |

### Environment Variables

```
EMBEDDING_PROVIDER=voyage
VOYAGE_EMBEDDING_MODEL=voyage-code-3
VOYAGE_API_KEY=pa-Vx5C... (set)
```

### Bug 1: `VOYAGE_EMBEDDING_MODEL` Env Var Is Ignored

**File:** `~/opc/scripts/core/db/embedding_service.py:609`

```python
elif provider == "voyage":
    voyage_model = model if model is not None else "voyage-3"  # ← hardcoded default
```

Both `store_learning.py:127` and `recall_learnings.py:272` call:
```python
EmbeddingService(provider=os.getenv("EMBEDDING_PROVIDER", "local"))
```

Neither passes a `model` parameter. So the `VOYAGE_EMBEDDING_MODEL` env var is **never read** by the embedding service. All embeddings — both storage and retrieval — use `voyage-3`, not `voyage-code-3`.

**Impact:** Storage and retrieval are **consistent with each other** (both use `voyage-3`), so search works. But the system is not using the model specified in the environment. The 164 learnings tagged `voyage-code-3` in metadata were likely embedded during the `re_embed_voyage.py` session that no longer exists — if those were actually embedded with `voyage-code-3`, they would be in a different embedding space from the `voyage-3` ones, causing similarity scores to be meaningless for those 164 records.

**Fix:** Either:
- (a) Read `VOYAGE_EMBEDDING_MODEL` in `store_learning.py` and `recall_learnings.py` and pass it as `model=` to `EmbeddingService`, or
- (b) Change the `EmbeddingService` default from `"voyage-3"` to `os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3")`, or
- (c) Accept `voyage-3` as the actual model and update the env var to match reality

**Memory reference:** Session `e7449240` (2026-03-21 16:01) — "VoyageEmbeddingProvider hardcodes 'voyage-3' for query embeddings regardless of VOYAGE_EMBEDDING_MODEL env var. Documents were re-embedded with voyage-code-3 but queries ran with voyage-3 — different embedding spaces making vector similarity scores meaningless."

### Bug 2: `input_type` Hardcoded to `"document"` for All Calls

**File:** `~/opc/scripts/core/db/embedding_service.py:304`

```python
json={
    "model": self.model,
    "input": texts,
    "input_type": "document",  # ← hardcoded for all calls
},
```

Voyage AI recommends using `input_type: "query"` for search queries and `"document"` for stored documents. The current code uses `"document"` for both storage and retrieval.

**Impact:** Slightly degraded search quality. Voyage optimizes embeddings differently for queries vs documents — queries get asymmetric treatment to improve retrieval. Using `"document"` for queries means the model doesn't apply this optimization.

**Fix:** Add an `input_type` parameter to `embed()` and `_call_api()`. `store_learning.py` should pass `input_type="document"`, `recall_learnings.py` should pass `input_type="query"`.

**Memory reference:** Session `e7449240-6f59-4a8f-8bad-968188139660` (2026-03-21 18:39) — "VoyageEmbeddingProvider hardcodes 'document' input_type for all embeddings."

### Possible Embedding Space Contamination

If the 164 `voyage-code-3`-tagged learnings were genuinely embedded with `voyage-code-3`, they exist in a **different vector space** from the ~2,031 `voyage-3` embeddings. Cosine similarity between vectors from different models is meaningless — these 164 records would never match queries correctly and would pollute hybrid search results.

**Diagnostic query:**
```sql
-- Check if voyage-code-3 embeddings cluster differently
SELECT metadata->>'embedding_model' as model,
       AVG(embedding <=> (SELECT embedding FROM archival_memory WHERE embedding IS NOT NULL LIMIT 1)) as avg_dist
FROM archival_memory
WHERE embedding IS NOT NULL
GROUP BY metadata->>'embedding_model';
```

**Fix if contaminated:** Re-embed the 164 `voyage-code-3` records with `voyage-3` (or re-embed everything with `voyage-code-3` and update all code to use it consistently).

---

## References

| Source | Location |
|--------|----------|
| Current daemon code | `~/opc/scripts/core/memory_daemon.py` |
| Feature documentation | `~/opc/docs/changes-from-upstream.md` |
| S3 archival reference code | `~/opc/scripts/core/backfill_archive.py` (untracked) |
| Faulthandler script | `~/Development/agentic-work/add_faulthandler.py` (untracked) |
| Memory learnings | PostgreSQL `archival_memory` table (query via `mcp__opc-memory__recall_learnings`) |
| Session transcripts | `~/.claude/projects/` (if not yet archived/deleted) |
