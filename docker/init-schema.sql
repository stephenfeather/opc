-- OPC (Opinionated Persistent Context) Database Schema
--
-- Based on Continuous-Claude-v3 with OPC-specific extensions for:
--   - Memory daemon extraction pipeline (sessions columns)
--   - Embedding provenance and deduplication (archival_memory columns)
--   - Session continuity snapshots (continuity table)
--   - Implementation plan indexing (plans table)
--   - Dedup rejection tracking (learning_rejections table)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =============================================================================
-- COORDINATION LAYER
-- =============================================================================

-- Sessions: Cross-terminal awareness
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    working_on TEXT,
    started_at TIMESTAMP DEFAULT NOW(),
    last_heartbeat TIMESTAMP DEFAULT NOW(),

    -- OPC: Memory daemon extraction pipeline
    memory_extracted_at TIMESTAMP,
    claude_session_id TEXT,
    transcript_path TEXT,
    exited_at TIMESTAMP,
    pid INTEGER,
    host_id TEXT,
    archived_at TIMESTAMP,
    archive_path TEXT,
    extraction_status TEXT DEFAULT 'pending',
    extraction_attempts INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_host ON sessions(host_id);

-- File Claims: Cross-terminal file locking
CREATE TABLE IF NOT EXISTS file_claims (
    file_path TEXT NOT NULL,
    project TEXT NOT NULL,
    session_id TEXT,
    claimed_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (file_path, project)
);

-- =============================================================================
-- MEMORY LAYER
-- =============================================================================

-- Archival Memory: Long-term learnings with embeddings
CREATE TABLE IF NOT EXISTS archival_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    agent_id TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- OPC: Embedding provenance and deduplication
    embedding_model TEXT DEFAULT 'bge',
    host_id TEXT,
    content_hash TEXT,

    -- OPC: Temporal decay tracking
    last_recalled TIMESTAMPTZ,
    recall_count INTEGER NOT NULL DEFAULT 0,

    -- OPC: Learning chains (superseded learnings)
    superseded_by UUID REFERENCES archival_memory(id),
    superseded_at TIMESTAMPTZ,

    -- OPC: Project relevance for contextual reranking
    project TEXT
);

CREATE INDEX IF NOT EXISTS idx_archival_session ON archival_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_archival_agent ON archival_memory(session_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_archival_created ON archival_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_archival_content_fts ON archival_memory
    USING gin(to_tsvector('english', content));
CREATE UNIQUE INDEX IF NOT EXISTS idx_archival_content_hash ON archival_memory(content_hash);
CREATE INDEX IF NOT EXISTS idx_archival_host ON archival_memory(host_id);
CREATE INDEX IF NOT EXISTS idx_archival_last_recalled ON archival_memory(last_recalled DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_archival_superseded ON archival_memory(superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_archival_active ON archival_memory(superseded_by) WHERE superseded_by IS NULL;
CREATE INDEX IF NOT EXISTS idx_archival_project ON archival_memory(project) WHERE project IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_archival_embedding_hnsw ON archival_memory
    USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Memory Tags: Structured tag storage for archival_memory entries
CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id UUID NOT NULL REFERENCES archival_memory(id) ON DELETE CASCADE,
    tag       TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (memory_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag);
CREATE INDEX IF NOT EXISTS idx_memory_tags_session ON memory_tags(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_tags_memory_id ON memory_tags(memory_id);

-- =============================================================================
-- HANDOFFS LAYER
-- =============================================================================

-- Handoffs: Session handoffs/task completions with embeddings for semantic search
CREATE TABLE IF NOT EXISTS handoffs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_name TEXT NOT NULL,
    session_uuid TEXT,
    file_path TEXT UNIQUE NOT NULL,
    format TEXT DEFAULT 'yaml',
    session_id TEXT,
    agent_id TEXT,
    root_span_id TEXT,
    jsonl_path TEXT,
    goal TEXT,
    what_worked TEXT,
    what_failed TEXT,
    key_decisions TEXT,
    outcome TEXT CHECK(outcome IN ('SUCCEEDED','PARTIAL_PLUS','PARTIAL_MINUS','FAILED','UNKNOWN')),
    outcome_notes TEXT,
    content TEXT,
    embedding VECTOR(1024),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    indexed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_handoffs_session ON handoffs(session_name);
CREATE INDEX IF NOT EXISTS idx_handoffs_session_uuid ON handoffs(session_uuid);
CREATE INDEX IF NOT EXISTS idx_handoffs_session_id ON handoffs(session_id);
CREATE INDEX IF NOT EXISTS idx_handoffs_root_span ON handoffs(root_span_id);
CREATE INDEX IF NOT EXISTS idx_handoffs_created ON handoffs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_handoffs_outcome ON handoffs(outcome);
CREATE INDEX IF NOT EXISTS idx_handoffs_goal_fts ON handoffs
    USING gin(to_tsvector('english', COALESCE(goal, '')));
CREATE INDEX IF NOT EXISTS idx_handoffs_embedding_hnsw ON handoffs
    USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- OPC EXTENSIONS
-- =============================================================================

-- Continuity: Session state snapshots for the continuity ledger system
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

-- Memory Feedback: Track whether recalled learnings were actually useful
CREATE TABLE IF NOT EXISTS memory_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learning_id UUID NOT NULL REFERENCES archival_memory(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    helpful BOOLEAN NOT NULL,
    context TEXT,
    source TEXT NOT NULL DEFAULT 'manual'
        CHECK (source IN ('manual', 'hook', 'auto')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_learning ON memory_feedback(learning_id);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON memory_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_helpful ON memory_feedback(helpful);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON memory_feedback(created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_unique_per_session
    ON memory_feedback(learning_id, session_id);

-- Backfill Log: Track extraction attempts by S3 UUID so sessions without
-- a DB row in `sessions` still get marked and aren't re-processed.
CREATE TABLE IF NOT EXISTS backfill_log (
    s3_uuid TEXT PRIMARY KEY,
    session_id TEXT,
    project TEXT,
    status TEXT NOT NULL,
    learnings_stored INTEGER DEFAULT 0,
    file_size_bytes BIGINT,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backfill_log_status ON backfill_log(status);

-- Learning Rejections: Dedup rejection details for extraction diagnostics
CREATE TABLE IF NOT EXISTS learning_rejections (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    similarity REAL,
    threshold REAL,
    existing_id TEXT,
    existing_session TEXT,
    project TEXT,
    learning_type TEXT,
    context TEXT,
    tags TEXT[],
    rejected_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_learning_rejections_session
    ON learning_rejections (session_id);

-- Plans: Indexed implementation plans for cross-session discovery
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
