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
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid() defaults used across multiple tables in this schema

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
    extraction_attempts INTEGER DEFAULT 0,
    last_error TEXT
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

    -- OPC: Push tracking (separate from recall to avoid metric pollution)
    push_count INTEGER NOT NULL DEFAULT 0,
    last_pushed_at TIMESTAMPTZ,

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
-- Functional index for the case-insensitive --project-first scoped pass
-- (issue #139): recall filters on LOWER(project) = $N, which only uses an
-- index whose expression matches. Keeps the scoped pass off a seq scan.
CREATE INDEX IF NOT EXISTS idx_archival_project_lower ON archival_memory (LOWER(project)) WHERE project IS NOT NULL;
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

-- Recall Log: Append-only per-recall-event log (issue #140). One row per
-- recall event with parallel arrays of the recalled rows' ids and projects,
-- so cross-project mis-scope frequency (issue #130) is measurable. Zero-result
-- recalls are logged too (empty arrays, result_count = 0) -- finding nothing
-- signals over-restrictive scoping. Never stores raw query text (privacy).
-- Retention: automated by the memory daemon (issue #146) -- it prunes rows
-- older than daemon.recall_log_retention_days (default 90) every
-- daemon.recall_log_prune_interval_hours (default 24) in bounded batches via
-- prune_recall_log(); set retention to 0 to disable. Equivalent manual prune:
-- `DELETE FROM recall_log WHERE created_at < NOW() - INTERVAL '90 days'`.
-- Analysis queries (time-scoped, e.g. INTERVAL '30 days', using the
-- created_at DESC index) live in scripts/migrations/add_recall_log.sql.
CREATE TABLE IF NOT EXISTS recall_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    caller_project TEXT,                 -- canonicalized; NULL = no project context
    recalled_ids UUID[] NOT NULL,
    recalled_projects TEXT[] NOT NULL,   -- NULL elements = unattributed memories
    result_count INTEGER NOT NULL,
    -- source is validated at the writer (record_recall) against the regex
    -- ^[a-z][a-z0-9_-]{0,31}$, NOT with a DB CHECK constraint: a CHECK
    -- violation would abort the whole INSERT and silently drop the entire
    -- append-only log row (losing the recall event). Invalid labels are
    -- dropped to NULL in Python instead, so the event is still recorded.
    source TEXT,                         -- short caller label (hook|mcp|cli); NULL = unknown
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_recall_log_created ON recall_log(created_at DESC);

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

-- =============================================================================
-- DOCUMENT-COLLECTION RAG LAYER
-- =============================================================================

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size_bytes BIGINT,
    page_count INTEGER,
    extraction_status TEXT NOT NULL DEFAULT 'extracted'
        CHECK (extraction_status IN
            ('extracted', 'skipped_unsupported', 'skipped_needs_ocr', 'error')),
    error TEXT,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (collection_name, file_path)
);

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_name);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(extraction_status);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_name TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('global', 'restricted')),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_collection ON document_chunks(collection_name);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_scope ON document_chunks(scope);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_content_fts ON document_chunks
    USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding_hnsw ON document_chunks
    USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- KNOWLEDGE GRAPH LAYER
-- =============================================================================

-- Knowledge Graph: Entities (nodes)
CREATE TABLE IF NOT EXISTS kg_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,                    -- canonical name (lowercase, normalized)
    display_name TEXT NOT NULL,            -- original casing for display
    entity_type TEXT NOT NULL,             -- file, module, tool, concept, etc.
    metadata JSONB DEFAULT '{}',
    embedding vector(1024),                -- optional: entity-level embedding
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mention_count INTEGER NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_entity_unique
    ON kg_entities(name, entity_type);
CREATE INDEX IF NOT EXISTS idx_kg_entity_type
    ON kg_entities(entity_type);
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        CREATE INDEX IF NOT EXISTS idx_kg_entity_name_trgm
            ON kg_entities USING gin(name gin_trgm_ops);
    ELSE
        RAISE NOTICE 'pg_trgm not available — skipping trigram index on kg_entities.name';
    END IF;
END
$$;

-- Knowledge Graph: Relationships (edges)
CREATE TABLE IF NOT EXISTS kg_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight FLOAT NOT NULL DEFAULT 1.0,
    memory_id UUID NOT NULL REFERENCES archival_memory(id) ON DELETE CASCADE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_edge_unique
    ON kg_edges(source_id, target_id, relation, memory_id);
CREATE INDEX IF NOT EXISTS idx_kg_edge_source ON kg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edge_target ON kg_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_kg_edge_relation ON kg_edges(relation);

-- Knowledge Graph: Entity-to-Learning mentions (join table)
CREATE TABLE IF NOT EXISTS kg_entity_mentions (
    entity_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES archival_memory(id) ON DELETE CASCADE,
    mention_type TEXT NOT NULL DEFAULT 'explicit',
    span_start INTEGER,
    span_end INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entity_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_mentions_memory
    ON kg_entity_mentions(memory_id);
CREATE INDEX IF NOT EXISTS idx_kg_mentions_entity
    ON kg_entity_mentions(entity_id);
