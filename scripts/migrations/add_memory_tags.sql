-- Migration: Create memory_tags table for tag-based filtering
-- Date: 2026-03-30
-- Purpose: Enable structured tag storage for archival_memory entries.
--          Tags were previously stored only in metadata JSON; this table
--          enables efficient tag-based lookups and filtering in recall queries.
--          The code in memory_service_pg.py already references this table
--          (INSERT, SELECT, DELETE) -- this migration creates the backing table.

CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id UUID NOT NULL REFERENCES archival_memory(id) ON DELETE CASCADE,
    tag       TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (memory_id, tag)
);

-- Index on tag for fast tag-based lookups (e.g., WHERE tag = ANY($1))
CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag);

-- Index on session_id for session-scoped tag queries
CREATE INDEX IF NOT EXISTS idx_memory_tags_session ON memory_tags(session_id);

-- Index on memory_id for FK lookups and joins
CREATE INDEX IF NOT EXISTS idx_memory_tags_memory_id ON memory_tags(memory_id);
