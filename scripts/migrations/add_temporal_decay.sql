-- Migration: Add temporal decay columns to archival_memory
-- Date: 2026-03-30
-- Purpose: Track when learnings are recalled and how often,
--          enabling decay-aware ranking in recall queries.

-- Add columns (idempotent - safe to run multiple times)
ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS last_recalled TIMESTAMPTZ;
ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS recall_count INTEGER NOT NULL DEFAULT 0;

-- Index for decay-aware queries (sort by last_recalled, filter by recall_count)
CREATE INDEX IF NOT EXISTS idx_archival_last_recalled ON archival_memory(last_recalled DESC NULLS LAST);
