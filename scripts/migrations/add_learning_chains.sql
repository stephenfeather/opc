-- Migration: Add learning chain columns to archival_memory
-- Date: 2026-03-30
-- Purpose: Enable learning chains where newer learnings supersede older ones.
--          Old learnings are marked with superseded_by pointing to the new one,
--          so recall filtering is trivial: WHERE superseded_by IS NULL.

-- Add columns (idempotent - safe to run multiple times)
ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS superseded_by UUID REFERENCES archival_memory(id);
ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ;

-- Index for looking up superseded learnings (e.g., chain traversal)
CREATE INDEX IF NOT EXISTS idx_archival_superseded ON archival_memory(superseded_by) WHERE superseded_by IS NOT NULL;

-- Index for filtering active (non-superseded) learnings in recall queries
CREATE INDEX IF NOT EXISTS idx_archival_active ON archival_memory(superseded_by) WHERE superseded_by IS NULL;
