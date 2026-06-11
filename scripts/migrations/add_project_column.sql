-- Migration: Add project column to archival_memory
-- Enables project-relevance boosting in contextual reranker

ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS project TEXT;
CREATE INDEX IF NOT EXISTS idx_archival_project ON archival_memory (project) WHERE project IS NOT NULL;
-- Case-insensitive scoped-pass index (issue #139): --project-first filters
-- on LOWER(project) = $N. IF NOT EXISTS keeps reruns idempotent.
CREATE INDEX IF NOT EXISTS idx_archival_project_lower ON archival_memory (LOWER(project)) WHERE project IS NOT NULL;
