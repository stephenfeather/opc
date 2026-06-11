-- Migration: Add project column to archival_memory
-- Enables project-relevance boosting in contextual reranker
--
-- This migration runs against EXISTING (populated) databases. A plain
-- CREATE INDEX takes a SHARE lock that blocks every write to
-- archival_memory (stores, and recall_count UPDATEs from record_recall)
-- for the whole build, stalling the live daemon/hooks (aegis MEDIUM-1).
-- Both index builds therefore use CREATE INDEX CONCURRENTLY, which does
-- not block writes.
--
-- IMPORTANT: CONCURRENTLY cannot run inside a transaction block. Apply
-- this file with `psql -f` (autocommit) -- NOT with `psql -1`/`BEGIN`.
-- If a CONCURRENTLY build is interrupted it can leave an INVALID index;
-- IF NOT EXISTS will then skip the rebuild, so on failure drop and retry:
--   DROP INDEX IF EXISTS idx_archival_project;
--   DROP INDEX IF EXISTS idx_archival_project_lower;
-- then re-run this migration.

ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS project TEXT;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archival_project ON archival_memory (project) WHERE project IS NOT NULL;
-- Case-insensitive scoped-pass index (issue #139): --project-first filters
-- on LOWER(project) = $N. IF NOT EXISTS keeps reruns idempotent.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archival_project_lower ON archival_memory (LOWER(project)) WHERE project IS NOT NULL;
