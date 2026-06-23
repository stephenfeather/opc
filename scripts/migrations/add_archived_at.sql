-- Migration: Add archived_at lifecycle column to archival_memory
-- Issue #63 Phase 2b Step 3 (stale-archive)
--
-- A STALE learning has no survivor, so superseded_by (which references a keeper)
-- cannot mark it retired. archived_at is a first-class lifecycle column:
--   archived_at IS NULL      -> active
--   archived_at IS NOT NULL  -> archived (stale-retired); the row stays for undo
-- Recall/active surfaces filter `AND archived_at IS NULL` IN ADDITION to
-- `superseded_by IS NULL`. The column is nullable with NO default so existing
-- rows default to active (NULL) without rewriting the table.
--
-- This migration runs against EXISTING (populated) databases. A plain
-- CREATE INDEX takes a SHARE lock that blocks every write to archival_memory
-- for the whole build (aegis MEDIUM-1), so the index uses CREATE INDEX
-- CONCURRENTLY, which does not block writes. Mirrors add_project_column.sql.
--
-- IMPORTANT: CONCURRENTLY cannot run inside a transaction block. Apply this
-- file with `psql -f` (autocommit) -- NOT with `psql -1`/`BEGIN`. If a
-- CONCURRENTLY build is interrupted it can leave an INVALID index; IF NOT
-- EXISTS will then skip the rebuild, so on failure drop and retry:
--   DROP INDEX IF EXISTS idx_archival_not_archived;
-- then re-run this migration.

ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;

-- Partial index mirroring idx_archival_active: keeps the active-row filter
-- (archived_at IS NULL) index-backed on a large corpus.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archival_not_archived
    ON archival_memory (archived_at) WHERE archived_at IS NULL;
