-- Migration: Add session_uuid column to handoffs table
-- Run: docker exec -i continuous-claude-postgres psql -U claude -d continuous_claude < docker/migrate-add-session-uuid.sql

ALTER TABLE handoffs ADD COLUMN IF NOT EXISTS session_uuid TEXT;
CREATE INDEX IF NOT EXISTS idx_handoffs_session_uuid ON handoffs(session_uuid);
