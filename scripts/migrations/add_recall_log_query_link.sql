-- Migration: link recall_log events to the originating query (recall-tuning
-- feedback loop). Adds the columns needed to join a memory_feedback row
-- (helpful/unhelpful on a recalled learning) back to the query that surfaced it,
-- so the feedback stream can grow the golden eval set.
--
-- Columns:
--   session_id  TEXT  -- sessions PK / Claude session id; the JOIN key to
--                        memory_feedback.session_id. Always recorded when the
--                        caller supplies it (the memory-awareness hook does).
--   query_hash  TEXT  -- canonical SHA-256 of the query (same digest function as
--                        archival_memory.content_hash). One-way, so it groups
--                        identical queries WITHOUT storing prompt text. Always
--                        safe to record.
--   query_text  TEXT  -- raw query text. Populated ONLY when the operator sets
--                        recall.log_query_text = true. DEFAULT is NULL, which
--                        preserves the deliberate "recall_log NEVER stores raw
--                        query text" privacy posture of add_recall_log.sql
--                        (issue #139/#140). Required to materialize a
--                        human-readable golden query from mined feedback.
--
-- Apply with `psql -f` (autocommit). ADD COLUMN IF NOT EXISTS is metadata-only
-- (no table rewrite, no row locks) and idempotent. The session_id index is a
-- plain CREATE INDEX IF NOT EXISTS; on an already-populated recall_log consider
-- CONCURRENTLY to avoid a write lock during the build.
--
-- Mining join (see scripts/benchmarks/mine_feedback_labels.py):
--   SELECT rl.query_text, rl.query_hash, mf.helpful, am.content_hash
--   FROM memory_feedback mf
--   JOIN recall_log rl
--     ON rl.session_id = mf.session_id
--    AND mf.learning_id = ANY(rl.recalled_ids)
--    AND rl.created_at <= mf.created_at
--    AND rl.created_at > mf.created_at - INTERVAL '1 day'
--   JOIN archival_memory am ON am.id = mf.learning_id;

ALTER TABLE recall_log ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE recall_log ADD COLUMN IF NOT EXISTS query_hash TEXT;
ALTER TABLE recall_log ADD COLUMN IF NOT EXISTS query_text TEXT;

CREATE INDEX IF NOT EXISTS idx_recall_log_session ON recall_log(session_id);
