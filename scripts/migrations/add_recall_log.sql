-- Migration: Add recall_log table (issue #140)
-- Append-only per-recall-event log so "how often does cross-project recall
-- mis-scope happen" (issue #130) is answerable with real data. Each recall
-- event writes ONE row with parallel arrays of the recalled rows' ids and
-- their projects (point-in-time, captured via UPDATE ... RETURNING).
--
-- Apply with `psql -f` (autocommit). This creates a NEW, empty table, so a
-- plain CREATE INDEX is fine -- there are no existing rows to lock and no
-- concurrent writers to block, so CONCURRENTLY is NOT needed here (unlike
-- add_project_column.sql, which indexed a populated table). IF NOT EXISTS
-- keeps the migration idempotent and safe to re-run.
--
-- Volume: ~1 row per user prompt (the memory-awareness hook fires per
-- prompt). At that rate the table grows slowly; prune old rows periodically,
-- e.g.:
--   DELETE FROM recall_log WHERE created_at < NOW() - INTERVAL '90 days';
--
-- Privacy: this table NEVER stores raw query text. Only canonical project
-- labels, recalled memory ids, a count, and a short caller label are kept --
-- prompt text is a leak class (#139 redactor) and is deliberately excluded.
--
-- Mis-scope analysis (answers #130): unnest the parallel projects array and
-- bucket each recalled row relative to the caller's project. A NULL recalled
-- project is "unattributed" (not mis-scoped); a non-NULL project that differs
-- from the caller is "mis-scoped".
--   SELECT caller_project,
--          COUNT(*) AS recalled_rows,
--          COUNT(*) FILTER (WHERE rp IS NULL) AS unattributed_rows,
--          COUNT(*) FILTER (WHERE rp IS NOT NULL AND rp <> caller_project)
--              AS mis_scoped_rows,
--          ROUND(100.0 * COUNT(*) FILTER (
--              WHERE rp IS NOT NULL AND rp <> caller_project
--          ) / COUNT(*), 1) AS mis_scope_pct
--   FROM recall_log
--   CROSS JOIN LATERAL unnest(recalled_projects) AS rp
--   WHERE caller_project IS NOT NULL
--   GROUP BY caller_project
--   ORDER BY mis_scope_pct DESC;

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
