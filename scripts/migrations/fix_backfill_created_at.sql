-- Migration: Retroactively fix created_at for backfilled learnings (issue #52)
--
-- Learnings extracted from backfilled (historical) sessions were stamped with
-- created_at = NOW() (the archival_memory DEFAULT) at extraction time, NOT the
-- original session time. The reranker's recency decay (exp(-age_days / 45))
-- then treats weeks-old content as age-zero, giving it a ~1.0 recency score
-- and a ~7x boost over genuinely recent learnings. push_learnings.py
-- (ORDER BY created_at DESC) and pattern_detector.py date math are biased the
-- same way.
--
-- Going forward, store_learning.py --source-time / CLAUDE_SOURCE_TIME stamps
-- created_at from the source session time. This migration repairs the EXISTING
-- backfilled rows so their created_at reflects the original session.
--
-- Source of truth: backfill_sessions.py inserts sessions with
-- working_on = 'backfill' and exited_at = JSONL mtime (the session end time).
-- archival_memory.session_id -> sessions.id recovers that time via JOIN.
--
-- VERIFIED PREDICATE (dev DB, 2026-06-11):
--   sessions.working_on = 'backfill'  distinguishes 152 backfilled sessions
--   (the live/non-backfill rows have working_on = '' or NULL). The JOIN +
--   only-move-earlier guard matched 90 archival_memory rows whose created_at
--   was 2-3 days later than the true session time.
--
-- SAFETY:
--   * Scoped to working_on = 'backfill' ONLY -- live-session rows are untouched.
--   * Only-move-EARLIER guard (s.exited_at < a.created_at) -- created_at is
--     never moved forward, so re-running cannot inflate recency.
--   * exited_at IS NOT NULL guard -- skips sessions with no recorded end time.
--   * Idempotent -- after the first run the guard makes the row set empty.
--
-- Apply with `psql -f` (autocommit). This is a bounded single-statement UPDATE
-- over ~90 rows; no locking concerns.
--
-- ---------------------------------------------------------------------------
-- DRY RUN (run this SELECT first; it mutates nothing). It reports the row
-- count and the created_at shift window so the operator can sanity-check the
-- blast radius before running the UPDATE below.
-- ---------------------------------------------------------------------------
--
--   SELECT
--       COUNT(*)                              AS rows_to_fix,
--       MIN(s.exited_at)                      AS min_session_time,
--       MAX(s.exited_at)                      AS max_session_time,
--       MIN(a.created_at - s.exited_at)       AS min_backward_shift,
--       MAX(a.created_at - s.exited_at)       AS max_backward_shift
--   FROM archival_memory a
--   JOIN sessions s ON a.session_id = s.id
--   WHERE s.working_on = 'backfill'
--     AND s.exited_at IS NOT NULL
--     AND s.exited_at < a.created_at;
--
-- ---------------------------------------------------------------------------
-- UPDATE
-- ---------------------------------------------------------------------------

UPDATE archival_memory AS a
SET created_at = s.exited_at
FROM sessions AS s
WHERE a.session_id = s.id
  AND s.working_on = 'backfill'
  AND s.exited_at IS NOT NULL
  AND s.exited_at < a.created_at;  -- only move EARLIER; never forward

-- Echo the number of rows repaired (0 on a re-run thanks to the guard above).
-- The number is reported by psql as "UPDATE <n>".
