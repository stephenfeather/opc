-- Migration: Add pool_size/fetch_k to recall_log (issue #228)
-- Selection-shape telemetry. recall_log (issue #140) already records
-- result_count (the final returned count) but never the candidate POOL the
-- backend produced, so selection rate (returned / available) was
-- uncomputable. This adds two nullable INTEGER columns:
--   pool_size -- the raw backend candidate pool (the over-fetch; when
--                reranking, compute_fetch_k = max(3*k, 50)) captured BEFORE
--                enrichment/tag-filter/rerank trim it.
--   fetch_k   -- the requested over-fetch ceiling.
-- Selection rate is computed at QUERY time, NOT stored as a derived column:
--   result_count::float / NULLIF(pool_size, 0)
--
-- Both columns are nullable with NO default. ADD COLUMN with no default is a
-- metadata-only change in PostgreSQL 11+ (no table rewrite, no row locks), so
-- this is cheap on a populated table. Existing pre-#228 rows (and the legacy
-- 5-column INSERT fallback in record_recall) stay valid -- a NULL here means
-- "rate unknown / pre-telemetry", which the query-time NULLIF handles.
--
-- IF NOT EXISTS keeps the migration idempotent and safe to re-run. No index is
-- needed: selection-rate analysis is time-scoped via the existing
-- idx_recall_log_created (created_at DESC) range scan.
--
-- Apply with `psql -f` (autocommit).

ALTER TABLE recall_log ADD COLUMN IF NOT EXISTS pool_size INTEGER;
ALTER TABLE recall_log ADD COLUMN IF NOT EXISTS fetch_k INTEGER;
