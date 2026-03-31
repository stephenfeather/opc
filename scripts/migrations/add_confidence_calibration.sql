-- Add confidence_calibrated_at column for tracking calibration state
-- This column is NULL for uncalibrated learnings, set to NOW() after calibration.

ALTER TABLE archival_memory
ADD COLUMN IF NOT EXISTS confidence_calibrated_at TIMESTAMPTZ;

-- Index for efficient backfill queries (find uncalibrated learnings)
CREATE INDEX IF NOT EXISTS idx_archival_memory_uncalibrated
ON archival_memory (created_at DESC)
WHERE confidence_calibrated_at IS NULL AND superseded_by IS NULL;
