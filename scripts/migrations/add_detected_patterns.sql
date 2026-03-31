-- Migration: Create detected_patterns and pattern_members tables
-- Date: 2026-03-30
-- Purpose: Store cross-session pattern detection results.
--          detected_patterns holds each detected cluster/pattern.
--          pattern_members maps learnings to patterns (many-to-many).
--          Pattern strength is computed dynamically via JOIN at recall time
--          to avoid MVCC bloat from frequent UPDATEs on archival_memory.

CREATE TABLE IF NOT EXISTS detected_patterns (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_type          TEXT NOT NULL,
    label                 TEXT NOT NULL,
    representative_id     UUID REFERENCES archival_memory(id),
    tags                  TEXT[] NOT NULL DEFAULT '{}',
    session_count         INTEGER NOT NULL,
    confidence            REAL NOT NULL,
    metadata              JSONB NOT NULL DEFAULT '{}',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id                UUID NOT NULL,
    synthesized_memory_id UUID REFERENCES archival_memory(id),
    superseded_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_detected_patterns_type
    ON detected_patterns(pattern_type);

CREATE INDEX IF NOT EXISTS idx_detected_patterns_run
    ON detected_patterns(run_id);

CREATE INDEX IF NOT EXISTS idx_detected_patterns_active
    ON detected_patterns(superseded_at)
    WHERE superseded_at IS NULL;

-- Many-to-many: learnings <-> patterns
CREATE TABLE IF NOT EXISTS pattern_members (
    pattern_id  UUID NOT NULL REFERENCES detected_patterns(id) ON DELETE CASCADE,
    memory_id   UUID NOT NULL REFERENCES archival_memory(id) ON DELETE CASCADE,
    distance    REAL,
    PRIMARY KEY (pattern_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_pattern_members_memory
    ON pattern_members(memory_id);
