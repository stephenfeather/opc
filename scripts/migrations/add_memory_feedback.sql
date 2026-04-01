-- Migration: Add memory_feedback table for tracking learning usefulness
-- Date: 2026-04-01

CREATE TABLE IF NOT EXISTS memory_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learning_id UUID NOT NULL REFERENCES archival_memory(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    helpful BOOLEAN NOT NULL,
    context TEXT,
    source TEXT DEFAULT 'manual',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_learning ON memory_feedback(learning_id);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON memory_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_helpful ON memory_feedback(helpful);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON memory_feedback(created_at DESC);

-- Prevent duplicate feedback for same learning in same session
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_unique_per_session
    ON memory_feedback(learning_id, session_id);
