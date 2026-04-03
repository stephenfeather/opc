-- Migration: Add learning_rejections table for dedup rejection tracking
-- Date: 2026-04-03

CREATE TABLE IF NOT EXISTS learning_rejections (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    similarity REAL,
    threshold REAL,
    existing_id TEXT,
    existing_session TEXT,
    project TEXT,
    learning_type TEXT,
    context TEXT,
    tags TEXT[],
    rejected_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_learning_rejections_session
    ON learning_rejections (session_id);
