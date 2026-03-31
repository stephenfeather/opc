-- Backfill memory_tags from existing metadata JSON tags
--
-- Populates the memory_tags table from archival_memory rows that have
-- tags stored in metadata->'tags' (JSON array of strings).
-- Idempotent: ON CONFLICT DO NOTHING skips already-backfilled rows.
--
-- Usage:
--   docker exec continuous-claude-postgres psql -U claude -d continuous_claude \
--     -f /path/to/backfill_memory_tags.sql

INSERT INTO memory_tags (memory_id, tag, session_id, created_at)
SELECT
    am.id,
    tag.value,
    am.session_id,
    am.created_at
FROM archival_memory am,
     jsonb_array_elements_text(am.metadata -> 'tags') AS tag(value)
WHERE am.metadata::jsonb ? 'tags'
  AND jsonb_array_length(am.metadata -> 'tags') > 0
ON CONFLICT (memory_id, tag) DO NOTHING;
