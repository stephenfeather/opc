-- Knowledge Graph: Entities (nodes)
CREATE TABLE IF NOT EXISTS kg_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,                    -- canonical name (lowercase, normalized)
    display_name TEXT NOT NULL,            -- original casing for display
    entity_type TEXT NOT NULL,             -- file, module, tool, concept, etc.
    metadata JSONB DEFAULT '{}',           -- extra attributes
    embedding vector(1024),                -- optional: entity-level embedding
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mention_count INTEGER NOT NULL DEFAULT 1
);

-- Unique constraint: one entity per (name, entity_type)
CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_entity_unique
    ON kg_entities(name, entity_type);

-- Fast lookup by type
CREATE INDEX IF NOT EXISTS idx_kg_entity_type
    ON kg_entities(entity_type);

-- Full-text search on entity names (requires pg_trgm extension)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        CREATE INDEX IF NOT EXISTS idx_kg_entity_name_trgm
            ON kg_entities USING gin(name gin_trgm_ops);
    ELSE
        RAISE NOTICE 'pg_trgm not available — skipping trigram index on kg_entities.name';
    END IF;
END
$$;

-- Knowledge Graph: Relationships (edges)
CREATE TABLE IF NOT EXISTS kg_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,               -- uses, contains, solves, related_to, etc.
    weight FLOAT NOT NULL DEFAULT 1.0,    -- co-occurrence strength
    memory_id UUID NOT NULL REFERENCES archival_memory(id) ON DELETE CASCADE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Replace old 3-column unique index with 4-column per-learning provenance index.
-- DROP first because IF NOT EXISTS is a no-op when the name already exists.
DROP INDEX IF EXISTS idx_kg_edge_unique;
CREATE UNIQUE INDEX idx_kg_edge_unique
    ON kg_edges(source_id, target_id, relation, memory_id);

-- Traversal indexes (both directions for undirected queries)
CREATE INDEX IF NOT EXISTS idx_kg_edge_source ON kg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edge_target ON kg_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_kg_edge_relation ON kg_edges(relation);

-- Knowledge Graph: Entity-to-Learning mentions (join table)
CREATE TABLE IF NOT EXISTS kg_entity_mentions (
    entity_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES archival_memory(id) ON DELETE CASCADE,
    mention_type TEXT NOT NULL DEFAULT 'explicit',  -- 'explicit', 'inferred'
    span_start INTEGER,                              -- character offset in content
    span_end INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entity_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_mentions_memory
    ON kg_entity_mentions(memory_id);
CREATE INDEX IF NOT EXISTS idx_kg_mentions_entity
    ON kg_entity_mentions(entity_id);
