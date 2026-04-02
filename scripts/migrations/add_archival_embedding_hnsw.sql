-- Migration: Add HNSW index on archival_memory embeddings
-- Enables approximate nearest-neighbor search for vector queries
-- (query expansion, hybrid RRF vector ranking, direct vector search).
-- Without this, ORDER BY embedding <=> $1::vector does a sequential scan.

CREATE INDEX IF NOT EXISTS idx_archival_embedding_hnsw ON archival_memory
    USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
