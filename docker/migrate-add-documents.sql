-- Migration: Document-Collection RAG layer (born-digital v1)
-- Adds documents + document_chunks tables. Safe to run repeatedly.
--
-- Apply against the dev container (named opc-postgres in docker-compose.yml):
--   docker exec -i opc-postgres psql -U claude -d continuous_claude \
--       < docker/migrate-add-documents.sql

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size_bytes BIGINT,
    page_count INTEGER,
    extraction_status TEXT NOT NULL DEFAULT 'extracted'
        CHECK (extraction_status IN
            ('extracted', 'skipped_unsupported', 'skipped_needs_ocr', 'error')),
    error TEXT,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (collection_name, file_path)
);

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_name);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(extraction_status);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_name TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('global', 'restricted')),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_collection ON document_chunks(collection_name);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_scope ON document_chunks(scope);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_content_fts ON document_chunks
    USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding_hnsw ON document_chunks
    USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
