-- Migration: HNSW index on archival_memory.embedding (issue #151)
--
-- Accelerates the BOUNDED vector query paths — those shaped
-- `ORDER BY embedding <=> $1::vector LIMIT k`: --mode vector search,
-- recency-weighted vector, query-expansion neighbors, and the store-time
-- dedup probes. pgvector's HNSW only speeds up bounded ORDER-BY-distance-LIMIT
-- queries.
--
-- NOTE: the hybrid RRF vector leg is NOT one of those shapes — it is a
-- windowed unbounded rank, `ROW_NUMBER() OVER (ORDER BY embedding <=> $2)`
-- with no LIMIT inside the CTE, so it remains a sequential rank over the
-- (model-filtered, currently ~6k-row) corpus regardless of this index. That
-- is cheap at today's corpus size; reshaping the RRF leg into a bounded
-- ANN-friendly form is tracked as a follow-up issue.
--
-- WHY THIS IS A SEPARATE FILE FROM add_archival_embedding_hnsw.sql:
-- The original migration uses a plain (blocking) CREATE INDEX and predates
-- the embedding-space canonicalization. Build the index AFTER the corpus is
-- canonicalized to a single space (run scripts/core/re_embed_voyage.py first,
-- bge -> voyage-code-3) so the graph is constructed over one uniform space.
-- Building before re-embed wastes work: re-embedding does not move rows, but
-- the post-#151 recall filter (AND embedding_model = $N) only matches rows in
-- the query's space, so an index built over a mixed corpus indexes vectors
-- the filter will never return.
--
-- IDEMPOTENT: IF NOT EXISTS makes reruns a no-op. If a prior CONCURRENTLY
-- build was interrupted it may leave an INVALID index that IF NOT EXISTS then
-- skips; on failure drop and retry:
--   DROP INDEX IF EXISTS idx_archival_embedding_hnsw;
-- then re-run this migration.
--
-- IMPORTANT: CONCURRENTLY cannot run inside a transaction block. Apply with
-- `psql -f` (autocommit) -- NOT with `psql -1`/`BEGIN`. On the ~6k-row corpus
-- the build takes seconds; CONCURRENTLY avoids the SHARE lock that would
-- otherwise stall live stores and recall_count UPDATEs during the build.
--
-- The opclass MUST be vector_cosine_ops: all recall SQL orders by the cosine
-- distance operator `<=>`, and an HNSW index is only used when its opclass
-- matches the query operator. m/ef_construction match docker/init-schema.sql.
--
-- VERIFY index use after applying (expect "Index Scan using
-- idx_archival_embedding_hnsw", not "Seq Scan"):
--   EXPLAIN (ANALYZE, BUFFERS)
--   SELECT id FROM archival_memory
--   WHERE embedding IS NOT NULL AND embedding_model = 'voyage-code-3'
--   ORDER BY embedding <=> '[0,0, ... ]'::vector
--   LIMIT 10;
--
-- OPTIONAL (post-canonicalization tuning): once the corpus is a single
-- space, a partial index `WHERE embedding_model = 'voyage-code-3'` is
-- marginally tighter, but for ~6k rows the global index below is sufficient.

-- STEP 1 (issue #151, round 1 FIX 1): ensure the embedding_model column
-- exists BEFORE anything filters on it. The recall model filter
-- (AND embedding_model = $N) and the explicit store-time column write both
-- key on this column; on a pre-migration DB the runtime degrades to
-- unfiltered SQL, but applying this ADD COLUMN turns the filter on. The
-- DEFAULT matches docker/init-schema.sql so existing rows are labeled 'bge'
-- (the historical space) until re-embedded. IF NOT EXISTS keeps reruns
-- idempotent; this is a plain (non-CONCURRENT) ALTER and so may run inside
-- the same psql -f invocation as the index build below.
ALTER TABLE archival_memory
    ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'bge';

-- STEP 2: the HNSW index itself.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archival_embedding_hnsw
    ON archival_memory
    USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
