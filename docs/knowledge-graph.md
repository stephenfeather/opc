# Knowledge Graph

The OPC Knowledge Graph (KG) is a structured, queryable layer over the unstructured `archival_memory` table. It extracts named entities and the relationships between them from each stored learning, persists them to dedicated tables, and uses that structure at recall time to enrich results and boost reranker scoring.

The KG sits *alongside* the embedding-based hybrid RRF recall — it does not replace it. Embeddings retrieve candidates by semantic similarity; the KG explains *why* a candidate is relevant by surfacing the entities and edges it shares with the query.

---

## Why a Knowledge Graph

Embedding-based recall is fuzzy by design. Two learnings that both mention `pyproject.toml`, `uv`, and the same error string will rank close on cosine similarity, but the system has no concept that all three are *the same kinds of things* across both records. The KG adds that concept:

- **Entities** are typed nouns (a file, a tool, an error, a language). Each canonical entity is stored once, regardless of how many learnings mention it.
- **Edges** are typed verbs between entities (`solves`, `uses`, `supersedes`, `conflicts_with`, `contains`, `related_to`). Edges carry a confidence weight and a back-reference to the memory that produced them.
- **Mentions** record which memory referenced which entity, with a count for hot-spot detection.

At recall time, the system extracts entities from the query, looks up shared entities and edges across candidate results, and rewards results that share high-salience entity types (e.g., `file`, `error`, `function`) more than generic ones (e.g., `language`).

---

## Architecture

### Schema

Three tables are created by `scripts/migrations/add_knowledge_graph.sql` (also bundled into `docker/init-schema.sql`):

| Table | Purpose | Key columns |
|---|---|---|
| `kg_entities` | Canonical entity registry, deduplicated by `(name, entity_type)` | `id`, `name` (canonical, lowercased), `display_name`, `entity_type`, `mention_count`, `metadata` |
| `kg_edges` | Typed relationships between entities, scoped to the memory that produced them | `source_id`, `target_id`, `relation`, `weight`, `memory_id` |
| `kg_entity_mentions` | Per-memory entity occurrences | `memory_id`, `entity_id`, `span_start`, `span_end` |

Indexes ensure fast per-memory enrichment lookups (`idx_kg_mentions_memory`) and edge dedup on `(source_id, target_id, relation, memory_id)`.

### Entity types (extracted heuristically)

`file`, `module`, `tool`, `language`, `library`, `error`, `concept`, `function`, `config_var`.

Extraction is regex- and dictionary-based — see `scripts/core/kg_extractor.py`. The dictionaries `KNOWN_TOOLS`, `KNOWN_LANGUAGES`, and `KNOWN_LIBRARIES` are the source of truth for those three types.

### Relation types (inferred between co-occurring entities)

`solves`, `uses`, `supersedes`, `conflicts_with`, `contains`, `related_to`.

Relations are inferred sentence-locally. Two entities in the same sentence with the right type combination produce a typed edge (e.g., `tool` + `error` in the same sentence → `tool solves error`). When no specific rule fires, a generic `related_to` edge with a lower confidence weight is created.

---

## Lifecycle

### Population on store

`store_learning._try_index_kg()` (`scripts/core/store_learning.py:502`) runs after each successful store on the postgres backend:

1. `extract_entities(content)` — regex + dictionary scan
2. `extract_relations(content, entities)` — sentence-local co-occurrence rules
3. `store_entities_and_edges(memory_id, entities, relations)` — idempotent upsert
4. Returns a `kg_stats` dict (entities/edges/mentions counts) merged into the store result

The path is **non-fatal** — extraction or insertion failures log a warning and the memory is still stored. The sqlite backend skips KG entirely.

A symmetric `_try_backfill_kg()` runs when a dedup hit matches an existing memory by `content_hash`, so re-storing the same content is idempotent and still enriches the graph.

### Enrichment on recall

`recall_learnings.enrich_with_kg_context()` runs after the existing pattern-strength enrichment:

1. After hybrid RRF retrieval returns candidates, fetch all KG entities and edges scoped to the candidate memory IDs in a single round-trip.
2. Group by `memory_id` and merge a `kg_context` field onto each candidate. The shape is:

```json
{
  "kg_context": {
    "entities": [{"id": "...", "name": "...", "type": "tool", "mention_count": 4}],
    "edges": [{"source": "ruff", "target": "lint", "relation": "uses", "weight": 0.7}]
  }
}
```

3. The reranker (`scripts/core/reranker.py`) computes a `kg_overlap` signal — a type-weighted Jaccard between query entities and result entities. Salient types (`file`, `error`, `function`) carry full weight; generic types (`language`) are down-weighted via `KG_TYPE_WEIGHTS`.
4. The signal is combined with the other reranker signals (recency, project, confidence, recall, type, tags, pattern) using `kg_weight` from `RerankerConfig` (default `0.05`).

### Query-side entity extraction

When `make_recall_context()` builds a `RecallContext`, it calls `kg_extractor.extract_entities(query)` and stores the result in `query_entities`. Failures are non-fatal — short or non-entity queries simply produce an empty set, in which case `kg_overlap` returns `0.0` and recall falls back to the other signals.

---

## Current state and limitations

- **Read path is live** — every recall against the postgres backend already calls `enrich_with_kg_context` and the reranker already weights `kg_overlap`. When the tables are empty (or for memories that have no rows in `kg_entity_mentions`), enrichment is a no-op and recall behaves exactly as before.
- **Write path triggers only on new stores** — historical memories are not retroactively indexed. A backfill script for existing rows is tracked in [#124](https://github.com/stephenfeather/opc/issues/124).
- **No multi-hop traversal** — enrichment is 1-hop only (entities + their direct edges within the same memory). Deeper graph traversal is deferred.
- **Heuristic extraction** — entity and relation extraction is rule-based, not LLM-based. False positives (e.g., generic words misidentified as `concept`) are filtered by a noise list and by the `_is_noise()` heuristic.
- **Postgres only** — the sqlite backend has no KG tables.

---

## Tuning

| Knob | Where | Default | Purpose |
|---|---|---|---|
| `kg_weight` | `RerankerConfig` (`scripts/core/config/models.py`) | `0.05` | Contribution of `kg_overlap` to the final reranker score |
| `KG_TYPE_WEIGHTS` | `scripts/core/reranker.py` | `language=0.4`, others=`1.0` | Per-type salience inside the Jaccard |
| `KG_MAX_EDGES_PER_MEMORY` | `scripts/core/recall_learnings.py` | `50` | Caps payload size; emits a warning when truncated |
| Known dictionaries | `KNOWN_TOOLS`, `KNOWN_LANGUAGES`, `KNOWN_LIBRARIES` (`scripts/core/kg_extractor.py`) | — | Add new tools/languages/libraries here to expand entity coverage |

After tuning weights, run `tests/test_recall_reranking.py` and confirm `RerankerConfig.total_signal_weight` still sums correctly.

---

## File map

| File | Role |
|---|---|
| `scripts/core/kg_extractor.py` | Entity + relation extraction, idempotent persistence |
| `scripts/core/store_learning.py` | `_try_index_kg`, `_try_backfill_kg` — write-path hooks |
| `scripts/core/recall_learnings.py` | `_fetch_kg_rows`, `build_kg_lookup`, `apply_kg_enrichment`, `enrich_with_kg_context` — read-path enrichment |
| `scripts/core/reranker.py` | `kg_overlap` signal + `KG_TYPE_WEIGHTS` |
| `scripts/migrations/add_knowledge_graph.sql` | Schema migration (also embedded in `docker/init-schema.sql`) |
| `tests/test_kg_extractor.py` | Extractor unit tests |
| `tests/test_kg_enrichment.py` | Enrichment unit tests |
| `tests/test_store_kg_integration.py` | Store-time integration tests |
| `tests/test_recall_kg_integration.py` | End-to-end recall + KG integration tests |
| `thoughts/shared/plans/kg-phase3-query-time-integration.md` | Original Phase 3 plan (historical) |

---

## References

- PR [#121](https://github.com/stephenfeather/opc/pull/121) — Phases 1, 2, 3 merged together
- Issue [#124](https://github.com/stephenfeather/opc/issues/124) — backfill for historical memories
- Issue [#120](https://github.com/stephenfeather/opc/issues/120) — ReDoS audit of extractor regexes
- Issue [#122](https://github.com/stephenfeather/opc/issues/122) — extractor correctness/efficiency follow-ups
