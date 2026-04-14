# Knowledge Graph — Phase 3: Query-Time Integration

**Branch:** `feature/knowledge-graph`
**Scope:** Read-side only. No schema changes, no new write paths.
**Predecessors:** Phase 1 (`b76330c`, KG schema + extractor), Phase 2 (`1e7b3b1`, store-time integration — `kg_stats` in store result).
**Status:** Draft — awaiting ARCHITECT review before implementation.

---

## 1. Goal

Enrich `recall_learnings` results with structured entity/edge context from the KG tables, and let the reranker weight candidates that share entity overlap with the query. Current recall is hybrid RRF + contextual reranker (project, recency, confidence, recall, type, tags, pattern). Phase 3 adds **one new signal** (`kg_overlap`) and **one new output field** (`kg_context`).

---

## 2. Non-Goals (scope discipline)

- ❌ No new KG tables, columns, or migrations. Read-side only.
- ❌ No changes to the KG extractor, `store_learning_v2`, or the `kg_stats` key.
- ❌ No new embedding computation at recall time. Query-side entity extraction reuses `kg_extractor.extract_entities()` already used at store time.
- ❌ No graph traversal beyond 1 hop. Multi-hop reasoning is deferred to a later phase.
- ❌ No changes to default reranker behavior when KG data is unavailable (sqlite backend, empty tables). Gracefully degrades to current behavior.

---

## 3. Architecture

### 3.1 Data fetch — mirror the pattern-enrichment pattern

`recall_learnings.enrich_with_pattern_strength()` (lines 328–352) is the exact template. Phase 3 adds a parallel `enrich_with_kg_context()` in the same file, following the same shape:

```
_fetch_kg_rows(result_ids: list[str]) -> list[dict]      # pure I/O
build_kg_lookup(rows: list[dict]) -> dict[str, dict]     # pure
apply_kg_enrichment(results, lookup) -> list[dict]       # pure
enrich_with_kg_context(results) -> list[dict]            # orchestrator
```

**SQL (single round trip, per-memory aggregation):**

```sql
SELECT m.memory_id AS id,
       ARRAY_AGG(DISTINCT jsonb_build_object(
         'id', e.id,
         'name', e.display_name,
         'type', e.entity_type,
         'mention_count', e.mention_count
       )) AS kg_entities,
       COALESCE((
         SELECT ARRAY_AGG(DISTINCT jsonb_build_object(
           'source', se.display_name,
           'target', te.display_name,
           'relation', ed.relation,
           'weight', ed.weight
         ))
         FROM kg_edges ed
         JOIN kg_entities se ON se.id = ed.source_id
         JOIN kg_entities te ON te.id = ed.target_id
         WHERE ed.memory_id = m.memory_id
       ), ARRAY[]::jsonb[]) AS kg_edges
FROM kg_entity_mentions m
JOIN kg_entities e ON e.id = m.entity_id
WHERE m.memory_id = ANY($1::uuid[])
GROUP BY m.memory_id;
```

- Indexes already in place: `idx_kg_mentions_memory`, `idx_kg_edge_unique(source_id, target_id, relation, memory_id)`. No new indexes needed — the primary access paths are covered.
- **Caveat:** the edge subquery is not covered by a single index on `memory_id`. Current `idx_kg_edge_unique` has `memory_id` as the **4th column**, which postgres can use for equality lookup via skip-scan but may not be optimal. I will benchmark with `EXPLAIN ANALYZE` during implementation; if the edge subquery exceeds ~10ms for k=50 candidates, I'll file a follow-up (not in this phase) to add `idx_kg_edge_memory` on `(memory_id)`. This is a deliberate deferral to respect read-side-only scope.

### 3.2 Reranker signal — `kg_overlap`

New pure function in `scripts/core/reranker.py`:

```python
def kg_overlap(result: dict, ctx: RecallContext) -> float:
    """Score based on entity overlap between query and result.

    Uses pre-computed ctx.query_entities (set[str] of canonical names) and
    result['kg_entities'] (list of dicts from enrichment). Returns 0..1.
    """
```

**Signal definition (weighted Jaccard over entity names, weighted by entity type salience):**

```
overlap = sum(w(e.type) for e in query_entities ∩ result_entities)
union   = sum(w(e.type) for e in query_entities ∪ result_entities)
score   = overlap / union if union > 0 else 0.0
```

Type weights (justification below):

| Entity type | Weight | Rationale |
|-------------|--------|-----------|
| `file`      | 1.0    | Highly specific — overlap is strong evidence |
| `module`    | 1.0    | Same as file |
| `library`   | 0.8    | Specific but often generic (`pytest`, `asyncpg`) |
| `tool`      | 0.6    | Common across many learnings |
| `language`  | 0.4    | Too coarse (`python` matches everything) |
| `concept`   | 0.5    | Ambiguous by design |
| `error`     | 0.9    | Rare + specific — strong signal |
| `config_var`| 0.7    | Specific |

Weights are a **dataclass field** `KG_TYPE_WEIGHTS` in `reranker.py`, so they're configurable later without schema thrash.

### 3.3 Reranker integration — new weight, preserve invariants

Add to `RerankerConfig` (`scripts/core/config/models.py`):

```python
kg_weight: float = 0.05           # default signal weight
```

And extend `total_signal_weight` property to include it. Default starting value of **0.05** matches `pattern_weight` (which is the most similar signal in character: structured, read-side, enriched post-fetch). Rationale:

- Starting low avoids regressing existing behavior for users without populated KG data.
- After measurement (Phase 3 includes an offline eval harness — see §5), weights can be retuned in a follow-up. The architecture treats signal weight as a config value, not a hardcoded constant.
- **Invariant preserved:** `retrieval_weight + Σ signal_weights = 1.0`. Raising `kg_weight` requires lowering another weight or accepting reduced retrieval contribution. Not done automatically — requires explicit follow-up.

The `rerank()` function (reranker.py:309) adds one line to compute `sig_kg` and one line in the weighted sum. `rerank_details` gains a `kg_overlap` key.

### 3.4 Query-side entity extraction

`make_recall_context()` (recall_learnings.py:165) gains:

```python
query_entities: set[str]  # canonical names from kg_extractor.extract_entities(query)
```

Extraction is **synchronous, pure, already exists** in `kg_extractor`. Cost is negligible (regex-based). Populated only when backend == postgres (sqlite path has no KG data to compare against).

---

## 4. Output enrichment — `kg_context` namespace

Per architect guidance, use `kg_context` (not `kg`, not `kg_stats`) for the recall-side output field. Each recall result dict gains:

```python
result["kg_context"] = {
    "entities": [
        {"id": "...", "name": "pytest", "type": "tool", "mention_count": 42},
        ...
    ],
    "edges": [
        {"source": "pytest", "target": "asyncpg", "relation": "used_with", "weight": 2.0},
        ...
    ],
    "query_overlap": {
        "matched_entities": ["pytest", "asyncpg"],  # intersection with query_entities
        "score": 0.67,                              # kg_overlap signal value
    },
}
```

**Namespace discipline:**
- `kg_stats` (store-side, Phase 2): stats dict from `store_entities_and_edges` — `{entities, edges, mentions}`.
- `kg_context` (recall-side, Phase 3): structured per-result context from KG tables.

These are different shapes with different purposes. Keeping them in separate keys avoids overloading one field.

When enrichment fails or yields no data, `kg_context` is **omitted** (not set to empty). Consumers must check `"kg_context" in result`. This matches the pattern-enrichment convention.

**Safety cap:** A module-level constant `KG_MAX_EDGES_PER_MEMORY = 50` caps the edge list per result. When a memory's edge count exceeds the cap, the top-50 by `weight` are kept and a `logger.warning` is emitted identifying the memory_id and total edge count. Typical usage is < 10 edges per memory; the cap exists to prevent pathological payload bloat on future high-connectivity learnings without breaking the output contract.

---

## 5. TDD test strategy

### 5.1 Unit tests

**`tests/test_kg_enrichment.py`** (new, ~8 tests):
- `_fetch_kg_rows` returns empty list for empty input
- `build_kg_lookup` groups entities + edges correctly by memory_id
- `apply_kg_enrichment` sets `kg_context` on matching results, omits for non-matches
- `apply_kg_enrichment` does not mutate input list
- `enrich_with_kg_context` returns unchanged list when backend is sqlite
- `enrich_with_kg_context` returns unchanged list when result_ids is empty
- `enrich_with_kg_context` gracefully handles DB connection error (non-fatal)
- `enrich_with_kg_context` correctly handles results with no KG entries (no `kg_context` key)

**`tests/test_reranker.py`** (extend, ~5 tests):
- `kg_overlap` returns 0.0 when query_entities is None/empty
- `kg_overlap` returns 0.0 when result has no `kg_context`
- `kg_overlap` returns 1.0 for identical entity sets
- `kg_overlap` type-weight calculation (verify `file` entity overlap scores higher than `language` overlap of same size)
- `rerank()` with `kg_weight > 0` boosts results with matching entities; zero weight is a no-op

### 5.2 Integration test

**`tests/test_recall_kg_integration.py`** (new, ~3 tests):
- Against seeded KG (fixture inserts ~5 learnings with entities/edges), verify recall returns `kg_context` populated correctly.
- Verify query containing a known entity name causes that entity's learnings to rank higher with `kg_weight=0.2` vs `kg_weight=0.0`.
- Verify sqlite backend returns results without `kg_context` and without error.

### 5.3 Offline eval (nice-to-have, not blocking)

`scripts/eval_kg_reranker.py` runs a fixed set of 20 handcrafted queries against a seeded KG with and without `kg_weight` > 0. Reports MRR / nDCG@5 delta. Used to justify future weight tuning. **Not** a hard gate for Phase 3 merge — weights start conservative at 0.05 regardless.

### 5.4 Coverage targets

`uv run pytest --cov=scripts/core/recall_learnings --cov=scripts/core/reranker --cov=scripts/core/kg_extractor` must show ≥ baseline coverage for all three files (baselines captured before this work starts).

---

## 6. Implementation phasing (TDD red → green → refactor)

1. **Commit A — fetch + enrichment plumbing**
   - Red: write `test_kg_enrichment.py` against stubbed `_fetch_kg_rows`.
   - Green: implement the four functions in `recall_learnings.py`.
   - Refactor: deduplicate with `enrich_with_pattern_strength` if a clean shared helper emerges (likely not — keep separate).

2. **Commit B — reranker signal**
   - Red: extend `test_reranker.py` with `kg_overlap` and `rerank` tests.
   - Green: add `kg_overlap()`, `KG_TYPE_WEIGHTS`, `kg_weight` to `RerankerConfig`, wire into `rerank()`.
   - Refactor: verify `total_signal_weight` math and existing tests still pass.

3. **Commit C — query-side entity extraction + wiring**
   - Red: extend `RecallContext` tests, `test_recall_kg_integration.py`.
   - Green: add `query_entities` to `RecallContext`, call `extract_entities` in `make_recall_context`, call `enrich_with_kg_context` in the recall pipeline.

4. **Commit D — adversarial round 1 findings** (required by workflow).
5. **Commit E — adversarial round 2 findings**.
6. **Commit F — adversarial round 3 findings**.
7. **Commit G — security audit findings + coverage report in PR body**.

Each commit stays small, reviewable, and keeps tests green.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Edge subquery perf on large KG | `EXPLAIN ANALYZE` during implementation; defer to follow-up if > 10ms |
| Existing recall tests over-index on result shape | Run full `tests/test_recall_*` early; if strict-dict assertions exist, update them alongside the change and note in commit message (per architect precedent) |
| `kg_overlap` signal fires on trivial entities (e.g. every learning mentions `python`) | Type weights down-rank `language` to 0.4; also considered but rejected: IDF-weighting over mention_count (more complex, defer) |
| Query entity extraction misses intent (short queries like "error") | Non-fatal — signal contributes 0 when no entities extracted. Other reranker signals compensate. |
| Reranker weight sum drifts from 1.0 | `RerankerConfig.__post_init__` can assert invariant; alternative: leave `total_signal_weight` computed and document that retrieval_weight is the remainder |
| KG tables empty (fresh install) | `enrich_with_kg_context` returns results unchanged; `kg_overlap` returns 0; recall behaves exactly as before Phase 3 |

---

## 8. Open questions for ARCHITECT

1. **Weight starting value** — `kg_weight = 0.05` is conservative. Acceptable, or prefer 0.10 since KG is a higher-signal feature than e.g. `recency`?
2. **Query entity extraction** — use full `kg_extractor.extract_entities()` (regex-heavy, extracts file paths) or a subset tuned for query strings (no file-path extraction)? My default is the full extractor for consistency with store-side.
3. **Edge inclusion in `kg_context`** — include all edges for the memory, or cap at top-N by weight? Default: all (typically < 10 per learning; cap if measured > 50).
4. **Follow-up scope** — if edge subquery perf is poor, file a follow-up issue rather than adding `idx_kg_edge_memory` in this PR. Confirm?

---

## 9. Success criteria (from assignment)

- ✅ Recall results include entity/edge context (§4)
- ✅ Reranker uses KG-derived signal with justified weights (§3.2, §3.3)
- ✅ Coverage on changed files ≥ existing baseline (§5.4)
- ✅ 3 adversarial review rounds pass (§6 commits D/E/F)
- ✅ PR merged after 2+ AI reviewer cycles

---

## 10. Files touched (expected)

| File | Change |
|------|--------|
| `scripts/core/recall_learnings.py` | +4 functions (fetch, build, apply, enrich), wire enrich into pipeline, extend RecallContext construction |
| `scripts/core/reranker.py` | +1 signal function, +1 constant dict, wire into `rerank()` |
| `scripts/core/config/models.py` | +1 field `kg_weight`, extend `total_signal_weight` |
| `tests/test_kg_enrichment.py` | NEW |
| `tests/test_recall_kg_integration.py` | NEW |
| `tests/test_reranker.py` | extend |
| `tests/test_recall_learnings.py` | extend for RecallContext changes |

Estimated net LOC: **+400 / -30**.
