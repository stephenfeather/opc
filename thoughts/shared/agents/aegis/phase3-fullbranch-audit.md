# Aegis Full-Branch Security Audit — feature/knowledge-graph

**Date:** 2026-04-14
**Target:** `feature/knowledge-graph` vs `origin/main`
**Scope:** All 24 files in the branch diff (Phases 1, 2, 3). Broader than the earlier Phase-3-scoped audit (4 files).
**PR:** [#121](https://github.com/stephenfeather/opc/pull/121)

## Verdict

**Risk: LOW.** 0 critical / 0 high / 0 medium / 2 new LOW findings. Branch is safe to merge from a security standpoint.

## Files covered

**Production code:**
- `scripts/core/kg_extractor.py` (new, 487 LOC — Phase 1)
- `scripts/core/store_learning.py` (+76 LOC — Phase 2 KG write-side)
- `scripts/core/recall_learnings.py` (+190 LOC — Phase 3)
- `scripts/core/reranker.py` (+127 LOC — Phase 3)
- `scripts/core/recall_formatters.py` (+7 LOC — Phase 3)
- `scripts/core/config/models.py` (+46 LOC — Phase 3)

**Schema:**
- `docker/init-schema.sql` (+67 LOC — Phase 1)
- `scripts/migrations/add_knowledge_graph.sql` (new, 72 LOC)

**Tests:** 8 files (read only, no secrets, no sensitive data).

## Positive confirmations

### SQL parameterization clean across all new SQL
- `kg_extractor.py:408-479` (write-side INSERT) — all `$N` bind params with explicit `::uuid` / `::jsonb` casts.
- `store_learning.py:546-559` — parameterized.
- `recall_learnings.py:415-449` (`_fetch_kg_rows`) — parameterized with `$1::uuid[]` + `$2::int`, already reviewed in Round E.
- No f-strings, `.format`, or `%` substitution against user data.
- `jsonb_build_object` prevents JSON injection even from `display_name`.

### No unsafe deserialization
No `pickle`, `yaml.unsafe_load`, or `eval` anywhere in diff.

### No command injection surface
No `subprocess` calls in KG files.

### No hardcoded secrets
`tests/test_kg_extractor.py:53-56` references the string `"VOYAGE_API_KEY"` as an extractor input fixture — it is a config-var-name literal, not a real secret value.

### Info disclosure clean
KG log sites use `str(e)[:200]` (bounded), no `exc_info=True` on new paths.

### Transactional safety
- `store_entities_and_edges` wraps upsert + mention + edge in a single `conn.transaction()`.
- Backfill path's TOCTOU on dedup is covered by the outer non-fatal handler at `store_learning.py:558`. FK violation on a racing delete is caught, not propagated.

### Read-side resource caps enforced at SQL + Python layers
- `KG_MAX_EDGES_PER_MEMORY=50` applied in SQL `LATERAL ... LIMIT $2` and rechecked in Python `build_kg_lookup` as defense-in-depth.
- Query length cap `_KG_QUERY_EXTRACTION_MAX_CHARS=4096` applied before `extract_entities` is called (fixed from LOW-1 in earlier Phase-3 audit).

## New LOW findings

### LOW-4 — ReDoS exposure in `_RE_FILE_PATH`
**File:** `scripts/core/kg_extractor.py:83-90`
**Description:** Pattern includes a nested quantifier `(?:[\w.~-]+/)+[\w.-]+`. Theoretically susceptible to catastrophic backtracking on pathological input.
**Severity:** LOW (advisory).
**Current mitigation:** already bounded by the 4096-char query length cap fixed in Commit G (LOW-1). Store-side input is operator-authored only.
**Remediation:** None required under current threat model. If learnings ever accept third-party input, wrap extractor calls in a subprocess + timeout boundary, or switch to the `regex` module with explicit match-time budget.

### LOW-5 — Unbounded entity/edge counts per learning on write
**File:** `scripts/core/kg_extractor.py:147-230` (`extract_entities`), `:261-358` (`extract_relations`), `:383-487` (`store_entities_and_edges`)
**Description:** No cap on entities produced per learning; `extract_relations` is O(N²) in entity count (already tracked in #122). Read-side is capped at 50 edges per memory, so recall latency is bounded. Risk is **write-time DB storage/write cost** only. Idempotent upserts prevent retry inflation.
**Severity:** LOW (advisory).
**Remediation (optional):** Introduce `KG_MAX_ENTITIES_PER_MEMORY` (e.g. 200) enforced in `extract_entities`. Folded into follow-up #122.

## Known-deferred items (acknowledged, not re-filed)

- **#122** — kg_extractor correctness/efficiency bundle (lstrip path corruption, regex compile in hot loop, sentence offset drift, O(N²) directory containment, metadata hardcoded to `"{}"`, entity_count docstring-vs-behavior mismatch, test assertion logic bug). Still visible in branch; scope-deferred per "Phase 3 is read-side only."
- **#123** — `gen_random_uuid()` requires `pgcrypto` at `docker/init-schema.sql:233` and `scripts/migrations/add_knowledge_graph.sql:3`. Phase 1 schema; scope-deferred.
- **LOW-1 / LOW-3** — already landed in commit `23acb48` (query length cap + `kg_context` trust-boundary comment).
- **#111** — unrelated pre-existing test failure on `origin/main`.

## Security checklist

| Category | Result |
|---|---|
| SQL injection (new SQL) | Clean |
| Unsafe deserialization (eval/pickle/yaml) | Clean |
| Command injection / shell | N/A |
| Path traversal | N/A (kg_extractor has zero file I/O — paths are strings only) |
| Hardcoded secrets | Clean |
| Info disclosure via logs | Acceptable |
| TOCTOU / races | Safe (single-txn upsert; outer non-fatal handler on dedup backfill) |
| Resource exhaustion (read) | Capped |
| Resource exhaustion (write) | Unbounded — see LOW-5 |
| Schema / extensions | `pg_trgm` guarded; `pgcrypto` implicit (#123) |
| Dependency changes | None |

## Recommendations

- **Immediate:** None.
- **Short-term:** Optionally add `KG_MAX_ENTITIES_PER_MEMORY` cap (fold into #122).
- **Long-term (if threat model expands to third-party learnings):** ReDoS timeout wrappers + store-side content caps; enforce `kg_context` trust boundary with an actual escape at any future HTML/markdown rendering sink.

## Conclusion

Branch is safe to merge. Two new LOW findings are advisory under the current internal-CLI threat model. The two already-filed follow-up issues (#122, #123) cover the out-of-scope Phase 1 findings. No critical, high, or medium items outstanding.
