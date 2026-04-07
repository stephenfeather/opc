# Plan: TDD+FP Compliance Refactor of scripts/core/

## Context

The `scripts/core/` directory contains 27 Python files (including 5 in `db/`) that form the OPC memory system. Many files lack test coverage, mix I/O with business logic, and violate FP principles (mutation, god functions, mixed concerns). This plan refactors each file for TDD+FP compliance following the dependency graph mapped by the scout agent.

Each file is treated as a separate stage following the Large Plan Workflow: git worktree, 3x adversarial review, security review, PR, 2x AI reviewer cycles.

## Completed Stages

| Stage | File | PR | Status |
|-------|------|----|--------|
| S1 | `track_stale_rate.py` | #17 | DONE |
| S2 | `extract_thinking_blocks.py` | #18 | DONE |
| S3 | `db/memory_protocol.py` | #19 | DONE |
| S4 | `db/postgres_pool.py` | #20 | DONE |
| S5 | `db/embedding_service.py` | #22 | DONE |
| S6 | `db/memory_factory.py` | #21 | DONE |
| S7 | `db/memory_service_pg.py` | #23 | DONE |
| S8 | `reranker.py` | #27 | DONE |
| S9 | `pattern_detector.py` | #26 | DONE |
| S10 | `recall_formatters.py` | #25 | DONE |
| S13 | `confidence_calibrator.py` | #30 | DONE |
| S11 | `extract_workflow_patterns.py` | #28 | DONE |
| S12 | `generate_mini_handoff.py` | #29 | DONE |
| S14 | `backfill_sessions.py` | #31 | DONE |
| S15 | `backfill_archive.py` | #33 | DONE |
| S16 | `backfill_learnings.py` | #34 | DONE |
| S17 | `artifact_index.py` | #35 | DONE |
| S18 | `artifact_mark.py` | #36 | DONE |
| S19 | `artifact_query.py` | #39 | DONE |
| S22 | `pattern_batch.py` | #38 | DONE |
| S28 | `memory_feedback.py` | #40 | DONE |
| S20 | `query_expansion.py` | #41 | DONE |
| S21 | `recall_backends.py` | #43 | DONE |
| S23 | `store_learning.py` | #42 | DONE |
| S24 | `recall_learnings.py` | #44 | DONE |
| S25 | `push_learnings.py` | #45 | DONE |
| S27 | `memory_metrics.py` | #48 | DONE |
| S29 | `pattern_report.py` | #47 | DONE |

## Cross-Cutting Changes

| Feature | PR | Files Touched | Notes |
|---------|-----|---------------|-------|
| Config system (`opc.toml`) | #24 | 11 consumer files + new `scripts/core/config/` package | Externalized hardcoded values in `store_learning`, `memory_daemon`, `reranker`, `pattern_detector`, `recall_backends`, `query_expansion`, `re_embed_voyage`, `backfill_archive`, `postgres_pool`, `embedding_providers`, `memory_metrics`. Future refactor stages do NOT need to re-wire config — it's already done. Exception: `memory_service_pg.py` values (limit, k) deferred to TODO (was blocked by S7). |

## Remaining Stages by Dependency Layer

### Layer 0 -- db/ Foundations (refactor first, no internal deps)

| Stage | File | Lines | Notes |
|-------|------|-------|-------|
| S3 | `db/memory_protocol.py` | 198 | No deps. Leaf. |
| S4 | `db/postgres_pool.py` | 345 | No deps. Imported by 11 files -- high impact, careful interface preservation. |
| S5 | `db/embedding_service.py` | 800 | No deps. Large -- split into `db/embedding_service.py` (core) + `db/embedding_providers.py` (provider implementations) if >500 lines post-refactor. |
| S6 | `db/memory_factory.py` | 148 | Depends on `db/memory_protocol.py` (S3 first). |
| S7 | `db/memory_service_pg.py` | 1292 | Depends on `db/postgres_pool.py` (S4 first). Too large -- must split. |

### Layer 1 -- Pure Domain Libraries (no scripts.core deps)

| Stage | File | Lines | Notes |
|-------|------|-------|-------|
| S8 | `reranker.py` | 391 | Already mostly pure functions. |
| S9 | `pattern_detector.py` | 611 | Large -- may need split at 500+ line threshold. |
| S10 | `recall_formatters.py` | 198 | Pure output formatting. |

### Layer 1 -- Standalone CLIs (no scripts.core deps, independent)

| Stage | File | Lines | Notes |
|-------|------|-------|-------|
| S11 | `extract_workflow_patterns.py` | 330 | Parses JSONL, no DB. |
| S12 | `generate_mini_handoff.py` | 469 | Parses JSONL, no DB. |
| S13 | `confidence_calibrator.py` | 419 | Uses asyncpg directly (not via db/). |
| S14 | `backfill_sessions.py` | 248 | Uses psycopg2 directly. |
| S15 | `backfill_archive.py` | 168 | Uses psycopg2 directly. |
| S16 | `backfill_learnings.py` | 491 | Uses subprocess + psycopg2. |
| S17 | `artifact_index.py` | 1129 | Too large -- must split. No scripts.core imports. |
| S18 | `artifact_mark.py` | 306 | No scripts.core imports. |
| S19 | `artifact_query.py` | 530 | Large -- split if needed. No scripts.core imports. |

### Layer 2 -- Middle Integrators (depend on Layer 0/1)

| Stage | File | Lines | Depends On |
|-------|------|-------|------------|
| S20 | `query_expansion.py` | 282 | `db/postgres_pool.py` |
| S21 | `recall_backends.py` | 619 | `db/postgres_pool.py`, `db/embedding_service.py`, `query_expansion.py` |
| S22 | `pattern_batch.py` | 500 | `pattern_detector.py`, `db/postgres_pool.py` |

### Layer 3 -- Heavy Integrators (depend on Layer 2)

| Stage | File | Lines | Depends On |
|-------|------|-------|------------|
| S23 | `store_learning.py` | 501 | `db/embedding_service.py`, `db/memory_factory.py`, `confidence_calibrator.py`, `recall_formatters.py` |
| S24 | `recall_learnings.py` | 429 | `recall_backends.py`, `recall_formatters.py`, `db/postgres_pool.py`, `query_expansion.py`, `reranker.py` |
| S25 | `push_learnings.py` | 296 | `db/postgres_pool.py`, `recall_formatters.py`, `recall_learnings.py` |
| S26 | `re_embed_voyage.py` | 266 | `db/embedding_service.py`, `db/postgres_pool.py` |
| S27 | `memory_metrics.py` | 588 | `db/postgres_pool.py` |
| S28 | `memory_feedback.py` | 273 | `db/postgres_pool.py` |
| S29 | `pattern_report.py` | 386 | `db/postgres_pool.py` |

### Layer 4 -- Orchestrator (last)

| Stage | File | Lines | Depends On |
|-------|------|-------|------------|
| S30 | `memory_daemon.py` | 1140 | `confidence_calibrator.py`, `extract_workflow_patterns.py`, `store_learning.py`, `generate_mini_handoff.py`, spawns `pattern_batch.py` as subprocess. Too large -- must split. |

## Per-Stage Workflow

Each stage follows the Large Plan Workflow from CLAUDE.md:

1. **Worktree**: `git worktree add .worktrees/refactor/tdd-fp-<filename> -b refactor/tdd-fp-<filename>`
2. **Red**: Write failing tests first (AAA pattern, F.I.R.S.T. qualities)
3. **Green**: Implement minimum code to pass
4. **Refactor**: Split pure logic from I/O, apply FP principles
5. **Adversarial review**: 3 rounds of `/codex:adversarial-review --wait --scope branch --base refactor/tdd-fp-<filename>`
6. **Security review**: `/security` on changed files
7. **PR**: `gh pr create` with body-file
    - PR Title will be in the format of 
    - {stage}: TDD+FP refactor of {file}
    - (example: S13: TDD+FP refactor of confidence_calibrator.py)
8. **AI reviewer cycles**: 2 rounds addressing Copilot/CodeRabbit comments
9. **Merge + cleanup**: Remove worktree, update this plan

## FP Refactoring Checklist (per file)

- [ ] Separate pure logic (`_core.py` suffix if split needed) from I/O handlers
- [ ] Functions under 20 lines where possible
- [ ] No global mutable state
- [ ] Return values instead of mutating arguments
- [ ] Comprehensions/higher-order functions over imperative loops for transforms
- [ ] Type hints on all function signatures
- [ ] Error handling: return Option/Result types in core, exceptions only in handlers
- [ ] Backwards-compatible re-exports if splitting files (`from .foo_core import *`)

## Files Requiring Splits (>500 lines post-refactor)

| File | Lines | Split Strategy |
|------|-------|---------------|
| `db/embedding_service.py` | 800 | `db/embedding_service.py` (interface) + `db/embedding_providers.py` (implementations) |
| `db/memory_service_pg.py` | 1292 | `db/memory_service_pg.py` (core) + `db/memory_service_queries.py` (SQL/queries) |
| `artifact_index.py` | 1129 | `artifact_index.py` (CLI) + `artifact_index_core.py` (pure logic) |
| `memory_daemon.py` | 1140 | `memory_daemon.py` (orchestration) + `memory_daemon_extractors.py` (extraction logic) |

## Parallelism

Stages within the same layer have no mutual dependencies and can be worked on in parallel by independent sessions:

- **Layer 0**: S3-S5 in parallel (S6 after S3, S7 after S4)
- **Layer 1 libs**: S8-S10 in parallel
- **Layer 1 CLIs**: S11-S19 all in parallel (and parallel with Layer 0)
- **Layer 2**: S20-S22 after their Layer 0/1 deps complete
- **Layer 3**: S23-S29 after their Layer 2 deps complete
- **Layer 4**: S30 last

## Verification

After each stage:
1. `uv run pytest tests/test_<filename>.py -x` -- new tests pass
2. `uv run pytest tests/ -x` -- no regressions
3. `uv run ruff check scripts/core/<filename>` -- lint clean
4. `uv run ruff check scripts/core/` -- no new lint issues introduced
5. Imports from other files still resolve (backwards compat)

## Source Reference

- Scout dependency graph: `.claude/cache/agents/scout/output-20260402-dependency-graph.md`
- Branch naming: `refactor/tdd-fp-<filename-without-extension>`
