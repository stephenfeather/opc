# OPC Commands Reference

A complete catalog of everything OPC exposes as a command. There are **three surfaces**, each with a different audience:

| Surface | Who calls it | How |
|---------|--------------|-----|
| **opc-memory MCP tools** | Claude / agents (in-session) | `mcp__opc-memory__<tool>` |
| **`opc` CLI** | Humans (terminal) | `opc <command>` |
| **Raw scripts** | Power users, hooks, cron | `uv run python scripts/...` |

The MCP tools and the `opc` CLI are two faces of the same memory system — the MCP server is the agent-facing entrypoint, and `opc` is the human-facing twin. Most `opc` subcommands shell out to the same argparse scripts the MCP tools wrap. The raw scripts include everything else: maintenance, benchmarks, migrations, external-service wrappers, and experiments that aren't registered in either.

> **Note on the MCP server source.** The `opc-memory` MCP server is defined in a separate repo at `~/Tools/opc-memory-mcp/main.py`, not in this repository. It imports and calls the same `scripts/core/` logic documented below.

See also: [recall API reference](recall-api-reference.md) for deep flag docs on recall/store/curation, [database reference](database-reference.md) for table schemas, and [knowledge-graph.md](knowledge-graph.md) for the KG layer.

---

## How to invoke

```bash
# Human CLI dispatcher — discover everything
opc --help
opc <command> --help          # per-command help

# Raw script (equivalent to most opc commands)
uv run python scripts/core/recall_learnings.py --query "..." --k 5

# Some scripts need the repo root on PYTHONPATH (flagged below)
PYTHONPATH=. uv run python scripts/core/push_learnings.py

# Module-style scripts
uv run python -m scripts.core.type_affinity --refresh --model-label local

# MCP tools are called by agents in-session, e.g.
#   mcp__opc-memory__recall_learnings(query="...", k=5)
```

---

## 1. opc-memory MCP tools (agent-facing)

18 tools, defined in `~/Tools/opc-memory-mcp/main.py`. These are what Claude calls during a session.

### Recall & Store

#### `recall_learnings`
Semantic search over stored learnings (hybrid RRF by default; text-only or vector-only modes available).

| Param | Type | Req | Notes |
|-------|------|-----|-------|
| `query` | str | ✅ | Search query |
| `k` | int | | Number of results (default 5) |
| `text_only` | bool | | Fast text search, no embeddings |
| `vector_only` | bool | | Pure vector search |
| `threshold` | float | | Similarity threshold 0.0–1.0 (default 0.2) |
| `project` | str | | Project name to boost relevance (auto-detected) |
| `tags` | str | | Space-separated tags to boost via reranker |
| `tags_strict` | bool | | Hard-filter to results sharing ≥1 tag |
| `json_full` | bool | | Include `recall_count`, `pattern_strength`, `pattern_tags` |
| `structured` | bool | | Group results by `learning_type` |

#### `store_learning`
Store a learning with embeddings for future recall.

| Param | Type | Req | Notes |
|-------|------|-----|-------|
| `content` | str | ✅ | The learning text |
| `learning_type` | str | | `WORKING_SOLUTION` (default), `ARCHITECTURAL_DECISION`, `CODEBASE_PATTERN`, `FAILED_APPROACH`, `ERROR_FIX`, `USER_PREFERENCE`, `OPEN_THREAD` |
| `session_id` | str | | Default `"mcp-session"` |
| `context` | str | | What this relates to |
| `tags` | str | | Comma-separated tags |
| `confidence` | str | | `high` / `medium` (default) / `low` |
| `supersedes` | str | | UUID of older learning this replaces |
| `project` | str | | Auto-detected; required when storing for a different project |

### Feedback

#### `store_feedback`
Record a helpful / not-helpful signal on a recalled learning to weight future recall.

| Param | Type | Req | Notes |
|-------|------|-----|-------|
| `learning_id` | str | ✅ | UUID of the learning |
| `helpful` | bool | ✅ | `true` = helpful |
| `session_id` | str | | Default `"cli"` |
| `context` | str | | Free-text explanation |
| `source` | str | | e.g. `"manual"` (default), `"auto"` |

#### `get_feedback`
Get all individual feedback entries for one learning. Param: `learning_id` (str, required).

#### `feedback_summary`
Aggregate feedback stats (totals, helpful/not-helpful breakdown, helpfulness rate, top-rated learnings). No parameters.

### Artifacts (handoffs, plans, continuity)

#### `index_artifacts`
Index handoff files, plans, and continuity ledgers into the artifact database.

| Param | Type | Req | Notes |
|-------|------|-----|-------|
| `mode` | str | | `all` (default), `handoffs`, `plans`, `continuity`, `file` |
| `file_path` | str | | Required when `mode=file` (absolute path preferred) |
| `project_dir` | str | | Base for resolving relative `file_path` |

#### `mark_handoff`
Mark a handoff with an outcome for success-rate tracking.

| Param | Type | Req | Notes |
|-------|------|-----|-------|
| `outcome` | str | ✅ | `SUCCEEDED`, `PARTIAL_PLUS`, `PARTIAL_MINUS`, `FAILED` |
| `handoff_id` | str | | Specific handoff ID (uses latest if empty) |
| `notes` | str | | Notes about the outcome |

#### `query_artifacts`
Search the Context Graph for precedent from past sessions.

| Param | Type | Req | Notes |
|-------|------|-----|-------|
| `query` | str | ✅ | Search query |
| `artifact_type` | str | | `handoffs`, `plans`, `continuity`, `all` (default) |
| `outcome` | str | | Filter by outcome |
| `limit` | int | | Max results (default 5) |
| `with_content` | bool | | Include full file content |
| `by_span_id` | str | | Fetch a handoff by Braintrust `root_span_id` |

### Documents (RAG collections)

#### `query_documents`
Semantic search over ingested document collections. Default is global-scope only; restricted collections require an explicit `collection` name.

| Param | Type | Req | Notes |
|-------|------|-----|-------|
| `text` | str | ✅ | Question / search text |
| `collection` | str | | Target a specific collection (only way to reach `restricted` collections) |
| `limit` | int | | Max results 1–100 (default 8) |

#### `list_document_collections`
List all registered document collections and their ingest stats. No parameters.

#### `scan_document_collection`
Incrementally ingest one collection or all of them (hash-based, idempotent). Provide either `name` (str) or `scan_all` (bool) — mutually exclusive.

#### `create_document_collection`
Register a new document collection.

| Param | Type | Req | Notes |
|-------|------|-----|-------|
| `name` | str | ✅ | Unique collection name |
| `path` | str | ✅ | Absolute path to the folder |
| `scope` | str | | `global` (default) or `restricted` |
| `extensions` | str | | Comma-separated (default `.pdf,.docx,.txt,.csv,.md,.html,.htm,.xml`) |
| `ocr` | bool | | Stored but ignored in v1 |

### Patterns & Metrics

#### `detect_patterns`
Cluster stored learnings and identify recurring cross-session patterns.

| Param | Type | Req | Notes |
|-------|------|-----|-------|
| `dry_run` | bool | | Preview without writing |
| `report` | bool | | Show last run's report instead of detecting |
| `min_confidence` | float | | Threshold 0.0–1.0 (default 0.3) |
| `use_llm` | bool | | Use LLM classifier instead of heuristic |

#### `memory_metrics`
System-wide memory statistics (counts, distributions, tags, stale rates, temporal range). Optional `period` (str): ISO date range `START:END`, e.g. `2026-03-01:2026-03-31`; omit for all-time.

### Daemon

#### `start_daemon` / `stop_daemon` / `daemon_status`
Start, stop, or check the background memory-extraction daemon. No parameters. `daemon_status` returns the PID if running.

---

## 2. The `opc` CLI (human-facing)

The `opc` dispatcher (`scripts/core/cli.py`) wraps 29 argparse scripts. Run `opc --help` to discover all of them, or `opc <command> --help` for per-command flags. Grouped by category below.

### Recall & Store

| Command | Script | Key flags |
|---------|--------|-----------|
| `opc recall` | `recall_learnings.py` | `--query/-q`, `--k`, `--provider {local,voyage}`, `--json` / `--json-full`, `--text-only` / `--vector-only`, `--threshold`, `--recency`, `--tags`, `--no-rerank`, `--llm-rerank`, `--project`, `--structured`, `--source` |
| `opc store` | `store_learning.py` | `--session-id` (req), `--type {WORKING_SOLUTION,…}`, `--content`, `--context`, `--tags`, `--confidence {high,medium,low}`, `--project`, `--auto-classify` |
| `opc feedback` | `memory_feedback.py` | subcommands: `store`, `get`, `summary` |
| `opc push` | `push_learnings.py` | `--project`, `--k`, `--json`, `--no-record`, `--max-chars` — surfaces never-recalled high-value learnings. *Requires `PYTHONPATH=.`* |

### Artifacts

| Command | Script | Key flags |
|---------|--------|-----------|
| `opc artifact query` | `artifact_query.py` | `[query]`, `--type {handoffs,plans,continuity,all}`, `--outcome`, `--limit`, `--json`, `--by-span-id`, `--with-content` |
| `opc artifact mark` | `artifact_mark.py` | `--handoff ID` / `--latest` / `--get-latest-id`, `--outcome {SUCCEEDED,PARTIAL_PLUS,PARTIAL_MINUS,FAILED}`, `--notes` |
| `opc artifact index` | `artifact_index.py` | `--handoffs`, `--plans`, `--continuity`, `--all`, `--file FILE` (single-file hook mode), `--json` |

### Patterns & Metrics

| Command | Script | Key flags |
|---------|--------|-----------|
| `opc pattern detect` / `opc pattern batch` | `pattern_batch.py` | `--dry-run`, `--report`, `--min-size`, `--min-samples`, `--min-confidence`, `--use-llm`, `--verbose` |
| `opc pattern report` | `pattern_report.py` | `--run-id`, `--json`, `--summary` |
| `opc metrics` | `memory_metrics.py` | `--json` / `--human`, `--period YYYY-MM-DD:YYYY-MM-DD` |
| `opc duplicate-density` | `duplicate_density.py` | `--output PNG`, `--dump-pairs THRESHOLD`, `--text-only`, `--limit` |
| `opc memory-review` | `memory_review.py` | `[project]`, `--min-recall`, `--threshold`, `--ef-search`, `--merge-timeout`, `--promote-only`, `--cleanup-only` — read-only |
| `opc memory-apply` | `memory_apply.py` | `[project]`, `--ids`, `--manifest`, `--execute` (dry-run default), `--merge`, `--pair ID_A:ID_B`, `--archive`, `--unpromote` |
| `opc confidence calibrate` | `confidence_calibrator.py` | `--session-id`, `--backfill`, `--dry-run`, `--json` |
| `opc type-affinity` | `type_affinity.py` | `--refresh`, `--model-label` (req), `--cache-path` — run as `python -m scripts.core.type_affinity` |

### Daemon

| Command | Script | Key flags |
|---------|--------|-----------|
| `opc daemon` | `memory_daemon.py` | `{start,stop,status}`, `--debug` |

### Extraction

| Command | Script | Key flags |
|---------|--------|-----------|
| `opc extract session` | `extract_session.py` | `--session-id` (req), `--dry-run`, `--model`, `--max-turns`, `--timeout`, `--verbose`. *Requires `PYTHONPATH=.`* |
| `opc extract thinking` | `extract_thinking_blocks.py` | `--jsonl` (req), `--filter`, `--output`, `--format {text,json}`, `--stats` |
| `opc extract workflow` | `extract_workflow_patterns.py` | `--jsonl` (req), `--format`, `--output`, `--patterns-only`, `--stats`, `--max-entries`, `--redact` |
| `opc handoff` | `generate_mini_handoff.py` | `--jsonl`, `--state-file`, `--session-id` (req), `--project-dir` (req), `--output`, `--format {yaml,json}` |

### Maintenance

| Command | Script | Key flags |
|---------|--------|-----------|
| `opc backfill sessions` | `backfill_sessions.py` | `--dry-run`, `--batch-size`, `--all`, `--after YYYY-MM-DD` |
| `opc backfill learnings` | `backfill_learnings.py` | `--dry-run`, `--limit`, `--project`, `--workers`, `--skip-no-db`. *Requires `PYTHONPATH=.`* |
| `opc backfill kg` | `backfill_kg.py` | `--dry-run`, `--limit`, `--since`, `--memory-id`, `--project`, `--recheck-no-entities`, `--batch-size`, `--max-consecutive-errors` |
| `opc backfill archive` | `backfill_archive.py` | `--dry-run` — archives JSONL files to S3 |
| `opc re-embed voyage` | `re_embed_voyage.py` | `--model {voyage-3,voyage-3-large,voyage-code-3}`, `--batch-size`, `--dry-run`, `--retry-failed` |

---

## 3. Raw scripts (not registered in `opc`)

These have working CLI entrypoints but are not in the `opc` dispatcher. Run them with `uv run python <path>`. Several need the repo root on the path (`PYTHONPATH=.`); flagged inline.

### `scripts/core/` — path-only

| Script | Description | Flags |
|--------|-------------|-------|
| `track_stale_rate.py` | Insert a daily stale-rate data point into the DB | none |
| `documents/cli.py` | Doc-RAG collection CLI (the `opc-docs` tool) | subcommands `create`, `scan`, `list`, `query`; global `--json` |

### `scripts/mcp/` — external-service wrapper CLIs

Each calls an external API and prints results. None are in `opc`.

| Script | Description | Key flags |
|--------|-------------|-----------|
| `firecrawl_scrape.py` | Web scraping via Firecrawl | `--url` or `--search`, `--format {markdown,html,text}`, `--limit`, `--main-only` |
| `github_search.py` | GitHub code/repo/issue/PR search | `--query` (req), `--type {code,repos,issues,prs}`, `--owner`, `--repo`, `--limit` |
| `morph_search.py` | Codebase search via Morph/WarpGrep | `--search`, `--path`, `--edit`, `--content` |
| `morph_apply.py` | Apply code edits via Morph Fast Apply | `--file` (req), `--instruction` (req), `--code_edit` (req), `--dry-run` |
| `perplexity_search.py` | AI search via Perplexity (5 modes) | `--ask` / `--search` / `--research` / `--reason` / `--deep` (exclusive), `--model`, `--recency`, `--max-results`, `--domains` |
| `nia_docs.py` | Nia API client | subcommands `oracle`, `search`, `repos`, `sources`, `papers`, `context` |

### `scripts/setup/`

| Script | Description | Flags |
|--------|-------------|-------|
| `wizard.py` | Interactive OPC setup wizard (prereqs, DB, API keys, `.env`) | interactive |
| `docker_setup.py` | Manage Docker/Podman stack (start, health, migrations) | interactive / positional |
| `math_features.py` | Toggle math feature dependencies | `--install`, `--status`, `--verify`, `--lean` |
| `personalization.py` | Import/export/generate user preference config | `--jsonl-dir`, `--import-file`, `--export-file`, `--configure` |
| `update.py` | OPC update wizard (git pull + migration steps) | interactive |

(`claude_integration.py` and `embedded_postgres.py` here are import-only libraries, not commands.)

### `scripts/benchmarks/` — reranker tuning (require `PYTHONPATH=.`)

| Script | Description | Key flags |
|--------|-------------|-----------|
| `run_rerank_benchmark.py` | A/B benchmark: reranker on vs off | `--queries`, `--output`, `--verbose`, `--sweep`, `--split {train,holdout,all}`, `--llm-rerank` |
| `tune_loop.py` | Automated keep/discard tuning loop over weight sweep | `--queries`, `--split`, `--latency-budget-ms`, `--journal`, `--apply` (writes winning weights to `opc.toml`) |
| `bootstrap_golden.py` | Interactive golden-set annotation tool | `--queries`, `--force` |
| `backfill_golden_hashes.py` | One-time migration: re-anchor golden set from UUIDs to content hashes | `--queries`, `--dry-run` |
| `mine_feedback_labels.py` | Mine recall-feedback events into golden-set label candidates | `--min-judgments`, `--window-hours`, `--output`, `--merge-into` |

### `scripts/migrations/`

| Script | Description | Flags |
|--------|-------------|-------|
| `backfill_project_column.py` | Backfill `archival_memory.project` (three-tier strategy) | `--tier {1,2,3,all}`, `--dry-run`, `--verbose` |
| `normalize_project_values.py` | Collapse fragmented project-name variants to canonical forms (#130) | `--apply` (dry-run default) |

### `scripts/tldr/`

| Script | Description | Flags |
|--------|-------------|-------|
| `build_symbol_index.py` | Build an AST-based symbol index for a path | `[path]`, `--hook` |
| `index_incremental.py` | Fast incremental indexer for new/changed files | `--dry-run`, `--timeout`, `--hook` |

### `scripts/client/`

| Script | Description |
|--------|-------------|
| `status.py` | Cross-platform Claude Code status line (context %, branch, current goal); one-line output |
| `tldr_stats.py` | TLDR token-savings dashboard (session cost, all-time savings, sparkline) |

### `scripts/` (top-level)

| Script | Description | Key flags |
|--------|-------------|-----------|
| `braintrust_analyze.py` | Analyze Braintrust sessions (stats, loop detection, token trends, LLM classify) | `--last-session`, `--sessions N`, `--agent-stats`, `--skill-stats`, `--detect-loops`, `--replay`, `--weekly-summary`, `--token-trends`, `--learn`, `--review`, `--reclassify`, `--project`, `--score` |
| `observe_agents.py` | Query live agent state (memory, blackboard, tasks, outputs) | `--what {memory,blackboard,tasks,outputs,all}`, `--query`, `--session-id`, `--json`, `--limit` |
| `recall_temporal_facts.py` | Semantic recall from the temporal-facts table | `--query` (req), `--k`, `--rerank`, `--provider`, `--rerank-provider`, `--session-id`, `--all-sessions` |
| `ragie_query.py` | Semantic search via the Ragie document store | `--query` (req), `--partition`, `--top-k`, `--rerank`, `--json` |
| `ragie_upload.py` | Upload documents to Ragie | `--file`, `--dir`, `--partition`, `--metadata`, `--extension` |
| `ragie_status.py` | Check Ragie ingestion status | `--doc-id`, `--list`, `--partition` |
| `repoprompt_async.py` | RepoPrompt async operations via tmux | `--action {start,status,result,kill}`, `--task`, `--command`, `--workspace`, `--timeout`, `--no-cleanup` |
| `loogle_search.py` | Mathlib type-signature search via Loogle | `[query]`, `--json`, `--status`, `--stop` |
| `loogle_server.py` | Keep a Loogle process warm for fast queries | daemon, no flags |
| `ast_grep_find.py` | AST-grep code search via MCP | `--pattern` (req), `--language`, `--path`, `--glob`, `--replace`, `--dry-run` — must run from repo root |
| `qlty_check.py` | Code-quality checks via Qlty MCP | `--check`, `--metrics`, `--smells`, `--fmt`, `--init`, `--plugins` — must run from repo root |
| `multi_tool_pipeline.py` | Demo: chain MCP tools in a workflow | `--repo-path`, `--max-commits` — must run from repo root |
| `research_implement_pipeline.py` | Research → implement MCP chaining pipeline | `--topic` (req), `--target-dir`, `--dry-run`, `--verbose` — must run from repo root |
| `test_research_pipeline.py` | Test harness for the research pipeline | `--test {…,all}`, `--keep-sandbox`, `--verbose`, `--sandbox-dir` |
| `benchmark_daemon.py` | Benchmark TLDR daemon vs CLI latency | hardcoded project path |
| `benchmark_tokens.py` | Benchmark token savings: raw files vs TLDR | none |

> Scripts marked *"must run from repo root"* import the in-repo MCP runtime and fail with an import error unless launched from the repository root (or with `PYTHONPATH=.`).

### Library-only directories (no commands)

- `scripts/cc_math/` — compute backends (`sympy_compute.py`, `numpy_compute.py`, `scipy_compute.py`, `mpmath_compute.py`, `pint_compute.py`, `z3_solve.py`, `shapely_compute.py`, `math_router.py`, `math_plot.py`, `math_tutor.py`, …). Imported by the math MCP tools; no `__main__` entrypoints.
- `scripts/` import-only helpers: `claude_spawn.py`, `stream_monitor.py`.

---

## Adding a new command

When you add a user-runnable script to `scripts/core/`, register it in the `opc` dispatcher (`scripts/core/cli.py`) and confirm `opc --help` lists it — keep dispatcher changes additive so existing `uv run python scripts/core/<script>.py` invocations from hooks, skills, and MCP wrappers keep working. When you change CLI flags or output formats for `recall_learnings.py`, `store_learning.py`, or other core scripts, update [docs/recall-api-reference.md](recall-api-reference.md) and this file.

### Unregistered CLIs worth promoting

These have real CLI value but aren't in `opc` yet — candidates for future registration:

| Script | Suggested command |
|--------|-------------------|
| `scripts/core/track_stale_rate.py` | `opc track stale-rate` |
| `scripts/core/documents/cli.py` | `opc docs {create,scan,list,query}` |
| `scripts/braintrust_analyze.py` | `opc braintrust` |
| `scripts/observe_agents.py` | `opc observe` |
| `scripts/recall_temporal_facts.py` | `opc recall-temporal` |
| `scripts/ragie_query.py` / `ragie_upload.py` / `ragie_status.py` | `opc ragie {query,upload,status}` |
| `scripts/setup/math_features.py` | `opc setup math` |
| `scripts/setup/personalization.py` | `opc setup personalize` |
| `scripts/migrations/*` | `opc migrate {project-column,normalize-projects}` |
| `scripts/tldr/*` | `opc tldr {build-index,index}` |
