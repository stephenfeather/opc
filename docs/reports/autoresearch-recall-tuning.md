# Borrowing autoresearch's loop to tune OPC recall

**Date:** 2026-06-25
**Subject:** How `karpathy/autoresearch`'s process maps onto OPC's lookup / query / suggestion tuning
**Sources:** `https://github.com/karpathy/autoresearch` (README + `program.md`); OPC code verified against this repo on 2026-06-25.

---

## TL;DR

autoresearch is not a research idea — it's a **loop discipline**: one frozen harness, one comparable metric, one modifiable surface, a fixed budget, an append-only journal, and a mechanical keep/discard rule the agent runs forever without asking permission. Its only genuinely hard requirement is a **cheap, automated, trustworthy metric** (`val_bpb`, which generates its own labels from training loss).

OPC is closer to this than expected. We already have the modifiable surface (`RerankerConfig` knobs in `opc.toml`), a real metric harness (`scripts/benchmarks/run_rerank_benchmark.py` with P@k / NDCG / MRR, a weight-sweep mode, **and a three-arm LLM-vs-reranker-vs-raw comparison**), and a secondary constraint to trade against (recall latency). What we are missing is exactly the part autoresearch makes non-negotiable: a **trustworthy, self-maintaining metric** and the **automated loop + journal** wrapped around it.

The single highest-leverage move is to **close the feedback dead-end** — `memory_feedback` is currently a reporting sink with zero pathway into scoring (verified: no `feedback` reference in `reranker.py` or `recall_learnings.py`). Turn it into the labeling source for a growing golden set, and our manual golden benchmark becomes an autoresearch-style cheap automated metric. Everything else (the loop, the journal, the keep/discard rule) is mechanical once the metric is trustworthy.

---

## 1. What autoresearch actually does (distilled to primitives)

Stripping away the ML-research framing, the process is six primitives:

| Primitive | autoresearch implementation |
|---|---|
| **Single comparable metric** | `val_bpb` — validation bits/byte, *vocab-size-independent* so every variant compares fairly. Lower = better. |
| **Frozen harness** | `prepare.py` is read-only: fixed data prep, dataloader, eval. The agent cannot touch how it's scored — so it can't cheat the metric. |
| **One modifiable surface** | Only `train.py` may change (model, optimizer, hyperparameters). No new dependencies. |
| **Fixed budget** | 5 min wall-clock per run → ~12 experiments/hr, ~100 overnight. A *secondary* metric, `peak_vram_mb`, is tracked alongside. |
| **Append-only journal** | `results.tsv`: `commit · val_bpb · memory_gb · status · description`, where `status ∈ {keep, discard, crash}`. Untracked by git. One commit per experiment. |
| **Mechanical decision rule** | Improved metric → keep the commit; else `git reset`. Tiebreaker: *all else equal, simpler is better*. Agent never stops, never asks. |

The scientific rigor (one variable at a time, baseline-relative, log everything) lives in `program.md`, the human-editable steering prompt — **not** baked into code. That separation matters: the *harness* enforces fairness, the *prompt* enforces method.

The key insight: autoresearch works because `val_bpb` is **cheap, objective, and self-generating** (it falls out of training). The agent can run 100 experiments overnight because no human is in the scoring loop.

---

## 2. Where OPC already matches (we're ~60% there)

We already have analogs to four of the six primitives:

- **Modifiable surface ✓** — `RerankerConfig` exposes the signal weights + `rrf_scale_factor` (= 25 in `opc.toml:89`) + the SQL recall boost, all overridable via `opc.toml`. `query_expansion.py` adds expansion on/off and top-N-terms. This is a clean, declarative knob surface — our `train.py`.
- **Metric harness ✓ (partial)** — `scripts/benchmarks/run_rerank_benchmark.py` computes P@k, NDCG@k, MRR and has a **weight-sweep mode** that fetches results once then reruns the in-process reranker across many `RerankerConfig` instances (the expensive DB round-trip happens once). It **already includes the LLM-selector as a third arm (Phase E, `--llm-rerank`)**, reporting LLM-vs-reranker precision deltas and per-query wall-clock latency. Historical A/B results live under `scripts/benchmarks/`.
- **Secondary constraint ✓** — we have a natural `peak_vram_mb` analog: **recall latency**. The hook runs on a ~5s `spawnSync` budget; the LLM selector (`scripts/core/llm_selector.py`) carries `LLM_SELECTOR_TIMEOUT = 10.0` and is deliberately **gated off the hook path** so its latency only lands on CLI/benchmark callers. Latency is the constraint we must trade quality against — exactly autoresearch's memory/quality trade.
- **Frozen-harness instinct ✓** — the model-filter clause (`embedding_model = 'voyage-code-3'`) and the "fetch once, rerun in-process" sweep design already show the right reflex: hold the expensive/variable parts fixed so comparisons are fair.

---

## 3. The gaps (the primitives we lack)

| Missing primitive | Current OPC state | Consequence |
|---|---|---|
| **Trustworthy metric** | Golden set is **UUID/`golden_ids`-anchored** to one DB instance (`bootstrap_golden.py` saves raw `golden_ids`), and small. Fresh DB breaks scoring; no re-anchoring. | Metric is fragile and tiny — a multi-weight sweep *will overfit*. |
| **Self-maintaining labels** | `memory_feedback` collects helpful/unhelpful signals but `memory_feedback.py` is **aggregation/reporting only** (`aggregate_feedback`, `compute_helpfulness_rate`); **zero pathway into any scoring function** (no `feedback` reference in `reranker.py` or `recall_learnings.py`). | We collect the exact data autoresearch wishes it had, and only report on it. |
| **The automated loop** | Benchmark + Phase-E comparison are **run by hand**; no CI, no iteration driver, no keep/discard automation. | No "100 experiments overnight." Even the LLM-arm comparison is a manual one-off. |
| **The journal** | A/B results exist as ad-hoc files; no structured `results.tsv`-style log of `config · metrics · keep/discard · why`. | No accumulating record of what was tried and rejected → we re-try dead ends. |
| **Coverage of the surface** | Sweep tunes reranker weights; Phase E measures the LLM arm. But **query-expansion on/off** is not ablated in the loop, and the **hook intent-extraction / conversational gate** has no eval at all. | The knobs that most affect the *suggestion* path are unmeasured. |
| **Staleness detection** | Type centroids (`infer_query_type`) degrade silently as the corpus evolves; no detection. | Metric quality rots without anyone noticing. |

---

## 4. Proposal: the OPC recall-tuning loop

Build a thin autoresearch-style harness around the benchmark we already have. Each piece maps directly to an autoresearch primitive.

### 4.1 The metric (`val_bpb` analog)

Pick **one headline number**: `NDCG@5` on the golden set. Report `P@5`, `MRR`, and **p95 recall latency** alongside it (latency is our `peak_vram_mb` — a co-metric we must not regress). A change is "better" only if NDCG@5 improves *and* p95 latency stays within budget. This gives the loop a single, fair, objective decision variable. The Phase-E machinery already emits per-query latency, so the co-metric plumbing largely exists.

### 4.2 The frozen harness (`prepare.py` analog)

The loop **may not edit**: the golden set, the metric definitions, or the retrieval SQL contract. To make retrieval/expansion changes comparable (they need DB round-trips, unlike the in-process reranker sweep), pin a **frozen DB snapshot** as the eval corpus — autoresearch's "one-time data prep." Same corpus, same queries, every run. The harness lives in a read-only path the tuning agent is instructed never to touch.

### 4.3 The modifiable surface (`train.py` analog)

A single `opc.toml` profile section the loop is allowed to mutate — **one knob per experiment**:
- reranker signal weights + `rrf_scale_factor` + SQL recall-boost coefficient
- query expansion: on/off, top-N terms, IDF refresh cadence
- LLM-selector enable + model + timeout (already implemented; already measurable via Phase E) — the loop's job is to put that measurement on a keep/discard footing, deciding when the selector's latency earns its precision gain

### 4.4 The journal (`results.tsv` analog)

Append-only, untracked, one row per experiment:

```
config_hash   ndcg@5   p@5   mrr   p95_latency_ms   status   description
```

`status ∈ {keep, discard, crash}`. `description` states the single variable changed and the hypothesis. This is the institutional memory autoresearch relies on — and it should feed back into OPC's own memory system as `WORKING_SOLUTION` / `FAILED_APPROACH` learnings so the tuning history is itself recallable.

### 4.5 The decision rule

Mechanical, copied verbatim from `program.md`:
- NDCG@5 improves within latency budget → **keep** (commit the `opc.toml` profile).
- Else → **discard** (revert).
- Tiebreaker: **all else equal, simpler is better** — equal NDCG with fewer signals / lower latency wins. (We already paid for this lesson: the #228 LLM-rerank adversarial rounds repeatedly found that broad try/except "simplifications" silently degraded to the reranker and corrupted the signal — the simplicity rule is about *fewer moving parts that earn their keep*, not blanket fallbacks.)

### 4.6 Autonomy & budget

autoresearch's "never stop, ~100 overnight" maps to a **bounded sweep or a `/loop`-driven self-paced run**: enumerate single-variable perturbations around the current baseline, run each through the harness, journal it, keep/revert. The latency budget caps per-experiment cost the way the 5-min wall-clock does. Unlike autoresearch we should **not** run unbounded — see the overfitting risk below.

---

## 5. The killer synergy: feedback → auto-labels

This is where the analogy pays off most. autoresearch's superpower is a metric that **generates its own labels** (training loss needs no human). OPC's blocker is that relevance judgments need humans — *except we are already collecting them and only reporting on them.*

`memory_feedback` (helpful/unhelpful, with context) is a stream of human relevance judgments tied to real queries. Wire it as the **labeling source** for the golden set:
- A `helpful=true` on a recalled ID for a given query intent → a positive relevance judgment.
- `helpful=false` → a hard negative (the most valuable label for ranking).
- Re-anchor by **(query intent, content hash)** instead of raw `golden_ids`/UUID, fixing the instance-specific fragility in one move.

This converts our metric from "fragile manual set" into autoresearch's "cheap automated and self-growing." It simultaneously closes the dead-end feedback loop, and gives the tuning loop fresh labels as the corpus and usage evolve — which also addresses centroid/label staleness.

---

## 6. Risks & disanalogies (where the metaphor breaks)

1. **Tiny eval set → overfitting.** autoresearch trains against a large data stream; a multi-weight sweep against a small golden set will memorize it. *Mitigation:* grow the set via feedback (§5), hold out a validation split, and treat a "win" as significant only across a cross-validated set. Do not run unbounded overnight on a small set — you'll tune to noise.
2. **Human-label dependence.** Our metric is only as honest as the feedback labels. autoresearch's loss can't be gamed; relevance judgments can be sparse or biased. *Mitigation:* require N judgments per query before it counts; track label coverage.
3. **DB reproducibility.** The reranker sweep is reproducible (fetch-once), but retrieval/expansion *and the live LLM arm* hit external state. *Mitigation:* the frozen snapshot in §4.2 for the retrieval leg; for the LLM arm, pin model + temperature and treat its latency/precision as a distribution, not a point.
4. **The suggestion path is barely measured.** The hook's intent extraction and conversational-turn gate have no eval (false-positive/negative rate unknown). Borrowing autoresearch here means *first building the metric* (a labeled set of prompts → should-surface / should-stay-silent) before any loop can tune it. This is greenfield, not a knob-tune.
5. **Don't let the agent edit the scorer.** autoresearch's whole integrity rests on `prepare.py` being read-only. Our equivalent guardrail: the tuning loop must be structurally barred from editing the golden set, the metric, or the snapshot.

---

## 7. Phased rollout

- **Phase 0 — Metric trust.** Re-anchor golden judgments to (intent, content-hash); wire `memory_feedback` → golden labels; add a held-out split. *Without this, the loop overfits — do not skip.*
- **Phase 1 — Journal + decision rule.** Add the `results.tsv`-style log and the mechanical keep/discard wrapper around the existing reranker sweep. Smallest possible change; reuses `run_rerank_benchmark.py`.
- **Phase 2 — Autonomy (bounded).** Drive single-variable perturbations across the weight surface via a self-paced loop; journal every run; surface the keep-list. Add p95 latency as the co-metric gate.
- **Phase 3 — Fold in the already-measured arms.** Query-expansion ablation and the **already-implemented** LLM selector (Phase E) come under the same automated keep/discard loop — turning today's manual three-arm comparison into a standing decision: enable the selector only where its measured latency buys ranking quality.
- **Phase 4 — Suggestion-path metric.** Build the labeled prompt set for the hook's gate, then tune it the same way.

---

## 8. Concrete next step

The cheapest first cut that proves the whole idea: **Phase 0 + Phase 1 on the reranker sweep only.** Re-anchor the golden set off content hashes, pipe `memory_feedback` rows in as labels, and wrap the existing sweep in a journal + keep/discard rule. That alone turns today's manual, write-only tuning into autoresearch's cheap-metric-plus-journal loop — using code we already have — and closes the feedback dead-end as a side effect.

Everything past that (autonomy, broader surface, suggestion-path metric) is incremental and only worth doing once the metric is trustworthy. autoresearch's lesson in one line: **the loop is trivial; earning a metric you can run 100 times without a human is the whole game.**
