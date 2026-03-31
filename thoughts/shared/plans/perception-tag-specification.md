# Perception Tag Specification

## Current State Analysis

### The Numbers

| Metric | Value |
|--------|-------|
| Total learnings | 2,233 |
| Learnings with `perception` tag | 1,691 (75.7%) |
| Learnings without `perception` | 538 (24.1%) |
| Learnings with no tags at all | 4 (0.2%) |
| `perception` as only tag | 0 |

The `perception` tag is the #1 most frequent tag by a factor of 4.5x (next is `testing` at 16.8%).

### Perception Rate by Learning Type

| Learning Type | With Perception | Total | % Tagged |
|---------------|-----------------|-------|----------|
| CODEBASE_PATTERN | 735 | 896 | 82.0% |
| ERROR_FIX | 603 | 778 | 77.5% |
| WORKING_SOLUTION | 297 | 418 | 71.1% |
| FAILED_APPROACH | 15 | 22 | 68.2% |
| ARCHITECTURAL_DECISION | 28 | 72 | 38.9% |
| USER_PREFERENCE | 12 | 45 | 26.7% |
| OPEN_THREAD | 1 | 5 | 20.0% |

Perception tags are heavily concentrated on the three high-volume types (CODEBASE_PATTERN, ERROR_FIX, WORKING_SOLUTION) which are exactly the types the memory-extractor agent produces most often.

### Sub-Signal Breakdown

Of 1,691 perception-tagged learnings:

| Sub-Signal Tag | Count | % of Perception |
|----------------|-------|-----------------|
| `insight` | 299 | 17.7% |
| `correction` | 209 | 12.4% |
| `debugging` | 173 | 10.2% |
| `realization` | 105 | 6.2% |
| `validation` | 74 | 4.4% |
| **Any sub-signal** | **830** | **49.1%** |
| **No sub-signal** | **861** | **50.9%** |

Half of perception-tagged learnings have no sub-signal explaining *what kind* of perception it was.

### What Non-Perception Learnings Look Like

Learnings without `perception` are typically:
- **Manual stores** via `store_learning.py` called directly by sessions or hooks
- **Pre-extraction learnings** stored before the memory-extractor pipeline existed
- **Project-specific patterns** (fa-wpmcp, bedrock, docker configs)

Average tag count is similar: 4.60 for perception vs 4.69 for non-perception.

## Root Cause

### Primary: Memory-extractor agent prompt hardcodes `perception` on every learning

In `~/.claude/agents/memory-extractor.md`, all example `store_learning.py` calls include `perception` as the first tag:

```bash
--tags "perception,correction,topic"
--tags "perception,insight,topic"
--tags "perception,debugging,approach"
```

The LLM follows these examples and adds `perception` to every learning it stores. Since the memory-extractor is the primary learning creation path (~76% of all learnings), the tag covers ~76% of all learnings.

### Secondary: Overly broad regex pre-filter

`scripts/core/extract_thinking_blocks.py` defines 35 perception signal patterns including extremely common words:
- `\bactually\b` - appears in nearly all technical thinking
- `\bwait,?\b` - common pause/reconsideration word
- `\binteresting\b` - generic observation
- `\bhmm\b` - common thinking filler

These patterns match ~80%+ of all thinking blocks, so almost everything passes the pre-filter.

### Tertiary: No quality gate

There is no validation step between the LLM deciding "this is a perception" and the tag being stored. The LLM is instructed to be selective (Rule 1: "Quality over quantity"), but the examples model always including `perception`.

## What "Perception" Was Intended To Mean

Based on the agent prompt's philosophy section:

> "A point of view is worth 80 IQ points" - Alan Kay

The tag was meant to identify **mental model shifts** -- moments where Claude's understanding changes:
- Realizations: "Oh, X was actually Y"
- Corrections: "I was wrong about..."
- Insights: "The pattern here is..."
- Surprises: "Unexpected that..."

In practice, it became a **source marker** meaning "extracted by the memory-extractor agent from thinking blocks" rather than a semantic signal about the learning's content.

## Proposed Specification

### Option A: Keep as-is, filter from IDF (Quick Fix)

Add `perception` to a static noise-tag exclusion list in pattern detection.

**Impact:**
- Pattern detector already has `detect_noise_tags()` which would catch it dynamically (bottom 10th percentile IDF)
- Does not fix the semantic confusion
- Tag remains in `memory_tags` table, consuming index space
- IDF weighting on other tags remains accurate since perception is excluded

**Effort:** 1 line of code
**Semantic fix:** No

### Option B: Tighten criteria to ~20-30% (Moderate)

Update the memory-extractor prompt to only tag `perception` when a learning represents a genuine mental model shift, not just any extracted learning. Remove `perception` from example tags; instead instruct the LLM to add it only when specific criteria are met.

**Criteria for applying `perception`:**
1. The learning describes a **correction** of a prior belief ("I was wrong", "turns out X not Y")
2. The learning captures a **surprise** ("unexpected", "didn't expect")
3. The learning describes a **realization** that changes approach ("now I see", "the real issue was")

**Do NOT apply when:**
- The learning is a straightforward pattern observation ("SAM looks for requirements.txt in CodeUri")
- The learning is a working solution without any model shift ("use vi.fn() for mocking")
- The learning is a factual error fix without a surprise element ("add --depth flag")

**Impact:**
- Estimated reduction to ~25-35% coverage (the 830 learnings with sub-signals like insight/correction/realization, minus procedural ones)
- Requires re-tagging existing learnings
- IDF weight of `perception` becomes meaningful for recall

**Effort:** Prompt update + migration script
**Semantic fix:** Yes

### Option C: Replace with specific sub-tags (Comprehensive)

Remove `perception` entirely. Replace with the sub-signals that already partially exist:

| New Tag | Replaces | Meaning | Estimated Count |
|---------|----------|---------|-----------------|
| `correction` | perception+correction | Prior belief was wrong | ~300 |
| `insight` | perception+insight | Pattern or cause discovered | ~350 |
| `realization` | perception+realization | Understanding clicked | ~150 |
| `validation` | perception+validation | Approach confirmed working | ~100 |
| *(no tag)* | perception alone | Just a normal learning | ~800 |

**Impact:**
- Each sub-tag has real discriminative power (5-15% coverage)
- Pattern detection can find clusters of corrections vs insights
- `correction` learnings are more useful for error prevention
- `validation` learnings are more useful for approach confidence
- 800 learnings lose a tag that was meaningless anyway

**Effort:** Prompt update + migration script + update any code that queries `perception`
**Semantic fix:** Yes, strongest

### Option D: Remove entirely

Delete `perception` from all learnings and the prompt.

**Impact:**
- Sub-signals (insight, correction, etc.) remain and are more useful
- 861 learnings that only had `perception` + topic tags lose no real signal
- Pattern detection improves immediately
- Simplest migration

**Effort:** DELETE + prompt update
**Semantic fix:** Yes, by deletion

## Recommendation: Option C (Replace with sub-tags)

**Rationale:**

1. **The sub-tags already exist** on 49% of perception learnings. We're not inventing new taxonomy -- we're promoting existing signals.

2. **Sub-tags have real discriminative power.** `correction` at 12% is useful for finding "what was I wrong about?" patterns. `validation` at 4% identifies confirmed-good approaches. `perception` at 76% tells you nothing.

3. **Backward compatible.** Code that queries `memory_tags WHERE tag = 'perception'` gets fewer results (only the un-migrated stragglers), not errors. The pattern detector's `detect_noise_tags()` would have excluded it anyway.

4. **The 861 learnings without sub-signals need classification**, not just tag removal. Many of them are genuine patterns that were tagged `perception` by default when the LLM couldn't find a better sub-signal. They should get `insight` (most common case for pattern observations) or no perception-family tag at all.

## Migration Plan

### Phase 1: Stop the bleeding (immediate)

1. **Update `~/.claude/agents/memory-extractor.md`:**
   - Remove `perception` from all example `--tags` values
   - Add explicit instructions: "Do NOT use the tag `perception`. Instead, use the specific sub-signal tag that applies: `correction`, `insight`, `realization`, or `validation`. If none applies, omit all perception-family tags."
   - Update the classification table to map Internal Type directly to tag

2. **Add `perception` to pattern detector noise exclusion** as an interim measure while migration runs:
   ```python
   # In pattern_detector.py detect_patterns()
   STATIC_NOISE_TAGS = {"perception"}
   noise_tags = detect_noise_tags(tag_idf, tag_noise_percentile) | STATIC_NOISE_TAGS
   ```

### Phase 2: Migrate existing data (batch)

Run a SQL migration to handle the three cases:

```sql
-- Case 1: Has perception + a sub-signal → just remove perception
DELETE FROM memory_tags
WHERE tag = 'perception'
AND memory_id IN (
  SELECT mt1.memory_id FROM memory_tags mt1
  JOIN memory_tags mt2 ON mt1.memory_id = mt2.memory_id
  WHERE mt1.tag = 'perception'
  AND mt2.tag IN ('insight', 'correction', 'realization', 'validation')
);
-- Estimated: ~830 rows

-- Case 2: Has perception + no sub-signal but has 'debugging' → add 'insight', remove 'perception'
-- (debugging is a method, not a perception type -- add insight for the pattern discovered)
INSERT INTO memory_tags (memory_id, tag, session_id)
SELECT mt.memory_id, 'insight', mt.session_id
FROM memory_tags mt
WHERE mt.tag = 'perception'
AND NOT EXISTS (SELECT 1 FROM memory_tags mt2 WHERE mt2.memory_id = mt.memory_id AND mt2.tag IN ('insight', 'correction', 'realization', 'validation'))
AND EXISTS (SELECT 1 FROM memory_tags mt3 WHERE mt3.memory_id = mt.memory_id AND mt3.tag = 'debugging')
ON CONFLICT DO NOTHING;

DELETE FROM memory_tags
WHERE tag = 'perception'
AND memory_id IN (
  SELECT memory_id FROM memory_tags WHERE tag = 'debugging'
)
AND memory_id NOT IN (
  SELECT memory_id FROM memory_tags WHERE tag IN ('correction', 'realization', 'validation')
);
-- Estimated: ~173 rows

-- Case 3: Has perception + no sub-signal and no debugging → remove perception
-- These are mostly straightforward observations that don't need a perception-family tag
DELETE FROM memory_tags
WHERE tag = 'perception'
AND memory_id NOT IN (
  SELECT memory_id FROM memory_tags WHERE tag IN ('insight', 'correction', 'realization', 'validation', 'debugging')
);
-- Estimated: ~688 rows
```

Also update `metadata->'tags'` to stay in sync:

```sql
-- Remove 'perception' from metadata tags JSON array
UPDATE archival_memory
SET metadata = jsonb_set(
  metadata,
  '{tags}',
  (SELECT jsonb_agg(elem) FROM jsonb_array_elements(metadata->'tags') AS elem WHERE elem != '"perception"')
)
WHERE metadata->'tags' @> '"perception"'::jsonb;
```

### Phase 3: Verify (post-migration)

```sql
-- Should return 0
SELECT count(*) FROM memory_tags WHERE tag = 'perception';

-- Sub-signals should now be the perception-family tags
SELECT tag, count(*) FROM memory_tags
WHERE tag IN ('insight', 'correction', 'realization', 'validation')
GROUP BY tag ORDER BY count(*) DESC;
```

### Phase 4: Clean up code

1. Remove `perception` from `PERCEPTION_SIGNALS` regex list name (rename to `THINKING_BLOCK_SIGNALS` since they're used for pre-filtering, not tagging)
2. Remove any hardcoded references to the `perception` tag string
3. Update the test fixture in `test_pattern_detector.py` that uses `perception` as a noise tag example

## Impact on Downstream Systems

| System | Current Impact | After Migration |
|--------|---------------|-----------------|
| **IDF weighting** | `perception` has near-zero IDF, drags down co-occurrence scores | Removed; all remaining tags have meaningful IDF |
| **Pattern detection** | `detect_noise_tags()` already filters it dynamically | No longer needed as noise; sub-tags carry signal |
| **Recall queries** | `--tags perception` returns 76% of everything | Sub-tag queries (`--tags correction`) return targeted results |
| **Tag aggregation** | Perception dominates every tag frequency report | Clean distribution with meaningful top tags |
| **memory_tags index** | 1,691 rows for one near-useless tag | ~860 rows for 4 useful sub-tags combined |
