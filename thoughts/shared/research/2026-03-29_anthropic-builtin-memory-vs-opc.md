# Anthropic Built-in Memory vs. OPC Extraction Pipeline

**Date:** 2026-03-29
**Sources:** Piebald-AI/claude-code-system-prompts (6 prompt files)

## What Their System Does

Anthropic's upcoming built-in memory writes to **structured markdown files** (one per session) with a fixed template. It operates in two modes:

### 1. Session Memory Update (Real-time, in-conversation)
- Updates a notes file **during** the conversation, not after
- Fixed sections: `Current State`, `Task specification`, `Files and Functions`, `Workflow`, `Errors & Corrections`, `Codebase and System Documentation`, `Learnings`, `Key results`, `Worklog`
- Uses the Edit tool to update sections in parallel
- Preserves template structure (headers + italic descriptions are immutable)
- Focuses on actionable detail: file paths, commands, technical specifics

### 2. Dream: Memory Consolidation (Post-session, background)
- 4-phase process: **Orient → Gather signal → Consolidate → Prune/index**
- Reads memory files and consolidates/deduplicates across sessions
- Merges related items, removes stale info, re-indexes for retrieval
- Operates on the file-based memory store

### 3. User/Feedback Memories (Separate from session notes)
- **User memories**: Role, goals, responsibilities — "who is this person"
- **Feedback memories**: Guidance on work approach — both what to avoid AND what to keep doing
- Checks for contradictions with team memories before saving
- Explicitly avoids negative judgements

### 4. Agent-specific Memory Instructions
- Domain-specialized: code reviewers track patterns/conventions, test runners track flaky tests, architects track codepaths
- Agents accumulate institutional knowledge during normal operations

---

## What We Extract vs. What They Extract

| Dimension | Our System | Their System | Gap? |
|-----------|-----------|-------------|------|
| **Perception shifts** ("aha moments") | Yes — core focus via regex signals | No — not explicitly targeted | **We're ahead** |
| **Current state / what's being worked on** | No | Yes (`Current State` section) | **Gap** |
| **Task specification / intent** | Partially (via handoffs) | Yes (dedicated section) | Minor gap |
| **Files and functions touched** | No (only in handoffs manually) | Yes (dedicated section) | **Gap** |
| **Workflow / commands used** | No | Yes (bash commands, order, interpretation) | **Gap** |
| **Errors and corrections** | Yes (ERROR_FIX type) | Yes (dedicated section + approach failures) | Comparable |
| **Codebase documentation** | No | Yes (system components, how they fit) | **Gap** |
| **Key results / outputs** | No | Yes (exact answers, tables, documents) | **Gap** |
| **Worklog / step-by-step** | No | Yes (terse step log) | **Gap** |
| **User preferences / who the user is** | Yes (USER_PREFERENCE type) | Yes (dedicated memory type) | Comparable |
| **Feedback on approach** | Partially (FAILED_APPROACH) | Yes — **both** success and failure explicitly | **Gap** — we bias toward failures |
| **Cross-memory dedup** | Yes (0.85 similarity) | Yes (consolidation phase) | Comparable |
| **Contradiction checking** | No | Yes (private vs team feedback) | **Gap** |
| **Real-time extraction** | No (post-session only) | Yes (during conversation) | **Gap** |

---

## Key Things Their System Extracts That We Miss

### 1. Success patterns (not just failures)
Their feedback memory explicitly says: "Record from failure AND success: if you only save corrections, you will avoid past mistakes but drift away from approaches the user has already validated, and may grow overly cautious."

Our system biases toward perception *changes* — things that surprised or corrected. We miss "this approach worked great, keep doing it."

### 2. Workflow knowledge (commands and their order)
They capture: "What bash commands are usually run and in what order? How to interpret their output?"

We don't extract workflow patterns from sessions at all. This is high-value for continuity.

### 3. Codebase documentation as a byproduct
Their system captures "What are the important system components? How do they work/fit together?" during normal sessions. We rely on separate `/research` or `/explore` to build this, not as automatic extraction.

### 4. Key results / exact outputs
If a session produced a specific answer, table, or document, their system captures it verbatim. We don't.

### 5. Current state / continuation context
Their `Current State` section captures "what's actively being worked on right now, pending tasks, immediate next steps" — basically our handoff `now:` and `next:` fields, but extracted automatically every session rather than manually.

### 6. Files-and-functions inventory
Automatic tracking of which files are important and why. We do this in handoffs but not in our automated extraction.

---

## Recommendations

1. **Add success extraction**: Add perception signals for validation/confirmation ("this works", "good approach", "as expected") and map to a `VALIDATED_APPROACH` or reuse `WORKING_SOLUTION` more aggressively
2. **Extract workflow patterns**: Parse tool_use calls from JSONL to capture command sequences (not just thinking blocks)
3. **Auto-generate mini-handoffs**: After extraction, also produce a structured "session summary" with current state, files touched, and next steps — bridging the gap between their real-time notes and our post-session extraction
4. **Add feedback balance check**: When storing FAILED_APPROACH, also look for the positive counterpart in the same session
5. **Consider real-time hooks**: Their biggest advantage is in-conversation updates. Our PostToolUse hooks could capture file/function context as it happens rather than reconstructing post-session

---

## Source Prompts Analyzed

1. `agent-prompt-dream-memory-consolidation.md` — 4-phase consolidation (Orient, Gather, Consolidate, Prune)
2. `agent-prompt-session-memory-update-instructions.md` — Real-time note-taking rules
3. `data-session-memory-template.md` — Fixed template with 10 sections
4. `system-prompt-agent-memory-instructions.md` — Domain-specific agent memory guidance
5. `system-prompt-description-part-of-memory-instructions.md` — User memory type definition
6. `system-prompt-memory-description-of-user-feedback.md` — Feedback memory type with contradiction checking

## Our Pipeline (for reference)

- **Extraction**: `scripts/core/extract_thinking_blocks.py` — regex-based perception signal detection on thinking blocks
- **Storage**: `scripts/core/store_learning.py` — PostgreSQL with BGE embeddings, 0.85 cosine dedup
- **Agent**: `~/.claude/agents/memory-extractor.md` — headless `claude -p` with Sonnet, 15 max turns
- **Daemon**: `scripts/core/memory_daemon.py` — watches for stale sessions, spawns extraction, archives to S3
