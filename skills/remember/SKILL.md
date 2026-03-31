---
name: remember
description: Store a learning, pattern, or decision in the memory system for future recall
user-invocable: true
---

# Remember - Store Learning in Memory

Store a learning, pattern, or decision in the memory system for future recall.

## Usage

```
/remember <what you learned>
```

Or with explicit type:

```
/remember --type WORKING_SOLUTION <what you learned>
```

## Examples

```
/remember TypeScript hooks require npm install before they work
/remember --type ARCHITECTURAL_DECISION Session affinity uses terminal PID
/remember --type FAILED_APPROACH Don't use subshell for store_learning command
```

## What It Does

1. Stores the learning in PostgreSQL with BGE embeddings
2. Auto-detects learning type if not specified
3. Extracts tags from content
4. Returns confirmation with ID

## Learning Types (Priority Order)

| Type | Use For |
|------|---------|
| `FAILED_APPROACH` | Something tried that didn't work |
| `ERROR_FIX` | Specific error diagnosed + fix found |
| `OPEN_THREAD` | Incomplete work to resume later |
| `USER_PREFERENCE` | User's preferred way of doing things |
| `ARCHITECTURAL_DECISION` | Deliberate choice between alternatives |
| `WORKING_SOLUTION` | Specific technique that solved a problem |
| `CODEBASE_PATTERN` | Observation about how things work (default) |

## Execution

When this skill is invoked, use the **opc-memory** MCP server:

```
Call MCP tool: mcp__opc-memory__store_learning
Parameters:
  content: "<ARGS>"                    (the learning content from user)
  learning_type: "<TYPE>"              (detected type, default: CODEBASE_PATTERN)
  session_id: "manual-YYYYMMDD-HHMM"   (current date/time)
  context: "manual entry via /remember"
  confidence: "medium"
```

This replaces the old bash command approach with direct MCP tool invocation.

## Auto-Type Detection (Priority-Ordered)

If no `--type` specified, classify the learning by checking these rules **top-to-bottom** and using the **FIRST match**:

### 1. FAILED_APPROACH — Something was tried and didn't work
**Test:** Does it describe a negative outcome?
- Signal words: "doesn't work", "breaks", "anti-pattern", "failed", "didn't work", "don't", "avoid", "causes issues"
- → If yes: `FAILED_APPROACH`

### 2. ERROR_FIX — A specific error was diagnosed and fixed
**Test:** Does it reference a specific error message, status code, exception, or failure symptom AND provide the resolution?
- Signal words: "error", "fix", "bug", "exception", "status code", "resolved by", "stack trace"
- → If yes: `ERROR_FIX`

### 3. OPEN_THREAD — Work is incomplete and must be resumed
**Test:** Does it describe something that still needs to be done?
- Signal words: "TODO", "not yet implemented", "still needs", "behind N migrations", "incomplete", "WIP"
- → If yes: `OPEN_THREAD`

### 4. USER_PREFERENCE — The user wants things done a specific way
**Test:** Is it prescriptive about how to do things?
- Signal words: "always use", "never do", "prefer X over Y", "user requires", "user wants", "convention is"
- → If yes: `USER_PREFERENCE`

### 5. ARCHITECTURAL_DECISION — A deliberate choice between alternatives
**Test:** Does it explain WHY one approach was chosen over another?
- Signal words: "chose X over Y because", "decision:", "instead of", "trade-off", "we went with"
- → If yes: `ARCHITECTURAL_DECISION`

### 6. WORKING_SOLUTION — A specific technique that solved a problem
**Test:** Does it describe an action someone took that succeeded?
- Signal words: "fixed by", "solved by", "recovered by", "works by", "the fix was", "solution:"
- → If yes: `WORKING_SOLUTION`

### 7. CODEBASE_PATTERN — Default/catch-all
**Test:** None of the above matched. It's an observation about how things work.
- Typical form: "when X, then Y" observations without a fix, failure, preference, or decision
- → Default: `CODEBASE_PATTERN`

**IMPORTANT:** Do NOT default to WORKING_SOLUTION — that's rule 6, not the catch-all. CODEBASE_PATTERN is the catch-all. The rules are ordered by specificity: easy-to-detect types (FAILED_APPROACH, ERROR_FIX, OPEN_THREAD) are checked first since they have strong signal words.
