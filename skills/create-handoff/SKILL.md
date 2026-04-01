---
name: create-handoff
description: Create handoff document for transferring work to another session
---

# Create Handoff

You are tasked with writing a handoff document to hand off your work to another agent in a new session. You will create a handoff document that is thorough, but also **concise**. The goal is to compact and summarize your context without losing any of the key details of what you're working on.


## Process
### 1. Filepath & Metadata
Use the following information to understand how to create your document:

**Determine the session name using this fallback chain (first match wins):**

1. **Existing handoffs** (most recently modified folder):
```bash
ls -td thoughts/shared/handoffs/*/ 2>/dev/null | head -1 | xargs basename
```

2. **Git worktree name** (if in a worktree, not the main repo):
```bash
basename "$(git worktree list --porcelain 2>/dev/null | head -1 | sed 's/^worktree //')" 2>/dev/null
```

3. **Git branch name** (kebab-cased):
```bash
git branch --show-current 2>/dev/null
```

4. **Directory name** (last resort):
```bash
basename "$CLAUDE_PROJECT_DIR"
```

Use the first non-empty result as the handoff folder name.

**Create your file under:** `thoughts/shared/handoffs/{session-name}/YYYY-MM-DD_HH-MM_description.yaml`, where:
- `{session-name}` is from existing handoffs (e.g., `open-source-release`) or `general` if none exist
- `YYYY-MM-DD` is today's date
- `HH-MM` is the current time in 24-hour format (no seconds needed)
- `description` is a brief kebab-case description

**Examples:**
- `thoughts/shared/handoffs/open-source-release/2026-01-08_16-30_memory-system-fix.yaml`
- `thoughts/shared/handoffs/general/2026-01-08_16-30_bug-investigation.yaml`

### 2. Write YAML handoff (~400 tokens vs ~2000 for markdown)

**CRITICAL: Use EXACTLY this YAML format. Do NOT deviate or use alternative field names.**

The `goal:` and `now:` fields are shown in the statusline - they MUST be named exactly this.

```yaml
---
session: {session-name from ledger}
date: YYYY-MM-DD
status: complete|partial|blocked
outcome: SUCCEEDED|PARTIAL_PLUS|PARTIAL_MINUS|FAILED
---

goal: {What this session accomplished - shown in statusline}
now: {What the NEXT session should do first - must be an INCOMPLETE action, never something already done}
test: {Command to verify this work, e.g., pytest tests/test_foo.py}

done_this_session:
  - task: {First completed task}
    files: [{file1.py}, {file2.py}]
  - task: {Second completed task}
    files: [{file3.py}]

blockers: [{any blocking issues}]

questions: [{unresolved questions for next session}]

decisions:
  - {decision_name}: {rationale}

findings:
  - {key_finding}: {details}

worked: [{approaches that worked - Problem → Solution format}]
failed: [{approaches that failed - what was tried, what broke, why to avoid}]

next:
  - {First next step}
  - {Second next step}

files:
  created: [{new files}]
  modified: [{changed files}]
```

**Field guide:**
- `goal:` + `now:` - REQUIRED, shown in statusline
- `now:` - CRITICAL: Must be the first INCOMPLETE action for the next session. Never set this to something that was just completed. Cross-reference against `done_this_session:` — if it appears there, it's done and should NOT be `now:`. Pick the first item from `next:` instead.
- `done_this_session:` - What was accomplished with file references
- `decisions:` - Important choices and rationale
- `findings:` - Key learnings
- `worked:` / `failed:` - What to repeat vs avoid (use Problem → Solution format: state what happened, what broke, and the fix)
- `next:` - Action items for next session (first item should match `now:`)

**DO NOT use alternative field names like `session_goal`, `objective`, `focus`, `current`, etc.**
**The statusline parser looks for EXACTLY `goal:` and `now:` - nothing else works.**
---

### 3. Mark Session Outcome (REQUIRED)

**Determine outcome automatically when possible:**

- If `blockers:` is empty AND `questions:` is empty → default to **SUCCEEDED**
- If `blockers:` is empty AND `questions:` has items → default to **PARTIAL_PLUS**
- If `blockers:` has items → ask the user using AskUserQuestion:

```
Question: "Session has blockers — how did it go?"
Options:
  - PARTIAL_PLUS: Mostly done, minor issues remain
  - PARTIAL_MINUS: Some progress, major issues remain
  - FAILED: Task abandoned or blocked
```

For the default cases (SUCCEEDED or PARTIAL_PLUS), inform the user of the outcome but do NOT prompt — just proceed.

After determining the outcome, use the **opc-memory** MCP server:

```
# First, index the handoff into the database
Call MCP tool: mcp__opc-memory__index_artifacts
Parameters:
  mode: "file"
  file_path: "thoughts/shared/handoffs/{session_name}/{filename}.yaml"

# Then mark the outcome
Call MCP tool: mcp__opc-memory__mark_handoff
Parameters:
  outcome: "<USER_CHOICE>"  (SUCCEEDED, PARTIAL_PLUS, PARTIAL_MINUS, or FAILED)
```

**IMPORTANT:** Replace `{session_name}` and `{filename}` with the actual values from step 1.

These MCP tools auto-detect the database (PostgreSQL if configured, SQLite fallback).

### 4. Confirm completion

After marking the outcome, respond to the user:

```
Handoff created! Outcome marked as [OUTCOME].

Resume in a new session with:
/resume_handoff path/to/handoff.yaml
```

---
##.  Additional Notes & Instructions
- **more information, not less**. This is a guideline that defines the minimum of what a handoff should be. Always feel free to include more information if necessary.
- **be thorough and precise**. include both top-level objectives, and lower-level details as necessary.
- **avoid excessive code snippets**. While a brief snippet to describe some key change is important, avoid large code blocks or diffs; do not include one unless it's necessary (e.g. pertains to an error you're debugging). Prefer using `/path/to/file.ext:line` references that an agent can follow later when it's ready, e.g. `packages/dashboard/src/app/dashboard/page.tsx:12-24`
