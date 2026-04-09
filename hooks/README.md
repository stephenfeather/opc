# Hooks

Claude Code hooks that enable skill auto-activation, file tracking, and validation.

**Zero runtime dependencies** - hooks are pre-bundled, just clone and go.

---

## Architecture

```
hooks/
├── src/              # TypeScript source (source of truth)
├── dist/             # Bundled *.mjs artifacts (built by esbuild)
└── package.json      # Build + dev dependencies (esbuild, vitest)

scripts/
└── deploy_hooks.sh   # Mirrors hooks/{src,dist} → $HOME/.claude/hooks/
```

**For users:** Clone the repo, then `cd hooks && npm install && npm run build`.
The `postbuild` step automatically mirrors `src/` and `dist/` into
`$HOME/.claude/hooks/` via `../scripts/deploy_hooks.sh`, which is where
Claude Code's `settings.json` loads hooks from. No manual `cp` required.

**For developers:** Edit `src/*.ts`, then `npm run build`. The mirror runs
automatically after every successful build.

> `~/.claude/hooks/src/` and `~/.claude/hooks/dist/` are **mirrors** of this
> tree — do not edit them directly. The deploy script uses `rsync --delete`,
> so any files there that do not exist in `opc/hooks/{src,dist}/` will be
> removed on the next build.

### Deploy entry points

| Command (from `hooks/`)            | What it does                                       |
|------------------------------------|----------------------------------------------------|
| `npm run build`                    | Build + auto-deploy (via `postbuild --auto`)       |
| `npm run deploy`                   | Deploy unconditionally (works from worktrees)      |
| `../scripts/deploy_hooks.sh`       | Standalone shell entry (unconditional)             |
| `../scripts/deploy_hooks.sh --auto`| Standalone shell entry (skips from worktrees)      |

**Worktree guard.** The `postbuild` step passes `--auto`, which skips
deploy when `OPC_ROOT` is inside any git worktree. Detection uses
`git rev-parse --git-dir` vs `--git-common-dir` — when they differ,
it's a linked worktree. A pathname check on `/.worktrees/` and
`/.claude/worktrees/` remains as a belt-and-suspenders fallback for
environments where git is unavailable. This keeps experimental branches
from stomping the live `~/.claude/hooks/` tree used by other Claude Code
sessions. If you *want* to deploy from a worktree, run `npm run deploy`
(no `--auto`) — that bypass is explicit and opt-in.

**Target validation.** The script refuses to run with `DEPLOY_TARGET`
pointing at `/`, `$HOME`, or `$HOME/.claude` (logical or physical path),
and requires the target's basename to be `hooks`. The parent is resolved
physically via `cd -P && pwd -P` so symlinks in the parent chain are
followed, but the target itself must not be a symlink — the script
refuses to follow one into an unrelated tree. These guards exist because
`rsync --delete` prunes any file in the target that isn't in the source.

**Lock serialization.** The script acquires an `mkdir`-based lock at
`$TMPDIR/opc-deploy-hooks.lock.d` before touching the target. The lock
dir contains a `pid` file. On contention the script:

1. Checks whether the owning PID is still alive via `kill -0`.
2. If alive — exits 5 (real contention), **regardless of lock age**.
   A running deploy is never preempted: stealing its lock would admit
   concurrent `rsync --delete` runs against the same target, which is
   exactly the failure this lock exists to prevent.
3. If dead or the PID file is missing — atomically `mv`s the stale
   lock to a unique quarantine name (`rename()` is atomic at the POSIX
   level, so two concurrent reclaimers cannot both win), then retries
   the acquire.

`rsync --delay-updates` additionally batches file renames into the
final moment of each sync, so in-flight Claude Code sessions see either
the old or new tree — not a half-updated mix.

**Wedged deploy recovery.** If a legitimate deploy genuinely hangs
(e.g. rsync stuck on a dead NFS mount), the lock stays until the user
manually investigates: `rm -rf $TMPDIR/opc-deploy-hooks.lock.d` after
confirming no running deploy. This is the intentional tradeoff — a
recoverable wedge is preferable to silent concurrent writes.

### Environment overrides

| Variable         | Default                         | Purpose                                      |
|------------------|---------------------------------|----------------------------------------------|
| `DEPLOY_TARGET`  | `$HOME/.claude/hooks`           | Mirror into a different tree (tests/install) |
| `TMPDIR`         | `/tmp`                          | Lock directory location                      |

If `$HOME/.claude/` does not exist (e.g. CI, fresh box without Claude Code
installed), the deploy step prints a skip message and exits 0 instead of
failing. Issue #105 tracked the drift that motivated this automation.

---

## What Are Hooks?

Hooks are scripts that run at specific points in Claude's workflow:
- **UserPromptSubmit**: When user submits a prompt
- **PreToolUse**: Before a tool executes
- **PostToolUse**: After a tool completes
- **SessionStart**: When a session starts/resumes
- **SessionEnd**: When a session ends
- **PreCompact**: Before context compaction
- **SubagentStop**: When a subagent completes

**Key insight:** Hooks can modify prompts, block actions, and track state - enabling features Claude can't do alone.

---

## Essential Hooks (Start Here)

### skill-activation-prompt (UserPromptSubmit)

**Purpose:** Automatically suggests relevant skills based on user prompts and file context

**How it works:**
1. Reads `skill-rules.json`
2. Matches user prompt against trigger patterns
3. Checks which files user is working with
4. Injects skill suggestions into Claude's context

**Why it's essential:** This is THE hook that makes skills auto-activate.

**Integration:**
```bash
# Just copy - no npm install needed!
cp -r .claude/hooks your-project/.claude/

# Make shell scripts executable
chmod +x your-project/.claude/hooks/*.sh
```

**Add to settings.json:**
```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/skill-activation-prompt.sh"
          }
        ]
      }
    ]
  }
}
```

---

### post-tool-use-tracker (PostToolUse)

**Purpose:** Tracks file changes and build attempts for context management

**How it works:**
1. Monitors Edit/Write/Bash tool calls
2. Records which files were modified
3. Captures build/test pass/fail for reasoning
4. Auto-detects project structure (frontend, backend, packages, etc.)

**Why it's essential:** Helps Claude understand what parts of your codebase are active.

**Add to settings.json:**
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|MultiEdit|Write|Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/post-tool-use-tracker.sh"
          }
        ]
      }
    ]
  }
}
```

---

## Continuity Hooks

### session-start-continuity (SessionStart)

**Purpose:** Loads continuity ledger on session start/resume/compact

### pre-compact-continuity (PreCompact)

**Purpose:** Auto-creates handoff document before context compaction

### session-end-cleanup (SessionEnd)

**Purpose:** Updates ledger timestamp, cleans old cache

### subagent-stop-continuity (SubagentStop)

**Purpose:** Logs agent output to ledger and cache for resumability

---

## Development

To modify hooks:

```bash
# Edit TypeScript source
vim src/skill-activation-prompt.ts

# Rebuild bundled JS and auto-deploy to ~/.claude/hooks/
cd hooks && npm run build

# Run hook tests
npm test
```

`npm run build` invokes esbuild, and the `postbuild` script runs
`../scripts/deploy_hooks.sh` to mirror `src/` and `dist/` into
`$HOME/.claude/hooks/`. To re-deploy without rebuilding, use
`npm run deploy`.

---

## For Claude Code

**When setting up hooks for a user:**

1. **Copy the hooks directory** - no npm install needed
2. **Make shell scripts executable:** `chmod +x .claude/hooks/*.sh`
3. **Add to settings.json** as shown above
4. **Verify after setup:**
   ```bash
   ls -la .claude/hooks/*.sh | grep rwx
   ```
