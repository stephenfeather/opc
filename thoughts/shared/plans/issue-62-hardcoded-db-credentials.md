# Issue #62 — Hardcoded DB Credentials in Fallback URLs

**Status:** Draft — awaiting ARCHITECT approval
**Author:** SECURITY_EXPERT
**Branch:** `fix/issue-62-hardcoded-credentials` (worktree:
`.worktrees/issue-62-hardcoded-credentials`)
**Base:** `origin/main` (282cebf — the merged #104 fix)
**Severity:** LOW (local-Docker context); OWASP credential-management
violation
**Complexity:** MEDIUM — multi-file + tests + TS source + regenerated
dist (Large Plan Workflow)

## Problem

Multiple repo files fall back to the literal string
`postgresql://claude:claude_dev@localhost:5432/continuous_claude`
when no DB env var is set. Anyone with repo access sees the dev
credentials. The `docker-compose.yml` is the legitimate source of those
credentials for local dev; the code should not duplicate them.

## Precedent Honored

The existing canonical env-var precedence, established by the S14
adversarial review and stored as memory
`s-mnm34zi8` (CODEBASE_PATTERN), is:

1. `CONTINUOUS_CLAUDE_DB_URL` (primary)
2. `DATABASE_URL` (fallback)
3. `OPC_POSTGRES_URL` (legacy, hooks only)

This plan keeps that precedence. No supersede.

## Audit — Confirmed Call Sites

### Production code with literal `claude:claude_dev` fallback (must fix)

| # | File | Line | Form |
|---|------|------|------|
| 1 | `scripts/core/backfill_sessions.py` | 50 | `or "postgresql://claude:claude_dev@..."` trailing fallback in `get_pg_url()` |
| 2 | `scripts/core/db/postgres_pool.py` | 44 | `_DEV_DEFAULT_URL = "postgresql://claude:claude_dev@..."` module constant; returned by `resolve_connection_url()` when `AGENTICA_ENV` ∈ local-dev |
| 3 | `scripts/migrations/backfill_project_column.py` | 175 | `os.environ.get("DATABASE_URL", "postgresql://claude:claude_dev@...")` — 2-arg `get` default |
| 4 | `hooks/src/shared/db-utils-pg.ts` | multiple (37, 181, 416, 497, 571, 618, 682, 727, 794, 844, 901, 968, 1060) | TS source emits Python heredoc / JS literal `... || "postgresql://claude:claude_dev@..."`  |

### Compiled artifacts regenerated from TS source (rebuild + commit)

| File | Notes |
|------|-------|
| `hooks/dist/file-claims.mjs` | 3 occurrences |
| `hooks/dist/heartbeat.mjs` | 2 |
| `hooks/dist/session-crash-recovery.mjs` | 3 |
| `hooks/dist/session-register.mjs` | 2 |
| `hooks/dist/peer-awareness.mjs` | 2 |
| `hooks/dist/session-clean-exit.mjs` | 2 |
| `hooks/dist/working-on-sync.mjs` | 1 (only checks `CONTINUOUS_CLAUDE_DB_URL`) |

`hooks/dist/` is **not** gitignored (verified) — the artifacts are
committed. After updating `hooks/src/`, run `npm run build` in
`hooks/ts/` and commit the regenerated dist.

### Environment-example file

| # | File | Line | Form |
|---|------|------|------|
| 5 | `.env.example` | 9 | `DATABASE_URL=postgresql://claude:claude_dev@...?gssencmode=disable` |

### Test assertions that hardcode the fallback URL

| # | File | Line(s) | What it asserts |
|---|------|---------|-----------------|
| 6 | `tests/test_backfill_sessions.py` | 86 | `assert get_pg_url() == "postgresql://claude:claude_dev@..."` — tests the fallback directly |
| 7 | `tests/test_postgres_pool.py` | 33, 187, 196, 250, 275 | `resolve_connection_url` returns `_DEV_DEFAULT_URL` when env unset + `AGENTICA_ENV=development`; and one negative assertion at 33 that an error string contains the URL |

## Design Options for ARCHITECT

The core choice is how to handle local developer ergonomics after the
literal default is removed. Three options:

### Option A — Strict removal (simplest, breaks developer flow)

Remove every literal. Every script requires `CONTINUOUS_CLAUDE_DB_URL`
or `DATABASE_URL` set before import. If unset, `get_pg_url()` /
`resolve_connection_url()` raises `ValueError` with an actionable
message (docker-compose command to run + env var to export).

Pros: zero hardcoded credentials anywhere; OWASP-compliant.
Cons: every first-time dev is blocked until they read the error and
export a var.

### Option B — Constructed dev default (keeps ergonomics, still a default)

Build the dev URL at runtime from parts env vars with empty-password
default:

```python
def _build_dev_url() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "claude")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    db = os.environ.get("POSTGRES_DB", "continuous_claude")
    if not password:
        raise ValueError("POSTGRES_PASSWORD not set; see .env.example")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"
```

Pros: no literal credential string anywhere; still "just works" for
anyone who has `POSTGRES_PASSWORD` exported from the docker-compose
`.env`.
Cons: surfaces complexity (6 new env-var names) for a marginal benefit;
password would still be `claude_dev` for every new dev who pulls the
docker-compose config.

### Option C (Recommended) — Keep env precedence, fail fast, point at docker

Remove the literal fallback. `resolve_connection_url()` and `get_pg_url()`
raise `ValueError` (or return `None` where callers already handle `None`)
when no env var is set, with a message that tells the dev exactly what
to do:

```
Database URL not set. Either:
  (a) run `docker compose -f docker/docker-compose.yml up -d` and export
      $(cat docker/.env | grep -v '^#' | xargs) before invoking this
      script, or
  (b) set CONTINUOUS_CLAUDE_DB_URL (preferred) or DATABASE_URL in your
      shell / launcher.
```

Pros: zero hardcoded credentials; clear error message; docker-compose
remains the dev-credentials source of truth (per prompt scope).
Cons: AGENTICA_ENV=development shortcut is deprecated — anyone relying
on "I have Docker up, I don't need to export anything" path gets an
error once.

**I recommend Option C.** It matches the issue intent (OWASP compliance)
and the prompt's "fail fast with actionable error message listing the
expected env vars" requirement.

## Implementation Plan (TDD) — Assuming Option C

### Commit 1 — RED: tests that lock the new contract

New + modified tests:

- `tests/test_postgres_pool.py`:
  - Update positive assertions (lines 187, 196, 250, 275) to expect
    either the env-set URL or `ValueError` on unset env.
  - Keep the existing "must-raise-for-non-local-env" test (already
    present).
  - Add a new test asserting no module in `scripts/core/` or
    `scripts/migrations/` contains the literal `claude:claude_dev` —
    a grep-style static invariant test.
  - Update line 33's error-string assertion: error no longer needs to
    contain the dev URL, but must mention the env vars.
- `tests/test_backfill_sessions.py:86`: change expectation — env unset
  should raise, env set should return.
- New `tests/test_no_hardcoded_credentials.py`:
  - Walks `scripts/`, `hooks/src/`, `src/`, fails if any non-test file
    contains the literal `claude:claude_dev`.
  - Also asserts `.env.example` uses a placeholder like `CHANGE_ME` or
    `YOUR_PASSWORD_HERE` rather than a real credential.

### Commit 2 — GREEN: production code

- `scripts/core/db/postgres_pool.py`: remove `_DEV_DEFAULT_URL`; change
  `resolve_connection_url` to raise on missing URL regardless of
  `AGENTICA_ENV`; update error message.
- `scripts/core/backfill_sessions.py:45-51`: remove the `or "postgresql://..."`
  trailing fallback; raise `ValueError` with actionable text (same
  message shape as postgres_pool for consistency).
- `scripts/migrations/backfill_project_column.py:175`: replace 2-arg
  `os.environ.get` with explicit fetch + raise.
- `hooks/src/shared/db-utils-pg.ts`:
  - Change the 13 occurrences of `|| "postgresql://claude:claude_dev@..."`
    to throw a clear error if all three env vars are unset. For the
    TypeScript getter: `if (!url) throw new Error("...")`. For the
    Python heredocs: emit a `sys.exit("ERROR: ...")`.

### Commit 3 — REFACTOR: hooks/dist regeneration

- `cd hooks/ts && npm run build`
- Commit the regenerated `hooks/dist/*.mjs` files (verify no literal
  `claude:claude_dev` remains in dist).

### Commit 4 — Documentation

- `.env.example`: change the `DATABASE_URL` line to a commented-out
  placeholder with explanation:
  ```
  # Primary DB URL — required. For local docker dev, export this from
  # docker/.env after `docker compose up`.
  CONTINUOUS_CLAUDE_DB_URL=
  # Optional fallback (legacy). Scripts check CONTINUOUS_CLAUDE_DB_URL
  # first; DATABASE_URL is read only if the primary is unset.
  # DATABASE_URL=
  ```
- `CLAUDE.md` project instructions: add one section under "Environment"
  documenting the env-var precedence contract (`CONTINUOUS_CLAUDE_DB_URL`
  > `DATABASE_URL` > legacy `OPC_POSTGRES_URL`; no code-level fallback).

### Review cycle

1. Three Gemini-pro adversarial rounds (Codex still quota-blocked):
   - R1: completeness — every literal removed, no grep hits remain
   - R2: error paths — does each fail-fast error message reach the user,
     does any caller swallow the raised ValueError silently
   - R3: API design — is the env contract documented for discoverability,
     any regression in developer onboarding flow
2. Aegis security-specialist pass (same pattern as #104)
3. `/security` against changed files
4. PR with before/after diff + coverage on changed files + review-trail
   documentation
5. ≥ 2 AI reviewer cycles (Copilot + CodeRabbit)

## Non-Goals

- No changes to `docker/docker-compose.yml` or `docker/.env` (per prompt
  scope — those ARE the dev credentials source of truth)
- No production deployment changes (those already use env vars)
- No password rotation (ops concern)
- No migration to Vault / AWS SSM / other secrets manager
- No refactor of `get_pg_url()` beyond the minimum needed to remove the
  fallback
- Per deferred-fix rule, unrelated TS / Python style issues discovered
  during audit are out of scope; file separate issues

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Removing `AGENTICA_ENV=development` fallback breaks dev onboarding | Dev friction | Medium | Error message tells user exactly how to fix; update README/CLAUDE.md |
| Missed literal in a file Grep didn't match | Remaining OWASP violation | Low | Static-invariant test walks every non-test file and asserts absence |
| Rebuilt `hooks/dist/` contains subtle changes beyond credential removal (e.g., esbuild version drift) | PR diff noise | Low | Run a rebuild from clean and diff against committed dist to confirm only expected changes |
| A hook script that previously "just worked" in a shell where only DATABASE_URL was unset now fails | User-visible hook failure | Medium | Test a full session lifecycle in the worktree after the changes land, verify hook error messages are human-readable |

## Success Criteria

- [ ] `rg "claude:claude_dev"` in the worktree returns zero hits outside
      `docker/` and `.worktrees/` (verified by static-invariant test).
- [ ] All DB-consuming scripts raise `ValueError`/`Error` with an
      actionable message when no env var is set.
- [ ] `.env.example` documents the canonical env var with a safe
      placeholder.
- [ ] `CLAUDE.md` documents the env-var precedence contract.
- [ ] Existing test suite passes; no new regressions.
- [ ] 3 Gemini-pro review rounds + Aegis audit clean.
- [ ] `/security` scan clean on changed files.
- [ ] PR merged after ≥ 2 AI reviewer cycles.

## Open Questions for ARCHITECT

1. **Option A vs B vs C?** I recommend C (strict removal + fail-fast
   with docker-compose hint). A is cleanest but worst UX. B keeps a
   default password shape and buys little.
2. **Deprecate `AGENTICA_ENV=development` shortcut?** Currently
   `postgres_pool.resolve_connection_url` returns the dev default iff
   `AGENTICA_ENV` ∈ local-dev envs. Under Option C, `AGENTICA_ENV` no
   longer gates a code fallback. OK to remove the branch entirely,
   or keep the env-var for other future purposes and just drop the
   default-URL behavior?
3. **`OPC_POSTGRES_URL` legacy env var — still needed?** Only referenced
   by hooks. If we can retire it as part of this cleanup, the hooks
   code gets simpler. Or is something still exporting it?
4. **Static-invariant test — assert absence of literal in `docker/`
   too, or exempt that directory?** Prompt says docker-compose is the
   dev-credentials source. I'd exempt `docker/` (the credentials live
   there on purpose) and `.worktrees/`, assert absence everywhere else.
   Confirm?
