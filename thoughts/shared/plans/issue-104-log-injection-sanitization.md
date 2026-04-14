# Issue #104 — Log Injection Sanitization

**Status:** Draft — awaiting ARCHITECT approval
**Author:** SECURITY_EXPERT
**Branch:** `fix/issue-104-log-injection` (worktree: `.worktrees/issue-104-log-injection`)
**Base:** `origin/main` (f0401bc)
**Severity:** LOW (requires DB write access already)
**Complexity:** MEDIUM (multi-file, security-sensitive → Large Plan Workflow)

## Problem

`memory_daemon.py` and `memory_daemon_extractors.py` interpolate DB-sourced
strings (session IDs, project paths, transcript paths, subprocess stderr)
directly into log messages via f-strings. An attacker who already has DB
write access can inject:

- `\n` / `\r` — forge/split log lines
- `\x1b` — ANSI escape sequences for terminal manipulation on an admin's TTY
- Other C0 control characters

## Primitives & Design

The primitive is "a rendered log field": a short, bounded, printable
representation of an arbitrary (possibly None, possibly binary) input. The
sanitizer is the single black-box boundary where untrusted bytes become
safe display text. One helper, one rule, one test suite.

### Module placement

New module: `scripts/core/log_safety.py`

- Minimal surface area (one public function `safe()`)
- No dependencies beyond stdlib
- Reusable anywhere in the codebase that logs DB/IPC/subprocess-sourced
  strings — not daemon-specific
- Importable without touching `memory_daemon` (avoids circular risk during
  future refactors)

### Helper contract

```python
def safe(s: object, *, max_len: int = 500) -> str:
    """Render an arbitrary value as a single-line, control-char-free log field.

    - None → "" (empty string)
    - Non-str → str(s) first (never raises; uses repr-safe coercion)
    - \n, \r, \x00-\x08, \x0b-\x1f, \x7f → replaced with \xNN markers
    - \t (0x09) preserved
    - Truncation: values longer than max_len are cut with an ellipsis marker
      "…[truncated N bytes]" to prevent log-bombing
    - Returns plain str, never bytes
    """
```

Rationale for each rule:

| Rule | Why |
|------|-----|
| `None` → `""` | Matches current behavior of `f"{None}"` giving `"None"` is LESS useful — empty makes forged fields obvious in `project=` |
| Strip CR/LF | Prevents log line forgery |
| Strip ESC (`\x1b`) | Prevents ANSI injection into admin terminals |
| Preserve TAB | Real data often contains tabs; not a control vector in tailing |
| `\xNN` markers | Reversible for forensics vs. naive stripping |
| Cap `max_len` | Prevents a 100MB transcript_path from flooding the log |

Alternative considered: use `unicodedata.category()` to strip all Cc/Cf.
Rejected: more complex, changes behavior on non-C0 categories (e.g., RTL
marks in legit project paths), higher maintenance risk. Simple byte-level
allowlist is sufficient for the LOW-severity threat model.

## Scope of Call-Site Changes

Audit completed against `scripts/core/memory_daemon.py` and
`scripts/core/memory_daemon_extractors.py` on the `origin/main` base
(f0401bc).

### `memory_daemon.py` — ~27 call sites

DB-sourced variables that appear in `log(f"...")` / `debug(f"...")`:

- `session_id`, `sid`, `s.id`, `s.pid` (from `sessions` table)
- `project`, `project_dir` (from `sessions.project`)
- `stderr_text` (subprocess output seeded from session JSONL content)
- `total`, `stdout`, `stderr` from pattern-batch subprocess

Representative lines (origin/main offsets):

- L584 `Extraction completed for {session_id}`
- L596 `stderr: {stderr_text}`
- L628 `Watchdog: killing stuck extraction ...`
- L657 `Dequeuing {session_id} (project={project or 'unknown'}, ...)`
- L682 `Queued {session_id} ...`
- L815 `Pattern detection completed: {stdout[:200]}`
- L836 `Pattern detection failed (rc={rc}): {stderr}`
- L884 `Skipping {s.id}: process {s.pid} still alive`
- L889 `Skipping {sid}: marked exited ...`
- L897 `Found {len(truly_stale)} stale sessions: ...`
- L918 daemon startup log
- L939 `Seeded last pattern run from DB: ...`
- L946 `Error in daemon loop: {e}` (exception text may reflect DB data)
- L1098 `Memory daemon stopping (PID {pid})` — PID is local, but wrap for
  consistency

### `memory_daemon_extractors.py` — ~25 call sites

All six extractor functions use the injected `log_fn(f"...")` pattern with
DB-sourced `session_id`, `project_dir`, `transcript_path`, and subprocess
stderr:

- L65, L71, L87, L112, L143, L150 — `run_extraction`
- L172, L186, L195, L208, L210, L213, L222, L224 — `archive_session_jsonl`
- L242, L248 — `calibrate_session_confidence`
- L271, L280, L306, L308, L310 — `extract_and_store_workflows`
- L331, L335, L350, L355, L357, L359 — `generate_mini_handoff`

### `memory_daemon_db.py`

`memory_daemon_db.py` does not perform logging with f-strings — it returns
typed results to the caller. No changes required there. Confirmed by grep
for `log(` / `debug(` / `logger.` / `print(` in that file.

### Broader audit

Before implementation, I will also grep other modules that `SELECT` from
`sessions`, `archival_memory`, or `detected_patterns` and log the
returned values (e.g., `pattern_detector.py`, `reranker.py`). Any hits
get the same `safe()` wrapper. Findings will be listed in the first commit
message so the adversarial reviewer can validate breadth.

## Implementation Plan (TDD)

### Commit 1 — RED: tests for `safe()`

`tests/test_log_safety.py` covering:

- `safe(None) == ""`
- `safe("") == ""`
- `safe("hello") == "hello"`
- `safe("a\nb") == "a\\x0ab"`
- `safe("a\r\nb")` strips both CR and LF
- `safe("\x1b[31mRED\x1b[0m")` strips ESCs
- `safe("a\tb") == "a\tb"` (tab preserved)
- `safe("\x00\x01\x7f")` — all C0 + DEL replaced
- `safe(42) == "42"`
- `safe(["a"]) == "['a']"`
- `safe("x" * 1000)` truncated, ends with truncation marker
- `safe(b"abc")` — bytes coerced via `str()` (yields `"b'abc'"`) then
  sanitized; documents and locks this behavior
- An object whose `__str__` raises still yields a finite string (fallback
  to `repr()` or a fixed sentinel; chosen during GREEN)

Target coverage: ≥ 95 % of `log_safety.py` (it's small, so ≥ 99 % is
realistic).

### Commit 2 — GREEN: implement `scripts/core/log_safety.py`

Minimum code to pass. Pure function, no I/O, no globals. ~40 LOC.

### Commit 3 — REFACTOR: apply at call sites

- `from scripts.core.log_safety import safe`
- Replace `f"... {session_id} ..."` → `f"... {safe(session_id)} ..."` at
  every identified site.
- Subprocess stderr: decode with `errors="replace"` then `safe(...)`.
- No behavior change for happy-path ASCII strings (existing tests stay
  green).

Run `uv run pytest tests/ -x` — must pass with no regressions.

### Commit 4 — adversarial fixes (rounds 1–3)

Three rounds of `/codex:adversarial-review --wait --scope branch --base
fix/issue-104-log-injection "Log injection sanitization for memory_daemon"`
with commits between rounds.

### Commit 5 — `/security` scan

Against changed files. Address any findings.

### PR

`gh pr create --body-file <tmp>` — body includes per-file coverage from
`uv run pytest --cov=scripts/core/log_safety --cov=scripts/core/memory_daemon --cov=scripts/core/memory_daemon_extractors --cov-report=term-missing`.

## Non-Goals

- No change to log *destinations* (rotating handler, faulthandler path)
- No change to what *fields* are logged
- No new dependencies
- No schema changes
- No refactor of the `log()` / `debug()` wrappers themselves
- Not fixing log injection in unrelated modules discovered during the
  broad audit if they are outside `scripts/core/`; those get separate
  GitHub issues per CLAUDE.md out-of-scope policy.

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Wrapper misses a call site | Residual log injection | Medium | Grep-based audit + adversarial-review rounds specifically asked to look for missed sites |
| `safe()` changes a happy-path log format that a downstream tool greps | Broken log parsing | Low | ASCII printable inputs are passed through unchanged |
| Truncation hides real diagnostic info | Harder debugging | Low | 500-char cap is generous for these fields; exception messages unchanged by truncation in practice |
| Import cycle | Daemon fails to start | Very low | `log_safety` has no internal imports |

## Success Criteria

- [ ] `safe()` implemented in `scripts/core/log_safety.py`
- [ ] All identified call sites use `safe()` on DB-sourced values
- [ ] Test coverage for `log_safety.py` ≥ 95 %
- [ ] Existing daemon tests pass unchanged
- [ ] 3 adversarial review rounds completed with findings addressed
- [ ] `/security` clean on changed files
- [ ] PR opened with coverage report in body
- [ ] 2 AI reviewer cycles (Copilot + CodeRabbit) addressed

## Open Questions for ARCHITECT

1. **Module placement** — OK with new `scripts/core/log_safety.py`, or
   prefer colocating in an existing `scripts/core/shared_*.py`? I chose
   a new file for replaceability (Steenberg black-box) and zero-coupling.
2. **`None` handling** — empty string vs. a literal `<none>` marker? I
   chose `""` because the issue spec says so, but `<none>` preserves the
   "field was present but null" signal for forensics.
3. **Broad audit scope** — limit to `scripts/core/` (daemon + immediate
   callers), or expand to `scripts/mcp/` and `src/runtime/` in this PR?
   I recommend scoping this PR to `scripts/core/` and filing a follow-up
   issue for any other hits, to keep the diff reviewable.
4. **Truncation marker format** — I proposed `…[truncated N bytes]`;
   fine, or prefer ASCII-only (`...[truncated N bytes]`) to avoid any
   unicode in log output?
