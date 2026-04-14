# Plan: Bump Python Floor to 3.13 (Issue #113)

**Author:** QA_ENGINEER (TMUX)
**Branch:** `fix/issue-113-python-313` (worktree `.worktrees/issue-113-python-313`, branched from `origin/main` @ `aebd209`)
**Workflow:** Large Plan (3 adversarial rounds + Aegis + /security + 2 AI reviewer cycles)
**Status:** Draft — awaiting ARCHITECT review before implementation

## 1. Problem

`tests/test_memory_protocol.py::TestMemoryBackendProtocol::test_protocol_is_a_protocol` fails on `origin/main` because it imports `typing.is_protocol`, which was added in Python **3.13**. Current `requires-python = ">=3.12"` lets uv resolve 3.12.13, producing an `ImportError`. Filed as #113.

**Chosen fix:** bump the Python floor to **3.13**. `typing.is_protocol` then lives in stdlib, and the test passes without code change.

## 2. Scope & Non-Goals

### In scope
- Bump `requires-python` to `>=3.13` in `pyproject.toml`
- Align stale `[tool.mypy] python_version` (currently `"3.11"`) and `[tool.ruff] target-version` (currently `"py311"`) to `"3.13"` / `"py313"`. Also align `[tool.black] target-version` (currently `["py311"]`) to `["py313"]` — found during survey, same class of staleness.
- Create `.python-version` file at repo root with `3.13` (prevents uv drift back to 3.12)
- Regenerate `uv.lock` against 3.13 resolver (`uv lock --upgrade`)
- Update `docker/Dockerfile.sandbox:17` from `python:3.12-slim` to `python:3.13-slim`
- Update dep marker `symbolica-agentica>=0.1.0; python_version >= '3.12'` (keep marker but update to `>= '3.13'` — it becomes a no-op since requires-python already enforces it, but leaving it correct for discoverability)

### Out of scope (explicitly)
- No runtime code changes unless a 3.13 deprecation surfaces a real failure during full-suite run
- No adoption of new 3.13 syntax (PEP 695 type statements, PEP 742 TypeIs, etc.)
- No bump to 3.14 (bleeding edge; 3.13 is the stable floor target)
- No CI workflow changes — **survey confirmed no `.github/workflows/` directory exists in the repo**
- No Dockerfile changes beyond the sandbox image version

## 3. Survey Results (from actual repo read)

| File / Location | Current | Planned |
|---|---|---|
| `pyproject.toml:6` | `requires-python = ">=3.12"` | `">=3.13"` |
| `pyproject.toml:62` dep marker | `python_version >= '3.12'` | `python_version >= '3.13'` |
| `pyproject.toml:94` `[tool.black]` | `target-version = ["py311"]` | `["py313"]` |
| `pyproject.toml:97` `[tool.mypy]` | `python_version = "3.11"` | `"3.13"` |
| `pyproject.toml:117` `[tool.ruff]` | `target-version = "py311"` | `"py313"` |
| `docker/Dockerfile.sandbox:17` | `FROM python:3.12-slim` | `FROM python:3.13-slim` |
| `.python-version` | *(absent)* | new file containing `3.13` |
| `uv.lock` | resolved against 3.12 | regenerate via `uv lock --upgrade` |
| `.github/workflows/*.yml` | **none exist** | N/A |
| Python-version shebangs (`#!…python3.12`) | **none found** | N/A |

Python 3.13.12 and 3.14.3 are installed locally (verified).

## 4. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A pinned dep doesn't ship 3.13 wheels | Medium | High — blocks install | Gemini-pro Round 1 (dep compat) before impl; if found, push ARCHITECT BLOCKED |
| Stdlib 3.13 behavior change breaks silent assumption | Medium | Medium | Gemini-pro Round 3 (stdlib diff 3.12→3.13); full suite must be green |
| Sandbox container image breaks at runtime | Low | Medium | Dockerfile only changes the base tag; no logic change. Verify image builds in post-impl step (optional — document if skipped). |
| `uv lock --upgrade` pulls in unintended major bumps | Low | Medium | Review lockfile diff for major-version jumps before commit; flag any to ARCHITECT |
| Dev contributors on 3.12 get immediate failure | High | Low | Documented in PR body; `.python-version` + `requires-python` will produce a clear error |
| `typing.is_protocol` behaves subtly differently on 3.13 vs `typing_extensions` fallback | Low | Low | Test already exists; full-suite proves it. |

## 5. Implementation Steps (after plan approval)

1. Edit `pyproject.toml` (lines 6, 62, 94, 97, 117)
2. Write `.python-version` with content `3.13`
3. `uv lock --upgrade` — inspect diff for major bumps
4. `uv sync` — verify env builds clean
5. `uv run pytest tests/ -x` — must be green, incl. `test_protocol_is_a_protocol`
6. Update `docker/Dockerfile.sandbox:17`
7. Commit atomically:
   - `chore(py): bump Python floor to 3.13 (#113)` — pyproject + .python-version
   - `chore(py): regenerate uv.lock for Python 3.13`
   - `chore(docker): bump sandbox base image to python:3.13-slim`

## 6. Adversarial Reviews (3 rounds, Gemini-pro substitute for Codex)

Codex quota exhausted → using `mcp__gemini-mcp__gemini-query` (model='pro', thinkingLevel='high').

- **Round 1 — Dep Compatibility:** For every pinned dep in `pyproject.toml`, does the minimum version support Python 3.13? Highlight any that require a version bump. Key candidates to probe: `torch>=2.9.1`, `psycopg2-binary>=2.9.0`, `sentence-transformers>=5.2.0`, `asyncpg>=0.31.0`, `scipy>=1.16.3`, `jq>=1.6.0` (C extension).
- **Round 2 — CI + Deploy Blast Radius:** Confirm survey findings (no `.github/workflows/`, only Dockerfile.sandbox, no shebangs). Probe for systemd units, launchd plists, deploy scripts, hook-runner configs, mise/asdf files, GitHub Actions reusable workflows referenced from elsewhere.
- **Round 3 — 3.12 → 3.13 Stdlib Drift:** Removed/deprecated APIs (`aifc`, `chunk`, `crypt`, `imghdr`, `mailcap`, `msilib`, `nis`, `nntplib`, `ossaudiodev`, `pipes`, `sndhdr`, `spwd`, `sunau`, `telnetlib`, `uu`, `xdrlib` were removed in 3.13). `pathlib.Path` added `full_match`/`walk` changes. `asyncio.TaskGroup` stricter. Warn changes. Probe the codebase for any of these.

## 7. Security (Aegis + /security)

After Gemini rounds pass:
- Dispatch `Task(subagent_type='aegis', prompt='Audit the Python 3.13 bump: changed defaults in hashlib, ssl, subprocess; new transitive deps pulled by uv lock; deprecation of removed stdlib modules. Write findings to thoughts/shared/agents/aegis/issue-113-audit.md')`
- Run `/security` on changed configs before PR

## 8. PR & Reviewer Cycles

- `gh pr create --body-file …` with:
  - Before/after Python (3.12 → 3.13) and reason (Fixes #113)
  - Full-suite result
  - Lockfile diff summary (major bumps flagged, if any)
  - Aegis report link
  - "Reviewed with Gemini-pro (Codex quota outage) + Aegis"
- Address **2 cycles** of Copilot / CodeRabbit / Gemini Code Assist comments

## 9. Success Criteria

- [ ] `requires-python = ">=3.13"` in `pyproject.toml`
- [ ] `.python-version` pins `3.13`
- [ ] `uv.lock` regenerated and committed
- [ ] `test_protocol_is_a_protocol` passes (no code change to source)
- [ ] Full suite green (previously 2156 passed + 1 failed + 1 skipped → expected 2157 passed + 1 skipped)
- [ ] `[tool.black|mypy|ruff]` versions aligned to 3.13
- [ ] Dockerfile.sandbox bumped
- [ ] 3 Gemini-pro review rounds completed with findings addressed
- [ ] Aegis audit done; `/security` clean
- [ ] PR merged after 2+ AI reviewer cycles

## 10. Rollback

If a dep fails 3.13 or uv resolution blows up:
1. Push `ARCHITECT BLOCKED:` with the specific failure
2. Options: (a) bump the offending dep to a 3.13-supporting version, (b) revert to `>=3.12` and choose the `typing_extensions` path on #113 instead
3. Worktree + branch can be nuked without touching `main`
