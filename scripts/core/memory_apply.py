#!/usr/bin/env python3
"""Promotion apply for the memory-review workflow (issue #63 Phase 2a).

Applies APPROVED promotion candidates from the read-only detector
(``scripts/core/memory_review.py``) to the always-loaded memory tiers. Safety rails:

* **Dry-run is the default.** Nothing is written unless ``execute=True``.
* **DB backup before any mutation.** ``backup_database`` runs ``pg_dump`` to a
  timestamped file; apply aborts if it fails.
* **Reversible & idempotent.** Promotion appends to the target file and tags the source
  ``archival_memory`` row's metadata with ``promoted_to`` — it never deletes the row, and
  re-applying an already-tagged learning is skipped.

Targets in Phase 2a: ``MEMORY.md`` (Claude auto-memory) and ``CLAUDE.md`` (opc-local).
``rules/`` (a separate repo) and merge/archive cleanup-apply are deferred to later phases.

The read-only detector is intentionally untouched: all write logic lives here.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.core.db.postgres_pool import close_pool, get_pool  # noqa: E402
from scripts.core.memory_review import PromotionCandidate, route_destination  # noqa: E402
from scripts.core.project_naming import canonicalize_project, project_from_path  # noqa: E402

# pg_dump backup target (the Dockerized PostgreSQL from docker/docker-compose.yml).
_BACKUP_CONTAINER = "continuous-claude-postgres"
_BACKUP_USER = "claude"
_BACKUP_DB = "continuous_claude"

# learning_type -> apply target (subset of memory_review routing supported in Phase 2a).
# The detector may route a candidate to "rules/" (USER_PREFERENCE) — that target is deferred
# (separate repo), so build_plan surfaces it as a skip rather than writing it.
_APPLY_ROUTING: dict[str, str] = {
    "CODEBASE_PATTERN": "MEMORY.md",
    "ARCHITECTURAL_DECISION": "CLAUDE.md",
}

_SLUG_MAXLEN = 60
_SLUG_FALLBACK = "promoted-learning"


# --- Data structures -------------------------------------------------------


@dataclass(frozen=True)
class ApplyAction:
    candidate: PromotionCandidate
    target: str | None
    skipped: bool
    skip_reason: str | None


@dataclass(frozen=True)
class ApplyPlan:
    actions: list[ApplyAction] = field(default_factory=list)
    dry_run: bool = True

    @property
    def applicable(self) -> list[ApplyAction]:
        return [a for a in self.actions if not a.skipped]


# --- Pure functions --------------------------------------------------------


def route_apply_target(learning_type: str) -> str | None:
    """Apply target for a learning_type, or None if unsupported/deferred in Phase 2a."""
    return _APPLY_ROUTING.get(learning_type)


def slugify(text: str) -> str:
    """Lowercase kebab slug, bounded length, with a stable fallback for empty input."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not cleaned:
        return _SLUG_FALLBACK
    if len(cleaned) > _SLUG_MAXLEN:
        cleaned = cleaned[:_SLUG_MAXLEN].rstrip("-")
    return cleaned or _SLUG_FALLBACK


def build_plan(
    candidates: list[PromotionCandidate],
    already_promoted: set[str],
    dry_run: bool,
) -> ApplyPlan:
    """Pure: map approved candidates to apply actions, marking idempotent/unsupported skips."""
    actions: list[ApplyAction] = []
    for c in candidates:
        if c.id in already_promoted:
            actions.append(
                ApplyAction(c, None, skipped=True, skip_reason="already promoted (provenance tag)")
            )
            continue
        target = route_apply_target(c.learning_type)
        if target is None:
            reason = f"target for {c.learning_type} is deferred" + (
                " (rules/ is a separate repo)" if c.destination == "rules/" else ""
            )
            actions.append(ApplyAction(c, None, skipped=True, skip_reason=reason))
            continue
        actions.append(ApplyAction(c, target, skipped=False, skip_reason=None))
    return ApplyPlan(actions=actions, dry_run=dry_run)


def _short(text: str, width: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def render_plan(plan: ApplyPlan) -> str:
    """Render the apply plan as a readable preview. Pure — writes nothing."""
    lines: list[str] = []
    banner = "DRY RUN — no changes written" if plan.dry_run else "EXECUTE — writing changes"
    lines.append(f"## Promotion Apply Plan ({banner})")
    applicable = plan.applicable
    lines.append(f"{len(applicable)} to apply, {len(plan.actions) - len(applicable)} skipped")
    lines.append("")
    for a in plan.actions:
        if a.skipped:
            lines.append(f"  skip [{a.candidate.id[:8]}] {a.skip_reason}")
            lines.append(f"        {_short(a.candidate.content, 60)}")
        else:
            lines.append(
                f"  → {a.target}  [{a.candidate.id[:8]}, {a.candidate.learning_type}, "
                f"recalled {a.candidate.recall_count}×]"
            )
            lines.append(f"        {_short(a.candidate.content)}")
    if plan.dry_run and applicable:
        lines.append("")
        lines.append("_Re-run with --execute to apply (a DB backup is taken first)._")
    return "\n".join(lines)


def memory_entry(candidate: PromotionCandidate) -> tuple[str, str]:
    """Build (filename, file_body) for a MEMORY.md promotion as a Claude-memory file."""
    slug = slugify(candidate.content)
    filename = f"promoted-{slug}.md"
    body = (
        "---\n"
        f"name: promoted-{slug}\n"
        f"description: Promoted from archival_memory (recalled {candidate.recall_count}×)\n"
        "metadata:\n"
        "  type: reference\n"
        f"  source_learning_id: {candidate.id}\n"
        "---\n\n"
        f"{candidate.content.strip()}\n"
    )
    return filename, body


def memory_index_line(candidate: PromotionCandidate, filename: str) -> str:
    """One-line MEMORY.md index pointer for a promoted memory file."""
    title = _short(candidate.content, 60)
    return f"- [{title}]({filename}) — promoted, recalled {candidate.recall_count}×"


def claude_md_block(candidate: PromotionCandidate) -> str:
    """Markdown block appended to CLAUDE.md for an ARCHITECTURAL_DECISION promotion."""
    return (
        f"- {candidate.content.strip()} "
        f"_(promoted from archival_memory {candidate.id[:8]}, recalled {candidate.recall_count}×)_"
    )


# --- I/O handlers ----------------------------------------------------------

_PROMOTED_IDS_SQL = """
    SELECT id::text AS id
    FROM archival_memory
    WHERE LOWER(project) = LOWER($1)
      AND metadata ? 'promoted_to'
"""

# Merge a provenance object into metadata without clobbering existing keys. The tier is
# bound as $2 (jsonb-encoded by asyncpg) so it can never carry SQL.
_PROVENANCE_SQL = """
    UPDATE archival_memory
    SET metadata = metadata || jsonb_build_object('promoted_to', $2::text)
    WHERE id = $1::uuid
"""


async def fetch_promoted_ids(pool, project: str) -> set[str]:
    """IDs already promoted (tagged with promoted_to) for idempotency."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(_PROMOTED_IDS_SQL, project)
    return {r["id"] for r in rows}


async def write_provenance(pool, learning_id: str, tier: str) -> None:
    """Tag a source row as promoted. Reversible (clear the key); never deletes the row."""
    async with pool.acquire() as conn:
        await conn.execute(_PROVENANCE_SQL, learning_id, tier)


_CANDIDATES_BY_IDS_SQL = """
    SELECT id::text AS id,
           content,
           recall_count,
           metadata->>'learning_type' AS learning_type
    FROM archival_memory
    WHERE LOWER(project) = LOWER($1)
      AND superseded_by IS NULL
      AND id::text = ANY($2::text[])
"""


async def fetch_candidates_by_ids(pool, project: str, ids: list[str]) -> list[PromotionCandidate]:
    """Resolve approved ids to promotion candidates (independent of the recall threshold)."""
    if not ids:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(_CANDIDATES_BY_IDS_SQL, project, list(ids))
    out: list[PromotionCandidate] = []
    for r in rows:
        lt = r["learning_type"]
        out.append(
            PromotionCandidate(
                id=r["id"],
                content=r["content"],
                recall_count=int(r["recall_count"]),
                learning_type=lt,
                destination=route_destination(lt) or "?",
            )
        )
    return out


# --- Filesystem + backup handlers ------------------------------------------


def _atomic_write(path: Path, content: str) -> None:
    """Write a file atomically (temp + rename) so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _append_line_if_absent(path: Path, line: str) -> bool:
    """Append a single line to a file if not already present. Returns True if appended."""
    existing = path.read_text() if path.exists() else ""
    if line in existing:
        return False
    base = existing if existing.endswith("\n") or not existing else existing + "\n"
    _atomic_write(path, base + line + "\n")
    return True


def backup_database(dest: Path, *, run=subprocess.run) -> Path:
    """pg_dump the archival_memory table to `dest`. Raises RuntimeError on failure.

    Always run before any --execute DB mutation so a bad apply is recoverable.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "exec",
        _BACKUP_CONTAINER,
        "pg_dump",
        "-U",
        _BACKUP_USER,
        "-d",
        _BACKUP_DB,
        "-t",
        "archival_memory",
    ]
    with open(dest, "w") as fh:
        result = run(cmd, stdout=fh, stderr=subprocess.PIPE)
    if getattr(result, "returncode", 1) != 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"database backup failed (pg_dump returncode={getattr(result, 'returncode', None)}); "
            "aborting apply"
        )
    return dest


def apply_memory_file(memory_dir: Path, candidate: PromotionCandidate) -> Path | None:
    """Create the promoted Claude-memory file + index pointer. Idempotent (None if present)."""
    filename, body = memory_entry(candidate)
    path = memory_dir / filename
    if path.exists():
        return None
    _atomic_write(path, body)
    _append_line_if_absent(memory_dir / "MEMORY.md", memory_index_line(candidate, filename))
    return path


_CLAUDE_SECTION = "## Promoted Decisions"


def append_claude_md(path: Path, candidate: PromotionCandidate) -> bool:
    """Append a decision block to CLAUDE.md under a Promoted section. Idempotent by id."""
    text = path.read_text() if path.exists() else ""
    marker = candidate.id[:8]
    if marker in text:
        return False
    if _CLAUDE_SECTION not in text:
        text = (
            (text.rstrip() + "\n\n" + _CLAUDE_SECTION + "\n")
            if text.strip()
            else (_CLAUDE_SECTION + "\n")
        )
    text = text.rstrip() + "\n" + claude_md_block(candidate) + "\n"
    _atomic_write(path, text)
    return True


# --- Orchestrator ----------------------------------------------------------


@dataclass(frozen=True)
class ApplyResult:
    plan: ApplyPlan
    applied: list[str] = field(default_factory=list)
    backup_path: Path | None = None


async def run_apply(
    pool,
    project: str,
    approved_ids: list[str],
    *,
    execute: bool,
    memory_dir: Path,
    claude_md_path: Path,
    backup_dir: Path,
    timestamp: str,
    run=subprocess.run,
) -> ApplyResult:
    """Resolve approved ids, plan, and (only if execute) back up then write + tag.

    Dry-run (execute=False) performs ZERO writes and takes NO backup. Under execute, the
    DB backup runs before the first mutation; each item writes its file then tags the
    source row (file-first so a tag never outlives a failed write). Items are independent
    and idempotent, so a partial failure is safe to re-run.
    """
    candidates = await fetch_candidates_by_ids(pool, project, approved_ids)
    already = await fetch_promoted_ids(pool, project)
    plan = build_plan(candidates, already, dry_run=not execute)

    if not execute or not plan.applicable:
        return ApplyResult(plan=plan, applied=[], backup_path=None)

    backup_path = backup_database(backup_dir / f"memory-apply-{timestamp}.sql", run=run)
    applied: list[str] = []
    for action in plan.applicable:
        c = action.candidate
        tier = action.target
        if tier is None:  # unreachable for applicable actions; satisfies the type + defensive
            continue
        if tier == "MEMORY.md":
            apply_memory_file(memory_dir, c)
        elif tier == "CLAUDE.md":
            append_claude_md(claude_md_path, c)
        await write_provenance(pool, c.id, tier)
        applied.append(c.id)
    return ApplyResult(plan=plan, applied=applied, backup_path=backup_path)


# --- CLI -------------------------------------------------------------------


def _project_dir() -> str:
    """The real project directory (worktree-aware via CLAUDE_PROJECT_DIR), for path defaults."""
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def default_memory_dir(project_dir: str) -> Path:
    """Claude auto-memory dir for a project path: ~/.claude/projects/<flattened>/memory."""
    flattened = project_dir.replace("/", "-")
    return Path.home() / ".claude" / "projects" / flattened / "memory"


def parse_ids(raw_ids: str | None, manifest: str | None) -> list[str]:
    """Approved ids from a comma list and/or a manifest file (one id per line)."""
    ids: list[str] = []
    if raw_ids:
        ids.extend(part.strip() for part in raw_ids.split(",") if part.strip())
    if manifest:
        for line in Path(manifest).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    # de-dup, preserve order
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply approved memory promotions (issue #63 Phase 2a). Dry-run by default."
    )
    p.add_argument(
        "project", nargs="?", default=None, help="Project (default: cwd, worktree-aware)"
    )
    p.add_argument("--ids", default=None, help="Comma-separated approved learning ids")
    p.add_argument("--manifest", default=None, help="File of approved ids (one per line)")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Perform the writes (default: dry-run, writes nothing). A DB backup runs first.",
    )
    p.add_argument("--memory-dir", default=None, help="Override Claude memory dir")
    p.add_argument("--claude-md", default=None, help="Override CLAUDE.md path")
    p.add_argument("--backup-dir", default=None, help="Override backup output dir")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project = (
        canonicalize_project(args.project) if args.project else project_from_path(_project_dir())
    )
    if not project:
        print("memory-apply: could not resolve a project. Pass one explicitly.", file=sys.stderr)
        return 2

    ids = parse_ids(args.ids, args.manifest)
    if not ids:
        print("memory-apply: no approved ids given (use --ids or --manifest).", file=sys.stderr)
        return 2

    project_dir = _project_dir()
    memory_dir = Path(args.memory_dir) if args.memory_dir else default_memory_dir(project_dir)
    claude_md = Path(args.claude_md) if args.claude_md else Path(project_dir) / "CLAUDE.md"
    backup_dir = Path(args.backup_dir) if args.backup_dir else Path(project_dir) / "backups"
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    # Echo resolved write targets so a wrong default is caught in dry-run before --execute.
    print(f"memory-apply: project={project}")
    print(f"  memory dir : {memory_dir}")
    print(f"  CLAUDE.md  : {claude_md}")
    if args.execute:
        print(f"  backup dir : {backup_dir}")
    print("")

    try:
        pool = await get_pool()
        result = await run_apply(
            pool,
            project,
            ids,
            execute=args.execute,
            memory_dir=memory_dir,
            claude_md_path=claude_md,
            backup_dir=backup_dir,
            timestamp=timestamp,
        )
    except ValueError as exc:
        print(f"memory-apply: configuration error: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as exc:
        # RuntimeError covers a failed pg_dump backup (apply aborts before any mutation).
        print(f"memory-apply: aborted ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    print(render_plan(result.plan))
    if args.execute:
        print("")
        print(f"Applied {len(result.applied)} promotion(s).")
        if result.backup_path:
            print(f"DB backup: {result.backup_path}")
    return 0


async def _cli_main(argv: list[str] | None = None) -> int:
    try:
        return await main(argv)
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(_cli_main()))
