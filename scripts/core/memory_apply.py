#!/usr/bin/env python3
"""Promotion apply for the memory-review workflow (issue #63 Phase 2a).

Applies APPROVED promotion candidates from the read-only detector
(``scripts/core/memory_review.py``) to the always-loaded memory tiers. Safety rails:

* **Dry-run is the default.** Nothing is written unless ``execute=True``.
* **Backups before any mutation.** A ``pg_dump`` of ``archival_memory`` AND a snapshot of
  the files about to change (``MEMORY.md``, ``CLAUDE.md``) are written first; apply aborts
  if the DB dump fails. Newly-created ``promoted-*.md`` files need no backup (undo = delete).
* **Serialized.** The execute phase holds a per-project flock so concurrent applies can't
  clobber each other's file edits.
* **Reversible & idempotent.** Promotion appends to the target file and tags the source row
  with STRUCTURED provenance (tier, exact target path, timestamp) — it never deletes the
  row, and re-applying an already-tagged learning is skipped.

Targets in Phase 2a: ``MEMORY.md`` (Claude auto-memory) and ``CLAUDE.md`` (opc-local).
``rules/`` (a separate repo), merge/archive cleanup-apply, and a one-shot operation-level
``unpromote``/``repair`` command are deferred to later phases — the structured provenance
written here is the enabler for that undo path.

The read-only detector is intentionally untouched: all write logic lives here.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from asyncpg.exceptions import PostgresError  # noqa: E402

from scripts.core.db.postgres_pool import close_pool, get_pool  # noqa: E402
from scripts.core.memory_review import PromotionCandidate, route_destination  # noqa: E402
from scripts.core.project_naming import canonicalize_project, project_from_path  # noqa: E402

# pg_dump backup target. The running container is `continuous-claude-postgres` (per the
# project CLAUDE.md and the live setup), which differs from the stale `container_name` in
# docker/docker-compose.yml — so the name is env-overridable to stay portable.
_BACKUP_CONTAINER = os.environ.get("OPC_PG_CONTAINER", "continuous-claude-postgres")
_BACKUP_USER = os.environ.get("OPC_PG_USER", "claude")
_BACKUP_DB = os.environ.get("OPC_PG_DB", "continuous_claude")

# learning_type -> apply target (subset of memory_review routing supported in Phase 2a).
# The detector may route a candidate to "rules/" (USER_PREFERENCE) — that target is deferred
# (separate repo), so build_plan surfaces it as a skip rather than writing it.
_APPLY_ROUTING: dict[str, str] = {
    "CODEBASE_PATTERN": "MEMORY.md",
    "ARCHITECTURAL_DECISION": "CLAUDE.md",
}

_SLUG_MAXLEN = 60
_SLUG_FALLBACK = "promoted-learning"

# Frontmatter key that records the source learning id in a promoted memory file. Used as
# the authoritative provenance marker for same-candidate vs slug-collision detection.
SOURCE_MARKER_KEY = "source_learning_id"


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


def _sanitize_content(text: str) -> str:
    """Neutralize a learning body that tries to forge an idempotency marker — defang the
    exact ``promoted_from_archival_memory`` token so stored content can't fake a promotion
    marker (or an HTML comment) and cause a different promotion to be wrongly skipped."""
    return text.replace("promoted_from_archival_memory", "promoted-from-archival-memory")


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


def _memory_name(candidate: PromotionCandidate) -> str:
    """Base Claude-memory file stem for a candidate (no extension)."""
    return f"promoted-{slugify(candidate.content)}"


def memory_entry(candidate: PromotionCandidate, name: str | None = None) -> tuple[str, str]:
    """Build (filename, file_body) for a MEMORY.md promotion as a Claude-memory file.

    ``name`` overrides the file stem (used to dodge slug collisions). The full source id is
    embedded in frontmatter as the authoritative provenance marker — apply uses it to tell a
    same-candidate re-apply from a different-candidate slug collision.
    """
    stem = name or _memory_name(candidate)
    filename = f"{stem}.md"
    # Frontmatter values are double-quoted so dynamic text (the recall count's ×, etc.)
    # is always a valid YAML scalar; the learning body lives below the frontmatter where
    # it cannot affect the block. stem/id are slug/uuid (no quote chars) by construction.
    body = (
        "---\n"
        f'name: "{stem}"\n'
        f'description: "Promoted from archival_memory (recalled {candidate.recall_count}×)"\n'
        "metadata:\n"
        "  type: reference\n"
        f'  {SOURCE_MARKER_KEY}: "{candidate.id}"\n'
        "---\n\n"
        f"{_sanitize_content(candidate.content.strip())}\n"
    )
    return filename, body


def memory_index_line(candidate: PromotionCandidate, filename: str) -> str:
    """One-line MEMORY.md index pointer for a promoted memory file."""
    title = _short(candidate.content, 60)
    return f"- [{title}]({filename}) — promoted, recalled {candidate.recall_count}×"


def claude_md_marker(candidate: PromotionCandidate) -> str:
    """Exact, structured idempotency marker carrying the FULL learning id (no substring traps)."""
    return f"<!-- promoted_from_archival_memory: {candidate.id} -->"


def claude_md_block(candidate: PromotionCandidate) -> str:
    """Markdown block appended to CLAUDE.md for an ARCHITECTURAL_DECISION promotion."""
    return (
        f"- {_sanitize_content(candidate.content.strip())} "
        f"_(promoted from archival_memory {candidate.id[:8]}, recalled {candidate.recall_count}×)_ "
        f"{claude_md_marker(candidate)}"
    )


# --- I/O handlers ----------------------------------------------------------

_PROMOTED_IDS_SQL = """
    SELECT id::text AS id
    FROM archival_memory
    WHERE LOWER(project) = LOWER($1)
      AND metadata ? 'promoted_to'
"""

# Merge a STRUCTURED provenance object into metadata (not a bare string) so a future
# repair/unpromote pass can find exactly what was written where. tier/target/timestamp are
# bound params ($2..$4) — jsonb_build_object never lets them carry SQL.
# Scoped to the SAME predicates the candidate was fetched under (project + not superseded)
# so the tag can only land on a current row. asyncpg returns "UPDATE <n>" — write_provenance
# verifies n == 1 so a row that was superseded/removed between fetch and write surfaces as an
# error instead of silently leaving the file promoted but the row untagged.
_PROVENANCE_SQL = """
    UPDATE archival_memory
    SET metadata = metadata || jsonb_build_object(
        'promoted_to', jsonb_build_object('tier', $2::text, 'target', $3::text, 'at', $4::text)
    )
    WHERE id = $1::uuid
      AND LOWER(project) = LOWER($5)
      AND superseded_by IS NULL
"""


async def fetch_promoted_ids(pool, project: str) -> set[str]:
    """IDs already promoted (tagged with promoted_to) for idempotency."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(_PROMOTED_IDS_SQL, project)
    return {r["id"] for r in rows}


async def write_provenance(
    pool, learning_id: str, *, project: str, tier: str, target: str, at: str
) -> None:
    """Tag a source row with structured promotion provenance (tier, exact target, timestamp).

    Reversible (clear the key) and never deletes the row. Verifies exactly one current row
    was tagged — a stale id, or a row superseded/removed between fetch and write, raises
    rather than silently reporting a promotion whose DB side never landed. Storing the exact
    target path + timestamp is what a Phase-2b unpromote/repair pass needs to undo this write.
    """
    async with pool.acquire() as conn:
        status = await conn.execute(_PROVENANCE_SQL, learning_id, tier, target, at, project)
    if status != "UPDATE 1":
        raise RuntimeError(
            f"provenance tag did not update exactly one row (status={status!r}) for "
            f"{learning_id} in project {project!r}: the row may have been superseded or removed"
        )


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
        try:
            f = os.fdopen(fd, "w")
        except (OSError, ValueError):
            os.close(fd)  # fdopen failed without taking ownership of fd — close it ourselves
            raise
        with f:
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
    # Dump to a temp file and rename only on success, so a spawn failure (docker missing)
    # or a non-zero pg_dump never leaves a misleading empty/partial backup at `dest`.
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".sql.tmp")
    os.close(fd)  # reopen by path so the file handle carries a real path name
    tmp = Path(tmp_name)
    try:
        with open(tmp, "w") as fh:
            result = run(cmd, stdout=fh, stderr=subprocess.PIPE)
        rc = getattr(result, "returncode", 1)
        if rc != 0:
            stderr = getattr(result, "stderr", b"") or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            raise RuntimeError(
                f"database backup failed (pg_dump returncode={rc}): {stderr[:500]}; aborting apply"
            )
        tmp.replace(dest)
        os.chmod(dest, 0o600)  # the dump carries all archival_memory content — owner-only
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def backup_files(backup_dir: Path, timestamp: str, files: list[Path]) -> list[Path]:
    """Snapshot the files an apply will MUTATE (MEMORY.md, CLAUDE.md) before --execute.

    The pg_dump only covers the DB table; these copies make the FILE side recoverable too.
    Newly-created promoted-*.md files need no backup (undo = delete them). Missing files are
    skipped (nothing to restore). Returns the list of backup copies written.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    copies: list[Path] = []
    for src in files:
        if not src.exists():
            continue
        dest = backup_dir / f"memory-apply-{timestamp}-{src.name}.bak"
        _atomic_write(dest, src.read_text())
        copies.append(dest)
    return copies


@contextmanager
def _apply_lock(lock_dir: Path, project: str):
    """Serialize apply per project: an flock so two concurrent applies can't clobber each
    other's MEMORY.md/CLAUDE.md edits or both pass the DB idempotency check.

    The lock is keyed to the (stable) memory dir + project, NOT the overridable backup dir,
    so two applies to the same memory tree serialize even with different --backup-dir."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-z0-9_-]+", "-", project.lower()) or "project"
    lock_path = lock_dir / f".memory-apply-{safe}.lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def apply_memory_file(memory_dir: Path, candidate: PromotionCandidate) -> Path | None:
    """Ensure the promoted Claude-memory file AND its MEMORY.md pointer exist.

    Returns the file Path once both artifacts are present (the caller tags provenance with
    that exact path only then), or None on a no-op failure. Reconciles a partial prior
    apply: if the file already exists but the index pointer is missing, the pointer is still
    appended. Slug collisions with a *different* source id get a collision-free filename (so
    one candidate never silently shadows another). Raises on I/O failure — the row is then
    left untagged and a re-run resumes safely.
    """
    base = _memory_name(candidate)
    path = memory_dir / f"{base}.md"
    if path.exists() and f'{SOURCE_MARKER_KEY}: "{candidate.id}"' not in path.read_text():
        # Same slug, different learning — disambiguate with the id so neither is shadowed.
        base = f"{base}-{candidate.id[:8]}"
        path = memory_dir / f"{base}.md"
    filename, body = memory_entry(candidate, name=base)
    if not path.exists():
        _atomic_write(path, body)
    # Always reconcile the index pointer (idempotent), even if the file pre-existed.
    _append_line_if_absent(memory_dir / "MEMORY.md", memory_index_line(candidate, filename))
    return path


_CLAUDE_SECTION = "## Promoted Decisions"


def append_claude_md(path: Path, candidate: PromotionCandidate) -> Path | None:
    """Ensure the decision block is present in CLAUDE.md under the Promoted section.

    Idempotency uses the EXACT structured marker carrying the full learning id (not an
    8-char substring), so unrelated text can never falsely suppress a promotion. Returns the
    CLAUDE.md Path once the block is present (caller tags provenance only then).
    """
    text = path.read_text() if path.exists() else ""
    if claude_md_marker(candidate) in text:
        return path  # already promoted (exact full-id marker) — present, nothing to do
    block = claude_md_block(candidate)
    if _CLAUDE_SECTION in text:
        # Insert directly under the section header so the block lands in the right section
        # even when other sections follow it (appending to EOF would misfile it).
        head, rest = text.split(_CLAUDE_SECTION, 1)
        sep = "" if rest.startswith("\n") else "\n"
        text = head + _CLAUDE_SECTION + "\n" + block + sep + rest
    else:
        prefix = (text.rstrip() + "\n\n") if text.strip() else ""
        text = prefix + _CLAUDE_SECTION + "\n" + block + "\n"
    _atomic_write(path, text)
    return path


# --- Orchestrator ----------------------------------------------------------


@dataclass(frozen=True)
class ApplyResult:
    plan: ApplyPlan
    applied: list[str] = field(default_factory=list)
    backup_path: Path | None = None
    failed: list[tuple[str, str]] = field(default_factory=list)


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

    Dry-run (execute=False) performs ZERO writes and takes NO backup. Under execute the
    whole mutation phase is serialized by a per-project flock (concurrent applies can't
    clobber each other's file edits); a DB-table dump AND a snapshot of the files about to
    be mutated are taken before the first write; each item writes its file then tags the
    source row with structured provenance (file-first, so a tag never outlives a failed
    write — and only the exact written path is recorded). Items are independent and
    idempotent, so a partial failure is safe to re-run.
    """
    candidates = await fetch_candidates_by_ids(pool, project, approved_ids)
    already = await fetch_promoted_ids(pool, project)
    plan = build_plan(candidates, already, dry_run=not execute)

    if not execute or not plan.applicable:
        return ApplyResult(plan=plan, applied=[], backup_path=None)

    with _apply_lock(memory_dir, project):
        backup_path = backup_database(backup_dir / f"memory-apply-{timestamp}.sql", run=run)
        backup_files(backup_dir, timestamp, [memory_dir / "MEMORY.md", claude_md_path])
        applied: list[str] = []
        failed: list[tuple[str, str]] = []
        for action in plan.applicable:
            c = action.candidate
            tier = action.target
            if tier is None:  # unreachable for applicable actions; satisfies type + defensive
                continue
            # File-first, tag-only-on-confirmed-success: the writer returns the exact target
            # path only once the artifact (file + index pointer, or CLAUDE.md block) is
            # present, so a skipped/partial write can never leave the row tagged-but-unwritten.
            # A write that raises propagates out untagged; the item is idempotent on re-run.
            if tier == "MEMORY.md":
                artifact = apply_memory_file(memory_dir, c)
            elif tier == "CLAUDE.md":
                artifact = append_claude_md(claude_md_path, c)
            else:
                artifact = None
            if artifact is None:
                continue
            try:
                await write_provenance(
                    pool, c.id, project=project, tier=tier, target=str(artifact), at=timestamp
                )
            except RuntimeError as exc:
                # Row vanished/superseded between fetch and tag: the file is written (and
                # idempotent), but we could not record provenance. Don't claim it applied;
                # surface it and keep going so one bad row doesn't block the rest.
                failed.append((c.id, str(exc)))
                continue
            applied.append(c.id)
    return ApplyResult(plan=plan, applied=applied, backup_path=backup_path, failed=failed)


# --- CLI -------------------------------------------------------------------


def _project_dir() -> str:
    """The real project directory (worktree-aware via CLAUDE_PROJECT_DIR), for path defaults."""
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def default_memory_dir(project_dir: str) -> Path:
    """Claude auto-memory dir for a project path: ~/.claude/projects/<flattened>/memory."""
    flattened = project_dir.replace("/", "-")
    return Path.home() / ".claude" / "projects" / flattened / "memory"


def default_backup_dir() -> Path:
    """Backups live OUTSIDE any git working tree (a pg_dump of the whole archival_memory
    table can carry secrets; in-repo it risks accidental commit + cross-project leakage)."""
    return Path.home() / ".claude" / "opc-backups"


_MANIFEST_MAX_BYTES = 256 * 1024
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def parse_ids(raw_ids: str | None, manifest: str | None) -> list[str]:
    """Approved ids from a comma list and/or a manifest file (one id per line).

    The manifest read is bounded (a huge/binary file is rejected rather than slurped).
    """
    ids: list[str] = []
    if raw_ids:
        ids.extend(part.strip() for part in raw_ids.split(",") if part.strip())
    if manifest:
        mpath = Path(manifest)
        if mpath.stat().st_size > _MANIFEST_MAX_BYTES:
            raise ValueError(
                f"--manifest file is too large ({mpath.stat().st_size} bytes > "
                f"{_MANIFEST_MAX_BYTES}); pass a list of ids, not arbitrary data"
            )
        for line in mpath.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    # de-dup, preserve order
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def validate_ids(ids: list[str]) -> tuple[list[str], list[str]]:
    """Split ids into (valid uuids, malformed). Malformed ids never reach SQL/filenames."""
    valid: list[str] = []
    invalid: list[str] = []
    for i in ids:
        # Canonicalize to lowercase: archival_memory.id::text is lowercase, so an uppercase
        # CLI uuid would otherwise silently match no candidate.
        if _UUID_RE.match(i):
            valid.append(i.lower())
        else:
            invalid.append(i)
    return valid, invalid


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

    try:
        ids = parse_ids(args.ids, args.manifest)
    except (OSError, ValueError) as exc:
        print(f"memory-apply: could not read ids: {exc}", file=sys.stderr)
        return 2
    ids, invalid = validate_ids(ids)
    for bad in invalid:
        print(f"memory-apply: ignoring malformed id (not a uuid): {bad!r}", file=sys.stderr)
    if not ids:
        print(
            "memory-apply: no valid approved ids given (use --ids or --manifest).",
            file=sys.stderr,
        )
        return 2

    project_dir = _project_dir()
    # Fail-closed guard (round 2): the DB project comes from the explicit arg, but the file
    # write roots derive from the working tree. If they disagree (e.g. `memory-apply other`
    # from the opc tree, or a stale CLAUDE_PROJECT_DIR), --execute would fetch one project's
    # candidates and write another's files. Require explicit --memory-dir AND --claude-md to
    # proceed in that case, so a mismatch can never silently target the wrong tree.
    cwd_project = project_from_path(project_dir)
    if args.execute and project != cwd_project and not (args.memory_dir and args.claude_md):
        print(
            f"memory-apply: refusing to --execute: requested project '{project}' differs from "
            f"the working tree's project '{cwd_project}'. Pass explicit --memory-dir and "
            "--claude-md so the write target is unambiguous.",
            file=sys.stderr,
        )
        return 2

    memory_dir = Path(args.memory_dir) if args.memory_dir else default_memory_dir(project_dir)
    claude_md = Path(args.claude_md) if args.claude_md else Path(project_dir) / "CLAUDE.md"
    backup_dir = Path(args.backup_dir) if args.backup_dir else default_backup_dir()
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
    except (OSError, RuntimeError, PostgresError) as exc:
        # RuntimeError covers a failed pg_dump backup (apply aborts before any mutation);
        # PostgresError is caught so a DB failure can't dump a traceback that echoes the DSN.
        print(f"memory-apply: aborted ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    resolved = {a.candidate.id for a in result.plan.actions}
    for missing in [i for i in ids if i not in resolved]:
        print(f"memory-apply: id not found / not a current learning: {missing}", file=sys.stderr)

    print(render_plan(result.plan))
    if args.execute:
        print("")
        print(f"Applied {len(result.applied)} promotion(s).")
        if result.backup_path:
            print(f"DB backup: {result.backup_path}")
        if result.failed:
            for fid, reason in result.failed:
                print(f"  ⚠️ not tagged [{fid[:8]}]: {reason}", file=sys.stderr)
            return 1
    return 0


async def _cli_main(argv: list[str] | None = None) -> int:
    try:
        return await main(argv)
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(_cli_main()))
