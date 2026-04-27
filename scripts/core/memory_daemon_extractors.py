"""Extraction subprocess and post-extraction pipeline for the memory daemon.

Moved from memory_daemon.py in Phase 4 of S30 TDD+FP refactor.
Only extraction logic lives here — scheduling (reap, watchdog, queue)
stays in the orchestrator per D13.

Dependencies:
- memory_daemon_core: StaleSession, _normalize_project, strip_yaml_frontmatter,
                      build_extraction_command, build_extraction_env, _ALLOWED_EXTRACTION_MODELS
- memory_daemon_db: pg_mark_archived
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.core.log_safety import safe
from scripts.core.memory_daemon_core import (
    build_extraction_command,
    build_extraction_env,
)

logger = logging.getLogger("memory-daemon")


# Fallback prompt used when CLAUDE_CONFIG_DIR/agents/memory-extractor.md is
# absent. Hoisted to a module-level constant so a single source of truth is
# importable by adjacent CLIs (e.g. scripts.core.extract_session) and a
# drift assertion can detect divergence at test time. Keep this byte-stable
# unless updating both production paths in lock-step.
_FALLBACK_AGENT_PROMPT = (
    "Extract learnings from this Claude Code session.\n"
    "Look for decisions, what worked, what failed, and patterns discovered.\n"
    "Store each learning using store_learning.py with appropriate type and tags."
)


# ---------------------------------------------------------------------------
# Step 4.1 — _is_extraction_blocked + extract_memories_impl
# ---------------------------------------------------------------------------


def is_extraction_blocked(project_dir: str) -> bool:
    """Return True if this project has opted out of memory extraction."""
    if not project_dir:
        return False
    sentinel = Path(project_dir) / ".claude" / "no-extract"
    return sentinel.exists()


def extract_memories_impl(
    session_id: str,
    project_dir: str,
    transcript_path: str | None,
    *,
    active_extractions: dict,
    subprocess_popen: Callable,
    is_blocked_fn: Callable[[str], bool],
    mark_extracted_fn: Callable[[str], None],
    mark_failed_fn: Callable[[str], None],
    log_fn: Callable[[str], None],
    daemon_cfg: Any,
    allowed_models: frozenset[str],
    strip_frontmatter_fn: Callable[[str], str],
) -> bool:
    """Run memory extraction for a session. Returns True if subprocess started.

    All collaborators are injected (D1) — subprocess_popen, mark_extracted,
    and _daemon_cfg are passed in, not imported directly. The module-level
    subprocess import is used only for subprocess.DEVNULL constants.
    """
    log_fn(
        f"Extracting memories for session {safe(session_id)} "
        f"(project={safe(project_dir or 'unknown')})"
    )

    if is_blocked_fn(project_dir):
        log_fn(
            f"Extraction blocked by .claude/no-extract sentinel "
            f"(project={safe(project_dir)}), marking as extracted (skip)"
        )
        mark_extracted_fn(session_id)
        return False

    # Use transcript_path from DB — no glob fallback
    jsonl_path = None
    if transcript_path:
        candidate = Path(transcript_path)
        if candidate.exists():
            jsonl_path = candidate

    if not jsonl_path:
        reason = "no transcript_path in DB" if not transcript_path else "file missing from disk"
        log_fn(
            f"No JSONL for session {safe(session_id)} "
            f"(project={safe(project_dir or 'unknown')}, {reason}), "
            f"marking as extracted (skip)"
        )
        mark_extracted_fn(session_id)
        return False

    # Run headless memory extraction
    try:
        config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
        agent_file = config_dir / "agents" / "memory-extractor.md"

        agent_prompt = ""
        if agent_file.exists():
            content = agent_file.read_text()
            agent_prompt = strip_frontmatter_fn(content)
        else:
            agent_prompt = _FALLBACK_AGENT_PROMPT

        if daemon_cfg.extraction_model not in allowed_models:
            log_fn(
                f"Invalid extraction_model '{safe(daemon_cfg.extraction_model)}', "
                f"must be one of {sorted(allowed_models)}. "
                f"Session {safe(session_id)} marked failed for retry."
            )
            mark_failed_fn(session_id)
            return False

        env = build_extraction_env(os.environ, project_dir)

        cmd = build_extraction_command(
            session_id=session_id,
            jsonl_path=str(jsonl_path),
            agent_prompt=agent_prompt,
            model=daemon_cfg.extraction_model,
            max_turns=daemon_cfg.extraction_max_turns,
        )

        proc = subprocess_popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        active_extractions[proc.pid] = (
            session_id,
            proc,
            jsonl_path,
            project_dir,
            time.time(),
        )
        log_fn(
            f"Started extraction for {safe(session_id)} "
            f"(pid={proc.pid}, file={safe(jsonl_path.name)}, "
            f"active={len(active_extractions)})"
        )
        return True
    except Exception as e:
        log_fn(f"Failed to start extraction: {safe(e)}")
        return False


# ---------------------------------------------------------------------------
# Step 4.2 — archive_session_jsonl
# ---------------------------------------------------------------------------


def archive_session_jsonl(
    session_id: str,
    jsonl_path: Path | None,
    *,
    log_fn: Callable[[str], None],
    mark_archived_fn: Callable[[str, str], None],
) -> None:
    """Compress and upload session JSONL to S3, then delete local copy."""
    bucket = os.environ.get("CLAUDE_SESSION_ARCHIVE_BUCKET")
    if not bucket:
        return

    if not jsonl_path or not jsonl_path.exists():
        log_fn(f"Archive skipped for {safe(session_id)}: JSONL not found")
        return

    project_name = jsonl_path.parent.name
    s3_key = f"s3://{bucket}/sessions/{project_name}/{jsonl_path.stem}.jsonl.zst"
    zst_path = jsonl_path.with_suffix(".jsonl.zst")

    try:
        result = subprocess.run(
            ["zstd", "-q", "--rm", str(jsonl_path)],
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            err = safe(result.stderr.decode(errors="replace"))
            log_fn(f"zstd failed for {safe(session_id)}: {err}")
            return

        result = subprocess.run(
            ["aws", "s3", "cp", str(zst_path), s3_key, "--quiet"],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            err = safe(result.stderr.decode(errors="replace"))
            log_fn(f"S3 upload failed for {safe(session_id)}: {err}")
            subprocess.run(
                ["zstd", "-d", "-q", "--rm", str(zst_path)],
                capture_output=True,
                timeout=300,
            )
            return

        zst_path.unlink(missing_ok=True)

        try:
            mark_archived_fn(session_id, s3_key)
        except Exception as e:
            log_fn(
                f"Archive DB update failed for {safe(session_id)} "
                f"(file already in S3): {safe(e)}"
            )

        log_fn(f"Archived {safe(session_id)} -> {safe(s3_key)}")

    except subprocess.TimeoutExpired:
        log_fn(f"Archive timeout for {safe(session_id)}")
        if zst_path.exists() and not jsonl_path.exists():
            try:
                subprocess.run(
                    ["zstd", "-d", "-q", "--rm", str(zst_path)],
                    capture_output=True,
                    timeout=300,
                )
            except (subprocess.TimeoutExpired, OSError) as restore_err:
                log_fn(f"Archive cleanup failed for {safe(session_id)}: {safe(restore_err)}")
    except Exception as e:
        log_fn(f"Archive error for {safe(session_id)}: {safe(e)}")


# ---------------------------------------------------------------------------
# Step 4.3a — _calibrate_session_confidence
# ---------------------------------------------------------------------------


def calibrate_session_confidence(session_id: str, log_fn: Callable[[str], None]) -> None:
    """Run confidence calibration on learnings from a completed extraction."""
    try:
        import asyncio

        from scripts.core.confidence_calibrator import calibrate_session

        result = asyncio.run(calibrate_session(session_id))
        stats = result["stats"]
        if stats["total"] > 0:
            # stats values come from calibrate_session which ultimately reads
            # DB-stored learning content; wrap for defense in depth even
            # though happy-path values are ints.
            log_fn(
                f"Confidence calibration for {safe(session_id)}: "
                f"{safe(stats['updated'])} updated, "
                f"{safe(stats['unchanged'])} unchanged"
            )
    except Exception as e:
        log_fn(f"Confidence calibration failed for {safe(session_id)}: {safe(e)}")


# ---------------------------------------------------------------------------
# Step 4.3b — _extract_and_store_workflows
# ---------------------------------------------------------------------------


def extract_and_store_workflows(
    session_id: str,
    jsonl_path: Path,
    project: str | None,
    log_fn: Callable[[str], None],
    normalize_project_fn: Callable[[str | None], str | None],
) -> None:
    """Extract workflow patterns and store as learnings. Non-fatal."""
    try:
        from scripts.core.extract_workflow_patterns import (
            detect_workflow_sequences,
            extract_tool_uses,
            format_pattern_as_learning,
        )
    except ImportError as e:
        log_fn(f"Workflow extraction unavailable: {safe(e)}")
        return

    try:
        tool_uses = extract_tool_uses(jsonl_path, max_entries=50_000)
        patterns = detect_workflow_sequences(tool_uses)
        successful = [p for p in patterns if p.get("success") is True]

        if not successful:
            log_fn(f"No successful workflow patterns for {safe(session_id)}")
            return

        from scripts.core.store_learning import store_learning_v2

        stored = 0
        for pattern in successful:
            content = format_pattern_as_learning(pattern)
            try:
                import asyncio

                project_name = normalize_project_fn(project) if project else None
                result = asyncio.run(
                    store_learning_v2(
                        session_id=session_id,
                        content=content,
                        learning_type="WORKING_SOLUTION",
                        context=project or "unknown",
                        tags=["workflow", pattern["pattern_type"]],
                        confidence="high",
                        project=project_name,
                    )
                )
                if result.get("success") and not result.get("skipped"):
                    stored += 1
            except Exception as e:
                log_fn(f"Failed to store workflow learning: {safe(e)}")

        log_fn(f"Stored {stored} workflow patterns for {safe(session_id)}")
    except Exception as e:
        log_fn(f"Workflow extraction failed for {safe(session_id)}: {safe(e)}")


# ---------------------------------------------------------------------------
# Step 4.3c — _generate_mini_handoff
# ---------------------------------------------------------------------------


def generate_mini_handoff(
    session_id: str,
    jsonl_path: Path,
    project: str | None,
    log_fn: Callable[[str], None],
) -> None:
    """Generate a mini-handoff YAML from session data. Non-fatal."""
    try:
        from scripts.core.generate_mini_handoff import (
            generate_handoff,
            write_handoff,
        )
    except ImportError as e:
        log_fn(f"Mini-handoff generation unavailable: {safe(e)}")
        return

    if not project:
        log_fn(f"Mini-handoff skipped for {safe(session_id)}: no project dir")
        return

    state_file = Path(project) / ".claude" / "cache" / "session-state" / f"{session_id}.jsonl"
    use_state_file = state_file.exists() and state_file.stat().st_size > 0

    try:
        handoff = generate_handoff(
            session_id=session_id,
            project_dir=project,
            jsonl_path=jsonl_path,
            state_file=state_file if use_state_file else None,
        )
        output_path = write_handoff(handoff, Path(project), session_id)
        source = "state_file" if use_state_file else "jsonl"
        log_fn(
            f"Mini-handoff written for {safe(session_id)} "
            f"(source={source}): {safe(output_path)}"
        )

        if use_state_file:
            try:
                state_file.unlink()
                log_fn(f"State file cleaned up for {safe(session_id)}")
            except OSError as cleanup_err:
                log_fn(f"State file cleanup failed for {safe(session_id)}: {safe(cleanup_err)}")
    except Exception as e:
        log_fn(f"Mini-handoff generation failed for {safe(session_id)}: {safe(e)}")


# ---------------------------------------------------------------------------
# Step 4.3d — _count_session_rejections
# ---------------------------------------------------------------------------

try:
    from scripts.core.store_learning import get_rejection_count
except ImportError:
    get_rejection_count = None  # type: ignore[assignment]


def count_session_rejections(session_id: str) -> int | None:
    """Count rejected learnings for a session. Returns None on error."""
    try:
        if get_rejection_count is None:
            return None
        return get_rejection_count(session_id)
    except Exception:
        return None
