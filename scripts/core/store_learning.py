#!/usr/bin/env python3
"""Store session learnings in PostgreSQL with pgvector embeddings.

Claude-native learning storage - called by stop-learnings hook or memory extractor.
Stores learnings in memory for semantic recall in future sessions.

Usage (legacy):
    uv run python opc/scripts/store_learning.py \
        --session-id "abc123" \
        --worked "Approach X worked well" \
        --failed "Y didn't work" \
        --decisions "Chose Z because..." \
        --patterns "Reusable technique..."

Usage (v2 - direct content):
    uv run python opc/scripts/store_learning.py \
        --session-id "abc123" \
        --type "WORKING_SOLUTION" \
        --context "hook development" \
        --tags "hooks,patterns" \
        --confidence "high" \
        --content "Pattern X works well for Y"

Learning Types:
    FAILED_APPROACH: Things that didn't work
    WORKING_SOLUTION: Successful approaches
    USER_PREFERENCE: User style/preferences
    CODEBASE_PATTERN: Discovered code patterns
    ARCHITECTURAL_DECISION: Design choices made
    ERROR_FIX: Error->solution pairs
    OPEN_THREAD: Unfinished work/TODOs

Environment:
    DATABASE_URL: PostgreSQL connection string
    VOYAGE_API_KEY: For embeddings (optional, falls back to local)
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_FAULT_HANDLER_LOG: Any | None = None


def _enable_faulthandler() -> None:
    """Enable faulthandler without breaking import if the log path is unavailable."""
    global _FAULT_HANDLER_LOG  # noqa: PLW0603
    crash_log = Path.home() / ".claude" / "logs" / "opc_crash.log"
    try:
        crash_log.parent.mkdir(parents=True, exist_ok=True)
        _FAULT_HANDLER_LOG = crash_log.open("a", encoding="utf-8")
        faulthandler.enable(file=_FAULT_HANDLER_LOG, all_threads=True)
    except OSError:
        faulthandler.enable(file=sys.stderr, all_threads=True)


_enable_faulthandler()

# Load global ~/.claude/.env first, then local .env
global_env = Path.home() / ".claude" / ".env"
if global_env.exists():
    load_dotenv(global_env)
load_dotenv()

# Add project to path
project_dir = os.environ.get("CLAUDE_PROJECT_DIR", str(Path(__file__).parent.parent.parent))
sys.path.insert(0, project_dir)

# Valid learning types for --type parameter
LEARNING_TYPES = [
    "FAILED_APPROACH",
    "WORKING_SOLUTION",
    "USER_PREFERENCE",
    "CODEBASE_PATTERN",
    "ARCHITECTURAL_DECISION",
    "ERROR_FIX",
    "OPEN_THREAD",
]

# Valid confidence levels
CONFIDENCE_LEVELS = ["high", "medium", "low"]

from scripts.core.config import get_config as _get_config  # noqa: E402
from scripts.core.db.embedding_service import EmbeddingService  # noqa: E402
from scripts.core.db.memory_factory import create_memory_service, get_default_backend  # noqa: E402

# Backward-compat: DEDUP_THRESHOLD was a module-level constant in the old code.
# Now computed live via _dedup_threshold(). This snapshot satisfies importers
# that read it at import time; live callers should use _dedup_threshold().
DEDUP_THRESHOLD = _get_config().dedup.threshold  # noqa: E402


# ===========================================================================
# Pure Functions
# ===========================================================================


def _dedup_threshold() -> float:
    """Return dedup threshold from config (cached after first get_config() call)."""
    return _get_config().dedup.threshold


def _pg_url() -> str | None:
    """Return PostgreSQL connection URL from environment, or None.

    Prefers CONTINUOUS_CLAUDE_DB_URL (canonical) over DATABASE_URL (fallback)
    to match the resolution order used by recall_learnings and memory_daemon.
    OPC_POSTGRES_URL is intentionally NOT included here — adding it would
    create split-brain behavior with consumers that don't check it yet.
    TODO: Unify all backend/URL resolution behind a shared function.
    """
    return os.environ.get("CONTINUOUS_CLAUDE_DB_URL") or os.environ.get("DATABASE_URL") or None


def detect_backend(env: dict[str, str], fallback: str | None = None) -> str:
    """Determine storage backend from environment variables.

    Takes an env dict and returns a backend name string. When no URL is found
    and no fallback is provided, delegates to get_default_backend() which
    reads os.environ and may raise.

    Matches the precedence used by recall_learnings.get_backend():
    CONTINUOUS_CLAUDE_DB_URL > DATABASE_URL > fallback.

    NOTE: AGENTICA_MEMORY_BACKEND and OPC_POSTGRES_URL are intentionally
    NOT checked here -- recall_learnings, memory_daemon, and confidence_calibrator
    don't all honor them yet, so adding them here would create split-brain.
    TODO: Unify all backend/URL resolution behind a shared function.
    """
    if env.get("CONTINUOUS_CLAUDE_DB_URL") or env.get("DATABASE_URL"):
        return "postgres"
    if fallback is not None:
        return fallback
    return get_default_backend()


def build_metadata(
    *,
    session_id: str,
    timestamp: datetime,
    learning_type: str | None = None,
    context: str | None = None,
    tags: list[str] | None = None,
    confidence: str | None = None,
    confidence_score: float | None = None,
    confidence_dimensions: dict[str, Any] | None = None,
    host_id: str | None = None,
    embedding_model: str | None = None,
    project: str | None = None,
    classification_reasoning: str | None = None,
) -> dict[str, Any]:
    """Build metadata dict for a learning entry.

    Pure function -- constructs a new dict from provided values.
    Omits keys whose values are None.
    """
    meta: dict[str, Any] = {
        "type": "session_learning",
        "session_id": session_id,
        "timestamp": timestamp.isoformat(),
    }
    if classification_reasoning:
        meta["classification_reasoning"] = classification_reasoning
        meta["classified_by"] = "llm_judge"
        meta["classified_at"] = timestamp.isoformat()
    if host_id:
        meta["host_id"] = host_id
    if embedding_model:
        meta["embedding_model"] = embedding_model
    if learning_type:
        meta["learning_type"] = learning_type
    if context:
        meta["context"] = context
    if tags:
        meta["tags"] = tags
    if confidence:
        meta["confidence"] = confidence
    if confidence_score is not None:
        meta["confidence_score"] = confidence_score
    if confidence_dimensions is not None:
        meta["confidence_dimensions"] = confidence_dimensions
    if project:
        meta["project"] = project
    return meta


def build_learning_content(
    *,
    worked: str,
    failed: str,
    decisions: str,
    patterns: str,
) -> str | None:
    """Assemble legacy learning content from four category strings.

    Pure function -- returns joined content string or None if all parts are empty/None.
    """
    parts: list[str] = []
    if worked and worked.lower() != "none":
        parts.append(f"What worked: {worked}")
    if failed and failed.lower() != "none":
        parts.append(f"What failed: {failed}")
    if decisions and decisions.lower() != "none":
        parts.append(f"Decisions: {decisions}")
    if patterns and patterns.lower() != "none":
        parts.append(f"Patterns: {patterns}")
    return "\n".join(parts) if parts else None


def check_dedup_result(
    *,
    existing: list[dict[str, Any]] | None,
    threshold: float,
    default_session: str = "",
) -> dict[str, Any] | None:
    """Check whether search results indicate a duplicate.

    Pure function -- returns dedup info dict if the top match meets the
    threshold, or None if no duplicate found.

    Args:
        default_session: Fallback session_id when the match row lacks one
            (e.g. session-scoped search_vector results).
    """
    if not existing:
        return None
    top_match = existing[0]
    similarity = top_match.get("similarity", 0)
    if similarity >= threshold:
        return {
            "similarity": similarity,
            "existing_session": top_match.get("session_id", default_session),
            "existing_id": str(top_match.get("id", "")),
        }
    return None


def parse_tags(tags_str: str | None) -> list[str] | None:
    """Parse comma-separated tag string into a list.

    Pure function -- returns list of stripped non-empty tags, or None.
    """
    if not tags_str:
        return None
    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    return tags if tags else None


def format_output(result: dict[str, Any], *, json_mode: bool) -> str:
    """Format a store result dict for CLI output.

    Pure function -- returns a printable string.
    """
    if json_mode:
        from scripts.core.recall_formatters import get_api_version

        output = {**result, "version": get_api_version()}
        return json.dumps(output)

    if result.get("skipped"):
        return f"~ Learning skipped: {result.get('reason', 'duplicate')}"
    if result["success"]:
        lines = [
            f"Learning stored (id: {result.get('memory_id', 'unknown')})",
            f"  Backend: {result.get('backend', 'unknown')}",
            f"  Content: {result.get('content_length', 0)} chars",
        ]
        return "\n".join(lines)
    return f"Failed to store learning: {result.get('error', 'unknown')}"


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for store_learning.

    Pure function -- returns parsed Namespace without side effects.
    """
    parser = argparse.ArgumentParser(description="Store session learnings in memory")
    parser.add_argument("--session-id", required=True, help="Session identifier")

    # Legacy parameters (v1)
    parser.add_argument("--worked", default="None", help="What worked well (legacy)")
    parser.add_argument("--failed", default="None", help="What failed or was tricky (legacy)")
    parser.add_argument("--decisions", default="None", help="Key decisions made (legacy)")
    parser.add_argument("--patterns", default="None", help="Reusable patterns (legacy)")

    # New v2 parameters
    parser.add_argument("--type", choices=LEARNING_TYPES, help="Learning type (v2)")
    parser.add_argument("--content", help="Direct content (v2)")
    parser.add_argument("--context", help="What this relates to (v2)")
    parser.add_argument("--tags", help="Comma-separated tags (v2)")
    parser.add_argument(
        "--confidence", choices=CONFIDENCE_LEVELS, help="Confidence level (v2)"
    )
    parser.add_argument(
        "--project",
        help="Project name for recall relevance (default: auto-detect from CLAUDE_PROJECT_DIR)",
    )

    # Host identification
    parser.add_argument("--host-id", help="Machine identifier for multi-system support")

    # Learning chains
    parser.add_argument("--supersedes", help="UUID of older learning this one replaces (v2)")

    # Auto-classification
    parser.add_argument(
        "--auto-classify",
        action="store_true",
        help="Auto-classify learning type via LLM (requires BRAINTRUST_API_KEY)",
    )

    # Output options
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    return parser.parse_args(argv)


# ===========================================================================
# I/O Helpers
# ===========================================================================


def _ensure_learning_rejections_table(cur: Any) -> None:
    """Create learning_rejections table if it does not exist."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS learning_rejections (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            similarity REAL,
            threshold REAL,
            existing_id TEXT,
            existing_session TEXT,
            project TEXT,
            learning_type TEXT,
            context TEXT,
            tags TEXT[],
            rejected_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_learning_rejections_session
        ON learning_rejections (session_id)
    """)


def _record_rejection(
    session_id: str,
    *,
    similarity: float | None = None,
    threshold: float | None = None,
    existing_id: str | None = None,
    existing_session: str | None = None,
    project: str | None = None,
    learning_type: str | None = None,
    context: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Record a rejected learning with dedup details.

    Inserts a row into learning_rejections so the daemon can query
    rejection counts and details after extraction completes.
    Non-fatal -- failures are logged and swallowed.
    """
    url = _pg_url()
    if not url:
        return
    try:
        import psycopg2

        conn = psycopg2.connect(url)
        cur = conn.cursor()
        _ensure_learning_rejections_table(cur)
        cur.execute(
            """
            INSERT INTO learning_rejections
                (session_id, similarity, threshold, existing_id,
                 existing_session, project, learning_type, context, tags)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                similarity,
                threshold,
                existing_id,
                existing_session,
                project,
                learning_type,
                context,
                tags,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("Failed to record rejection: %s", e)


def get_rejection_count(session_id: str) -> int:
    """Return the number of rejected (skipped) learnings for a session.

    Returns 0 if no rejections recorded or on error.
    """
    url = _pg_url()
    if not url:
        return 0
    try:
        import psycopg2

        conn = psycopg2.connect(url)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM learning_rejections WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


# ===========================================================================
# Async I/O: Auto-classification
# ===========================================================================


async def _try_auto_classify(
    content: str,
    learning_type: str | None,
    context: str | None,
) -> tuple[str | None, str | None]:
    """Attempt LLM auto-classification of learning type.

    Returns (learning_type, classification_reasoning) tuple.
    Non-fatal -- returns original type on failure.
    """
    try:
        from scripts.braintrust_analyze import classify_learning

        result = await classify_learning(
            content, existing_type=learning_type, context=context
        )
        if not result.get("error"):
            logger.info(
                "Auto-classified as %s (%s)",
                result["learning_type"],
                result.get("reasoning"),
            )
            return result["learning_type"], result.get("reasoning")
        logger.warning(
            "Auto-classification failed: %s. Using fallback type: %s",
            result["error"],
            learning_type or "WORKING_SOLUTION",
        )
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)[:80]
        logger.warning(
            "braintrust_analyze not available for auto-classification "
            "(missing: %s)", missing
        )
    except Exception as e:
        logger.warning("Auto-classification error: %s", str(e)[:100])
    return learning_type, None


def _try_calibrate_confidence(
    content: str,
) -> tuple[str | None, float | None, dict[str, Any] | None]:
    """Attempt confidence calibration.

    Returns (confidence, score, dimensions) tuple.
    Non-fatal -- returns (None, None, None) on failure.
    """
    try:
        from scripts.core.confidence_calibrator import calibrate_confidence

        cal = calibrate_confidence(content)
        return cal["confidence"], cal["score"], cal["dimensions"]
    except Exception:
        return None, None, None


# ===========================================================================
# Async I/O: Knowledge Graph Indexing (non-fatal side effects)
# ===========================================================================


async def _try_index_kg(memory_id: str, content: str) -> dict[str, Any] | None:
    """Extract entities/relations from content and store KG edges.

    Non-fatal -- returns None on any failure. Only runs for postgres backend.
    """
    try:
        from scripts.core.kg_extractor import (
            extract_entities,
            extract_relations,
            store_entities_and_edges,
        )

        entities = extract_entities(content)
        if not entities:
            return None
        relations = extract_relations(content, entities)
        kg_stats = await store_entities_and_edges(memory_id, entities, relations)
        logger.info(
            "KG indexed: %d entities, %d edges, %d mentions",
            kg_stats["entities"],
            kg_stats["edges"],
            kg_stats["mentions"],
        )
        return kg_stats
    except Exception as e:
        logger.warning("KG extraction failed (non-fatal): %s", str(e)[:200])
        return None


async def _try_backfill_kg(content_hash: str, content: str) -> None:
    """Backfill KG for an existing learning matched by content_hash.

    store_entities_and_edges is idempotent, so repeated calls are safe.
    Non-fatal -- logs and returns on any failure.
    """
    try:
        from scripts.core.db.postgres_pool import get_pool
        from scripts.core.kg_extractor import (
            extract_entities,
            extract_relations,
            store_entities_and_edges,
        )

        pool = await get_pool()
        async with pool.acquire() as conn:
            existing_id = await conn.fetchval(
                "SELECT id FROM archival_memory WHERE content_hash = $1",
                content_hash,
            )
        if not existing_id:
            return
        existing_id_str = str(existing_id)
        entities = extract_entities(content)
        if not entities:
            return
        relations = extract_relations(content, entities)
        await store_entities_and_edges(existing_id_str, entities, relations)
        logger.info("KG backfilled for dedup learning %s", existing_id_str)
    except Exception as e:
        logger.warning("KG backfill on dedup failed (non-fatal): %s", str(e)[:200])


# ===========================================================================
# Async I/O: Store Handlers
# ===========================================================================


async def store_learning_v2(
    session_id: str,
    content: str,
    learning_type: str | None = None,
    context: str | None = None,
    tags: list[str] | None = None,
    confidence: str | None = None,
    host_id: str | None = None,
    supersedes: str | None = None,
    project: str | None = None,
    auto_classify: bool = False,
) -> dict[str, Any]:
    """Store learning with v2 metadata schema and deduplication.

    Orchestrates: embedding -> dedup check -> auto-classify -> calibrate -> store.
    All pure logic is delegated to extracted functions.
    """
    if not content or not content.strip():
        return {"success": False, "error": "No content provided"}

    backend = detect_backend(dict(os.environ))
    memory = None

    try:
        memory = await create_memory_service(backend=backend, session_id=session_id)

        # Compute content hash for deduplication
        content_hash = hashlib.sha256(content.strip().encode()).hexdigest()

        # Generate embedding
        embed_provider = os.getenv("EMBEDDING_PROVIDER", "local")
        embedder = EmbeddingService(provider=embed_provider)
        embedding = await embedder.embed(content)

        # Determine embedding model name for metadata
        embedding_model = None
        if hasattr(embedder, "_provider") and hasattr(embedder._provider, "model"):
            embedding_model = embedder._provider.model

        # Semantic dedup check
        threshold = _dedup_threshold()
        try:
            if hasattr(memory, "search_vector_global"):
                existing = await memory.search_vector_global(
                    embedding, threshold=threshold, limit=1
                )
            else:
                logger.debug(
                    "search_vector_global unavailable, using session-scoped dedup"
                )
                existing = await memory.search_vector(embedding, limit=1)

            dedup = check_dedup_result(
                existing=existing, threshold=threshold, default_session=session_id
            )
            if dedup is not None:
                reason = (
                    f"duplicate (similarity: {dedup['similarity']:.2f},"
                    f" session: {dedup['existing_session']})"
                )
                logger.info(
                    "Dedup rejected: session=%s similarity=%.3f "
                    "existing_id=%s threshold=%.2f",
                    session_id,
                    dedup["similarity"],
                    dedup["existing_id"],
                    threshold,
                )
                _record_rejection(
                    session_id,
                    similarity=dedup["similarity"],
                    threshold=threshold,
                    existing_id=dedup["existing_id"],
                    existing_session=dedup["existing_session"],
                    project=project,
                    learning_type=learning_type,
                    context=context,
                    tags=tags,
                )
                return {
                    "success": True,
                    "skipped": True,
                    "reason": reason,
                    "existing_id": dedup["existing_id"],
                }
        except Exception:
            pass  # If search fails, proceed with storing

        # Auto-classify if requested and type is missing/default
        classification_reasoning = None
        if auto_classify and (not learning_type or learning_type == "WORKING_SOLUTION"):
            learning_type, classification_reasoning = await _try_auto_classify(
                content, learning_type, context
            )

        # Auto-calibrate confidence if not explicitly provided
        confidence_score_val = None
        confidence_dimensions = None
        if not confidence:
            confidence, confidence_score_val, confidence_dimensions = (
                _try_calibrate_confidence(content)
            )

        # Build metadata via pure function
        metadata = build_metadata(
            session_id=session_id,
            timestamp=datetime.now(UTC),
            learning_type=learning_type,
            context=context,
            tags=tags,
            confidence=confidence,
            confidence_score=confidence_score_val,
            confidence_dimensions=confidence_dimensions,
            host_id=host_id,
            embedding_model=embedding_model,
            project=project,
            classification_reasoning=classification_reasoning,
        )

        # Store with embedding and content_hash dedup
        memory_id = await memory.store(
            content,
            metadata=metadata,
            embedding=embedding,
            content_hash=content_hash,
            host_id=host_id,
            supersedes=supersedes if backend == "postgres" else None,
            tags=tags if backend == "postgres" else None,
            project=project if backend == "postgres" else None,
        )

        # content_hash dedup returns empty string
        if not memory_id:
            if backend == "postgres":
                await _try_backfill_kg(content_hash, content)
            _record_rejection(
                session_id,
                project=project,
                learning_type=learning_type,
                context=context,
                tags=tags,
            )
            return {
                "success": True,
                "skipped": True,
                "reason": "duplicate (content_hash match)",
            }

        # Knowledge graph indexing (non-fatal, postgres-only)
        kg_stats: dict[str, Any] | None = None
        if backend == "postgres":
            kg_stats = await _try_index_kg(memory_id, content)

        result_dict: dict[str, Any] = {
            "success": True,
            "memory_id": memory_id,
            "backend": backend,
            "content_length": len(content),
            "embedding_dim": len(embedding),
        }
        if supersedes and backend == "postgres":
            result_dict["superseded"] = supersedes
        if kg_stats:
            result_dict["kg_stats"] = kg_stats
        return result_dict

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        if memory is not None:
            await memory.close()


async def store_learning(
    session_id: str,
    worked: str,
    failed: str,
    decisions: str,
    patterns: str,
) -> dict[str, Any]:
    """Store learning in PostgreSQL with embedding (legacy interface).

    Delegates content assembly to build_learning_content() and backend
    detection to detect_backend().
    """
    learning_content = build_learning_content(
        worked=worked, failed=failed, decisions=decisions, patterns=patterns
    )
    if learning_content is None:
        return {"success": False, "error": "No learning content provided"}

    # Metadata for filtering/retrieval
    metadata = {
        "type": "session_learning",
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "categories": {
            "worked": bool(worked and worked.lower() != "none"),
            "failed": bool(failed and failed.lower() != "none"),
            "decisions": bool(decisions and decisions.lower() != "none"),
            "patterns": bool(patterns and patterns.lower() != "none"),
        },
    }

    backend = detect_backend(dict(os.environ))
    memory = None

    try:
        memory = await create_memory_service(backend=backend, session_id=session_id)

        embedder = EmbeddingService(provider=os.getenv("EMBEDDING_PROVIDER", "local"))
        embedding = await embedder.embed(learning_content)

        memory_id = await memory.store(
            learning_content, metadata=metadata, embedding=embedding
        )

        return {
            "success": True,
            "memory_id": memory_id,
            "backend": backend,
            "content_length": len(learning_content),
            "embedding_dim": len(embedding),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        if memory is not None:
            await memory.close()


# ===========================================================================
# CLI Entrypoint
# ===========================================================================


async def main() -> None:
    """CLI entrypoint -- parses args, routes to v2 or legacy, prints output."""
    args = parse_cli_args()

    # Auto-detect project from environment
    project = (
        args.project or os.environ.get("CLAUDE_PROJECT_DIR", "").rsplit("/", 1)[-1] or None
    )

    # Determine which mode to use: v2 if --content is provided, else legacy
    if args.content:
        tags = parse_tags(args.tags)
        result = await store_learning_v2(
            session_id=args.session_id,
            content=args.content,
            learning_type=args.type,
            context=args.context,
            tags=tags,
            confidence=args.confidence,
            host_id=args.host_id,
            supersedes=args.supersedes,
            project=project,
            auto_classify=args.auto_classify,
        )
    else:
        result = await store_learning(
            session_id=args.session_id,
            worked=args.worked,
            failed=args.failed,
            decisions=args.decisions,
            patterns=args.patterns,
        )

    output = format_output(result, json_mode=args.json)
    print(output)

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
