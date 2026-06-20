#!/usr/bin/env python3
"""opc-docs — CLI for the document-collection RAG layer.

Subcommands:
    create   Register a new collection in the YAML registry.
    scan     Ingest one collection (or --all). Re-scans are incremental.
    list     Show every collection and its ingest stats.
    query    Scoped semantic search across ingested documents.

Invocation (matches other OPC scripts):
    uv run python scripts/core/documents/cli.py <subcommand> ...

The cron job calls `scan --all`; manual `scan <name>` forces one collection.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure the repo root is importable when run directly as a script
# (pytest sets pythonpath, but `uv run python scripts/core/documents/cli.py`
# does not). Resolve from this file's location so the worktree / cron / main
# tree each import their own package — do NOT use CLAUDE_PROJECT_DIR here, as
# it may point at a different tree than the one this script lives in.
_project_dir = str(Path(__file__).resolve().parents[3])
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# Load global ~/.claude/.env first, then local .env (matches store_learning.py).
_global_env = Path.home() / ".claude" / ".env"
if _global_env.exists():
    load_dotenv(_global_env)
load_dotenv()

from scripts.core.db.embedding_service import EmbeddingService  # noqa: E402
from scripts.core.documents.db import collection_stats  # noqa: E402
from scripts.core.documents.ingest import ingest_collection  # noqa: E402
from scripts.core.documents.query import query_documents  # noqa: E402
from scripts.core.documents.registry import (  # noqa: E402
    Collection,
    RegistryError,
    append_collection,
    load_registry,
)


def _build_embedder() -> EmbeddingService:
    """Construct the embedding service the same way store_learning_v2 does."""
    return EmbeddingService(provider=os.getenv("EMBEDDING_PROVIDER", "local"))


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for opc-docs."""
    parser = argparse.ArgumentParser(prog="opc-docs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared --json flag: every subcommand can emit machine-readable JSON to
    # stdout instead of human text, so the opc-memory MCP can wrap them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON to stdout instead of human-readable text",
    )

    p_create = sub.add_parser("create", help="register a new collection", parents=[common])
    p_create.add_argument("name", help="unique collection name")
    p_create.add_argument("--path", required=True, help="folder to track")
    p_create.add_argument(
        "--scope",
        required=True,
        choices=["global", "restricted"],
        help="retrieval scope: 'global' surfaces by default, 'restricted' only when targeted",
    )
    p_create.add_argument(
        "--extensions",
        default=".pdf,.docx,.txt,.csv,.md,.html,.htm,.xml",
        help="comma-separated file extensions to ingest",
    )
    p_create.add_argument(
        "--ocr",
        action="store_true",
        help="request OCR (stored but ignored in v1 — born-digital only)",
    )

    p_scan = sub.add_parser("scan", help="ingest one collection or --all", parents=[common])
    p_scan.add_argument("name", nargs="?", default=None, help="collection to scan")
    p_scan.add_argument("--all", action="store_true", help="scan every collection")

    sub.add_parser("list", help="list collections and ingest stats", parents=[common])

    p_query = sub.add_parser("query", help="scoped semantic search", parents=[common])
    p_query.add_argument("text", help="the question / search text")
    p_query.add_argument(
        "--collection",
        default=None,
        help="target one collection by name — the ONLY way to reach a "
        "restricted collection; the default search is global-only",
    )
    p_query.add_argument("--limit", type=int, default=8, help="max results")

    return parser


def _cmd_create(args: argparse.Namespace) -> int:
    collection = Collection(
        name=args.name,
        path=args.path,
        scope=args.scope,
        extensions=[e.strip() for e in args.extensions.split(",") if e.strip()],
        ocr=args.ocr,
    )
    try:
        append_collection(None, collection)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "name": collection.name,
                    "scope": collection.scope,
                    "path": collection.path,
                    "extensions": collection.extensions,
                    "ocr": collection.ocr,
                    "status": "registered",
                }
            )
        )
    else:
        print(f"registered collection '{collection.name}' ({collection.scope})")
    return 0


async def _scan_all(targets: list[Collection]) -> list:
    # One event loop for the whole run: the asyncpg pool binds to the loop that
    # created it, so a separate asyncio.run() per collection would leave the
    # second collection acquiring dead connections ("event loop is closed").
    embedder = _build_embedder()
    reports = []
    for collection in targets:
        reports.append(await ingest_collection(collection, embedder))
    return reports


def _format_report(report) -> str:
    """One-line human-readable summary of an ingest report."""
    return (
        f"[{report.collection}] ingested={report.ingested} "
        f"unchanged={report.skipped_unchanged} rescoped={report.rescoped} "
        f"needs_ocr={report.needs_ocr} unsupported={report.skipped_unsupported} "
        f"too_large={report.skipped_too_large} purged={report.purged} "
        f"errors={report.errors}"
    )


def _cmd_scan(args: argparse.Namespace) -> int:
    try:
        collections = load_registry(None)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.all:
        targets = collections
    else:
        if not args.name:
            print("error: provide a collection name or --all", file=sys.stderr)
            return 1
        targets = [c for c in collections if c.name == args.name]
        if not targets:
            print(f"error: unknown collection '{args.name}'", file=sys.stderr)
            return 1
    try:
        reports = asyncio.run(_scan_all(targets))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps([dataclasses.asdict(r) for r in reports]))
    else:
        for report in reports:
            print(_format_report(report))
    return 0


async def _list_all(collections: list[Collection]) -> list[tuple[Collection, dict]]:
    # Single event loop — see the note in _scan_all about the pool/loop binding.
    rows = []
    for collection in collections:
        stats = await collection_stats(collection.name)
        rows.append((collection, stats))
    return rows


def _serialize_last_scan(value) -> str | None:
    """Render a last-scan timestamp as ISO-8601, or None when never scanned."""
    return value.isoformat() if value is not None else None


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        collections = load_registry(None)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not collections:
        print("[]" if args.json else "no collections registered")
        return 0
    rows = asyncio.run(_list_all(collections))
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": collection.name,
                        "scope": collection.scope,
                        "path": collection.path,
                        "document_count": stats["document_count"],
                        "chunk_count": stats["chunk_count"],
                        "last_scanned_at": _serialize_last_scan(stats["last_scanned_at"]),
                    }
                    for collection, stats in rows
                ]
            )
        )
    else:
        for collection, stats in rows:
            print(
                f"{collection.name}  scope={collection.scope}  "
                f"path={collection.path}  docs={stats['document_count']}  "
                f"chunks={stats['chunk_count']}  last_scan={stats['last_scanned_at']}"
            )
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    embedder = _build_embedder()
    results = asyncio.run(
        query_documents(
            args.text,
            embedder,
            collection=args.collection,
            limit=args.limit,
        )
    )
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "content": result.content,
                        "file_path": result.file_path,
                        "page_number": result.page_number,
                        "collection": result.collection,
                        "similarity": result.similarity,
                    }
                    for result in results
                ]
            )
        )
        return 0
    if not results:
        print("no matches")
        return 0
    for result in results:
        page = f" p.{result.page_number}" if result.page_number else ""
        print(f"[{result.similarity:.2f}] {result.collection}{page}  {result.file_path}")
        print(f"    {result.content[:300]}")
    return 0


def run(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    dispatch = {
        "create": _cmd_create,
        "scan": _cmd_scan,
        "list": _cmd_list,
        "query": _cmd_query,
    }
    return dispatch[args.command](args)


def main() -> None:
    """CLI entrypoint."""
    sys.exit(run())


if __name__ == "__main__":
    main()
