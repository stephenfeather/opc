"""Interactive golden set bootstrap tool for benchmark queries.

Runs each benchmark query with a large k, presents results with content
previews, and lets the user mark relevant results. Saves golden_ids back
to the query file for precision@k evaluation.

Usage:
    uv run python scripts/benchmarks/bootstrap_golden.py
    uv run python scripts/benchmarks/bootstrap_golden.py --queries custom.json
    uv run python scripts/benchmarks/bootstrap_golden.py --force  # re-annotate all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from pathlib import Path


FETCH_K = 20  # candidates shown per query


async def fetch_results(
    query: str,
    k: int,
    project: str | None = None,
    tags: list[str] | None = None,
) -> list[dict]:
    """Run a recall query and return full result dicts."""
    cmd = [
        sys.executable,
        "scripts/core/recall_learnings.py",
        "--query", query,
        "--k", str(k),
        "--json-full",
        "--no-rerank",
    ]
    if project:
        cmd.extend(["--project", project])
    if tags:
        cmd.extend(["--tags", *tags])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        print(f"  ERROR: {stderr.decode().strip()}", file=sys.stderr)
        return []

    data = json.loads(stdout)
    return data.get("results", [])


def format_preview(content: str, width: int = 80, lines: int = 3) -> str:
    """Create a compact content preview."""
    # Collapse whitespace and wrap
    cleaned = " ".join(content.split())
    wrapped = textwrap.wrap(cleaned, width=width)
    preview = "\n      ".join(wrapped[:lines])
    if len(wrapped) > lines:
        preview += " ..."
    return preview


def display_results(results: list[dict]) -> None:
    """Print numbered results with previews."""
    for i, r in enumerate(results):
        rid = r.get("id", "?")[:8]
        score = r.get("score", r.get("raw_score", 0.0))
        content = r.get("content", "")
        metadata = r.get("metadata", {})
        learning_type = metadata.get("learning_type", "?")
        project = metadata.get("project", "?")

        print(f"  [{i + 1:2d}] {rid}  score={score:.4f}  "
              f"type={learning_type}  project={project}")
        print(f"      {format_preview(content)}")
        print()


def prompt_selection(num_results: int) -> list[int] | None:
    """Prompt user to select relevant results.

    Returns list of 0-based indices, or None to skip.
    """
    print("  Mark relevant results (comma-separated numbers, "
          "'all', or 'skip'):")
    try:
        response = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if response in ("skip", "s", ""):
        return None
    if response == "all":
        return list(range(num_results))

    try:
        indices = []
        for part in response.split(","):
            part = part.strip()
            if "-" in part:
                # Range: "1-5"
                start, end = part.split("-", 1)
                for n in range(int(start), int(end) + 1):
                    if 1 <= n <= num_results:
                        indices.append(n - 1)
            else:
                n = int(part)
                if 1 <= n <= num_results:
                    indices.append(n - 1)
        return sorted(set(indices))
    except ValueError:
        print("  Invalid input, skipping.")
        return None


async def bootstrap(
    queries_path: Path,
    force: bool = False,
) -> None:
    """Run the interactive bootstrap process."""
    query_data = json.loads(queries_path.read_text())
    queries = query_data["queries"]

    annotated = 0
    skipped = 0
    already_done = 0

    print(f"Loaded {len(queries)} queries from {queries_path}")
    print(f"Fetching top {FETCH_K} results per query (no reranking)")
    print("=" * 60)
    print()

    for i, q in enumerate(queries):
        qid = q["id"]
        query_text = q["query"]
        existing_golden = q.get("golden_ids", [])

        if existing_golden and not force:
            already_done += 1
            continue

        print(f"--- [{i + 1}/{len(queries)}] {qid}: \"{query_text}\" ---")
        if existing_golden:
            print(f"  (has {len(existing_golden)} existing golden_ids, "
                  "re-annotating due to --force)")

        results = await fetch_results(
            query_text,
            FETCH_K,
            project=q.get("project"),
            tags=q.get("tags") or None,
        )

        if not results:
            print("  No results returned. Skipping.")
            skipped += 1
            print()
            continue

        display_results(results)
        selection = prompt_selection(len(results))

        if selection is None:
            skipped += 1
            print("  Skipped.")
        else:
            golden_ids = [
                results[idx].get("id", "")
                for idx in selection
                if results[idx].get("id")
            ]
            q["golden_ids"] = golden_ids
            annotated += 1
            print(f"  Marked {len(golden_ids)} results as relevant.")

        print()

    # Write back
    queries_path.write_text(json.dumps(query_data, indent=2) + "\n")

    print("=" * 60)
    print(f"Done. Annotated: {annotated}, Skipped: {skipped}, "
          f"Already done: {already_done}")
    print(f"Updated {queries_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive golden set bootstrap for benchmark queries"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("scripts/benchmarks/rerank_queries.json"),
        help="Path to benchmark query JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate queries that already have golden_ids",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.queries.exists():
        print(f"Error: {args.queries} not found", file=sys.stderr)
        return 1
    asyncio.run(bootstrap(args.queries, force=args.force))
    return 0


if __name__ == "__main__":
    sys.exit(main())
