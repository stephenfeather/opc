"""Tests for the opc-docs CLI argument parsing and dispatch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.core.documents.cli import build_parser, run
from scripts.core.documents.ingest import IngestReport
from scripts.core.documents.query import QueryResult


def test_cli_runs_as_script_without_pythonpath(tmp_path: Path) -> None:
    """The CLI must self-bootstrap sys.path when run as `python cli.py ...`.

    Regression guard: the in-process `run()` tests pass because pytest puts the
    repo root on the path, so they do not catch a broken script entrypoint.
    Invoke the file directly from an unrelated CWD with PYTHONPATH stripped —
    it must still resolve `scripts.core.documents.*` and exit 0.
    """
    repo_root = Path(__file__).resolve().parents[1]
    cli = repo_root / "scripts" / "core" / "documents" / "cli.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(cli), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "scoped semantic search" in result.stdout


def test_parser_create_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "caleb",
            "--path",
            "~/Documents/Feather, Caleb",
            "--scope",
            "restricted",
            "--extensions",
            ".pdf,.docx",
        ]
    )
    assert args.command == "create"
    assert args.name == "caleb"
    assert args.scope == "restricted"
    assert args.extensions == ".pdf,.docx"


def test_parser_scan_all_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["scan", "--all"])
    assert args.command == "scan"
    assert args.all is True
    assert args.name is None


def test_parser_query_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["query", "who flagged meds", "--collection", "caleb", "--limit", "5"])
    assert args.command == "query"
    assert args.text == "who flagged meds"
    assert args.collection == "caleb"
    assert args.limit == 5


def test_run_create_appends_to_registry(tmp_path: Path, monkeypatch) -> None:
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    exit_code = run(
        ["create", "caleb", "--path", "/tmp/caleb", "--scope", "restricted", "--extensions", ".pdf"]
    )
    assert exit_code == 0
    assert reg.exists()
    assert "caleb" in reg.read_text()


def test_run_create_duplicate_returns_error(tmp_path: Path, monkeypatch) -> None:
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    run(["create", "x", "--path", "/tmp/x", "--scope", "global", "--extensions", ".txt"])
    exit_code = run(
        ["create", "x", "--path", "/tmp/x", "--scope", "global", "--extensions", ".txt"]
    )
    assert exit_code == 1


def test_run_scan_named_collection_calls_ingest(tmp_path: Path, monkeypatch) -> None:
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    run(["create", "c", "--path", str(tmp_path), "--scope", "global", "--extensions", ".txt"])
    with (
        patch(
            "scripts.core.documents.cli.ingest_collection",
            new=AsyncMock(),
        ) as mock_ingest,
        patch("scripts.core.documents.cli._build_embedder", return_value=object()),
    ):
        exit_code = run(["scan", "c"])
    assert exit_code == 0
    mock_ingest.assert_awaited_once()


def test_run_scan_all_scans_every_collection(tmp_path: Path, monkeypatch) -> None:
    # Regression: scan --all must process every registered collection within a
    # single event loop. A prior version called asyncio.run() per collection,
    # which crashed on the 2nd collection ("event loop is closed") because the
    # asyncpg pool binds to the first loop.
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    run(["create", "a", "--path", str(tmp_path), "--scope", "global", "--extensions", ".txt"])
    run(["create", "b", "--path", str(tmp_path), "--scope", "restricted", "--extensions", ".txt"])
    with (
        patch("scripts.core.documents.cli.ingest_collection", new=AsyncMock()) as mock_ingest,
        patch("scripts.core.documents.cli._build_embedder", return_value=object()),
    ):
        exit_code = run(["scan", "--all"])
    assert exit_code == 0
    assert mock_ingest.await_count == 2
    scanned = {call.args[0].name for call in mock_ingest.await_args_list}
    assert scanned == {"a", "b"}


def test_run_scan_malformed_registry_returns_error(tmp_path: Path, monkeypatch) -> None:
    # A malformed registry must yield a clean non-zero exit, not a traceback.
    reg = tmp_path / "reg.yaml"
    reg.write_text("- not\n- a mapping\n")
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    assert run(["scan", "--all"]) == 1
    assert run(["list"]) == 1


def test_run_scan_unknown_collection_returns_error(tmp_path: Path, monkeypatch) -> None:
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    exit_code = run(["scan", "does-not-exist"])
    assert exit_code == 1


def test_run_query_calls_query_documents(tmp_path: Path, monkeypatch) -> None:
    with (
        patch(
            "scripts.core.documents.cli.query_documents",
            new=AsyncMock(return_value=[]),
        ) as mock_query,
        patch("scripts.core.documents.cli._build_embedder", return_value=object()),
    ):
        exit_code = run(["query", "find the nurse"])
    assert exit_code == 0
    mock_query.assert_awaited_once()


# --- --json output -----------------------------------------------------------


def test_parser_json_flag_on_every_subcommand() -> None:
    parser = build_parser()
    # default off
    assert parser.parse_args(["list"]).json is False
    # present and parseable on each subcommand
    assert parser.parse_args(["list", "--json"]).json is True
    assert parser.parse_args(["query", "q", "--json"]).json is True
    assert parser.parse_args(["scan", "--all", "--json"]).json is True
    assert (
        parser.parse_args(["create", "c", "--path", "/tmp/c", "--scope", "global", "--json"]).json
        is True
    )


def test_run_query_json_emits_full_content(capsys) -> None:
    results = [
        QueryResult(
            content="full chunk text that must not be truncated " * 10,
            file_path="/docs/a.pdf",
            page_number=3,
            collection="caleb",
            similarity=0.91,
        )
    ]
    with (
        patch(
            "scripts.core.documents.cli.query_documents",
            new=AsyncMock(return_value=results),
        ),
        patch("scripts.core.documents.cli._build_embedder", return_value=object()),
    ):
        exit_code = run(["query", "meds", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "content": results[0].content,
            "file_path": "/docs/a.pdf",
            "page_number": 3,
            "collection": "caleb",
            "similarity": 0.91,
        }
    ]


def test_run_query_json_empty_is_empty_array(capsys) -> None:
    with (
        patch(
            "scripts.core.documents.cli.query_documents",
            new=AsyncMock(return_value=[]),
        ),
        patch("scripts.core.documents.cli._build_embedder", return_value=object()),
    ):
        exit_code = run(["query", "nothing", "--json"])
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == []


def test_run_list_json_output(tmp_path: Path, monkeypatch, capsys) -> None:
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    run(
        ["create", "caleb", "--path", "/tmp/caleb", "--scope", "restricted", "--extensions", ".pdf"]
    )
    capsys.readouterr()  # discard setup output so only the --json line is captured
    scanned = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    stats = {"document_count": 4, "chunk_count": 42, "last_scanned_at": scanned}
    with patch(
        "scripts.core.documents.cli.collection_stats",
        new=AsyncMock(return_value=stats),
    ):
        exit_code = run(["list", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "name": "caleb",
            "scope": "restricted",
            "path": "/tmp/caleb",
            "document_count": 4,
            "chunk_count": 42,
            "last_scanned_at": scanned.isoformat(),
        }
    ]


def test_run_list_json_empty_registry(tmp_path: Path, monkeypatch, capsys) -> None:
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    assert run(["list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_run_list_json_null_last_scan(tmp_path: Path, monkeypatch, capsys) -> None:
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    run(["create", "c", "--path", "/tmp/c", "--scope", "global", "--extensions", ".txt"])
    capsys.readouterr()  # discard setup output so only the --json line is captured
    stats = {"document_count": 0, "chunk_count": 0, "last_scanned_at": None}
    with patch(
        "scripts.core.documents.cli.collection_stats",
        new=AsyncMock(return_value=stats),
    ):
        assert run(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["last_scanned_at"] is None


def test_run_scan_json_output(tmp_path: Path, monkeypatch, capsys) -> None:
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    run(["create", "c", "--path", str(tmp_path), "--scope", "global", "--extensions", ".txt"])
    capsys.readouterr()  # discard setup output so only the --json line is captured
    report = IngestReport(collection="c", ingested=2, skipped_unchanged=1, purged=3)
    with (
        patch(
            "scripts.core.documents.cli.ingest_collection",
            new=AsyncMock(return_value=report),
        ),
        patch("scripts.core.documents.cli._build_embedder", return_value=object()),
    ):
        exit_code = run(["scan", "c", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "collection": "c",
            "ingested": 2,
            "skipped_unchanged": 1,
            "skipped_unsupported": 0,
            "skipped_too_large": 0,
            "needs_ocr": 0,
            "errors": 0,
            "rescoped": 0,
            "purged": 3,
        }
    ]


def test_run_create_json_output(tmp_path: Path, monkeypatch, capsys) -> None:
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    exit_code = run(
        [
            "create",
            "caleb",
            "--path",
            "/tmp/caleb",
            "--scope",
            "restricted",
            "--extensions",
            ".pdf,.docx",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "name": "caleb",
        "scope": "restricted",
        "path": "/tmp/caleb",
        "extensions": [".pdf", ".docx"],
        "ocr": False,
        "status": "registered",
    }


def test_run_create_json_emits_normalized_extensions(tmp_path: Path, monkeypatch, capsys) -> None:
    # Regression: create --json must emit the canonical collection that is
    # actually written to the registry (extensions normalized to leading-dot),
    # not the raw argparse values. Otherwise an MCP consumer that trusts the
    # JSON sees "pdf" while the registry holds ".pdf".
    reg = tmp_path / "reg.yaml"
    monkeypatch.setenv("OPC_DOC_REGISTRY", str(reg))
    exit_code = run(
        [
            "create",
            "c",
            "--path",
            "/tmp/c",
            "--scope",
            "global",
            "--extensions",
            "pdf,docx",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["extensions"] == [".pdf", ".docx"]
    # The JSON contract must match what the registry actually stored.
    assert ".pdf" in reg.read_text()
    assert "\n- pdf\n" not in reg.read_text()
