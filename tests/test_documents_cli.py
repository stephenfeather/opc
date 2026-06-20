"""Tests for the opc-docs CLI argument parsing and dispatch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.core.documents.cli import build_parser, run


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
