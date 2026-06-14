from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.core import cli


def test_top_level_help_lists_grouped_commands() -> None:
    text = cli.format_help()

    assert "opc - Opinionated Persistent Context" in text
    assert "Recall & Store:" in text
    assert "  recall" in text
    assert "Artifacts:" in text
    assert "artifact query" in text
    assert "Patterns & Metrics:" in text
    assert "pattern detect" in text
    assert "Daemon:" in text
    assert "daemon" in text


def test_prefix_help_lists_only_matching_subcommands() -> None:
    text = cli.format_help(("artifact",))

    assert "opc artifact" in text
    assert "artifact query" in text
    assert "artifact mark" in text
    assert "artifact index" in text
    assert "recall" not in text


def test_resolve_command_uses_longest_registered_prefix() -> None:
    key, command, remainder = cli.resolve_command(
        ["artifact", "query", "--type", "handoffs", "auth"]
    )

    assert key == ("artifact", "query")
    assert command is not None
    assert command.script == "artifact_query.py"
    assert remainder == ["--type", "handoffs", "auth"]


def test_main_forwards_to_script_with_remaining_args() -> None:
    completed = SimpleNamespace(returncode=17)
    with patch.object(cli.subprocess, "run", return_value=completed) as run:
        result = cli.main(["recall", "--query", "auth patterns", "--k", "5"])

    assert result == 17
    run.assert_called_once()
    forwarded = run.call_args.args[0]
    assert forwarded[0] == cli.sys.executable
    assert forwarded[1].endswith("scripts/core/recall_learnings.py")
    assert forwarded[2:] == ["--query", "auth patterns", "--k", "5"]


def test_main_prints_prefix_help_without_forwarding(capsys) -> None:
    with patch.object(cli.subprocess, "run") as run:
        result = cli.main(["pattern", "--help"])

    captured = capsys.readouterr()
    assert result == 0
    assert "opc pattern" in captured.out
    assert "pattern detect" in captured.out
    run.assert_not_called()


def test_cli_style_core_scripts_are_registered_or_explicitly_excluded() -> None:
    core_dir = Path(__file__).resolve().parent.parent / "scripts" / "core"
    registered = {command.script for command in cli.COMMANDS.values()}

    cli_style_scripts = set()
    for path in core_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports_argparse = any(
            (isinstance(node, ast.Import) and any(alias.name == "argparse" for alias in node.names))
            or (
                isinstance(node, ast.ImportFrom)
                and node.module == "argparse"
            )
            for node in ast.walk(tree)
        )
        has_main_guard = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in ast.walk(tree)
        )
        if imports_argparse and has_main_guard:
            cli_style_scripts.add(path.name)

    missing = cli_style_scripts - registered - cli.EXCLUDED_CORE_SCRIPTS
    assert missing == set()
