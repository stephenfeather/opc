"""Documentation contract for the deterministic session-audit command."""

from pathlib import Path


def test_session_audit_is_cataloged_with_cli_api_limits_and_exit_semantics() -> None:
    project = Path(__file__).resolve().parents[1]
    commands = (project / "docs" / "commands-reference.md").read_text(encoding="utf-8")
    api = (project / "docs" / "recall-api-reference.md").read_text(encoding="utf-8")

    assert "**27 commands** across **26 unique argparse scripts**" in commands
    assert "### Session Analysis" in commands
    assert "`opc session audit`" in commands
    assert "`session_audit_cli.py`" in commands
    for flag in (
        "--jsonl",
        "--format",
        "--output",
        "--no-thinking",
        "--judge",
        "--judge-model",
    ):
        assert f"`{flag}" in commands or f"`{flag}`" in commands

    assert "# Session Audit API Reference" in api
    assert "network-free" in api
    assert "64 MiB" in api
    assert "8 MiB" in api
    assert "125,000" in api
    assert "25,000" in api
    assert "| 0 |" in api
    assert "| 1 |" in api
    assert "| 2 |" in api
    assert "| 3 |" in api
    assert "| 4 |" in api
    assert "| 5 |" in api
    assert "not a confidence interval" in api
    assert "User-only" in api
    assert "ANTHROPIC_API_KEY" in api
    assert "raw jsonl is never uploaded" in api.casefold()
    assert "short session's entire normalized semantic content" in api.casefold()
    assert "best-effort masking, not anonymization" in api
    assert "claude-sonnet-5" in api
