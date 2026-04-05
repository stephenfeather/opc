"""Tests for memory daemon extraction subprocess security hardening.

Verifies:
- --allowedTools Bash,Read is present in subprocess args
- --dangerously-skip-permissions is still present
- Invalid extraction_model values are rejected before Popen
- Valid model values pass validation
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_active_extractions():
    """Clear active_extractions before each test to avoid cross-test pollution."""
    from scripts.core.memory_daemon import active_extractions

    active_extractions.clear()
    yield
    active_extractions.clear()


@pytest.fixture()
def tmp_jsonl(tmp_path: Path) -> Path:
    """Create a temporary JSONL file that extract_memories can find."""
    jsonl = tmp_path / "session.jsonl"
    jsonl.write_text('{"type":"message"}\n')
    return jsonl


def _make_mock_proc(pid: int = 12345) -> MagicMock:
    mock = MagicMock()
    mock.pid = pid
    return mock


# ── Allowed tools flag ──────────────────────────────────────────────


@patch("scripts.core.memory_daemon.subprocess.Popen")
@patch("scripts.core.memory_daemon._is_extraction_blocked", return_value=False)
@patch("scripts.core.memory_daemon.mark_extracted")
def test_popen_args_contain_allowed_tools(
    _mark, _blocked, mock_popen, tmp_jsonl
):
    """--allowedTools Bash,Read must appear in the subprocess args."""
    from scripts.core.memory_daemon import extract_memories

    mock_popen.return_value = _make_mock_proc()

    result = extract_memories("sess-1", "/tmp/proj", str(tmp_jsonl))

    assert result is True
    args_list = mock_popen.call_args[0][0]
    assert "--allowedTools" in args_list
    idx = args_list.index("--allowedTools")
    assert args_list[idx + 1] == "Bash,Read"


@patch("scripts.core.memory_daemon.subprocess.Popen")
@patch("scripts.core.memory_daemon._is_extraction_blocked", return_value=False)
@patch("scripts.core.memory_daemon.mark_extracted")
def test_popen_args_contain_dangerously_skip_permissions(
    _mark, _blocked, mock_popen, tmp_jsonl
):
    """--dangerously-skip-permissions must still be present."""
    from scripts.core.memory_daemon import extract_memories

    mock_popen.return_value = _make_mock_proc()

    result = extract_memories("sess-2", "/tmp/proj", str(tmp_jsonl))

    assert result is True
    args_list = mock_popen.call_args[0][0]
    assert "--dangerously-skip-permissions" in args_list


# ── Model validation ────────────────────────────────────────────────


@patch("scripts.core.memory_daemon.subprocess.Popen")
@patch("scripts.core.memory_daemon._is_extraction_blocked", return_value=False)
@patch("scripts.core.memory_daemon.mark_extracted")
def test_invalid_model_rejects_and_skips_popen(
    _mark, _blocked, mock_popen, tmp_jsonl
):
    """An extraction_model not in the allowlist must return False without calling Popen."""
    from scripts.core.memory_daemon import _daemon_cfg, extract_memories

    original_model = _daemon_cfg.extraction_model
    try:
        # Use object.__setattr__ because _daemon_cfg may be a frozen/slots object
        object.__setattr__(_daemon_cfg, "extraction_model", "gpt-evil")
        result = extract_memories("sess-3", "/tmp/proj", str(tmp_jsonl))
    finally:
        object.__setattr__(_daemon_cfg, "extraction_model", original_model)

    assert result is False
    mock_popen.assert_not_called()


@pytest.mark.parametrize("model", ["sonnet", "haiku", "opus"])
@patch("scripts.core.memory_daemon.subprocess.Popen")
@patch("scripts.core.memory_daemon._is_extraction_blocked", return_value=False)
@patch("scripts.core.memory_daemon.mark_extracted")
def test_valid_models_pass_validation(
    _mark, _blocked, mock_popen, model, tmp_jsonl
):
    """Each model in the allowlist should pass validation and reach Popen."""
    from scripts.core.memory_daemon import _daemon_cfg, extract_memories

    original_model = _daemon_cfg.extraction_model
    try:
        object.__setattr__(_daemon_cfg, "extraction_model", model)
        mock_popen.return_value = _make_mock_proc()
        result = extract_memories(f"sess-{model}", "/tmp/proj", str(tmp_jsonl))
    finally:
        object.__setattr__(_daemon_cfg, "extraction_model", original_model)

    assert result is True
    mock_popen.assert_called_once()
    args_list = mock_popen.call_args[0][0]
    assert "--model" in args_list
    idx = args_list.index("--model")
    assert args_list[idx + 1] == model


# ── Allowed models constant ─────────────────────────────────────────


def test_allowed_extraction_models_is_frozenset():
    """The allowlist must be a frozenset with exactly the expected models."""
    from scripts.core.memory_daemon import _ALLOWED_EXTRACTION_MODELS

    assert isinstance(_ALLOWED_EXTRACTION_MODELS, frozenset)
    assert _ALLOWED_EXTRACTION_MODELS == {"sonnet", "haiku", "opus"}
