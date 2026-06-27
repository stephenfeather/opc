"""Tests for the math-router → compute-script invocation subsystem.

Regression coverage for issue #255: after the compute modules moved under
``scripts/cc_math/``, both the in-module imports and the command paths the
router generated still pointed at the old ``scripts/`` location, so no route
could actually be executed. These tests pin:

1. The compute modules import cleanly (correct ``scripts.cc_math.*`` paths).
2. ``build_command`` emits a path that exists on disk.
3. An intent can be routed → built → executed end-to-end with a real result.
"""

from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.cc_math import math_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.cc_math.numpy_compute",
        "scripts.cc_math.scipy_compute",
        "scripts.cc_math.mpmath_compute",
        "scripts.cc_math.sympy_baseline_validation",
    ],
)
def test_compute_modules_importable(module_name: str) -> None:
    """Each compute module imports without ModuleNotFoundError.

    Before the fix these raised ``No module named 'scripts.math_base'``.
    """
    module = importlib.import_module(module_name)
    assert module is not None


def test_build_command_emits_existing_path() -> None:
    """build_command must reference a script path that exists on disk."""
    command = math_router.build_command(
        "numpy_compute.py", "det", {"matrix": "[[1,2],[3,4]]"}
    )

    # Path is the token immediately after "uv run python".
    parts = shlex.split(command)
    assert parts[:3] == ["uv", "run", "python"]
    script_path = parts[3]

    assert (PROJECT_ROOT / script_path).exists(), (
        f"Generated script path does not exist: {script_path}"
    )


def test_route_produces_runnable_path() -> None:
    """A routed intent yields a command whose script path exists."""
    match = math_router.route("numpy det [[1,2],[3,4]]")

    assert match.script == "numpy_compute.py"
    assert match.subcommand == "det"

    script_path = shlex.split(match.command)[3]
    assert (PROJECT_ROOT / script_path).exists()


def test_route_executes_end_to_end() -> None:
    """Route → build → execute the generated command and assert a real result.

    This is the guard that the subsystem can't silently rot again: it runs the
    exact command string the router produces and checks the computed value.
    """
    match = math_router.route("numpy det [[1,2],[3,4]]")

    result = subprocess.run(
        shlex.split(match.command),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    assert result.returncode == 0, (
        f"Command failed: {match.command}\nstderr:\n{result.stderr}"
    )

    payload = json.loads(result.stdout)
    # det([[1,2],[3,4]]) == -2 (numpy returns -2.0000000000000004)
    assert "error" not in payload, payload
    assert abs(float(payload["result"]) - (-2.0)) < 1e-6, payload


def _clean_home_env(tmp_home: Path) -> dict[str, str]:
    """Env with HOME pointed at a fresh dir lacking ``~/.claude/logs``.

    Reproduces the clean CI/agent-sandbox condition where the crash-log path
    does not yet exist, and guarantees imports resolve via ``PYTHONPATH``.
    """
    return {
        **os.environ,
        "HOME": str(tmp_home),
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.cc_math.math_base",
        "scripts.cc_math.numpy_compute",
        "scripts.cc_math.scipy_compute",
        "scripts.cc_math.mpmath_compute",
        "scripts.cc_math.sympy_compute",
        "scripts.cc_math.sympy_baseline_validation",
    ],
)
def test_modules_import_under_clean_home(
    module_name: str, tmp_path: Path
) -> None:
    """Modules must import even when ``~/.claude/logs`` does not exist.

    Before the crash-logging hardening, the module-level
    ``faulthandler.enable(open("~/.claude/logs/opc_crash.log"))`` raised
    ``FileNotFoundError`` at import in a clean HOME, defeating the path fix.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=_clean_home_env(tmp_path),
    )
    assert result.returncode == 0, (
        f"Import failed under clean HOME: {module_name}\nstderr:\n{result.stderr}"
    )


def test_routed_command_executes_under_clean_home(tmp_path: Path) -> None:
    """The routed compute command runs to a real result in a clean HOME.

    Runs the module the way the router targets it (direct file invocation) with
    a fresh HOME so the crash-log directory is absent, proving the subsystem is
    executable in clean CI/sandbox environments (issue #255).
    """
    match = math_router.route("numpy det [[1,2],[3,4]]")
    # Drop the leading "uv run " — execute with the test interpreter directly so
    # the run does not depend on uv's HOME-rooted caches under the temp HOME.
    args = shlex.split(match.command)
    assert args[:3] == ["uv", "run", "python"]
    cmd = [sys.executable, *args[3:]]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=_clean_home_env(tmp_path),
    )
    assert result.returncode == 0, (
        f"Command failed under clean HOME: {cmd}\nstderr:\n{result.stderr}"
    )
    payload = json.loads(result.stdout)
    assert "error" not in payload, payload
    assert abs(float(payload["result"]) - (-2.0)) < 1e-6, payload
