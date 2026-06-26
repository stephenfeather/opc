"""Regression tests for issue #204: math_router used `os` without importing it.

The module's top-level crash-logging setup references `os.path.expanduser(...)`,
so importing the module raised `NameError: name 'os' is not defined` before the
fix. These tests are hermetic: HOME is redirected to a temp dir so the real
`~/.claude/logs/opc_crash.log` is never read or written, and the fallback
branches are exercised by simulating failures rather than touching the real
environment.
"""

import importlib
import io

import pytest


def _raise_oserror(*args, **kwargs):
    """Simulate a read-only / restricted filesystem."""
    raise OSError("simulated read-only environment")


def _raise_unsupported(*args, **kwargs):
    """Simulate faulthandler rejecting a non-fd stderr stream."""
    raise io.UnsupportedOperation("fileno")


@pytest.fixture
def math_router(monkeypatch, tmp_path):
    """Import math_router under a hermetic HOME (temp dir, never the real one)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    module = importlib.import_module("scripts.cc_math.math_router")
    # Reload under the patched HOME in case another test imported it first, so
    # the module-level crash-logging block re-executes against tmp_path.
    return importlib.reload(module)


def test_math_router_imports_with_os_available(math_router, tmp_path):
    """#204 regression: module imports cleanly and `os` is resolvable at scope.

    The crash log appearing under the temp home proves the module-level
    `os.path`/`os.makedirs` logic executed without depending on, or mutating,
    the real user home.
    """
    assert hasattr(math_router, "os")
    assert (tmp_path / ".claude" / "logs" / "opc_crash.log").exists()


def test_crash_logging_falls_back_to_stderr_when_path_unwritable(
    math_router, monkeypatch
):
    """If the log directory can't be created, setup falls back to stderr without
    raising (so importing the module stays safe in read-only environments)."""
    calls = []
    monkeypatch.setattr(math_router.os, "makedirs", _raise_oserror)
    monkeypatch.setattr(
        math_router.faulthandler, "enable", lambda *a, **k: calls.append(k)
    )

    math_router._enable_crash_logging()  # must not raise

    # The file branch was skipped; only the stderr fallback (no file kwarg) ran.
    assert calls == [{"all_threads": True}]


def test_crash_logging_degrades_to_noop_when_stderr_unusable(
    math_router, monkeypatch
):
    """If both the log path and the stderr fallback are unusable, setup degrades
    to a no-op rather than letting module import fail."""
    monkeypatch.setattr(math_router.os, "makedirs", _raise_oserror)
    monkeypatch.setattr(math_router.faulthandler, "enable", _raise_unsupported)

    # Every diagnostic channel fails; the helper must still swallow and return.
    math_router._enable_crash_logging()


def test_crash_logging_falls_back_when_home_unresolvable(
    math_router, monkeypatch, tmp_path
):
    """In a home-less environment, `expanduser` returns the literal `~/...`
    path. The helper must not create a stray `~` tree in the cwd; it must skip
    the file branch and fall back to stderr."""
    # Simulate an unresolvable home: expanduser leaves the path unchanged.
    monkeypatch.setattr(math_router.os.path, "expanduser", lambda p: p)
    calls = []
    monkeypatch.setattr(
        math_router.faulthandler, "enable", lambda *a, **k: calls.append(k)
    )
    monkeypatch.chdir(tmp_path)

    math_router._enable_crash_logging()  # must not raise

    # File branch skipped (path not absolute); only the stderr fallback ran.
    assert calls == [{"all_threads": True}]
    # No literal "~" directory leaked into the working directory.
    assert not (tmp_path / "~").exists()
