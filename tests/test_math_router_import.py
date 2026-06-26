"""Regression test for issue #204: math_router used `os` without importing it.

The module's top-level crash-logging setup references `os.path.expanduser(...)`,
so importing the module raised `NameError: name 'os' is not defined` before the
fix. This test is hermetic: it redirects HOME to a temp dir so it never reads or
writes the real `~/.claude/logs/opc_crash.log`.
"""

import importlib


def test_math_router_imports_with_os_available(monkeypatch, tmp_path):
    """Importing math_router must not raise, and `os` must be resolvable.

    Redirecting HOME proves the module-level `os.path`/`os.makedirs` logic
    executes (the crash log appears under the temp home) without depending on,
    or mutating, the real user home.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    module = importlib.import_module("scripts.cc_math.math_router")
    # Reload under the patched HOME in case another test imported it first, so
    # the module-level crash-logging block re-executes against tmp_path.
    module = importlib.reload(module)

    assert hasattr(module, "os")
    assert (tmp_path / ".claude" / "logs" / "opc_crash.log").exists()
