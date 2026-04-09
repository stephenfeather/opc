"""Tests for memory_daemon startup concerns: fd setup, --debug flag, faulthandler init.

Covers the fixes for Issue #99:
  - _setup_daemon_fds: replaces the broken sys.std*.close() pattern in
    _run_as_daemon that left kernel fd-table holes at 0/1/2 and corrupted
    child subprocess stdio (observed in production as Fatal Python error
    init_sys_streams in pattern_batch children).
  - --debug CLI flag and debug() helper for diagnostic elevation.
  - Subprocess spawn sites pass stdin=DEVNULL and propagate
    MEMORY_DAEMON_DEBUG=1 into child env when DEBUG is set.
  - faulthandler.enable() is function-gated, not called at module import.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock


class TestSetupDaemonFds:
    """_setup_daemon_fds redirects fd 0/1/2 to /dev/null for daemonization.

    The function is parameterized on os.open / os.dup2 / os.close so tests
    can verify the exact syscall sequence without touching real file
    descriptors.
    """

    def test_opens_devnull_read_write(self):
        from scripts.core.memory_daemon import _setup_daemon_fds

        fake_open = MagicMock(return_value=7)
        _setup_daemon_fds(
            os_open_fn=fake_open,
            os_dup2_fn=MagicMock(),
            os_close_fn=MagicMock(),
        )

        fake_open.assert_called_once_with(os.devnull, os.O_RDWR)

    def test_dup2s_devnull_onto_all_std_fds(self):
        from scripts.core.memory_daemon import _setup_daemon_fds

        fake_dup2 = MagicMock()
        _setup_daemon_fds(
            os_open_fn=MagicMock(return_value=7),
            os_dup2_fn=fake_dup2,
            os_close_fn=MagicMock(),
        )

        dup2_calls = [c.args for c in fake_dup2.call_args_list]
        assert (7, 0) in dup2_calls
        assert (7, 1) in dup2_calls
        assert (7, 2) in dup2_calls
        assert fake_dup2.call_count == 3

    def test_closes_devnull_fd_when_above_2(self):
        from scripts.core.memory_daemon import _setup_daemon_fds

        fake_close = MagicMock()
        _setup_daemon_fds(
            os_open_fn=MagicMock(return_value=7),
            os_dup2_fn=MagicMock(),
            os_close_fn=fake_close,
        )

        fake_close.assert_called_once_with(7)

    def test_does_not_close_when_devnull_landed_at_std_fd(self):
        """Edge case: if /dev/null happens to land at fd 0, 1, or 2 (meaning
        those fds were already closed before the call), we must NOT close it
        afterward — that would re-create the fd-table hole we're fixing.
        """
        from scripts.core.memory_daemon import _setup_daemon_fds

        fake_close = MagicMock()
        _setup_daemon_fds(
            os_open_fn=MagicMock(return_value=2),
            os_dup2_fn=MagicMock(),
            os_close_fn=fake_close,
        )

        fake_close.assert_not_called()

    def test_does_not_touch_sys_module_wrappers(self):
        """_setup_daemon_fds must not call sys.std*.close(). Closing the
        Python wrappers is what caused the original bug by releasing the
        underlying fds without replacement.
        """
        from scripts.core.memory_daemon import _setup_daemon_fds

        original_stdin = sys.stdin
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        _setup_daemon_fds(
            os_open_fn=MagicMock(return_value=7),
            os_dup2_fn=MagicMock(),
            os_close_fn=MagicMock(),
        )

        # The Python wrapper objects must still be the same and still open.
        assert sys.stdin is original_stdin
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        assert not sys.stdin.closed
        assert not sys.stdout.closed
        assert not sys.stderr.closed


class TestDebugFlag:
    """--debug flag + DEBUG module state + debug() helper.

    The helper must short-circuit when DEBUG is False so that noisy
    diagnostic logging costs nothing in normal operation.
    """

    def test_module_exposes_debug_bool(self):
        import scripts.core.memory_daemon as mod

        assert hasattr(mod, "DEBUG")
        assert isinstance(mod.DEBUG, bool)

    def test_debug_helper_exists(self):
        import scripts.core.memory_daemon as mod

        assert callable(mod.debug)

    def test_debug_helper_is_noop_when_disabled(self, monkeypatch):
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "DEBUG", False)
        captured: list[str] = []
        monkeypatch.setattr(mod, "log", lambda m: captured.append(m))

        mod.debug("this should not appear")

        assert captured == []

    def test_debug_helper_logs_when_enabled(self, monkeypatch):
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "DEBUG", True)
        captured: list[str] = []
        monkeypatch.setattr(mod, "log", lambda m: captured.append(m))

        mod.debug("hello diagnostic")

        assert len(captured) == 1
        assert "hello diagnostic" in captured[0]
        assert "DEBUG" in captured[0]

    def test_main_debug_flag_sets_module_debug_and_propagates_env(self, monkeypatch):
        """Running main(["start", "--debug"]) must set mod.DEBUG=True and
        export MEMORY_DAEMON_DEBUG=1 so child subprocesses inherit it.
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "DEBUG", False)
        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", "0")
        monkeypatch.setattr(mod, "start_daemon", lambda: 0)
        monkeypatch.setattr(sys, "argv", ["memory_daemon.py", "start", "--debug"])

        rc = mod.main()

        assert rc == 0
        assert mod.DEBUG is True
        assert os.environ.get("MEMORY_DAEMON_DEBUG") == "1"

    def test_main_without_debug_flag_leaves_debug_false(self, monkeypatch):
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "DEBUG", False)
        monkeypatch.delenv("MEMORY_DAEMON_DEBUG", raising=False)
        monkeypatch.setattr(mod, "start_daemon", lambda: 0)
        monkeypatch.setattr(sys, "argv", ["memory_daemon.py", "start"])

        mod.main()

        assert mod.DEBUG is False
        assert "MEMORY_DAEMON_DEBUG" not in os.environ


class TestFaulthandlerGating:
    """faulthandler.enable() must be function-gated, not called at import time.

    Addresses Issues #55 / #57: the previous module-level
    ``faulthandler.enable(file=open(...), all_threads=True)`` at line 51
    leaked a file descriptor on every import and ran unconditionally from
    any script that imported memory_daemon.
    """

    def test_module_has_enable_faulthandler_function(self):
        import scripts.core.memory_daemon as mod

        assert callable(mod._enable_faulthandler)

    def test_module_does_not_call_faulthandler_enable_at_import_scope(self):
        """Walk the module AST and assert any ``faulthandler.enable(...)`` call
        is nested inside a function definition, not at module level.
        """
        import ast
        from pathlib import Path

        import scripts.core.memory_daemon as mod

        source = Path(mod.__file__).read_text()
        tree = ast.parse(source)

        module_level_enables: list[int] = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                fn = node.value.func
                if (
                    isinstance(fn, ast.Attribute)
                    and fn.attr == "enable"
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "faulthandler"
                ):
                    module_level_enables.append(node.lineno)

        assert not module_level_enables, (
            f"faulthandler.enable(...) found at module scope on lines "
            f"{module_level_enables}. Must be wrapped in a function and called "
            f"only from main() or _run_as_daemon()."
        )

    def test_enable_faulthandler_is_idempotent(self, monkeypatch):
        """Calling _enable_faulthandler twice must not re-open the log file."""
        import scripts.core.memory_daemon as mod

        # Reset the internal handle so the first call does real work.
        monkeypatch.setattr(mod, "_faulthandler_log_file", None, raising=False)

        enable_calls: list = []

        def fake_enable(*args, **kwargs):
            enable_calls.append((args, kwargs))

        monkeypatch.setattr(mod.faulthandler, "enable", fake_enable)

        mod._enable_faulthandler()
        first_handle = mod._faulthandler_log_file
        mod._enable_faulthandler()
        second_handle = mod._faulthandler_log_file

        assert first_handle is second_handle
        assert len(enable_calls) == 1


class TestPatternDetectionSpawn:
    """_run_pattern_detection_batch must harden the child subprocess stdio.

    Defense in depth for Issue #99: even though the primary fix is
    _setup_daemon_fds, the spawn site itself should explicitly pass
    stdin=subprocess.DEVNULL so a future daemonization regression cannot
    re-introduce the EBADF crash in pattern_batch children.
    """

    def _fresh_state(self, mod):
        """Replace _daemon_state with a clean instance so the spawn path fires."""
        return mod.create_daemon_state()

    def test_spawn_passes_stdin_devnull(self, monkeypatch):
        import subprocess

        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))
        # Prevent any stray exception from polluting ~/.claude/memory-daemon.log
        monkeypatch.setattr(mod, "log", lambda _m: None)

        captured: dict = {}

        class _FakeProc:
            pid = 99999  # real Popen exposes .pid; debug logging may read it

            def poll(self):
                return None

        def fake_popen(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

        mod._run_pattern_detection_batch()

        assert "kwargs" in captured, "Popen was not called"
        assert captured["kwargs"].get("stdin") == subprocess.DEVNULL

    def test_spawn_propagates_debug_env_to_child(self, monkeypatch):
        """When MEMORY_DAEMON_DEBUG=1 is in os.environ, the child process must
        see it — either because env= was passed with the key, or because env=
        was omitted (Popen default inherits parent os.environ).
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))
        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", "1")
        monkeypatch.setattr(mod, "log", lambda _m: None)

        captured: dict = {}

        class _FakeProc:
            pid = 99999  # real Popen exposes .pid; debug logging may read it

            def poll(self):
                return None

        def fake_popen(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

        mod._run_pattern_detection_batch()

        kwargs = captured["kwargs"]
        if "env" in kwargs and kwargs["env"] is not None:
            assert kwargs["env"].get("MEMORY_DAEMON_DEBUG") == "1"
        # When env= is omitted, the child inherits os.environ by default,
        # which we've set via monkeypatch.setenv above.
        assert os.environ.get("MEMORY_DAEMON_DEBUG") == "1"

    def test_spawn_happy_path_does_not_log_errors(self, monkeypatch):
        """Regression guard: the original implementation eagerly evaluated
        ``f"...pid={state.pattern_proc.pid}"`` inside a ``debug()`` call, which
        raised AttributeError on a test mock that lacked a ``.pid`` attribute.
        The production exception handler silently swallowed the error and
        logged it to ~/.claude/memory-daemon.log, polluting the real daemon
        log. This test would have caught that by asserting no error log on
        the happy path.
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))

        captured_log: list[str] = []
        monkeypatch.setattr(mod, "log", lambda m: captured_log.append(m))

        class _FakeProc:
            pid = 12345

            def poll(self):
                return None

        monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: _FakeProc())

        mod._run_pattern_detection_batch()

        error_logs = [m for m in captured_log if "error" in m.lower()]
        assert error_logs == [], (
            f"_run_pattern_detection_batch should not log errors on the happy path. "
            f"Got: {error_logs}"
        )
