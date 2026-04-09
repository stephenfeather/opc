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


class TestReserveLowFds:
    """_reserve_low_fds() prevents the rotating log handler from landing on
    fd 0/1/2 and being silently destroyed by _setup_daemon_fds() later.

    Addresses Codex Round 1 Finding 1 (HIGH): the module-level
    ``_logger = _setup_logging()`` call opens the rotating log file at
    import time. If the process is launched with any of fd 0/1/2 already
    closed, the kernel assigns that slot to the log file. _setup_daemon_fds()
    then dup2s /dev/null over it, silently nulling out daemon logging in
    exactly the degraded-stdio scenario this patch is supposed to harden.
    """

    def test_reserve_low_fds_reserves_all_three_when_closed(self):
        from scripts.core.memory_daemon import _reserve_low_fds

        # Simulate: fd 0, 1, 2 all closed. os.open returns 0, then 1,
        # then 2, then 3. _reserve_low_fds keeps 0/1/2 and closes 3.
        opened: list[int] = []
        closed: list[int] = []
        fd_sequence = iter([0, 1, 2, 3])

        def fake_open(path, flags):
            opened.append(next(fd_sequence))
            return opened[-1]

        def fake_close(fd):
            closed.append(fd)

        _reserve_low_fds(os_open_fn=fake_open, os_close_fn=fake_close)

        assert opened == [0, 1, 2, 3]
        assert closed == [3]

    def test_reserve_low_fds_noop_when_std_fds_already_open(self):
        from scripts.core.memory_daemon import _reserve_low_fds

        opened: list[int] = []
        closed: list[int] = []

        def fake_open(path, flags):
            opened.append(3)  # normal case: lowest free fd is 3
            return 3

        def fake_close(fd):
            closed.append(fd)

        _reserve_low_fds(os_open_fn=fake_open, os_close_fn=fake_close)

        # Opened once, got back a high fd, closed it, returned.
        assert opened == [3]
        assert closed == [3]

    def test_reserve_low_fds_partial_when_only_fd0_closed(self):
        from scripts.core.memory_daemon import _reserve_low_fds

        opened: list[int] = []
        closed: list[int] = []
        fd_sequence = iter([0, 3])  # fd 0 closed, 1/2 open → os.open gives 0 then 3

        def fake_open(path, flags):
            opened.append(next(fd_sequence))
            return opened[-1]

        def fake_close(fd):
            closed.append(fd)

        _reserve_low_fds(os_open_fn=fake_open, os_close_fn=fake_close)

        assert opened == [0, 3]
        assert closed == [3]  # 0 is kept; 3 is released

    def test_reserve_low_fds_tolerates_os_error(self):
        from scripts.core.memory_daemon import _reserve_low_fds

        def raising_open(path, flags):
            raise OSError("too many fds")

        closed: list[int] = []

        # Must not raise.
        _reserve_low_fds(
            os_open_fn=raising_open,
            os_close_fn=lambda fd: closed.append(fd),
        )

    def test_reserve_low_fds_not_called_at_module_scope(self):
        """Codex Round 3 Finding 3: _reserve_low_fds() must not run at import
        time. Any process that imports memory_daemon for its helpers would
        otherwise have its fd table silently mutated, which is especially
        damaging for fork-after-import callers that expect the original
        stdio semantics.
        """
        import ast
        from pathlib import Path

        import scripts.core.memory_daemon as mod

        tree = ast.parse(Path(mod.__file__).read_text())

        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_reserve_low_fds"
            ):
                raise AssertionError(
                    f"_reserve_low_fds() must not be called at module scope "
                    f"(found at line {node.lineno}). Call it from _run_as_daemon()."
                )

    def test_logger_is_none_at_module_scope_not_setup_eagerly(self):
        """Codex Round 3 Finding 3: module scope must not call
        _setup_logging() and must not hold an open log file handle. The
        binding ``_logger = None`` is the only acceptable module-level
        initialization; the real logger is lazy-initialized on first log()
        call or explicitly from _run_as_daemon().
        """
        import ast
        from pathlib import Path

        import scripts.core.memory_daemon as mod

        tree = ast.parse(Path(mod.__file__).read_text())

        for node in tree.body:
            if not (isinstance(node, (ast.Assign, ast.AnnAssign))):
                continue
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            else:
                targets = [node.target]
                value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "_logger":
                    # _logger = <something> at module scope — must be None.
                    assert isinstance(value, ast.Constant) and value.value is None, (
                        f"Module-scope _logger must be initialized to None "
                        f"(at line {node.lineno}), not _setup_logging() or any "
                        f"other eager call. Lazy-init from log() instead."
                    )
                    return

        raise AssertionError("Module-scope _logger = None binding not found")

    def test_run_as_daemon_reserves_fds_before_setup_daemon_fds(self):
        """The daemon bootstrap path must reserve low fds BEFORE
        _setup_daemon_fds() runs. If _setup_daemon_fds runs first and
        happens to land /dev/null at a fd the log handler already holds,
        the log file gets silently clobbered.
        """
        import ast
        from pathlib import Path

        import scripts.core.memory_daemon as mod

        tree = ast.parse(Path(mod.__file__).read_text())

        run_as_daemon_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_run_as_daemon":
                run_as_daemon_fn = node
                break

        assert run_as_daemon_fn is not None, "_run_as_daemon function not found"

        reserve_line: int | None = None
        setup_fds_line: int | None = None
        for subnode in ast.walk(run_as_daemon_fn):
            if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                if subnode.func.id == "_reserve_low_fds" and reserve_line is None:
                    reserve_line = subnode.lineno
                if subnode.func.id == "_setup_daemon_fds" and setup_fds_line is None:
                    setup_fds_line = subnode.lineno

        assert reserve_line is not None, (
            "_run_as_daemon must call _reserve_low_fds() to guard against " "degraded-stdio startup"
        )
        assert (
            setup_fds_line is not None
        ), "_run_as_daemon must call _setup_daemon_fds() for daemonization"
        assert reserve_line < setup_fds_line, (
            f"_reserve_low_fds must run BEFORE _setup_daemon_fds in _run_as_daemon. "
            f"Got reserve at line {reserve_line}, setup_fds at line {setup_fds_line}."
        )

    def test_log_lazy_initializes_logger_when_none(self, monkeypatch):
        """log() must lazy-initialize _logger if the module-scope binding
        is still None — otherwise non-daemon callers cannot log.
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_logger", None)

        fake_logger = MagicMock()
        fake_setup = MagicMock(return_value=fake_logger)
        monkeypatch.setattr(mod, "_setup_logging", fake_setup)

        mod.log("hello lazy")

        fake_setup.assert_called_once()
        fake_logger.info.assert_called_once_with("hello lazy")
        assert mod._logger is fake_logger


class TestSecurityHardening:
    """Aegis audit findings LOW-1, LOW-2, LOW-4 from the /security review of #99.

    Symlink TOCTOU hardening for log file opens, and graceful degradation
    when the kernel fd table is exhausted at daemonization time.
    """

    def test_open_log_file_secure_uses_nofollow_and_restrictive_mode(self, monkeypatch, tmp_path):
        """_open_log_file_secure must use O_NOFOLLOW (defeats symlink redirect)
        and mode 0o600 (not world-readable). Verifies the syscall signature
        via a mocked os.open.
        """
        from scripts.core.memory_daemon import _open_log_file_secure

        # Capture the real os.open BEFORE patching. The fake must NOT
        # call anything that goes through os.open indirectly (e.g.
        # ``Path.touch`` internally calls ``os.open`` and would recurse
        # into the monkeypatch). Doing the file creation via real_os_open
        # with O_CREAT in one shot avoids the recursion entirely.
        real_os_open = os.open
        real_path = str(tmp_path / "real.log")

        captured: dict = {}

        def fake_os_open(path, flags, mode=0o777):
            captured["path"] = path
            captured["flags"] = flags
            captured["mode"] = mode
            return real_os_open(
                real_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )

        monkeypatch.setattr(os, "open", fake_os_open)

        handle = _open_log_file_secure(tmp_path / "target.log", mode="ab")
        try:
            # O_NOFOLLOW defeats symlink TOCTOU.
            assert captured["flags"] & os.O_NOFOLLOW, "must use O_NOFOLLOW"
            # O_APPEND because we asked for "ab".
            assert captured["flags"] & os.O_APPEND
            assert captured["flags"] & os.O_CREAT
            assert captured["flags"] & os.O_WRONLY
            # Umask-free restrictive mode.
            assert captured["mode"] == 0o600
        finally:
            handle.close()

    def test_open_log_file_secure_refuses_to_follow_symlink(self, tmp_path):
        """End-to-end: if the target is a symlink to a file the attacker
        controls, _open_log_file_secure must raise rather than follow it.
        """
        from scripts.core.memory_daemon import _open_log_file_secure

        target = tmp_path / "real.log"
        target.write_text("attacker-controlled")
        link = tmp_path / "link.log"
        link.symlink_to(target)

        import pytest

        with pytest.raises(OSError):
            _open_log_file_secure(link, mode="ab")

        # The attacker-controlled file was NOT appended to.
        assert target.read_text() == "attacker-controlled"

    def test_open_log_file_secure_creates_new_file(self, tmp_path):
        """Happy path: the helper creates and opens a new log file."""
        from scripts.core.memory_daemon import _open_log_file_secure

        path = tmp_path / "new.log"
        handle = _open_log_file_secure(path, mode="ab")
        try:
            handle.write(b"hello\n")
            handle.flush()
        finally:
            handle.close()

        assert path.exists()
        assert path.read_bytes() == b"hello\n"
        # Mode check: owner-only read/write.
        assert path.stat().st_mode & 0o777 == 0o600

    def test_open_log_file_secure_tolerates_missing_o_nofollow(self, monkeypatch, tmp_path):
        """PR #106 Copilot P1/P3: os.O_NOFOLLOW is POSIX-only. On Windows
        the constant does not exist and any code that references it at
        runtime raises AttributeError. _open_log_file_secure must fall
        back gracefully via getattr so Windows daemon startup works.
        """
        import scripts.core.memory_daemon as mod

        # Simulate a platform where O_NOFOLLOW does not exist.
        if hasattr(os, "O_NOFOLLOW"):
            monkeypatch.delattr(os, "O_NOFOLLOW")

        # Must not raise AttributeError on O_NOFOLLOW lookup.
        handle = mod._open_log_file_secure(tmp_path / "nofollow_absent.log", mode="ab")
        try:
            assert (tmp_path / "nofollow_absent.log").exists()
        finally:
            handle.close()

    def test_open_log_file_secure_fchmods_existing_files(self, monkeypatch, tmp_path):
        """PR #106 Copilot P2: os.open's mode arg only applies on create.
        If the log file already exists with broader permissions, the
        helper must call os.fchmod(fd, 0o600) on a best-effort basis to
        enforce owner-only mode on upgrade paths.
        """
        import scripts.core.memory_daemon as mod

        target = tmp_path / "preexisting.log"
        target.write_text("old data")
        target.chmod(0o644)  # simulate loose permissions from old version
        assert target.stat().st_mode & 0o777 == 0o644

        handle = mod._open_log_file_secure(target, mode="ab")
        try:
            mode = target.stat().st_mode & 0o777
            assert mode == 0o600, f"expected 0o600, got 0o{mode:o}"
        finally:
            handle.close()

    def test_open_log_file_secure_survives_fchmod_failure(self, monkeypatch, tmp_path):
        """os.fchmod can fail on read-only filesystems or some network
        mounts. Best-effort: a failing fchmod must not prevent opening
        the file.
        """
        import scripts.core.memory_daemon as mod

        def raising_fchmod(_fd, _mode):
            raise PermissionError("read-only fs")

        monkeypatch.setattr(os, "fchmod", raising_fchmod)

        handle = mod._open_log_file_secure(tmp_path / "ro_mount.log", mode="ab")
        try:
            assert (tmp_path / "ro_mount.log").exists()
        finally:
            handle.close()

    def test_log_emits_one_time_stderr_warning_on_setup_failure(self, monkeypatch, capsys):
        """PR #106 CodeRabbit CR2: when _setup_logging() fails, log() should
        emit a one-time diagnostic warning to stderr so operators running
        CLI commands (status, stop) see the configuration error. In the
        daemon itself stderr is /dev/null after _setup_daemon_fds, so the
        warning is silently discarded — but the CLI paths benefit.
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_logger", None)
        monkeypatch.setattr(mod, "_log_setup_warning_emitted", False, raising=False)

        def failing_setup():
            raise OSError(13, "Permission denied: ~/.claude/memory-daemon.log")

        monkeypatch.setattr(mod, "_setup_logging", failing_setup)

        # First call: should emit the warning.
        mod.log("first message")
        err = capsys.readouterr().err
        assert "memory_daemon" in err.lower() or "log" in err.lower()
        assert "Permission denied" in err or "log setup failed" in err.lower()

        # Second call: should NOT emit again (once-only).
        mod.log("second message")
        err2 = capsys.readouterr().err
        assert err2 == "", f"warning should be one-time, got: {err2!r}"

    def test_setup_daemon_fds_survives_fd_exhaustion(self, monkeypatch):
        """Aegis LOW-4: if os.open raises OSError (fd exhaustion), the
        function must return gracefully rather than propagating the
        exception out of _run_as_daemon and killing the daemon with an
        unlogged traceback.
        """
        from scripts.core.memory_daemon import _setup_daemon_fds

        def raising_open(path, flags):
            raise OSError(24, "Too many open files")

        dup2_calls: list = []

        def fake_dup2(src, dst):
            dup2_calls.append((src, dst))

        # Must not raise.
        _setup_daemon_fds(
            os_open_fn=raising_open,
            os_dup2_fn=fake_dup2,
            os_close_fn=lambda _fd: None,
        )

        # And must not have attempted dup2 with an invalid fd.
        assert dup2_calls == []


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

        Hermetic: _enable_faulthandler is monkeypatched to a no-op so
        main() does not open the real ~/.claude/logs/opc_crash.log file
        (Codex Round 1 Finding 3).
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "DEBUG", False)
        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", "0")
        monkeypatch.setattr(mod, "_enable_faulthandler", lambda: None)
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
        monkeypatch.setattr(mod, "_enable_faulthandler", lambda: None)
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

    def test_enable_faulthandler_is_idempotent(self, monkeypatch, tmp_path):
        """Calling _enable_faulthandler twice must not re-open the log file.

        Hermetic: Path.home() is redirected to tmp_path so the real
        ~/.claude/logs/opc_crash.log is not touched. The real file handle
        we open is closed in the finally block so nothing leaks across
        tests (Codex Round 1 Finding 3).
        """
        import scripts.core.memory_daemon as mod

        # Redirect Path.home() so _enable_faulthandler writes under tmp_path.
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Reset the internal handle so the first call does real work.
        monkeypatch.setattr(mod, "_faulthandler_log_file", None, raising=False)

        enable_calls: list = []

        def fake_enable(*args, **kwargs):
            enable_calls.append((args, kwargs))

        monkeypatch.setattr(mod.faulthandler, "enable", fake_enable)

        try:
            mod._enable_faulthandler()
            first_handle = mod._faulthandler_log_file
            mod._enable_faulthandler()
            second_handle = mod._faulthandler_log_file

            assert first_handle is second_handle
            assert len(enable_calls) == 1

            # Verify the file was created under tmp_path, not real home.
            crash_log = tmp_path / ".claude" / "logs" / "opc_crash.log"
            assert crash_log.exists(), f"Expected {crash_log} to be created"
        finally:
            # Close the real file handle so it does not leak; monkeypatch
            # will then restore the module attribute.
            handle = mod._faulthandler_log_file
            if handle is not None:
                try:
                    handle.close()
                except (OSError, ValueError):
                    # OSError: close() syscall failure during teardown.
                    # ValueError: "I/O operation on closed file" if
                    # something else closed it first. Narrowed per
                    # CodeRabbit CR1 on PR #106.
                    pass


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

    def test_spawn_without_debug_does_not_pass_verbose(self, monkeypatch):
        """Release mode (DEBUG=False): pattern_batch is not invoked with
        --verbose, so it defaults to INFO level logging.
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))
        monkeypatch.setattr(mod, "DEBUG", False)
        monkeypatch.setattr(mod, "log", lambda _m: None)

        captured: dict = {}

        class _FakeProc:
            pid = 12345

            def poll(self):
                return None

        def fake_popen(*args, **kwargs):
            captured["argv"] = args[0] if args else kwargs.get("args")
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

        mod._run_pattern_detection_batch()

        argv = captured["argv"]
        assert "--verbose" not in argv
        assert "-v" not in argv

    def test_spawn_with_debug_passes_verbose_to_pattern_batch(self, monkeypatch):
        """Debug mode (DEBUG=True): pattern_batch is invoked with --verbose so
        it materially raises its log level. Env-var inheritance alone is
        insufficient — pattern_batch.py does not read MEMORY_DAEMON_DEBUG.
        (Codex Round 1 Finding 2.)
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))
        monkeypatch.setattr(mod, "DEBUG", True)
        monkeypatch.setattr(mod, "log", lambda _m: None)

        captured: dict = {}

        class _FakeProc:
            pid = 12345

            def poll(self):
                return None

        def fake_popen(*args, **kwargs):
            captured["argv"] = args[0] if args else kwargs.get("args")
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

        mod._run_pattern_detection_batch()

        argv = captured["argv"]
        assert "--verbose" in argv, f"Expected --verbose in argv, got: {argv}"

    def test_spawn_with_debug_redirects_stderr_to_log_file(self, monkeypatch, tmp_path):
        """Codex Round 2 Finding: DEBUG mode enables --verbose in the child,
        which floods stderr with DEBUG-level logging. With stderr=PIPE, the
        pipe buffer (64KB) can fill up and deadlock the child on write.

        Fix: when DEBUG is on, redirect stderr to an append-mode log file
        instead of piping it. The daemon does not need to read it; operators
        can tail ~/.claude/logs/pattern_batch_verbose.log.
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))
        monkeypatch.setattr(mod, "DEBUG", True)
        monkeypatch.setattr(mod, "log", lambda _m: None)

        captured: dict = {}

        class _FakeProc:
            pid = 12345

            def poll(self):
                return None

        def fake_popen(*args, **kwargs):
            captured["argv"] = args[0] if args else kwargs.get("args")
            captured["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

        mod._run_pattern_detection_batch()

        stderr_arg = captured["kwargs"].get("stderr")
        assert (
            stderr_arg is not mod.subprocess.PIPE
        ), "stderr must not be PIPE in DEBUG mode — would deadlock on verbose child"
        # The stderr arg must be a writable file-like object.
        assert hasattr(
            stderr_arg, "write"
        ), f"stderr should be a file object in DEBUG mode, got {type(stderr_arg).__name__}"
        # The verbose log file should have been created under tmp_path
        verbose_log = tmp_path / ".claude" / "logs" / "pattern_batch_verbose.log"
        assert verbose_log.exists(), f"{verbose_log} should be created"

    def test_spawn_without_debug_keeps_stderr_pipe(self, monkeypatch):
        """Release mode keeps stderr=PIPE for backward compatibility with
        the existing error-reporting path in _check_pattern_detection.
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))
        monkeypatch.setattr(mod, "DEBUG", False)
        monkeypatch.setattr(mod, "log", lambda _m: None)

        captured: dict = {}

        class _FakeProc:
            pid = 12345

            def poll(self):
                return None

        def fake_popen(*args, **kwargs):
            captured["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

        mod._run_pattern_detection_batch()

        assert captured["kwargs"].get("stderr") is mod.subprocess.PIPE

    def test_debug_stderr_handle_stashed_on_pattern_proc(self, monkeypatch, tmp_path):
        """The file handle used for stderr must be reachable for later
        cleanup via state.pattern_proc._debug_stderr_handle.
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))
        monkeypatch.setattr(mod, "DEBUG", True)
        monkeypatch.setattr(mod, "log", lambda _m: None)

        captured: dict = {}

        class _FakeProc:
            pid = 12345

            def poll(self):
                return None

        def fake_popen(*args, **kwargs):
            captured["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

        mod._run_pattern_detection_batch()

        proc = mod._ensure_daemon_state().pattern_proc
        handle = getattr(proc, "_debug_stderr_handle", None)
        assert handle is not None
        assert handle is captured["kwargs"]["stderr"]

        # Cleanup so the tmp_path file can be removed
        handle.close()

    def test_check_pattern_detection_closes_debug_stderr_handle(self, monkeypatch, tmp_path):
        """When _check_pattern_detection reaps a DEBUG-spawned process, it
        must close the stashed stderr file handle so we do not leak fds.
        """
        import io

        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))
        monkeypatch.setattr(mod, "log", lambda _m: None)

        # Simulate a completed debug spawn: pattern_proc exists with a
        # closeable stderr handle attached and poll() returning 0.
        closed = []

        class _FakeHandle(io.BytesIO):
            def close(self):
                closed.append(True)
                super().close()

        handle = _FakeHandle()

        class _FakeProc:
            stdout = io.BytesIO(b'{"patterns_detected": 0}')
            stderr = None  # redirected to file, not a pipe

            def poll(self):
                return 0

        proc = _FakeProc()
        proc._debug_stderr_handle = handle

        state = mod._ensure_daemon_state()
        state.pattern_proc = proc

        mod._check_pattern_detection()

        assert closed == [True], "_check_pattern_detection must close _debug_stderr_handle"
        assert state.pattern_proc is None

    def test_spawn_closes_debug_stderr_handle_on_popen_failure(self, monkeypatch, tmp_path):
        """Codex Round 3 Finding 2: if Popen raises after the verbose log is
        opened, the file handle must be closed. Otherwise repeated launch
        failures accumulate fds until the daemon hits its limit — most
        damaging in exactly the degraded environments where --debug is
        needed.
        """
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))
        monkeypatch.setattr(mod, "DEBUG", True)
        monkeypatch.setattr(mod, "log", lambda _m: None)

        # Track calls to close() on the verbose log handle.
        class _TrackedFile:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

            def write(self, _data):
                return 0

            def fileno(self):
                return 99

        tracked: list[_TrackedFile] = []

        # _run_pattern_detection_batch calls _open_log_file_secure (which
        # internally uses os.open + os.fdopen) to open the verbose log.
        # Intercept that helper directly so the test does not depend on
        # the specific fd plumbing inside.
        def fake_secure_open(path, mode="ab"):
            h = _TrackedFile()
            tracked.append(h)
            return h

        monkeypatch.setattr(mod, "_open_log_file_secure", fake_secure_open)

        # Force Popen to raise AFTER the verbose log file is opened.
        def fake_popen(*args, **kwargs):
            raise OSError("simulated spawn failure")

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

        mod._run_pattern_detection_batch()

        # Spawn failed → state.pattern_proc must not hold anything
        assert mod._ensure_daemon_state().pattern_proc is None

        # Exactly one verbose log handle was opened, and it was closed
        # before the function returned (no fd leak).
        assert len(tracked) == 1, f"Expected 1 verbose log open, got {len(tracked)}"
        assert tracked[0].closed, "Verbose log handle must be closed after Popen failure"

    def test_check_pattern_detection_survives_null_stderr_on_failure(self, monkeypatch):
        """When stderr was redirected to a file (proc.stderr is None), the
        failure path must not crash with AttributeError on None.read().
        """
        import io

        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "_daemon_state", self._fresh_state(mod))

        log_calls: list[str] = []
        monkeypatch.setattr(mod, "log", lambda m: log_calls.append(m))

        class _FakeProc:
            stdout = io.BytesIO(b"")
            stderr = None  # redirected, no pipe

            def poll(self):
                return 1  # non-zero → failure path

        state = mod._ensure_daemon_state()
        state.pattern_proc = _FakeProc()

        # Must not raise.
        mod._check_pattern_detection()

        # Should have logged a failure message (content format is flexible).
        assert any(
            "failed" in m.lower() for m in log_calls
        ), f"Expected failure log, got: {log_calls}"

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
