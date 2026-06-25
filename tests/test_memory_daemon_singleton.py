"""Tests for memory_daemon singleton startup lock and readiness handshake (Issue #102).

The daemon start path previously had two race/observability defects:

  1. No singleton handshake: two concurrent ``start`` calls could each observe
     "not running" via ``is_running()`` before either wrote the PID file, and
     both would fork — producing duplicate daemons.
  2. False-positive success: the parent printed "Memory daemon started" before
     the grandchild reached ``ensure_schema()`` / ``recover_stalled_extractions()``,
     so a DB-init failure became a silent-absent daemon reported as success.

The fix adds:
  - ``_acquire_singleton_lock()`` — an exclusive non-blocking ``flock`` taken
    BEFORE forking. A second concurrent ``start`` gets ``BlockingIOError`` and
    bails out instead of forking. The lock fd is inherited by the grandchild
    and held for the daemon's lifetime (``flock`` is per open-file-description
    and survives ``fork``).
  - A readiness handshake over ``os.pipe()``: the parent blocks until the
    grandchild signals ready (after basic DB init succeeds) or failed.
  - The PID file is written only AFTER basic init succeeds (via the
    ``daemon_loop(on_ready=...)`` callback), so the PID never points at a
    daemon that died during init.

Mirrors the dependency-injection test style used throughout
test_memory_daemon_startup.py: os.* syscalls are injected so the syscall
sequence can be asserted without touching real file descriptors.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


class TestAcquireSingletonLock:
    """_acquire_singleton_lock() serializes daemon startup via flock(LOCK_EX|LOCK_NB)."""

    def test_returns_fd_on_successful_lock(self, tmp_path):
        from scripts.core.memory_daemon import _acquire_singleton_lock

        fake_open = MagicMock(return_value=7)
        fake_flock = MagicMock()
        fake_close = MagicMock()

        fd = _acquire_singleton_lock(
            lock_path=tmp_path / "daemon.lock",
            os_open_fn=fake_open,
            flock_fn=fake_flock,
            os_close_fn=fake_close,
        )

        assert fd == 7
        fake_flock.assert_called_once_with(7)
        # On success the fd is RETAINED (held for the daemon lifetime).
        fake_close.assert_not_called()

    def test_returns_none_and_closes_fd_when_lock_held(self, tmp_path):
        """A second concurrent start sees BlockingIOError (EWOULDBLOCK) and
        must return None after releasing the fd it opened.
        """
        from scripts.core.memory_daemon import _acquire_singleton_lock

        fake_open = MagicMock(return_value=7)
        fake_close = MagicMock()

        def held_flock(_fd):
            raise BlockingIOError("lock held")

        fd = _acquire_singleton_lock(
            lock_path=tmp_path / "daemon.lock",
            os_open_fn=fake_open,
            flock_fn=held_flock,
            os_close_fn=fake_close,
        )

        assert fd is None
        fake_close.assert_called_once_with(7)

    def test_opens_lock_file_with_create_and_owner_only_mode(self, tmp_path):
        from scripts.core.memory_daemon import _acquire_singleton_lock

        captured: dict = {}

        def fake_open(path, flags, mode=0o777):
            captured["path"] = path
            captured["flags"] = flags
            captured["mode"] = mode
            return 7

        _acquire_singleton_lock(
            lock_path=tmp_path / "daemon.lock",
            os_open_fn=fake_open,
            flock_fn=MagicMock(),
            os_close_fn=MagicMock(),
        )

        assert captured["flags"] & os.O_CREAT, "lock file must be created if absent"
        assert captured["mode"] == 0o600, "lock file must be owner-only"

    def test_real_flock_is_mutually_exclusive(self, tmp_path):
        """End-to-end with the real flock_fn: the first acquire holds the lock,
        a second acquire on the same path returns None.
        """
        from scripts.core.memory_daemon import _acquire_singleton_lock

        lock_path = tmp_path / "daemon.lock"

        first = _acquire_singleton_lock(lock_path=lock_path)
        assert first is not None, "first acquire should succeed"
        try:
            second = _acquire_singleton_lock(lock_path=lock_path)
            assert second is None, "second acquire must fail while first holds the lock"
        finally:
            os.close(first)

        # After the first lock is released, a fresh acquire succeeds again.
        third = _acquire_singleton_lock(lock_path=lock_path)
        assert third is not None
        os.close(third)


class TestReadinessSignals:
    """_signal_ready / _signal_failed / _wait_for_ready implement the handshake."""

    def test_signal_ready_writes_ready_byte_and_closes_writer(self):
        from scripts.core.memory_daemon import _signal_ready

        read_fd, write_fd = os.pipe()
        try:
            _signal_ready(write_fd)
            assert os.read(read_fd, 1) == b"R"
            # Writer is closed → EOF on the next read.
            assert os.read(read_fd, 1) == b""
        finally:
            os.close(read_fd)
        # Double-closing the writer proves _signal_ready already closed it.
        with pytest.raises(OSError):
            os.close(write_fd)

    def test_signal_failed_writes_fail_byte_and_closes_writer(self):
        from scripts.core.memory_daemon import _signal_failed

        read_fd, write_fd = os.pipe()
        try:
            _signal_failed(write_fd)
            assert os.read(read_fd, 1) == b"F"
        finally:
            os.close(read_fd)
        with pytest.raises(OSError):
            os.close(write_fd)

    def test_signal_failed_tolerates_already_closed_writer(self):
        """If the parent already gave up and closed its read end, signaling
        failure must not raise out of the grandchild.
        """
        from scripts.core.memory_daemon import _signal_failed

        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        os.close(write_fd)
        # Writing to a fully closed pipe fd must be swallowed.
        _signal_failed(write_fd)  # must not raise

    def test_wait_for_ready_returns_ready_on_ready_byte(self):
        from scripts.core.memory_daemon import _wait_for_ready

        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"R")
        os.close(write_fd)
        assert _wait_for_ready(read_fd, timeout=1.0) == "ready"

    def test_wait_for_ready_returns_failed_on_fail_byte(self):
        from scripts.core.memory_daemon import _wait_for_ready

        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"F")
        os.close(write_fd)
        assert _wait_for_ready(read_fd, timeout=1.0) == "failed"

    def test_wait_for_ready_returns_failed_when_grandchild_dies_without_signaling(self):
        """Closed pipe with no byte (grandchild crashed) → EOF → failed."""
        from scripts.core.memory_daemon import _wait_for_ready

        read_fd, write_fd = os.pipe()
        os.close(write_fd)  # simulate grandchild death before signaling
        assert _wait_for_ready(read_fd, timeout=1.0) == "failed"

    def test_wait_for_ready_returns_timeout_distinct_from_failure(self):
        """A timeout must be reported as 'timeout', NOT 'failed' — the daemon may
        still be initializing and must not be presumed dead (R1 finding)."""
        from scripts.core.memory_daemon import _wait_for_ready

        read_fd, write_fd = os.pipe()
        try:
            # Never write → select times out.
            assert _wait_for_ready(read_fd, timeout=0.05) == "timeout"
        finally:
            os.close(write_fd)

    def test_wait_for_ready_closes_read_fd(self):
        from scripts.core.memory_daemon import _wait_for_ready

        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"R")
        os.close(write_fd)
        _wait_for_ready(read_fd, timeout=1.0)
        # read_fd should already be closed by _wait_for_ready.
        with pytest.raises(OSError):
            os.close(read_fd)


class TestStartDaemonSingletonGuard:
    """start_daemon acquires the lock BEFORE forking; a held lock aborts startup."""

    def test_refuses_to_fork_when_lock_held(self, monkeypatch, capsys):
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "is_running", lambda: (False, None))
        monkeypatch.setattr(mod, "_acquire_singleton_lock", lambda: None)

        fork_calls: list = []
        monkeypatch.setattr(mod.os, "fork", lambda: fork_calls.append(1))

        rc = mod.start_daemon()

        assert rc == 0
        assert fork_calls == [], "start_daemon must not fork when the lock is held"
        out = capsys.readouterr().out.lower()
        assert "already" in out, f"expected an 'already starting/running' message, got: {out!r}"

    def test_still_reports_already_running_via_pid_check(self, monkeypatch, capsys):
        """The fast-path is_running() check is preserved for a friendly message."""
        import scripts.core.memory_daemon as mod

        monkeypatch.setattr(mod, "is_running", lambda: (True, 4242))
        lock_calls: list = []
        monkeypatch.setattr(mod, "_acquire_singleton_lock", lambda: lock_calls.append(1))

        rc = mod.start_daemon()

        assert rc == 0
        assert lock_calls == [], "no need to take the lock if already running"
        out = capsys.readouterr().out
        assert "4242" in out


class TestDaemonLoopOnReady:
    """daemon_loop(on_ready=...) fires the callback after ALL setup succeeds."""

    def test_on_ready_fires_after_all_setup_before_serving(self, monkeypatch):
        """R2 Finding 3: readiness must fire only after every abort-capable
        setup step — schema, recovery, AND _seed_last_pattern_run — so a failure
        in the later steps is reported as failure, not a phantom success.
        """
        import scripts.core.memory_daemon as mod

        order: list[str] = []

        class _BreakLoopError(Exception):
            pass

        monkeypatch.setattr(mod, "ensure_schema", lambda: order.append("schema"))
        monkeypatch.setattr(mod, "recover_stalled_extractions", lambda: order.append("recover"))

        def _seed():
            order.append("seed")
            return 0.0

        monkeypatch.setattr(mod, "_seed_last_pattern_run", _seed)
        monkeypatch.setattr(mod, "use_postgres", lambda: False)
        monkeypatch.setattr(mod, "log", lambda _m: None)
        monkeypatch.setattr(mod, "daemon_tick", lambda: None)
        monkeypatch.setattr(mod, "_daemon_state", mod.create_daemon_state(), raising=False)

        def _break_sleep(*_a, **_k):
            raise _BreakLoopError()

        monkeypatch.setattr(mod.time, "sleep", _break_sleep)

        def on_ready():
            order.append("ready")

        with pytest.raises(_BreakLoopError):
            mod.daemon_loop(on_ready=on_ready)

        assert order == ["schema", "recover", "seed", "ready"], (
            f"on_ready must fire after ALL setup (incl. seed), immediately before "
            f"the tick loop; got {order}"
        )

    def test_on_ready_not_called_when_setup_after_recover_raises(self, monkeypatch):
        """R2 Finding 3: if a setup step after recover (here, seed) raises,
        on_ready must NOT fire — the exception propagates so the caller signals
        failure rather than the daemon reporting a phantom success.
        """
        import scripts.core.memory_daemon as mod

        called: list[str] = []

        monkeypatch.setattr(mod, "ensure_schema", lambda: None)
        monkeypatch.setattr(mod, "recover_stalled_extractions", lambda: None)

        def _seed_boom():
            raise RuntimeError("seed failed")

        monkeypatch.setattr(mod, "_seed_last_pattern_run", _seed_boom)
        monkeypatch.setattr(mod, "use_postgres", lambda: False)
        monkeypatch.setattr(mod, "log", lambda _m: None)
        monkeypatch.setattr(mod, "_daemon_state", mod.create_daemon_state(), raising=False)

        with pytest.raises(RuntimeError, match="seed failed"):
            mod.daemon_loop(on_ready=lambda: called.append("ready"))

        assert called == [], "on_ready must not fire when post-recover setup raises"

    def test_default_on_ready_none_is_backward_compatible(self, monkeypatch):
        """daemon_loop() with no on_ready must still run init and the loop."""
        import scripts.core.memory_daemon as mod

        ran: list[str] = []

        class _BreakLoopError(Exception):
            pass

        monkeypatch.setattr(mod, "ensure_schema", lambda: ran.append("schema"))
        monkeypatch.setattr(mod, "recover_stalled_extractions", lambda: ran.append("recover"))
        monkeypatch.setattr(mod, "_seed_last_pattern_run", lambda: 0.0)
        monkeypatch.setattr(mod, "use_postgres", lambda: False)
        monkeypatch.setattr(mod, "log", lambda _m: None)
        monkeypatch.setattr(mod, "daemon_tick", lambda: None)
        monkeypatch.setattr(mod, "_daemon_state", mod.create_daemon_state(), raising=False)

        def _break_sleep(*_a, **_k):
            raise _BreakLoopError()

        monkeypatch.setattr(mod.time, "sleep", _break_sleep)

        with pytest.raises(_BreakLoopError):
            mod.daemon_loop()  # no on_ready — must not raise TypeError

        assert ran == ["schema", "recover"]


class TestRunAsDaemonReadiness:
    """_run_as_daemon wires PID write + readiness signal through on_ready."""

    def _patch_bootstrap(self, monkeypatch, mod):
        monkeypatch.setattr(mod, "_harden_daemon_environment", lambda: None)
        monkeypatch.setattr(mod, "_reserve_low_fds", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_setup_logging", lambda: MagicMock())
        monkeypatch.setattr(mod, "_setup_daemon_fds", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_logger", MagicMock(), raising=False)
        monkeypatch.setattr(mod, "log", lambda _m: None)

    def test_writes_pid_early_and_signals_ready_on_success(self, monkeypatch, tmp_path):
        import scripts.core.memory_daemon as mod

        self._patch_bootstrap(monkeypatch, mod)
        pid_file = tmp_path / "daemon.pid"
        monkeypatch.setattr(mod, "PID_FILE", pid_file)

        seen: dict = {}

        def fake_loop(on_ready=None):
            # R2: the PID is written EARLY, before daemon_loop runs, so it must
            # already exist when the loop is entered — before readiness fires.
            seen["pid_before_ready"] = pid_file.exists()
            if on_ready is not None:
                on_ready()
            seen["pid_after_ready"] = pid_file.exists()
            seen["pid_content"] = pid_file.read_text() if pid_file.exists() else None

        monkeypatch.setattr(mod, "daemon_loop", fake_loop)

        read_fd, write_fd = os.pipe()
        try:
            mod._run_as_daemon(ready_write_fd=write_fd, lock_fd=None)
            assert seen["pid_before_ready"] is True, "PID must be written before init"
            assert seen["pid_after_ready"] is True
            assert seen["pid_content"] == str(os.getpid())
            assert os.read(read_fd, 1) == b"R"
        finally:
            os.close(read_fd)

    def test_signals_failed_and_cleans_up_pid_when_init_raises(self, monkeypatch, tmp_path):
        import scripts.core.memory_daemon as mod

        self._patch_bootstrap(monkeypatch, mod)
        pid_file = tmp_path / "daemon.pid"
        monkeypatch.setattr(mod, "PID_FILE", pid_file)

        def fake_loop(on_ready=None):
            raise RuntimeError("schema unreachable")

        monkeypatch.setattr(mod, "daemon_loop", fake_loop)

        read_fd, write_fd = os.pipe()
        try:
            # Must swallow the init error (already reported via the pipe).
            mod._run_as_daemon(ready_write_fd=write_fd, lock_fd=None)
            assert os.read(read_fd, 1) == b"F"
            # The PID is written early but the finally-clause unlinks it when the
            # daemon dies during init, so nothing lingers pointing at a corpse.
            assert not pid_file.exists(), "PID must be cleaned up when init fails"
        finally:
            os.close(read_fd)

    def test_does_not_release_lock_fd_before_returning(self, monkeypatch, tmp_path):
        """R3: _run_as_daemon must NOT close the singleton lock fd. Closing it
        while the process is still alive (before the caller's os._exit) would
        release the flock after the PID is already unlinked, opening a duplicate
        -start window. The lock must stay held until the kernel closes the fd at
        true process exit, so here the fd remains open after _run_as_daemon
        returns (os.close succeeds rather than raising EBADF).
        """
        import scripts.core.memory_daemon as mod

        self._patch_bootstrap(monkeypatch, mod)
        pid_file = tmp_path / "daemon.pid"
        monkeypatch.setattr(mod, "PID_FILE", pid_file)
        monkeypatch.setattr(mod, "daemon_loop", lambda on_ready=None: None)

        # A real fd to stand in for the inherited lock fd.
        lock_r, lock_w = os.pipe()
        try:
            mod._run_as_daemon(ready_write_fd=None, lock_fd=lock_w)
            # The PID is removed on exit, but the lock fd is deliberately left
            # OPEN (held until the kernel reclaims it at process exit).
            assert not pid_file.exists()
            os.close(lock_w)  # must not raise — fd is still open
        finally:
            os.close(lock_r)

    def test_propagates_failure_exit_code_on_post_readiness_crash(self, monkeypatch, tmp_path):
        """R3: a daemon_loop escape after readiness must yield a non-zero exit
        code so the grandchild os._exit's with failure rather than masking the
        crash as success.
        """
        import scripts.core.memory_daemon as mod

        self._patch_bootstrap(monkeypatch, mod)
        monkeypatch.setattr(mod, "PID_FILE", tmp_path / "daemon.pid")

        def fake_loop(on_ready=None):
            if on_ready is not None:
                on_ready()  # readiness reached
            raise RuntimeError("loop escaped after readiness")

        monkeypatch.setattr(mod, "daemon_loop", fake_loop)

        rc = mod._run_as_daemon(ready_write_fd=None, lock_fd=None)
        assert rc == 1, "post-readiness crash must surface a non-zero exit code"

    def test_clean_return_yields_zero_exit_code(self, monkeypatch, tmp_path):
        import scripts.core.memory_daemon as mod

        self._patch_bootstrap(monkeypatch, mod)
        monkeypatch.setattr(mod, "PID_FILE", tmp_path / "daemon.pid")
        monkeypatch.setattr(mod, "daemon_loop", lambda on_ready=None: None)

        rc = mod._run_as_daemon(ready_write_fd=None, lock_fd=None)
        assert rc == 0

    def test_survives_parent_timeout_that_closed_the_pipe(self, monkeypatch, tmp_path):
        """R1 regression: if the parent timed out and closed its read end before
        the grandchild finished init, the readiness signal hits a broken pipe.
        That BrokenPipeError must NOT be misclassified as init failure — the
        daemon successfully initialized, so its PID must be written, it must NOT
        be signaled as failed, and _run_as_daemon must not log an init error.
        """
        import scripts.core.memory_daemon as mod

        self._patch_bootstrap(monkeypatch, mod)
        logged: list[str] = []
        monkeypatch.setattr(mod, "log", lambda m: logged.append(m))
        pid_file = tmp_path / "daemon.pid"
        monkeypatch.setattr(mod, "PID_FILE", pid_file)

        # Parent already gave up: read end closed before on_ready fires.
        read_fd, write_fd = os.pipe()
        os.close(read_fd)

        seen: dict = {}

        def fake_loop(on_ready=None):
            # Init succeeded; fire readiness against the closed pipe, then the
            # daemon would proceed into its tick loop normally.
            if on_ready is not None:
                on_ready()
            seen["pid_during_loop"] = pid_file.exists()
            seen["pid_content"] = pid_file.read_text() if pid_file.exists() else None

        monkeypatch.setattr(mod, "daemon_loop", fake_loop)

        # Must not raise despite the broken readiness pipe.
        mod._run_as_daemon(ready_write_fd=write_fd, lock_fd=None)

        # The healthy daemon wrote its PID and ran the loop — not aborted.
        assert seen["pid_during_loop"] is True
        assert seen["pid_content"] == str(os.getpid())
        # And it was NOT treated as an init failure.
        assert not any(
            "exited with error" in m.lower() for m in logged
        ), f"broken readiness pipe must not be logged as a daemon error; got {logged}"
