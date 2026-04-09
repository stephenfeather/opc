"""RED-phase tests for Issue #96 — memory_daemon_core.py DEBUG gating.

These tests define the behavioral contract for DEBUG-gated diagnostic
logging in `scripts/core/memory_daemon_core.py`. They MUST fail until
kraken's Phase 3 implementation lands.

Regression baseline: `tests/test_memory_daemon_core.py` remains
unchanged. This file is additive and only tests the NEW surface:

1. A module-level `debug(msg)` helper in `memory_daemon_core`.
2. Call-site DEBUG gating inside `build_extraction_command` (argv dump).
3. Call-site DEBUG gating inside `build_extraction_env`
   (env KEY NAMES only — never values).
4. Module import hygiene (no I/O at import time).
5. Structural absence of silent `except Exception: pass` blocks.

Module docstring notes (for future reviewers — Learning #5 / R8):

- Broad `except Exception` handlers in logging code paths are
  INTENTIONAL by design. These tests DO NOT assert narrow exception
  types in the log path. If a narrowing suggestion appears in review,
  the correct response is "see PR #106 defense of M2/M5 — log-path
  broad catches are defensive".

Hermeticity notes (Learning #3 / F3 / R3; updated Round 3):

- Every test in this module is protected by the autouse
  function-scoped ``_hermetic_log`` fixture, which stubs
  ``memory_daemon.log`` (and defensively ``memory_daemon_core.log``
  if it exists) to a no-op lambda for the duration of each test.
  This is ACTIVE hermeticity (replace the I/O sink) rather than
  OBSERVATIONAL (watch a file size), and it replaces the earlier
  module-scope ``no_real_log_file`` fixture. Round 3 eliminated
  ``no_real_log_file`` because it had two failure modes: silent
  pass on clean machines (the pre-test byte-count check was
  skipped when the log file did not exist) and false positives on
  shared dev boxes (a concurrent Claude session writing during the
  test run would trigger the check).

- Tests that need to inspect captured log messages inject the
  ``log_spy`` fixture explicitly. ``log_spy`` reinstalls a
  list-appender on top of ``_hermetic_log``'s no-op stub; pytest's
  function-scoped fixture ordering means the later ``setattr`` wins.

- Do NOT monkeypatch `builtins.open` (Learning #4). The core module
  avoids I/O, so there is no open call to intercept anyway. For the
  import-hygiene tests, we monkeypatch the high-level helpers
  (`memory_daemon._open_log_file_secure`, `memory_daemon._setup_logging`)
  to tripwires and assert they were never invoked during import.

RED-phase failure-mode guard:

- Several tests in this module would pass trivially on the
  pre-implementation source — e.g., "silent when DEBUG off" holds
  because there is NO logging at all today, and "return value
  unchanged" holds because DEBUG state is ignored today. To ensure
  every test in RED phase fails for the right reason (feature
  missing, not feature present and correct), those tests call
  `_require_debug_feature()` as their first statement. It raises
  AssertionError if `memory_daemon_core.debug` does not yet exist.
"""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

import scripts.core.memory_daemon as memory_daemon
import scripts.core.memory_daemon_core as memory_daemon_core

CORE_MODULE_PATH = Path(memory_daemon_core.__file__)


def _require_debug_feature():
    """Fail-loud guard: assert the Issue #96 feature surface exists.

    Used by tests that would otherwise pass trivially on the pre-
    implementation source because they check properties that already
    hold (e.g., "no logging emitted" is trivially true when there is
    no logging code at all). With this guard, those tests fail in
    the RED phase with a clear message, then pass in GREEN phase
    when the feature lands AND the invariant still holds.
    """
    assert hasattr(memory_daemon_core, "debug"), (
        "memory_daemon_core.debug helper is missing — Issue #96 "
        "implementation has not landed. This test protects both a "
        "pre-existing invariant AND the presence of the new feature; "
        "it will pass when both hold."
    )


# ---------------------------------------------------------------------------
# Fixtures (§5 of the plan)
# ---------------------------------------------------------------------------


@pytest.fixture
def log_spy(monkeypatch):
    """F1: Capture log calls without touching the real daemon log file.

    Patches BOTH `memory_daemon_core.log` (if present) AND
    `memory_daemon.log` defensively, per R2. Returns a list that
    accumulates every message that either `log` call receives.
    """
    messages: list[str] = []

    def _spy(msg):
        messages.append(msg)

    # Always patch the sibling module's log — it definitely exists.
    monkeypatch.setattr(memory_daemon, "log", _spy, raising=True)

    # Patch core.log if/when the implementation adds one. Use
    # raising=False so we don't fail the RED phase just because the
    # attribute is not there yet — test 10 is what asserts its
    # existence.
    monkeypatch.setattr(memory_daemon_core, "log", _spy, raising=False)

    return messages


@pytest.fixture
def debug_on(monkeypatch):
    """F2 variant: force DEBUG=True via env var for the test duration."""
    monkeypatch.setenv("MEMORY_DAEMON_DEBUG", "1")
    # Also flip the sibling module's cached DEBUG constant so a lazy-
    # import-based implementation sees the change. Harmless if the
    # implementation reads the env var directly.
    monkeypatch.setattr(memory_daemon, "DEBUG", True, raising=False)
    yield
    # monkeypatch reverses both on teardown.


@pytest.fixture
def debug_off(monkeypatch):
    """F2 variant: force DEBUG=False via env unset."""
    monkeypatch.delenv("MEMORY_DAEMON_DEBUG", raising=False)
    monkeypatch.setattr(memory_daemon, "DEBUG", False, raising=False)
    yield


@pytest.fixture
def mock_with_eager_trap():
    """F4: MagicMock whose attribute access raises on the trap attr.

    Critical per R1: the trap must be on an ATTRIBUTE reference, not
    on `__repr__`. The test for C5 wants to prove that the DEBUG=False
    path never touches the attribute — a `__repr__`-only trap can be
    bypassed by an implementation that stringifies without repr.
    """
    trap = MagicMock(name="eager_eval_trap")
    # Attach a PropertyMock that raises on ANY access to `.trapped_attr`.
    type(trap).trapped_attr = PropertyMock(
        side_effect=RuntimeError("eager-eval leak detected")
    )
    return trap


@pytest.fixture(scope="module")
def core_source_ast():
    """F5: Parse memory_daemon_core.py once per module for structural tests."""
    source = CORE_MODULE_PATH.read_text()
    return ast.parse(source)


@pytest.fixture(autouse=True)
def _hermetic_log(monkeypatch):
    """F3: Autouse hermeticity — stub ``log()`` to a no-op for every test.

    Every test in this module gets ``memory_daemon.log`` (and, defensively,
    ``memory_daemon_core.log`` if it exists) replaced with a no-op
    lambda. This prevents any test from accidentally writing to the real
    ``~/.claude/memory-daemon.log`` file, even if the test forgets to
    inject the ``log_spy`` fixture.

    Tests that need to inspect captured log messages inject ``log_spy``
    explicitly. ``log_spy`` reinstalls its own list-appender on top of
    this no-op stub via ``monkeypatch.setattr``, and pytest's function-
    scoped fixture ordering means the later setattr wins — so log_spy's
    capture behavior is preserved.

    Replaces the previous module-scope ``no_real_log_file`` fixture,
    which had two failure modes (Codex Round 3 MEDIUM finding):

    1. **Silent pass on clean machines.** If ``~/.claude/memory-daemon.log``
       did not exist pre-test (``before = None``), the post-test check
       was skipped entirely — a leak that created the file would not be
       caught.

    2. **False positives on shared dev boxes.** The fixture watched a
       shared home-directory path, so a concurrent Claude session
       writing to the log during the test module's run would false-
       positive the entire module.

    This replacement eliminates both failure modes by making hermeticity
    active (stub the I/O sink) rather than observational (watch the
    file size).
    """
    from scripts.core import memory_daemon as _daemon

    monkeypatch.setattr(_daemon, "log", lambda _m: None, raising=True)
    # Defensive: core module may not bind its own `log` attribute,
    # so allow the setattr to no-op if the attribute does not exist.
    monkeypatch.setattr(
        memory_daemon_core, "log", lambda _m: None, raising=False
    )


# ---------------------------------------------------------------------------
# §4.1 Module import hygiene (tests 1–4)
# ---------------------------------------------------------------------------


class TestImportHygiene:
    """§4.1 — importing memory_daemon_core must be side-effect-free."""

    def test_import_does_not_call_setup_logging(self, monkeypatch):
        """Test 1: _setup_logging is not invoked during core import.

        Arrangement: patch memory_daemon._setup_logging to a tripwire,
        then reload memory_daemon_core. The reload must not propagate
        to the sibling module's setup path.
        """
        calls: list[str] = []

        def _tripwire(*args, **kwargs):
            calls.append("_setup_logging called")
            raise RuntimeError(
                "IMPORT HYGIENE VIOLATION: _setup_logging was invoked "
                "during core module import"
            )

        monkeypatch.setattr(memory_daemon, "_setup_logging", _tripwire)
        importlib.reload(memory_daemon_core)
        assert calls == [], (
            f"Expected zero _setup_logging calls during core import, "
            f"got {calls}"
        )
        # RED-phase fail-loud guard: without this, the test passes
        # trivially because core has no logging at all today. The
        # guard forces a fail until the feature lands.
        _require_debug_feature()

    def test_import_does_not_open_log_file(self, monkeypatch):
        """Test 2: _open_log_file_secure is not invoked during core import.

        Learning #4: patch the HIGH-LEVEL helper, not builtins.open.

        PR #110 Cycle 1 C1 (Copilot + CodeRabbit): the tripwire raises
        immediately rather than delegating to the real helper. An
        earlier iteration called ``original(*args, **kwargs)`` from
        inside the tripwire, which meant a regression that actually
        tripped this path would perform real filesystem I/O against
        ``~/.claude/memory-daemon.log`` — undermining the hermeticity
        guarantee the test is supposed to enforce. Raising ensures
        the failure mode is loud and I/O-free.
        """
        calls: list[tuple] = []

        def _tripwire(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError(
                "IMPORT HYGIENE VIOLATION: _open_log_file_secure was "
                "invoked during core module import"
            )

        monkeypatch.setattr(memory_daemon, "_open_log_file_secure", _tripwire)
        importlib.reload(memory_daemon_core)
        assert calls == [], (
            f"Expected zero _open_log_file_secure calls during core "
            f"import, got {len(calls)} calls: {calls}"
        )
        # RED-phase fail-loud guard.
        _require_debug_feature()

    def test_import_does_not_read_debug_env_var_destructively(self, monkeypatch):
        """Test 3: reading MEMORY_DAEMON_DEBUG at import is read-only.

        Observable: set the env to a sentinel, import, assert the
        env var is unchanged afterwards. This asserts the import path
        does not mutate os.environ.
        """
        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", "sentinel-001")
        importlib.reload(memory_daemon_core)
        assert os.environ.get("MEMORY_DAEMON_DEBUG") == "sentinel-001"
        # The module must also expose a `debug` helper — this is the
        # first test that asserts the new surface exists at all.
        assert hasattr(memory_daemon_core, "debug"), (
            "memory_daemon_core.debug helper is missing — Issue #96 "
            "implementation has not landed"
        )

    def test_reimport_is_idempotent(self):
        """Test 4: importlib.reload does not raise and does not produce I/O."""
        # First reload — capture any exception.
        try:
            importlib.reload(memory_daemon_core)
            importlib.reload(memory_daemon_core)
        except Exception as exc:  # pragma: no cover — diagnostic path
            pytest.fail(f"reimport raised: {exc!r}")
        # After two reloads, the debug helper must still exist.
        assert hasattr(memory_daemon_core, "debug"), (
            "debug helper missing after reload"
        )


# ---------------------------------------------------------------------------
# §4.2 DEBUG state resolution (tests 5–9)
# ---------------------------------------------------------------------------


class TestDebugStateResolution:
    """§4.2 — MEMORY_DAEMON_DEBUG env var controls logging."""

    def test_debug_off_when_env_unset(self, monkeypatch, log_spy):
        """Test 5: delenv → calling debug-gated code emits nothing."""
        monkeypatch.delenv("MEMORY_DAEMON_DEBUG", raising=False)
        monkeypatch.setattr(memory_daemon, "DEBUG", False, raising=False)
        memory_daemon_core.debug("should not appear")
        assert log_spy == [], (
            f"Expected zero log calls with env unset, got {log_spy}"
        )

    def test_debug_on_when_env_is_1(self, monkeypatch, log_spy):
        """Test 6: MEMORY_DAEMON_DEBUG=1 → calling debug() emits ≥1 log."""
        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", "1")
        monkeypatch.setattr(memory_daemon, "DEBUG", True, raising=False)
        memory_daemon_core.debug("hello-6")
        assert any("hello-6" in str(m) for m in log_spy), (
            f"Expected log containing 'hello-6' with DEBUG=1, got {log_spy}"
        )

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "on"])
    def test_debug_on_when_env_is_truthy_case_insensitive(
        self, monkeypatch, log_spy, value
    ):
        """Test 7: all truthy tokens enable debug."""
        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", value)
        monkeypatch.setattr(memory_daemon, "DEBUG", True, raising=False)
        memory_daemon_core.debug(f"token-{value}")
        assert any(f"token-{value}" in str(m) for m in log_spy), (
            f"Expected debug ENABLED for MEMORY_DAEMON_DEBUG={value!r}, "
            f"got empty log"
        )

    @pytest.mark.parametrize(
        "value", ["0", "false", "FALSE", "no", "off", "", "garbage"]
    )
    def test_debug_off_when_env_is_falsy(self, monkeypatch, log_spy, value):
        """Test 8: all falsy tokens disable debug."""
        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", value)
        monkeypatch.setattr(memory_daemon, "DEBUG", False, raising=False)
        memory_daemon_core.debug(f"token-{value}")
        matching = [m for m in log_spy if f"token-{value}" in str(m)]
        assert matching == [], (
            f"Expected debug DISABLED for MEMORY_DAEMON_DEBUG={value!r}, "
            f"got log calls: {matching}"
        )

    @pytest.mark.skip(
        reason="R4/C3: lazy-import fallback path is optional per "
        "implementation choice. Kraken is free to choose env-var-only "
        "(the plan recommends this). Re-enable this test if a future "
        "PR switches memory_daemon_core to the lazy-import path."
    )
    def test_lazy_import_fallback_if_used(self, monkeypatch, log_spy):
        """Test 9: mutating memory_daemon.DEBUG at runtime toggles behavior."""
        monkeypatch.delenv("MEMORY_DAEMON_DEBUG", raising=False)
        monkeypatch.setattr(memory_daemon, "DEBUG", True, raising=False)
        memory_daemon_core.debug("lazy-path")
        assert any("lazy-path" in str(m) for m in log_spy)


# ---------------------------------------------------------------------------
# §4.3 debug() helper short-circuit (tests 10–13)
# ---------------------------------------------------------------------------


class TestDebugHelperBehavior:
    """§4.3 — the module-level debug() helper."""

    def test_debug_helper_exists(self):
        """Test 10: hasattr(memory_daemon_core, 'debug')."""
        assert hasattr(memory_daemon_core, "debug"), (
            "memory_daemon_core.debug helper is missing"
        )
        assert callable(memory_daemon_core.debug), (
            "memory_daemon_core.debug exists but is not callable"
        )

    def test_debug_helper_is_noop_when_off(self, debug_off, log_spy):
        """Test 11: debug('x') with DEBUG=False → zero log calls."""
        memory_daemon_core.debug("noop-probe")
        matching = [m for m in log_spy if "noop-probe" in str(m)]
        assert matching == [], (
            f"Expected zero log calls with DEBUG off, got {matching}"
        )

    def test_debug_helper_calls_log_when_on(self, debug_on, log_spy):
        """Test 12: debug('x') with DEBUG=True → exactly one log call containing 'x'."""
        memory_daemon_core.debug("emit-probe")
        matching = [m for m in log_spy if "emit-probe" in str(m)]
        assert len(matching) >= 1, (
            f"Expected ≥1 log call containing 'emit-probe' with DEBUG=True, "
            f"got {log_spy}"
        )

    def test_debug_helper_swallows_exceptions_from_log(
        self, debug_on, monkeypatch
    ):
        """Test 13: if log() raises, debug() does not propagate.

        Learning #5 / R8: broad exception handling in log paths is
        intentional. This test asserts the defensive swallow.
        """
        def _raising_log(msg):
            raise RuntimeError("simulated log failure")

        monkeypatch.setattr(memory_daemon, "log", _raising_log, raising=True)
        monkeypatch.setattr(
            memory_daemon_core, "log", _raising_log, raising=False
        )
        # Must not raise.
        try:
            memory_daemon_core.debug("swallow-test")
        except RuntimeError as exc:
            pytest.fail(
                f"debug() propagated exception from log(): {exc!r}. "
                f"Expected defensive swallow per Learning #5."
            )


# ---------------------------------------------------------------------------
# §4.4 Eager-evaluation safety at call sites (tests 14–15)
# LOAD-BEARING — this is the PR #106 regression guard. See R1.
# ---------------------------------------------------------------------------


class TestEagerEvaluationSafety:
    """§4.4 — f-string arguments must not be evaluated when DEBUG is off.

    Per R1, the trap is set on an ATTRIBUTE of the mock (not just
    `__repr__`). This proves the gate prevents attribute resolution.
    """

    def test_call_site_does_not_eagerly_eval_when_debug_off(
        self, debug_off, log_spy, mock_with_eager_trap
    ):
        """Test 14: DEBUG=False → debug() with a trap-expression does not fire.

        We pass a string built by calling `.trapped_attr` on the mock,
        but ONLY inside the debug() call site. If the debug() helper
        short-circuits (gates) BEFORE evaluating its argument, this
        test passes. If it eagerly evaluates, the PropertyMock raises.

        Implementation note: the debug helper must accept a format
        string + positional args OR a callable producing the message.
        The test uses the callable form to force laziness; if the
        helper only accepts strings, the argument is pre-computed by
        the caller and the gate cannot protect. That is a design flag
        for kraken — see plan §6 and C5.
        """
        # Build a thunk that would raise if evaluated.
        def _thunk():
            return f"trap={mock_with_eager_trap.trapped_attr}"

        # Try the callable form first. If debug() does not accept a
        # callable, fall back to calling the helper at the gate — that
        # is, check a module-level gate BEFORE calling debug() at all.
        # Either way, with DEBUG off, no exception should escape.
        try:
            # Preferred API: debug accepts a callable. Kraken may choose.
            memory_daemon_core.debug(_thunk)
        except TypeError:
            # Fallback API: debug only accepts strings. In that case the
            # caller (not the helper) is responsible for guarding. The
            # module-level `DEBUG` attribute must exist so callers can
            # short-circuit.
            assert hasattr(memory_daemon_core, "DEBUG"), (
                "debug() rejects callables AND module has no DEBUG "
                "constant for call-site guarding — no way to prevent "
                "eager eval. This fails the C5 contract."
            )
            if memory_daemon_core.DEBUG:
                pytest.fail(
                    "DEBUG should be False with env unset, but module "
                    "reports True"
                )
            # Gate path: the test itself guards, mirroring what a well-
            # written call site would do.
            if memory_daemon_core.DEBUG:
                memory_daemon_core.debug(_thunk())
        except RuntimeError as exc:
            if "eager-eval leak" in str(exc):
                pytest.fail(
                    "EAGER-EVAL LEAK: debug() evaluated its argument "
                    "even though DEBUG is off. This is the PR #106 "
                    "regression."
                )
            raise

        # Assert the trap was never tripped — no message containing
        # 'trap=' should appear in the log spy.
        leaked = [m for m in log_spy if "trap=" in str(m)]
        assert leaked == [], (
            f"Unexpected debug output with DEBUG off: {leaked}"
        )

    def test_call_site_does_eagerly_eval_when_debug_on(
        self, debug_on, log_spy, mock_with_eager_trap
    ):
        """Test 15: DEBUG=True → trap IS evaluated (proves gate is the protection).

        This is the complement of test 14. If DEBUG=True and the debug
        helper is called with a thunk that references the trap, the
        PropertyMock must raise. That proves the DEBUG=False case's
        protection is the gate, not the helper swallowing all errors.
        """
        def _thunk():
            return f"trap={mock_with_eager_trap.trapped_attr}"

        # With DEBUG on, evaluating the thunk should raise RuntimeError.
        # The helper may propagate OR swallow (per test 13). We assert
        # that the attribute access WAS attempted — meaning the
        # PropertyMock.side_effect was triggered. If the mock was never
        # accessed (no call to .trapped_attr), the gate is broken in
        # the OTHER direction: DEBUG=True doesn't actually emit.
        try:
            memory_daemon_core.debug(_thunk)
        except TypeError:
            # Fallback API path: caller evaluates the string.
            if getattr(memory_daemon_core, "DEBUG", False):
                with pytest.raises(RuntimeError, match="eager-eval leak"):
                    memory_daemon_core.debug(_thunk())
                return
            else:
                pytest.fail(
                    "DEBUG should be True with env=1, but module reports "
                    "False"
                )
        except RuntimeError as exc:
            assert "eager-eval leak" in str(exc), (
                f"Unexpected RuntimeError: {exc!r}"
            )
            return  # Expected path: trap was evaluated and raised.

        # If we reach here, the call returned normally. That means
        # either (a) debug() swallowed the RuntimeError — acceptable
        # per test 13 — or (b) debug() never touched the thunk.
        # Inspect the PropertyMock call count via its recorded access.
        try:
            # Touching trapped_attr here will raise, confirming the
            # trap is still armed. But what we care about is whether
            # debug() touched it during the call above. We can't
            # directly observe that without recording — so assert the
            # SWALLOW path is the reason we got here.
            _ = mock_with_eager_trap.trapped_attr
        except RuntimeError:
            # Trap still armed as expected.
            pass
        # If debug() silently did nothing with DEBUG on, the log spy
        # would be empty AND no exception escaped. That is a failure —
        # DEBUG=True must either emit (and potentially raise from the
        # thunk) or at least attempt the emit.
        # We accept an empty spy ONLY if the trap was actually invoked
        # (which would have raised and been swallowed per test 13).
        # There is no way to distinguish "swallowed after eval" from
        # "never called" without instrumentation, so we require the
        # implementation to emit SOMETHING visible: either the log
        # message (if thunk didn't raise) or raising. The simplest
        # check: the spy is non-empty OR a RuntimeError escaped
        # (handled above).
        assert log_spy, (
            "DEBUG=True but debug() with a thunk produced neither a "
            "log message nor a raised exception — gate is inert"
        )


# ---------------------------------------------------------------------------
# §4.5 build_extraction_command argv logging (tests 16–19)
# ---------------------------------------------------------------------------


class TestBuildExtractionCommandLogging:
    """§4.5 — argv dump when DEBUG on.

    R6 reminder: this test asserts paths ARE PRESENT in the log
    (diagnostic value). Do NOT copy the "absent" assertion from the
    env-value test below.
    """

    def test_build_extraction_command_logs_argv_when_debug_on(
        self, debug_on, log_spy
    ):
        """Test 16: argv log contains model, session_id, jsonl_path."""
        memory_daemon_core.build_extraction_command(
            session_id="sess-x",
            jsonl_path="/tmp/x.jsonl",
            agent_prompt="prompt",
            model="sonnet",
            max_turns=15,
        )
        joined = " ".join(str(m) for m in log_spy)
        assert "sonnet" in joined, (
            f"Expected 'sonnet' in debug log with DEBUG=on, got: {log_spy}"
        )
        assert "sess-x" in joined, (
            f"Expected 'sess-x' in debug log with DEBUG=on, got: {log_spy}"
        )
        assert "/tmp/x.jsonl" in joined, (
            f"Expected '/tmp/x.jsonl' in debug log with DEBUG=on, "
            f"got: {log_spy}"
        )

    def test_build_extraction_command_silent_when_debug_off(
        self, debug_off, log_spy
    ):
        """Test 17: DEBUG off → build_extraction_command emits zero log calls."""
        # RED-phase guard: without this, test passes trivially because
        # build_extraction_command has no log calls at all today.
        _require_debug_feature()
        memory_daemon_core.build_extraction_command(
            session_id="sess-y",
            jsonl_path="/tmp/y.jsonl",
            agent_prompt="prompt",
            model="haiku",
            max_turns=10,
        )
        core_origin = [m for m in log_spy if "sess-y" in str(m)]
        assert core_origin == [], (
            f"Expected zero log calls from build_extraction_command with "
            f"DEBUG off, got: {core_origin}"
        )

    def test_build_extraction_command_return_value_unchanged_on(
        self, monkeypatch, log_spy
    ):
        """Test 18: return value is byte-equal DEBUG on vs off."""
        # RED-phase guard: return-value-stable holds trivially when the
        # function ignores DEBUG. The test must fail until DEBUG is
        # actually wired in.
        _require_debug_feature()
        kwargs = dict(
            session_id="sess-z",
            jsonl_path="/tmp/z.jsonl",
            agent_prompt="prompt-z",
            model="opus",
            max_turns=20,
        )
        monkeypatch.delenv("MEMORY_DAEMON_DEBUG", raising=False)
        monkeypatch.setattr(memory_daemon, "DEBUG", False, raising=False)
        off_result = memory_daemon_core.build_extraction_command(**kwargs)

        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", "1")
        monkeypatch.setattr(memory_daemon, "DEBUG", True, raising=False)
        on_result = memory_daemon_core.build_extraction_command(**kwargs)

        assert off_result == on_result, (
            f"Return value must not depend on DEBUG state. "
            f"off={off_result} on={on_result}"
        )

    def test_build_extraction_command_argv_log_is_single_message(
        self, debug_on, log_spy
    ):
        """Test 19: argv is dumped as ONE message, not fragmented per arg."""
        memory_daemon_core.build_extraction_command(
            session_id="sess-single",
            jsonl_path="/tmp/single.jsonl",
            agent_prompt="p",
            model="sonnet",
            max_turns=5,
        )
        # Find messages mentioning the session id.
        mentions = [m for m in log_spy if "sess-single" in str(m)]
        # There should be exactly ONE message containing the session
        # id — fragmentation would split the argv into multiple logs.
        assert len(mentions) == 1, (
            f"Expected single argv log message, got {len(mentions)}: "
            f"{mentions}"
        )

    def test_build_extraction_command_does_not_log_prompt_body(
        self, debug_on, log_spy
    ):
        """Test 19b (Codex Round 2 #3): agent_prompt body must NEVER appear in log.

        SECURITY/NOISE contract: the argv includes the full
        memory-extractor system prompt (loaded from
        CLAUDE_CONFIG_DIR/agents/memory-extractor.md at runtime). A
        naive `argv: {cmd}` dump leaks multi-KB of operator-specific
        prompt content into the daemon log on every extraction start,
        burying the useful triage fields.

        Codex recommendation: log only structured diagnostic fields
        (session_id, model, max_turns, jsonl_path, prompt_len). This
        test enforces that the prompt BODY — identified by a tripwire
        substring — is absent from the log, while the triage fields
        ARE present.
        """
        tripwire = "SYSTEM PROMPT BODY — DO NOT LOG zzz-PROMPT-TRIPWIRE-003"
        memory_daemon_core.build_extraction_command(
            session_id="sess-test",
            jsonl_path="/tmp/fake.jsonl",
            agent_prompt=tripwire,
            model="sonnet",
            max_turns=15,
        )
        joined = " ".join(str(m) for m in log_spy)

        # Triage fields MUST be present for the log to be useful.
        assert "sess-test" in joined, (
            f"Expected 'sess-test' (session_id) in structured debug "
            f"log, got: {log_spy}"
        )
        assert "sonnet" in joined, (
            f"Expected 'sonnet' (model) in structured debug log, "
            f"got: {log_spy}"
        )
        assert "/tmp/fake.jsonl" in joined, (
            f"Expected '/tmp/fake.jsonl' (jsonl_path) in structured "
            f"debug log, got: {log_spy}"
        )
        assert "max_turns=15" in joined, (
            f"Expected 'max_turns=15' in structured debug log, "
            f"got: {log_spy}"
        )

        # The prompt BODY must NOT appear in the log, in whole or part.
        # Defense in depth: check both the full tripwire and a
        # distinctive substring so any format that echoes any part of
        # the prompt fails the test.
        assert "zzz-PROMPT-TRIPWIRE-003" not in joined, (
            f"NOISE/LEAK VIOLATION: agent_prompt body tripwire "
            f"'zzz-PROMPT-TRIPWIRE-003' leaked into debug log. The "
            f"argv dump must log prompt metadata (e.g., prompt_len) "
            f"only, not the prompt body. Log: {log_spy}"
        )
        assert "SYSTEM PROMPT BODY" not in joined, (
            f"NOISE/LEAK VIOLATION: agent_prompt body substring "
            f"'SYSTEM PROMPT BODY' leaked into debug log. Even a "
            f"partial echo is forbidden — log structured metadata "
            f"only. Log: {log_spy}"
        )


# ---------------------------------------------------------------------------
# §4.6 build_extraction_env key-only logging (tests 20–25)
# SECURITY: values must never appear in logs. R5 — use tripwire strings.
# ---------------------------------------------------------------------------


class TestBuildExtractionEnvLogging:
    """§4.6 — env-key logging with value redaction.

    SECURITY CONTRACT (C8): keys are diagnostic signal, values are
    secrets. The tests use TRIPWIRE string values per R5 to avoid
    false positives on common path substrings.
    """

    # Tripwire values that are very unlikely to collide with any
    # legitimate log content.
    TRIPWIRE_SECRET = "zAbC123SECRETxyz"
    TRIPWIRE_PATH = "zzz-TRIPWIRE-001-path"
    TRIPWIRE_PROJECT = "yyy-TRIPWIRE-002-project"

    def _make_base_env(self):
        return {
            "AWS_SECRET_ACCESS_KEY": self.TRIPWIRE_SECRET,
            "PATH": self.TRIPWIRE_PATH,
        }

    def test_build_extraction_env_logs_daemon_owned_keys_only_when_debug_on(
        self, debug_on, log_spy
    ):
        """Test 20: DEBUG on → log exposes daemon-owned keys only.

        Codex Round 3 HIGH finding: the previous implementation dumped
        ``sorted(env.keys())`` which enumerated ALL parent-process env
        var names — reconnaissance value even with values redacted.
        The tightened format reveals only:
        - ``CLAUDE_MEMORY_EXTRACTION=1`` (daemon-set constant)
        - ``CLAUDE_PROJECT_DIR=set``/``unset`` (presence, never value)
        - ``env_var_count=N`` (sanity signal)
        Parent-env key names like ``AWS_SECRET_ACCESS_KEY`` or ``PATH``
        must NEVER appear in the log. This is the security invariant.
        """
        memory_daemon_core.build_extraction_env(
            self._make_base_env(), self.TRIPWIRE_PROJECT
        )
        joined = " ".join(str(m) for m in log_spy)
        # Security invariant: parent-env keys must NOT leak.
        assert "AWS_SECRET_ACCESS_KEY" not in joined, (
            f"SECURITY VIOLATION: parent-env key 'AWS_SECRET_ACCESS_KEY' "
            f"leaked into log: {log_spy}"
        )
        assert "PATH" not in joined, (
            f"SECURITY VIOLATION: parent-env key 'PATH' leaked into log: "
            f"{log_spy}"
        )
        # Daemon-owned signals must be present.
        assert "CLAUDE_MEMORY_EXTRACTION" in joined, (
            f"Expected 'CLAUDE_MEMORY_EXTRACTION' in log, got: {log_spy}"
        )
        assert "CLAUDE_PROJECT_DIR" in joined, (
            f"Expected 'CLAUDE_PROJECT_DIR' in log, got: {log_spy}"
        )
        assert "env_var_count=" in joined, (
            f"Expected 'env_var_count=' sanity signal in log, got: {log_spy}"
        )

    def test_build_extraction_env_does_not_leak_parent_env_keys(
        self, debug_on, log_spy
    ):
        """Test 20b: tripwire guard against re-introducing enumeration.

        Load-bearing regression guard: if a future refactor re-adds
        ``sorted(env.keys())`` or ``repr(env)`` to the debug log, this
        unique tripwire key name will surface in the log and fail the
        test. The tripwire string is deliberately unique so it cannot
        collide with legitimate log content.
        """
        tripwire_key = "OPC_KEY_TRIPWIRE_FIND_ME"
        base_env = dict(self._make_base_env())
        base_env[tripwire_key] = "value-doesnt-matter"
        memory_daemon_core.build_extraction_env(
            base_env, self.TRIPWIRE_PROJECT
        )
        joined = " ".join(str(m) for m in log_spy)
        assert tripwire_key not in joined, (
            f"REGRESSION: tripwire parent-env key {tripwire_key!r} "
            f"leaked into log — build_extraction_env is enumerating "
            f"env.keys() again. Got: {log_spy}"
        )

    def test_build_extraction_env_does_not_log_values(
        self, debug_on, log_spy
    ):
        """Test 21: DEBUG on → log does NOT contain value substrings.

        R5: assert TRIPWIRE strings are absent. Do NOT grep for common
        paths like '/usr/bin' — those are too ambiguous.
        """
        # RED-phase guard: "values are absent" holds trivially when
        # there is no logging. The test must fail until the feature
        # exists so we prove DEBUG=True emits SOMETHING and that
        # something does not include tripwires.
        _require_debug_feature()
        memory_daemon_core.build_extraction_env(
            self._make_base_env(), self.TRIPWIRE_PROJECT
        )
        joined = " ".join(str(m) for m in log_spy)
        # Additional fail-loud: with DEBUG=on AND the feature landed,
        # there should be at least ONE log message. Without this, a
        # broken implementation that emits nothing would pass this
        # test trivially.
        assert log_spy, (
            "Expected DEBUG=on to emit at least one log message from "
            f"build_extraction_env, got empty spy: {log_spy}"
        )
        assert self.TRIPWIRE_SECRET not in joined, (
            f"SECURITY VIOLATION: secret value {self.TRIPWIRE_SECRET!r} "
            f"leaked into log: {log_spy}"
        )
        assert self.TRIPWIRE_PATH not in joined, (
            f"SECURITY VIOLATION: PATH value {self.TRIPWIRE_PATH!r} "
            f"leaked into log: {log_spy}"
        )

    def test_build_extraction_env_logs_claude_project_dir_presence_only(
        self, debug_on, log_spy
    ):
        """Test 22: CLAUDE_PROJECT_DIR reported as presence, never value.

        Under the tightened format, CLAUDE_PROJECT_DIR is logged as
        ``CLAUDE_PROJECT_DIR=set`` when a project_dir is passed (and
        ``CLAUDE_PROJECT_DIR=unset`` when it is None). The actual
        directory path must NEVER appear in the log.
        """
        memory_daemon_core.build_extraction_env(
            self._make_base_env(), self.TRIPWIRE_PROJECT
        )
        joined = " ".join(str(m) for m in log_spy)
        assert "CLAUDE_PROJECT_DIR=set" in joined, (
            f"Expected 'CLAUDE_PROJECT_DIR=set' in log, got: {log_spy}"
        )
        assert self.TRIPWIRE_PROJECT not in joined, (
            f"SECURITY VIOLATION: CLAUDE_PROJECT_DIR value "
            f"{self.TRIPWIRE_PROJECT!r} leaked into log: {log_spy}"
        )

    def test_build_extraction_env_silent_when_debug_off(
        self, debug_off, log_spy
    ):
        """Test 23: DEBUG off → zero log calls from build_extraction_env."""
        # RED-phase guard: trivially silent today.
        _require_debug_feature()
        memory_daemon_core.build_extraction_env(
            self._make_base_env(), self.TRIPWIRE_PROJECT
        )
        # Filter to messages likely originating from this function.
        env_related = [
            m
            for m in log_spy
            if "CLAUDE_MEMORY_EXTRACTION" in str(m)
            or "AWS_SECRET_ACCESS_KEY" in str(m)
        ]
        assert env_related == [], (
            f"Expected zero env-related log calls with DEBUG off, "
            f"got: {env_related}"
        )

    def test_build_extraction_env_return_value_unchanged_on(
        self, monkeypatch, log_spy
    ):
        """Test 24: return dict equal under DEBUG on vs off."""
        # RED-phase guard: return equality trivially holds when DEBUG
        # is ignored. Require the feature to exist so the equality
        # test is meaningful.
        _require_debug_feature()
        base = self._make_base_env()

        monkeypatch.delenv("MEMORY_DAEMON_DEBUG", raising=False)
        monkeypatch.setattr(memory_daemon, "DEBUG", False, raising=False)
        off_result = memory_daemon_core.build_extraction_env(
            base, self.TRIPWIRE_PROJECT
        )

        monkeypatch.setenv("MEMORY_DAEMON_DEBUG", "1")
        monkeypatch.setattr(memory_daemon, "DEBUG", True, raising=False)
        on_result = memory_daemon_core.build_extraction_env(
            base, self.TRIPWIRE_PROJECT
        )

        assert off_result == on_result, (
            f"Return dict must not depend on DEBUG state. "
            f"off={off_result} on={on_result}"
        )

    def test_build_extraction_env_does_not_mutate_base_env_even_with_logging(
        self, debug_on, log_spy
    ):
        """Test 25: base_env is not mutated even when logging is active."""
        # RED-phase guard: non-mutation holds today because there is
        # no logging. Require the feature so we actually prove the
        # invariant holds WITH logging active.
        _require_debug_feature()
        base = self._make_base_env()
        snapshot = dict(base)
        memory_daemon_core.build_extraction_env(base, self.TRIPWIRE_PROJECT)
        assert base == snapshot, (
            f"build_extraction_env mutated base_env. "
            f"before={snapshot} after={base}"
        )

    # ------------------------------------------------------------------
    # PR #110 Cycle 1 M1 — inherited CLAUDE_PROJECT_DIR must not leak
    # ------------------------------------------------------------------
    #
    # CodeRabbit MAJOR (outside-diff) finding. Production call chain:
    #   memory_daemon.queue_or_extract(s.id, s.project or "", ...)
    #   → extract_memories → _extract_memories_impl
    #   → build_extraction_env(os.environ, project_dir)
    # When a stale session's project is None, the caller passes ""
    # (empty string, falsy). The old ``if project_dir:`` branch was
    # skipped, so any ``CLAUDE_PROJECT_DIR`` already present in the
    # daemon's ``os.environ`` (inherited from the launching shell)
    # silently propagated to the child extraction subprocess. The
    # DEBUG log ALSO lied — it reported ``CLAUDE_PROJECT_DIR=unset``
    # while the returned env dict actually contained a stale value.
    #
    # These three tests lock in the post-fix contract:
    #   1. An inherited key is removed when caller supplies no project.
    #   2. An inherited value does NOT override the caller's explicit
    #      project_dir (existing behavior — regression guard).
    #   3. The DEBUG log presence-marker matches the env dict after
    #      the fix (no more log/reality mismatch).

    def test_build_extraction_env_does_not_inherit_parent_claude_project_dir(
        self,
    ):
        """Test 25a (M1): inherited CLAUDE_PROJECT_DIR is dropped on None.

        Arrangement: base_env has a stale ``CLAUDE_PROJECT_DIR`` from
        the parent process environment. Caller passes ``project_dir=None``
        (production equivalent: stale session with no project).

        Expected (post-fix): the returned env dict does NOT contain
        ``CLAUDE_PROJECT_DIR`` at all. The stale parent value is
        discarded so the extraction subprocess does not silently
        inherit it.

        Before the fix, the key remained in the returned dict because
        ``env = dict(base_env)`` copied it and the ``if project_dir:``
        branch was skipped.
        """
        base_env = {"CLAUDE_PROJECT_DIR": "/parent-leaked"}
        result_env = memory_daemon_core.build_extraction_env(
            base_env, None
        )
        assert "CLAUDE_PROJECT_DIR" not in result_env, (
            f"CLAUDE_PROJECT_DIR from parent env leaked into child env "
            f"even though caller passed project_dir=None. "
            f"Got result_env['CLAUDE_PROJECT_DIR']="
            f"{result_env.get('CLAUDE_PROJECT_DIR')!r}"
        )

    def test_build_extraction_env_caller_project_dir_wins_over_inherited(
        self,
    ):
        """Test 25b (M1): explicit project_dir overrides inherited value.

        Regression guard for the existing behavior: when the caller
        passes an explicit project_dir AND the parent env also has
        one, the caller's value must win. This test should pass both
        before and after the fix — it is included to prevent the
        M1 fix from accidentally swinging the pendulum the other way.
        """
        base_env = {"CLAUDE_PROJECT_DIR": "/parent-leaked"}
        result_env = memory_daemon_core.build_extraction_env(
            base_env, "/explicit"
        )
        assert result_env["CLAUDE_PROJECT_DIR"] == "/explicit", (
            f"Caller-supplied project_dir='/explicit' should win over "
            f"inherited '/parent-leaked'. "
            f"Got: {result_env.get('CLAUDE_PROJECT_DIR')!r}"
        )

    def test_build_extraction_env_log_matches_env_when_parent_had_project_dir(
        self, debug_on, log_spy
    ):
        """Test 25c (M1): log and returned env agree when parent leaked.

        Before the fix, two failures occurred simultaneously:
          (a) result_env contained the parent's stale value, AND
          (b) the DEBUG log said ``CLAUDE_PROJECT_DIR=unset``.
        The log was lying about the subprocess environment.

        After the fix, both the log AND the dict agree: neither
        contains CLAUDE_PROJECT_DIR when the caller passes None, even
        if base_env had one inherited. This test asserts both halves
        of the contract in one place.
        """
        base_env = {"CLAUDE_PROJECT_DIR": "/parent-leaked"}
        result_env = memory_daemon_core.build_extraction_env(
            base_env, None
        )

        # Contract half 1: the returned dict is consistent with "unset".
        assert "CLAUDE_PROJECT_DIR" not in result_env, (
            f"result_env still contains stale parent CLAUDE_PROJECT_DIR: "
            f"{result_env.get('CLAUDE_PROJECT_DIR')!r}"
        )

        # Contract half 2: the DEBUG log agrees — it reports "unset"
        # AND it is not lying about what the subprocess will see.
        joined = " ".join(str(m) for m in log_spy)
        assert "CLAUDE_PROJECT_DIR=unset" in joined, (
            f"Expected 'CLAUDE_PROJECT_DIR=unset' in DEBUG log when "
            f"caller passes project_dir=None, got: {log_spy}"
        )
        # And the log must NOT have leaked the stale parent value.
        assert "/parent-leaked" not in joined, (
            f"SECURITY VIOLATION: stale parent CLAUDE_PROJECT_DIR value "
            f"'/parent-leaked' leaked into DEBUG log: {log_spy}"
        )


# ---------------------------------------------------------------------------
# §4.7 Structural audit — no silent pass-excepts (test 26)
# ---------------------------------------------------------------------------


class TestStructuralAudit:
    """§4.7 — AST walk to verify no silent exception handlers.

    R8: broad excepts in LOG PATHS are intentional. This test
    specifically looks for `except Exception: pass` (bare swallow)
    which is the anti-pattern. Log paths that catch broadly and then
    DO something (log a short summary, etc.) are acceptable and not
    flagged.
    """

    def test_module_has_no_silent_pass_excepts(self, core_source_ast):
        """Test 26: zero ExceptHandler nodes whose body is just `pass`."""
        silent_handlers: list[int] = []
        for node in ast.walk(core_source_ast):
            if not isinstance(node, ast.ExceptHandler):
                continue
            # Check if the body is a single `pass` statement, or a
            # single Constant/Ellipsis — both are silent swallows.
            body = node.body
            if len(body) == 1:
                stmt = body[0]
                if isinstance(stmt, ast.Pass):
                    silent_handlers.append(node.lineno)
                elif isinstance(stmt, ast.Expr) and isinstance(
                    stmt.value, ast.Constant
                ):
                    # `except: ...` or `except: "docstring"` — silent.
                    silent_handlers.append(node.lineno)
        assert silent_handlers == [], (
            f"Found silent pass-except handlers at lines "
            f"{silent_handlers}. Issue #96 must not introduce silent "
            f"swallows."
        )
        # ALSO assert: the module has AT LEAST ONE reference to
        # `debug(` or `if DEBUG` — proving the implementation landed.
        # This makes the structural test fail-loud in RED phase even
        # though the AST walk above returns clean for the
        # pre-implementation source.
        source = CORE_MODULE_PATH.read_text()
        has_debug_call = "debug(" in source
        has_debug_gate = "DEBUG" in source
        assert has_debug_call or has_debug_gate, (
            "memory_daemon_core.py has no reference to `debug(` or "
            "`DEBUG` — Issue #96 implementation has not landed"
        )
