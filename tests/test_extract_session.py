"""Tests for ``scripts/core/extract_session`` (issue #128).

The CLI re-invokes the memory daemon's extraction subprocess against a single
session for testing/debugging. All collaborators are injected or stubbed:

* ``subprocess.Popen`` is replaced with a ``FakePopen`` per test.
* ``postgres_pool.get_pool`` is stubbed with a fake pool whose ``acquire``
  context yields a ``FakeConn`` returning fixed ``fetchrow`` / ``fetchval``
  values.
* ``build_extraction_env`` and ``build_extraction_command`` are imported
  from ``memory_daemon_core`` unmodified — tests assert their wiring rather
  than re-implement them.

Coverage target on ``scripts/core/extract_session`` is >= 80%.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pytest

from scripts.core import extract_session

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakePopen:
    """Minimal subprocess.Popen stand-in.

    ``stderr`` is a BytesIO that yields a fixed list of lines on iteration so
    the streaming loop terminates deterministically.
    """

    def __init__(
        self,
        cmd: list[str],
        *,
        stdout: Any = None,
        stderr: Any = None,
        env: dict | None = None,
        stderr_lines: list[bytes] | None = None,
        return_code: int = 0,
        raises_on_wait: BaseException | None = None,
    ) -> None:
        self.cmd = cmd
        self.passed_env = env
        self.passed_stdout = stdout
        self.passed_stderr = stderr
        self.pid = 4242
        self._return_code = return_code
        self._raises_on_wait = raises_on_wait
        self.stderr = io.BytesIO(b"".join(stderr_lines or []))
        self.terminated = False
        self.killed = False
        self.entered = False
        self.exited = False

    def wait(self, timeout: float | None = None) -> int:
        if self._raises_on_wait is not None:
            raise self._raises_on_wait
        return self._return_code

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    @property
    def returncode(self) -> int:
        return self._return_code

    # subprocess.Popen is a context manager; the production code now relies on
    # __enter__/__exit__ for deterministic PIPE cleanup.
    def __enter__(self) -> FakePopen:
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class FakeConn:
    def __init__(self, *, row: dict | None = None, fetchval_value: int = 0) -> None:
        self._row = row
        self._fetchval = fetchval_value
        self.last_query: str | None = None
        self.last_args: tuple = ()

    async def fetchrow(self, query: str, *args: Any) -> dict | None:
        self.last_query = query
        self.last_args = args
        return self._row

    async def fetchval(self, query: str, *args: Any) -> int:
        self.last_query = query
        self.last_args = args
        return self._fetchval


class FakePool:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    def acquire(self):  # pragma: no cover - thin wrapper
        @asynccontextmanager
        async def _cm():
            yield self._conn

        return _cm()


@pytest.fixture
def session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def good_row(session_id: str, tmp_path) -> dict:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n")
    return {
        "id": session_id,
        "project": str(tmp_path),
        "transcript_path": str(transcript),
    }


@pytest.fixture
def patch_pool(monkeypatch):
    """Helper: install a FakePool returning the supplied FakeConn."""

    def install(conn: FakeConn) -> None:
        async def _get_pool() -> FakePool:
            return FakePool(conn)

        monkeypatch.setattr(extract_session, "get_pool", _get_pool)

    return install


def _run_main_sync(argv: list[str]) -> int:
    """Invoke the async run_main inside a fresh event loop for tests."""
    return asyncio.run(extract_session.run_main(argv))


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_requires_session_id() -> None:
    with pytest.raises(SystemExit):
        extract_session.parse_args([])


def test_parse_args_defaults(session_id: str) -> None:
    ns = extract_session.parse_args(["--session-id", session_id])
    assert ns.session_id == session_id
    assert ns.dry_run is False
    assert ns.model is None
    assert ns.max_turns is None
    assert ns.timeout is None
    assert ns.verbose is False
    # The --no-mark-extracted flag was removed (was a no-op); the CLI never
    # marks sessions extracted by design.
    assert not hasattr(ns, "no_mark_extracted")


def test_parse_args_overrides(session_id: str) -> None:
    ns = extract_session.parse_args(
        [
            "--session-id", session_id,
            "--dry-run",
            "--model", "opus",
            "--max-turns", "5",
            "--timeout", "120",
            "--verbose",
        ]
    )
    assert ns.dry_run is True
    assert ns.model == "opus"
    assert ns.max_turns == 5
    assert ns.timeout == 120
    assert ns.verbose is True


def test_parse_args_no_longer_accepts_no_mark_extracted(session_id: str) -> None:
    """The no-op --no-mark-extracted flag was removed; argparse must reject."""
    with pytest.raises(SystemExit):
        extract_session.parse_args(
            ["--session-id", session_id, "--no-mark-extracted"]
        )


# ---------------------------------------------------------------------------
# validate_session_id (pure)
# ---------------------------------------------------------------------------


def test_validate_session_id_accepts_valid_uuid(session_id: str) -> None:
    assert extract_session.validate_session_id(session_id) == session_id


def test_validate_session_id_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        extract_session.validate_session_id("not-a-uuid")


def test_validate_session_id_rejects_empty() -> None:
    with pytest.raises(ValueError):
        extract_session.validate_session_id("")


# ---------------------------------------------------------------------------
# redact_env (pure)
# ---------------------------------------------------------------------------


def test_redact_env_redacts_secret_keys() -> None:
    env = {
        "PATH": "/usr/bin",
        "OPENAI_API_KEY": "sk-secret",
        "VOYAGE_API_KEY": "vk-secret",
        "DATABASE_URL": "postgres://user:pass@host/db",
        "GITHUB_TOKEN": "ghp_xxx",
        "MY_PASSWORD": "hunter2",
        "INNOCENT_VAR": "ok",
    }
    redacted = extract_session.redact_env(env)
    assert env["OPENAI_API_KEY"] == "sk-secret"  # original untouched
    assert redacted["OPENAI_API_KEY"] == "***REDACTED***"
    assert redacted["VOYAGE_API_KEY"] == "***REDACTED***"
    assert redacted["DATABASE_URL"] == "***REDACTED***"
    assert redacted["GITHUB_TOKEN"] == "***REDACTED***"
    assert redacted["MY_PASSWORD"] == "***REDACTED***"
    assert redacted["PATH"] == "/usr/bin"
    assert redacted["INNOCENT_VAR"] == "ok"


def test_redact_env_redacts_expanded_token_set() -> None:
    """The expanded redaction list (AUTH/CREDENTIAL/PRIVATE/CERT) is honored."""
    env = {
        "AUTH_HEADER": "Bearer xyz",
        "BASIC_AUTH": "user:pass",
        "GCP_CREDENTIAL_FILE": "/etc/secrets/gcp.json",
        "MY_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----",
        "CLIENT_CERT": "/etc/ssl/client.pem",
        "INNOCENT_VAR": "ok",
    }
    redacted = extract_session.redact_env(env)
    assert redacted["AUTH_HEADER"] == "***REDACTED***"
    assert redacted["BASIC_AUTH"] == "***REDACTED***"
    assert redacted["GCP_CREDENTIAL_FILE"] == "***REDACTED***"
    assert redacted["MY_PRIVATE_KEY"] == "***REDACTED***"
    assert redacted["CLIENT_CERT"] == "***REDACTED***"
    assert redacted["INNOCENT_VAR"] == "ok"


# ---------------------------------------------------------------------------
# summarize_env_for_verbose (pure)
# ---------------------------------------------------------------------------


def test_summarize_env_only_emits_allowlisted_keys() -> None:
    env = {
        "CLAUDE_PROJECT_DIR": "/proj",
        "CLAUDE_MEMORY_EXTRACTION": "1",
        "OPC_DEBUG": "1",
        "PYTHONPATH": "/x",
        "HOME": "/Users/foo",
        "PATH": "/usr/bin",
        "GITHUB_TOKEN": "ghp_xxx",
    }
    visible, hidden = extract_session.summarize_env_for_verbose(env)
    assert set(visible) == {
        "CLAUDE_PROJECT_DIR",
        "CLAUDE_MEMORY_EXTRACTION",
        "OPC_DEBUG",
        "PYTHONPATH",
    }
    # Visible values still subject to redaction (no secret tokens here).
    assert visible["CLAUDE_MEMORY_EXTRACTION"] == "1"
    assert hidden == 3  # HOME, PATH, GITHUB_TOKEN


def test_summarize_env_redacts_secrets_in_visible_keys() -> None:
    env = {
        "CLAUDE_API_KEY": "sk-xxx",  # allowlisted prefix BUT secret-bearing key
        "OPC_DEBUG": "1",
    }
    visible, _ = extract_session.summarize_env_for_verbose(env)
    assert visible["CLAUDE_API_KEY"] == "***REDACTED***"
    assert visible["OPC_DEBUG"] == "1"


# ---------------------------------------------------------------------------
# redact_long_tokens_for_dry_run (pure)
# ---------------------------------------------------------------------------


def test_redact_long_tokens_replaces_oversized_argv_tokens() -> None:
    big = "X" * 500
    cmd = ["claude", "-p", "--append-system-prompt", big, "trailer"]
    out = extract_session.redact_long_tokens_for_dry_run(cmd)
    assert out[0] == "claude"
    assert out[3] == "<token: 500 chars>"
    assert out[4] == "trailer"


def test_redact_long_tokens_leaves_short_tokens_intact() -> None:
    cmd = ["claude", "-p", "short", "another"]
    assert extract_session.redact_long_tokens_for_dry_run(cmd) == cmd


# ---------------------------------------------------------------------------
# fetch_session_row / count_session_memories (async DB helpers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_session_row_returns_dict(
    patch_pool, good_row: dict, session_id: str
) -> None:
    conn = FakeConn(row=good_row)
    patch_pool(conn)
    row = await extract_session.fetch_session_row(session_id)
    assert row == good_row
    assert conn.last_args == (session_id,)


@pytest.mark.asyncio
async def test_fetch_session_row_returns_none_when_missing(
    patch_pool, session_id: str
) -> None:
    conn = FakeConn(row=None)
    patch_pool(conn)
    assert await extract_session.fetch_session_row(session_id) is None


@pytest.mark.asyncio
async def test_count_session_memories_returns_int(
    patch_pool, session_id: str
) -> None:
    conn = FakeConn(fetchval_value=7)
    patch_pool(conn)
    assert await extract_session.count_session_memories(session_id) == 7


@pytest.mark.asyncio
async def test_count_session_memories_returns_zero_when_null(
    patch_pool, session_id: str
) -> None:
    conn = FakeConn(fetchval_value=None)  # type: ignore[arg-type]
    patch_pool(conn)
    assert await extract_session.count_session_memories(session_id) == 0


# ---------------------------------------------------------------------------
# run_main — orchestrator
# ---------------------------------------------------------------------------


def _install_orchestrator_doubles(
    monkeypatch,
    *,
    row: dict | None,
    fake_popen_kwargs: dict | None = None,
    counts: tuple[int, int] = (0, 0),
) -> dict:
    """Install fakes on extract_session for run_main tests.

    Returns a dict capturing call data. Note that ``run_main`` is async so
    callers must invoke ``_run_main_sync`` (which spins ``asyncio.run`` per
    test) — the fixture stubs the async DB helpers directly to avoid the
    asyncpg / cached-pool path entirely.
    """
    captured: dict = {
        "popen_calls": [],
        "fetchrow_calls": 0,
        "fetchval_calls": 0,
        "popens": [],
    }

    async def fake_fetch(session_id: str) -> dict | None:
        captured["fetchrow_calls"] += 1
        captured["last_session_id"] = session_id
        return row

    async def fake_count(session_id: str) -> int:
        captured["fetchval_calls"] += 1
        return counts[1] if captured["fetchval_calls"] >= 2 else counts[0]

    monkeypatch.setattr(extract_session, "fetch_session_row", fake_fetch)
    monkeypatch.setattr(extract_session, "count_session_memories", fake_count)

    def fake_popen(cmd, **kwargs):
        captured["popen_calls"].append({"cmd": cmd, **kwargs})
        proc = FakePopen(cmd, **(fake_popen_kwargs or {}), **kwargs)
        captured["popens"].append(proc)
        return proc

    monkeypatch.setattr(extract_session.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        extract_session, "load_agent_prompt", lambda: "PROMPT_BODY"
    )
    return captured


def test_run_main_dry_run_does_not_spawn(
    monkeypatch, capsys, session_id: str, good_row: dict
) -> None:
    captured = _install_orchestrator_doubles(monkeypatch, row=good_row)
    rc = _run_main_sync(["--session-id", session_id, "--dry-run", "--verbose"])
    assert rc == 0
    assert captured["popen_calls"] == []
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "claude" in out
    # --verbose summary names the CLAUDE_ allowlist prefix entries.
    assert "CLAUDE_MEMORY_EXTRACTION" in out


def test_run_main_dry_run_redacts_long_argv_tokens(
    monkeypatch, capsys, session_id: str, good_row: dict
) -> None:
    """The agent prompt argv token must be redacted in dry-run output."""
    _install_orchestrator_doubles(monkeypatch, row=good_row)
    monkeypatch.setattr(
        extract_session, "load_agent_prompt", lambda: "P" * 5000
    )
    _run_main_sync(["--session-id", session_id, "--dry-run"])
    out = capsys.readouterr().out
    assert "<token: 5000 chars>" in out
    # The raw prompt body must NOT appear.
    assert "P" * 1000 not in out


def test_run_main_missing_session_returns_1(
    monkeypatch, capsys, session_id: str
) -> None:
    _install_orchestrator_doubles(monkeypatch, row=None)
    rc = _run_main_sync(["--session-id", session_id])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no session" in err.lower() or "not found" in err.lower()


def test_run_main_missing_transcript_path(
    monkeypatch, capsys, session_id: str, tmp_path
) -> None:
    row = {
        "id": session_id,
        "project": str(tmp_path),
        "transcript_path": None,
    }
    _install_orchestrator_doubles(monkeypatch, row=row)
    rc = _run_main_sync(["--session-id", session_id])
    assert rc == 1
    err = capsys.readouterr().err
    assert "transcript_path" in err.lower()


def test_run_main_transcript_file_missing_on_disk(
    monkeypatch, capsys, session_id: str, tmp_path
) -> None:
    row = {
        "id": session_id,
        "project": str(tmp_path),
        "transcript_path": str(tmp_path / "does-not-exist.jsonl"),
    }
    _install_orchestrator_doubles(monkeypatch, row=row)
    rc = _run_main_sync(["--session-id", session_id])
    assert rc == 1
    err = capsys.readouterr().err
    assert "transcript" in err.lower()
    assert "not found" in err.lower() or "missing" in err.lower()


def test_run_main_invalid_model_returns_1(
    monkeypatch, capsys, session_id: str, good_row: dict
) -> None:
    _install_orchestrator_doubles(monkeypatch, row=good_row)
    rc = _run_main_sync(["--session-id", session_id, "--model", "gpt4"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "model" in err.lower()


def test_run_main_invalid_uuid_returns_1(monkeypatch, capsys) -> None:
    _install_orchestrator_doubles(monkeypatch, row=None)
    rc = _run_main_sync(["--session-id", "not-a-uuid"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "uuid" in err.lower() or "invalid" in err.lower()


def test_run_main_happy_path_returns_subprocess_exit_code(
    monkeypatch, capsys, session_id: str, good_row: dict
) -> None:
    captured = _install_orchestrator_doubles(
        monkeypatch,
        row=good_row,
        fake_popen_kwargs={
            "return_code": 0,
            "stderr_lines": [b"extracting...\n", b"done.\n"],
        },
        counts=(3, 5),
    )
    rc = _run_main_sync(["--session-id", session_id])
    assert rc == 0
    assert len(captured["popen_calls"]) == 1
    call = captured["popen_calls"][0]
    assert isinstance(call["cmd"], list)
    assert call["cmd"][0] == "claude"
    assert call["env"] is not None
    assert call["env"]["CLAUDE_MEMORY_EXTRACTION"] == "1"
    out = capsys.readouterr().out
    assert "delta" in out.lower() or "+2" in out or "new_memories=2" in out


def test_run_main_subprocess_nonzero_propagates(
    monkeypatch, capsys, session_id: str, good_row: dict
) -> None:
    _install_orchestrator_doubles(
        monkeypatch,
        row=good_row,
        fake_popen_kwargs={"return_code": 7},
    )
    rc = _run_main_sync(["--session-id", session_id])
    assert rc == 7


def test_run_main_timeout_kills_process(
    monkeypatch, capsys, session_id: str, good_row: dict
) -> None:
    import subprocess as real_subprocess

    captured = _install_orchestrator_doubles(
        monkeypatch,
        row=good_row,
        fake_popen_kwargs={
            "return_code": 124,
            "raises_on_wait": real_subprocess.TimeoutExpired(cmd="claude", timeout=1),
        },
    )
    rc = _run_main_sync(["--session-id", session_id, "--timeout", "1"])
    assert rc == 124
    err = capsys.readouterr().err
    assert "timeout" in err.lower() or "timed out" in err.lower()
    assert captured["popen_calls"][0]["env"] is not None


def test_run_main_model_override_propagates(
    monkeypatch, session_id: str, good_row: dict
) -> None:
    captured = _install_orchestrator_doubles(
        monkeypatch,
        row=good_row,
        fake_popen_kwargs={"return_code": 0, "stderr_lines": []},
    )
    _run_main_sync(["--session-id", session_id, "--model", "haiku"])
    cmd = captured["popen_calls"][0]["cmd"]
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "haiku"


def test_run_main_max_turns_override_propagates(
    monkeypatch, session_id: str, good_row: dict
) -> None:
    captured = _install_orchestrator_doubles(
        monkeypatch,
        row=good_row,
        fake_popen_kwargs={"return_code": 0, "stderr_lines": []},
    )
    _run_main_sync(["--session-id", session_id, "--max-turns", "3"])
    cmd = captured["popen_calls"][0]["cmd"]
    idx = cmd.index("--max-turns")
    assert cmd[idx + 1] == "3"


# ---------------------------------------------------------------------------
# load_agent_prompt
# ---------------------------------------------------------------------------


def test_load_agent_prompt_uses_file_when_present(monkeypatch, tmp_path) -> None:
    config_dir = tmp_path / ".claude"
    (config_dir / "agents").mkdir(parents=True)
    agent_file = config_dir / "agents" / "memory-extractor.md"
    agent_file.write_text("---\nmeta: x\n---\nactual prompt body\n")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    prompt = extract_session.load_agent_prompt()
    assert "actual prompt body" in prompt
    assert "meta: x" not in prompt


def test_load_agent_prompt_fallback_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    prompt = extract_session.load_agent_prompt()
    assert "Extract learnings" in prompt


def test_fallback_prompt_matches_daemon_invariant() -> None:
    """The CLI fallback must BE the daemon fallback (same import).

    Tightened per qa CHANGES_REQUESTED on review-10: the previous version
    compared ``extract_session._FALLBACK_AGENT_PROMPT`` to a hand-typed
    duplicate inside the test body, which was a tautology. Now we import
    the constant from both modules and assert object identity — the CLI
    re-exports the daemon's constant via ``from ... import``, so they must
    refer to the same string.
    """
    from scripts.core.extract_session import _FALLBACK_AGENT_PROMPT as CLI_FB
    from scripts.core.memory_daemon_extractors import (
        _FALLBACK_AGENT_PROMPT as DAEMON_FB,
    )

    assert CLI_FB is DAEMON_FB


# ---------------------------------------------------------------------------
# Aegis polish (task 9)
# ---------------------------------------------------------------------------


def test_parse_args_rejects_negative_timeout(session_id: str) -> None:
    """argparse must reject --timeout -1 with exit code 2."""
    with pytest.raises(SystemExit) as exc:
        extract_session.parse_args(["--session-id", session_id, "--timeout", "-1"])
    assert exc.value.code == 2


def test_parse_args_accepts_zero_timeout_as_no_timeout(session_id: str) -> None:
    """--timeout 0 is documented as 'no timeout' and parses successfully."""
    ns = extract_session.parse_args(["--session-id", session_id, "--timeout", "0"])
    assert ns.timeout == 0


def test_parse_args_rejects_zero_max_turns(session_id: str) -> None:
    """argparse must reject --max-turns 0 (must be >= 1)."""
    with pytest.raises(SystemExit) as exc:
        extract_session.parse_args(["--session-id", session_id, "--max-turns", "0"])
    assert exc.value.code == 2


def test_parse_args_rejects_negative_max_turns(session_id: str) -> None:
    with pytest.raises(SystemExit) as exc:
        extract_session.parse_args(
            ["--session-id", session_id, "--max-turns", "-5"]
        )
    assert exc.value.code == 2


def test_parse_args_rejects_non_integer_timeout(session_id: str) -> None:
    with pytest.raises(SystemExit) as exc:
        extract_session.parse_args(
            ["--session-id", session_id, "--timeout", "not-a-number"]
        )
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Cycle-1 regressions (PR #129)
# ---------------------------------------------------------------------------


def test_run_main_uses_popen_as_context_manager(
    monkeypatch, session_id: str, good_row: dict
) -> None:
    """The actual production Popen instance must enter and exit the with-block.

    Tightened per the MINOR finding on PR #129 cycle 1: the previous version
    of this test built a fresh FakePopen and walked the dunders manually,
    which did not prove ``run_main`` triggered the context manager on the
    instance returned by the factory.
    """
    captured = _install_orchestrator_doubles(
        monkeypatch,
        row=good_row,
        fake_popen_kwargs={"return_code": 0, "stderr_lines": [b"ok\n"]},
    )
    _run_main_sync(["--session-id", session_id])
    assert len(captured["popens"]) == 1
    proc = captured["popens"][0]
    assert proc.entered is True
    assert proc.exited is True


def test_run_main_back_to_back_calls_share_no_loop_state(
    monkeypatch, session_id: str, good_row: dict
) -> None:
    """BUG 1 regression: two calls in succession must not raise loop errors.

    The previous implementation called ``asyncio.run`` three times per
    invocation, which left the cached asyncpg pool bound to a closed loop
    on the second call. ``run_main`` is async now and the entry point uses
    a single ``asyncio.run``; in tests we use ``_run_main_sync`` which mimics
    that contract. Calling it twice in a row exercises the regression.
    """
    second_session = str(uuid.uuid4())
    rows = {good_row["id"]: good_row, second_session: {**good_row, "id": second_session}}

    async def fake_fetch(sid: str) -> dict | None:
        return rows.get(sid)

    async def fake_count(_sid: str) -> int:
        return 0

    monkeypatch.setattr(extract_session, "fetch_session_row", fake_fetch)
    monkeypatch.setattr(extract_session, "count_session_memories", fake_count)
    monkeypatch.setattr(extract_session, "load_agent_prompt", lambda: "P")
    monkeypatch.setattr(
        extract_session.subprocess,
        "Popen",
        lambda cmd, **kw: FakePopen(
            cmd, return_code=0, stderr_lines=[], **kw
        ),
    )

    rc1 = _run_main_sync(["--session-id", session_id])
    rc2 = _run_main_sync(["--session-id", second_session])
    assert rc1 == 0
    assert rc2 == 0
    # And confirm the file uses asyncio.run only ONCE (in main()).
    src = (
        __import__("inspect").getsource(extract_session)
    )
    assert src.count("asyncio.run(") == 1


def test_run_main_timeout_fires_even_when_stderr_blocks(
    monkeypatch, session_id: str, good_row: dict
) -> None:
    """BUG 2 regression: a child holding stderr open must still hit --timeout.

    Models a subprocess that emits some stderr, then blocks on stderr (the
    iterator never returns). The timeout must fire in roughly the configured
    window — a serial drain in the main thread would block proc.wait and
    miss the deadline entirely.
    """
    import subprocess as real_subprocess

    class HangingStderr:
        """Iterator that yields some lines, then blocks on the next __next__.

        Released by the parent test when the production code calls .close()
        (which subprocess.Popen.__exit__ does on the PIPE).
        """

        def __init__(self, lines: list[bytes]) -> None:
            self._lines = list(lines)
            self._released = threading.Event()
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._lines:
                return self._lines.pop(0)
            # Block until released; mirror the real "child has stderr open".
            self._released.wait(timeout=10)
            raise StopIteration

        def close(self):
            self.closed = True
            self._released.set()

    class HangingFakePopen:
        """Like FakePopen but with a stderr that blocks the streaming thread."""

        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.env = kwargs.get("env")
            self.stderr = HangingStderr([b"line one\n", b"line two\n"])
            self._wait_calls = 0
            self.killed = False
            self.entered = False
            self.exited = False

        def wait(self, timeout=None):
            self._wait_calls += 1
            if timeout is None:
                # Block forever (test would never call this without a timeout).
                time.sleep(60)
                return 0
            # Sleep up to the timeout to mirror real behavior, then raise.
            time.sleep(min(timeout, 5))
            raise real_subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout)

        def kill(self):
            self.killed = True
            self.stderr.close()

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *exc):
            self.exited = True
            self.stderr.close()

        @property
        def returncode(self):
            return 124

    captured: dict = {"popens": []}

    async def fake_fetch(_sid: str) -> dict:
        return good_row

    async def fake_count(_sid: str) -> int:
        return 0

    def fake_popen(cmd, **kw):
        proc = HangingFakePopen(cmd, **kw)
        captured["popens"].append(proc)
        return proc

    monkeypatch.setattr(extract_session, "fetch_session_row", fake_fetch)
    monkeypatch.setattr(extract_session, "count_session_memories", fake_count)
    monkeypatch.setattr(extract_session, "load_agent_prompt", lambda: "P")
    monkeypatch.setattr(extract_session.subprocess, "Popen", fake_popen)

    start = time.perf_counter()
    rc = _run_main_sync(["--session-id", session_id, "--timeout", "1"])
    elapsed = time.perf_counter() - start
    assert rc == 124
    # Comfortable upper bound: even with the in-test sleep up to min(timeout, 5)
    # the call should return inside a few seconds, not hang on stderr.
    assert elapsed < 8, f"timeout did not fire promptly (took {elapsed:.2f}s)"
    proc = captured["popens"][0]
    assert proc.killed is True
    assert proc.exited is True


def test_run_main_zero_timeout_treated_as_no_timeout(
    monkeypatch, session_id: str, good_row: dict
) -> None:
    """--timeout 0 must NOT cause TimeoutExpired (treated as 'no timeout')."""
    captured: dict = {"wait_timeouts": []}

    class RecordingFakePopen(FakePopen):
        def wait(self, timeout: float | None = None) -> int:
            captured["wait_timeouts"].append(timeout)
            return 0

    async def fake_fetch(_sid: str) -> dict:
        return good_row

    async def fake_count(_sid: str) -> int:
        return 0

    monkeypatch.setattr(extract_session, "fetch_session_row", fake_fetch)
    monkeypatch.setattr(extract_session, "count_session_memories", fake_count)
    monkeypatch.setattr(extract_session, "load_agent_prompt", lambda: "P")
    monkeypatch.setattr(
        extract_session.subprocess,
        "Popen",
        lambda cmd, **kw: RecordingFakePopen(
            cmd, stderr_lines=[], return_code=0, **kw
        ),
    )

    rc = _run_main_sync(["--session-id", session_id, "--timeout", "0"])
    assert rc == 0
    assert captured["wait_timeouts"] == [None]
