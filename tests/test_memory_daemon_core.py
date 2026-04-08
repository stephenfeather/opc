"""Tests for memory_daemon_core — pure functions and constants.

All functions in memory_daemon_core.py are side-effect-free:
they take data in and return new data out without I/O, mutation,
or global state. I/O boundaries (like _is_process_alive) are
injected as predicates.
"""

from __future__ import annotations

import pytest

from scripts.core.memory_daemon_core import (
    StaleSession,
    _ALLOWED_EXTRACTION_MODELS,
    _normalize_project,
    build_extraction_command,
    build_extraction_env,
    build_s3_key,
    build_zst_path,
    filter_truly_stale_sessions,
    strip_yaml_frontmatter,
    validate_extraction_model,
)


# ── StaleSession NamedTuple ────────────────────────────────────────


class TestStaleSession:
    """StaleSession supports both named attribute and positional access."""

    def test_named_access(self):
        s = StaleSession(
            id="sess-1",
            project="myproj",
            transcript_path="/tmp/t.jsonl",
            pid=1234,
            exited_at=None,
        )
        assert s.id == "sess-1"
        assert s.project == "myproj"
        assert s.transcript_path == "/tmp/t.jsonl"
        assert s.pid == 1234
        assert s.exited_at is None

    def test_positional_access(self):
        """Existing daemon_loop code uses row[0], row[3], row[4]."""
        s = StaleSession("sess-2", "proj", "/tmp/x.jsonl", 5678, None)
        assert s[0] == "sess-2"
        assert s[3] == 5678
        assert s[4] is None

    def test_tuple_unpacking(self):
        """Existing code does `for sid, proj, *_ in truly_stale`."""
        s = StaleSession("sess-3", "proj", None, None, None)
        sid, proj, *rest = s
        assert sid == "sess-3"
        assert proj == "proj"
        assert len(rest) == 3


# ── _ALLOWED_EXTRACTION_MODELS ─────────────────────────────────────


class TestAllowedExtractionModels:
    def test_is_frozenset(self):
        assert isinstance(_ALLOWED_EXTRACTION_MODELS, frozenset)

    def test_contains_expected_models(self):
        assert _ALLOWED_EXTRACTION_MODELS == {"sonnet", "haiku", "opus"}

    def test_immutable(self):
        with pytest.raises(AttributeError):
            _ALLOWED_EXTRACTION_MODELS.add("gpt-evil")  # type: ignore[attr-defined]


# ── _normalize_project ─────────────────────────────────────────────


class TestNormalizeProject:
    def test_none_returns_none(self):
        assert _normalize_project(None) is None

    def test_empty_string_returns_none(self):
        assert _normalize_project("") is None

    def test_normal_path_returns_leaf(self):
        assert _normalize_project("/Users/dev/myproject") == "myproject"

    def test_worktree_path_returns_parent_of_worktrees(self):
        assert _normalize_project(
            "/Users/dev/myproject/.worktrees/refactor/feat-x"
        ) == "myproject"

    def test_worktree_at_root_returns_name(self):
        """Edge case: .worktrees at index 0 in parts."""
        result = _normalize_project("/.worktrees/something")
        # idx=1 (.worktrees), idx-1=0 ("/"), parts[0]="/"
        # p.name would be "something"
        assert result is not None

    def test_single_component_path(self):
        assert _normalize_project("myproject") == "myproject"


# ── validate_extraction_model ──────────────────────────────────────


class TestValidateExtractionModel:
    def test_valid_model(self):
        allowed = frozenset({"sonnet", "haiku", "opus"})
        assert validate_extraction_model("sonnet", allowed) is True

    def test_invalid_model(self):
        allowed = frozenset({"sonnet", "haiku", "opus"})
        assert validate_extraction_model("gpt-evil", allowed) is False

    def test_empty_string(self):
        allowed = frozenset({"sonnet", "haiku", "opus"})
        assert validate_extraction_model("", allowed) is False

    def test_empty_allowlist(self):
        assert validate_extraction_model("sonnet", frozenset()) is False


# ── build_extraction_command ───────────────────────────────────────


class TestBuildExtractionCommand:
    def test_contains_model_flag(self):
        cmd = build_extraction_command(
            "sess-1", "/tmp/s.jsonl", "Extract learnings", "sonnet", 15
        )
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "sonnet"

    def test_contains_allowed_tools(self):
        cmd = build_extraction_command(
            "sess-1", "/tmp/s.jsonl", "Extract learnings", "sonnet", 15
        )
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Bash,Read"

    def test_contains_dangerously_skip_permissions(self):
        cmd = build_extraction_command(
            "sess-1", "/tmp/s.jsonl", "Extract learnings", "sonnet", 15
        )
        assert "--dangerously-skip-permissions" in cmd

    def test_contains_max_turns(self):
        cmd = build_extraction_command(
            "sess-1", "/tmp/s.jsonl", "Extract learnings", "opus", 25
        )
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "25"

    def test_contains_session_id_and_jsonl_in_prompt(self):
        cmd = build_extraction_command(
            "sess-42", "/tmp/s.jsonl", "Extract learnings", "sonnet", 15
        )
        last_arg = cmd[-1]
        assert "sess-42" in last_arg
        assert "/tmp/s.jsonl" in last_arg

    def test_starts_with_claude(self):
        cmd = build_extraction_command(
            "sess-1", "/tmp/s.jsonl", "prompt", "sonnet", 15
        )
        assert cmd[0] == "claude"
        assert cmd[1] == "-p"


# ── build_extraction_env ───────────────────────────────────────────


class TestBuildExtractionEnv:
    def test_sets_memory_extraction_flag(self):
        env = build_extraction_env({}, "/tmp/proj")
        assert env["CLAUDE_MEMORY_EXTRACTION"] == "1"

    def test_sets_project_dir_when_provided(self):
        env = build_extraction_env({}, "/tmp/proj")
        assert env["CLAUDE_PROJECT_DIR"] == "/tmp/proj"

    def test_omits_project_dir_when_none(self):
        env = build_extraction_env({}, None)
        assert "CLAUDE_PROJECT_DIR" not in env

    def test_omits_project_dir_when_empty(self):
        env = build_extraction_env({}, "")
        assert "CLAUDE_PROJECT_DIR" not in env

    def test_does_not_mutate_base_env(self):
        base = {"PATH": "/usr/bin"}
        env = build_extraction_env(base, "/tmp/proj")
        assert "CLAUDE_MEMORY_EXTRACTION" not in base
        assert "CLAUDE_MEMORY_EXTRACTION" in env
        assert env["PATH"] == "/usr/bin"


# ── strip_yaml_frontmatter ────────────────────────────────────────


class TestStripYamlFrontmatter:
    def test_strips_frontmatter(self):
        content = "---\ntitle: test\n---\nBody text here"
        assert strip_yaml_frontmatter(content) == "Body text here"

    def test_no_frontmatter_passthrough(self):
        content = "Just plain content"
        assert strip_yaml_frontmatter(content) == "Just plain content"

    def test_only_opening_delimiter(self):
        content = "---\nsome stuff without closing"
        result = strip_yaml_frontmatter(content)
        assert result == content

    def test_empty_string(self):
        assert strip_yaml_frontmatter("") == ""

    def test_empty_frontmatter(self):
        content = "---\n---\nBody"
        assert strip_yaml_frontmatter(content) == "Body"


# ── build_s3_key and build_zst_path ───────────────────────────────


class TestBuildS3Key:
    def test_format(self):
        key = build_s3_key("my-bucket", "myproject", "session-abc")
        assert key == "s3://my-bucket/sessions/myproject/session-abc.jsonl.zst"


class TestBuildZstPath:
    def test_suffix(self):
        from pathlib import Path

        result = build_zst_path(Path("/tmp/sessions/s.jsonl"))
        assert result == Path("/tmp/sessions/s.jsonl.zst")


# ── filter_truly_stale_sessions ───────────────────────────────────


class TestFilterTrulyStaleSessions:
    """Uses mock predicates — no real os.kill calls."""

    def _make_session(self, sid, pid=None, exited_at=None):
        return StaleSession(
            id=sid, project="proj", transcript_path=None,
            pid=pid, exited_at=exited_at,
        )

    def test_alive_pid_goes_to_still_alive(self):
        sessions = [self._make_session("s1", pid=1234)]
        truly_stale, newly_dead, still_alive = filter_truly_stale_sessions(
            sessions, is_alive=lambda _pid: True,
        )
        assert still_alive == ["s1"]
        assert truly_stale == []
        assert newly_dead == []

    def test_dead_pid_no_exited_at_goes_to_newly_dead(self):
        sessions = [self._make_session("s1", pid=1234, exited_at=None)]
        truly_stale, newly_dead, still_alive = filter_truly_stale_sessions(
            sessions, is_alive=lambda _pid: False,
        )
        assert newly_dead == ["s1"]
        assert still_alive == []
        assert truly_stale == []

    def test_dead_pid_with_exited_at_goes_to_truly_stale(self):
        sessions = [self._make_session("s1", pid=1234, exited_at="2026-01-01")]
        truly_stale, newly_dead, still_alive = filter_truly_stale_sessions(
            sessions, is_alive=lambda _pid: False,
        )
        assert len(truly_stale) == 1
        assert truly_stale[0].id == "s1"
        assert newly_dead == []
        assert still_alive == []

    def test_none_pid_treated_as_dead(self):
        sessions = [self._make_session("s1", pid=None, exited_at="2026-01-01")]
        truly_stale, newly_dead, still_alive = filter_truly_stale_sessions(
            sessions, is_alive=lambda _pid: False,
        )
        assert len(truly_stale) == 1

    def test_mixed_sessions(self):
        sessions = [
            self._make_session("alive", pid=100),
            self._make_session("new-dead", pid=200, exited_at=None),
            self._make_session("stale", pid=300, exited_at="2026-01-01"),
        ]
        alive_pids = {100}
        truly_stale, newly_dead, still_alive = filter_truly_stale_sessions(
            sessions, is_alive=lambda pid: pid in alive_pids,
        )
        assert still_alive == ["alive"]
        assert newly_dead == ["new-dead"]
        assert [s.id for s in truly_stale] == ["stale"]

    def test_empty_input(self):
        truly_stale, newly_dead, still_alive = filter_truly_stale_sessions(
            [], is_alive=lambda _: True,
        )
        assert truly_stale == []
        assert newly_dead == []
        assert still_alive == []
