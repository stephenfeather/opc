"""Tests for scripts/deploy_hooks.sh — Issue #105.

Exercises the bash deploy script via subprocess with isolated tempdir fixtures.
Each test builds a fake `hooks/{src,dist}/` tree under a tmp_path, drops the
real deploy_hooks.sh into a parallel `scripts/` dir so OPC_ROOT resolves
correctly, and sets DEPLOY_TARGET to a separate tempdir whose basename is
`hooks` (required by the script's safety guard).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_SCRIPT = REPO_ROOT / "scripts" / "deploy_hooks.sh"


def _build_fixture(
    tmp_path: Path, *, opc_subpath: str = "opc"
) -> tuple[Path, Path, Path]:
    """Create a fake OPC root with scripts/deploy_hooks.sh and hooks/{src,dist}/.

    Returns (opc_root, script_path, target_root).
    """
    opc_root = tmp_path / opc_subpath
    scripts_dir = opc_root / "scripts"
    hooks_src = opc_root / "hooks" / "src"
    hooks_dist = opc_root / "hooks" / "dist"
    scripts_dir.mkdir(parents=True)
    hooks_src.mkdir(parents=True)
    hooks_dist.mkdir(parents=True)

    script_path = scripts_dir / "deploy_hooks.sh"
    shutil.copy(REAL_SCRIPT, script_path)
    script_path.chmod(0o755)

    (hooks_src / "sample.ts").write_text("export const x = 1;\n")
    (hooks_src / "shared").mkdir()
    (hooks_src / "shared" / "util.ts").write_text("export const y = 2;\n")
    (hooks_dist / "sample.mjs").write_text("export const x = 1;\n")

    target_parent = tmp_path / "claude-home"
    target_parent.mkdir()
    target = target_parent / "hooks"  # basename must be "hooks"

    return opc_root, script_path, target


def _run(
    script_path: Path,
    target: Path | str,
    *,
    args: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the deploy script with DEPLOY_TARGET set; capture both streams.

    HOME is pointed at /tmp/nonexistent-home so the fallback path never
    collides with the developer's real ~/.claude/hooks.
    """
    env: dict[str, str] = {
        "DEPLOY_TARGET": str(target),
        "HOME": "/tmp/nonexistent-home",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(script_path), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# --- Happy path --------------------------------------------------------------


class TestHappyPath:
    def test_mirrors_src_and_dist(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(tmp_path)

        result = _run(script, target)

        assert result.returncode == 0, result.stderr
        assert (target / "src" / "sample.ts").read_text() == "export const x = 1;\n"
        assert (target / "src" / "shared" / "util.ts").read_text() == "export const y = 2;\n"
        assert (target / "dist" / "sample.mjs").read_text() == "export const x = 1;\n"
        assert "mirrored src/ and dist/" in result.stdout

    def test_idempotent_second_run(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(tmp_path)

        first = _run(script, target)
        second = _run(script, target)

        assert first.returncode == 0
        assert second.returncode == 0
        assert (target / "src" / "sample.ts").exists()


# --- rsync --delete behavior -------------------------------------------------


class TestDeleteBehavior:
    def test_removes_stale_file_in_target_src(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(tmp_path)
        (target / "src").mkdir(parents=True, exist_ok=True)
        stale = target / "src" / "old-hook.ts"
        stale.write_text("// should be deleted\n")

        result = _run(script, target)

        assert result.returncode == 0, result.stderr
        assert not stale.exists(), "rsync --delete should have removed the stale file"

    def test_removes_stale_file_in_target_dist(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(tmp_path)
        (target / "dist").mkdir(parents=True, exist_ok=True)
        stale = target / "dist" / "old-hook.js"
        stale.write_text("// stale\n")

        result = _run(script, target)

        assert result.returncode == 0, result.stderr
        assert not stale.exists()


# --- Guard clauses -----------------------------------------------------------


class TestGuards:
    def test_empty_dist_errors(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(tmp_path)
        for child in (opc_root / "hooks" / "dist").iterdir():
            child.unlink()

        result = _run(script, target)

        assert result.returncode == 1
        assert "run 'cd hooks && npm run build' first" in result.stderr

    def test_empty_src_errors(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(tmp_path)
        shutil.rmtree(opc_root / "hooks" / "src")
        (opc_root / "hooks" / "src").mkdir()

        result = _run(script, target)

        assert result.returncode == 1
        assert "src is empty or missing" in result.stderr

    def test_missing_claude_root_skips_cleanly(self, tmp_path: Path) -> None:
        opc_root, script, _real_target = _build_fixture(tmp_path)
        missing = tmp_path / "does-not-exist" / "hooks"

        result = _run(script, missing)

        assert result.returncode == 0
        assert "skipping" in result.stdout
        assert not missing.exists(), "script should not create target when parent is missing"


# --- Finding #1: --auto worktree guard ---------------------------------------


class TestAutoModeWorktreeGuard:
    def test_auto_skips_from_dot_worktrees_path(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(
            tmp_path, opc_subpath="project/.worktrees/experiment"
        )

        result = _run(script, target, args=("--auto",))

        assert result.returncode == 0
        assert "skipping auto-deploy" in result.stdout
        assert not (target / "src" / "sample.ts").exists(), "should not have deployed"

    def test_auto_skips_from_claude_worktrees_path(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(
            tmp_path, opc_subpath="project/.claude/worktrees/agent-abc"
        )

        result = _run(script, target, args=("--auto",))

        assert result.returncode == 0
        assert "skipping auto-deploy" in result.stdout
        assert not (target / "src" / "sample.ts").exists()

    def test_auto_deploys_from_non_worktree_path(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(
            tmp_path, opc_subpath="home/user/opc"
        )

        result = _run(script, target, args=("--auto",))

        assert result.returncode == 0, result.stderr
        assert (target / "src" / "sample.ts").exists()
        assert "mirrored src/ and dist/" in result.stdout

    def test_explicit_deploy_still_runs_from_worktree(self, tmp_path: Path) -> None:
        """Without --auto, the worktree check is bypassed — user opt-in."""
        opc_root, script, target = _build_fixture(
            tmp_path, opc_subpath="project/.worktrees/experiment"
        )

        result = _run(script, target)  # no --auto

        assert result.returncode == 0, result.stderr
        assert (target / "src" / "sample.ts").exists()


# --- Finding #2: DEPLOY_TARGET validation ------------------------------------


class TestTargetValidation:
    def test_rejects_non_hooks_basename(self, tmp_path: Path) -> None:
        opc_root, script, _target = _build_fixture(tmp_path)
        bad = tmp_path / "claude-home" / "not-hooks"

        result = _run(script, bad)

        assert result.returncode == 4
        assert "basename must be 'hooks'" in result.stderr
        assert not bad.exists()

    def test_rejects_root(self, tmp_path: Path) -> None:
        opc_root, script, _target = _build_fixture(tmp_path)

        result = _run(script, "/")

        assert result.returncode == 4
        assert "unsafe target" in result.stderr

    def test_rejects_home(self, tmp_path: Path) -> None:
        """If DEPLOY_TARGET resolves to $HOME itself, refuse."""
        opc_root, script, _target = _build_fixture(tmp_path)
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()

        result = _run(
            script,
            str(fake_home),
            extra_env={"HOME": str(fake_home)},
        )

        assert result.returncode == 4
        assert "unsafe target" in result.stderr

    def test_rejects_home_dot_claude(self, tmp_path: Path) -> None:
        """If DEPLOY_TARGET resolves to $HOME/.claude (parent of hooks), refuse."""
        opc_root, script, _target = _build_fixture(tmp_path)
        fake_home = tmp_path / "fake-home"
        (fake_home / ".claude").mkdir(parents=True)

        result = _run(
            script,
            str(fake_home / ".claude"),
            extra_env={"HOME": str(fake_home)},
        )

        assert result.returncode == 4
        assert "unsafe target" in result.stderr


# --- Finding #3: lock serialization ------------------------------------------


class TestLockSerialization:
    def test_held_lock_causes_skip(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(tmp_path)
        # Pre-create the lock dir so the script sees contention.
        lock_dir = tmp_path / "opc-deploy-hooks.lock.d"
        lock_dir.mkdir()

        result = _run(
            script,
            target,
            extra_env={"TMPDIR": str(tmp_path)},
        )

        assert result.returncode == 5
        assert "another deploy is in progress" in result.stderr
        # Target should not have been touched.
        assert not (target / "src" / "sample.ts").exists()

    def test_lock_released_on_success(self, tmp_path: Path) -> None:
        opc_root, script, target = _build_fixture(tmp_path)
        lock_parent = tmp_path / "lock-test"
        lock_parent.mkdir()

        result = _run(
            script,
            target,
            extra_env={"TMPDIR": str(lock_parent)},
        )

        assert result.returncode == 0, result.stderr
        # Lock dir should be gone after the script exits cleanly.
        assert not (lock_parent / "opc-deploy-hooks.lock.d").exists()

    def test_lock_released_on_failure(self, tmp_path: Path) -> None:
        """Even when the script exits non-zero, the lock dir is cleaned up."""
        opc_root, script, _target = _build_fixture(tmp_path)
        lock_parent = tmp_path / "lock-test"
        lock_parent.mkdir()
        bad_target = tmp_path / "claude-home" / "not-hooks"  # triggers exit 4

        result = _run(
            script,
            bad_target,
            extra_env={"TMPDIR": str(lock_parent)},
        )

        # exit 4 happens BEFORE the lock is acquired, so the lock should
        # never have been created.
        assert result.returncode == 4
        assert not (lock_parent / "opc-deploy-hooks.lock.d").exists()


# --- Deploy target override --------------------------------------------------


class TestDeployTargetOverride:
    def test_respects_deploy_target_env(self, tmp_path: Path) -> None:
        opc_root, script, _default_target = _build_fixture(tmp_path)
        alternate_parent = tmp_path / "alternate"
        alternate_parent.mkdir()
        custom = alternate_parent / "hooks"  # basename must be "hooks"

        result = _run(script, custom)

        assert result.returncode == 0, result.stderr
        assert (custom / "src" / "sample.ts").exists()
        assert (custom / "dist" / "sample.mjs").exists()

    def test_default_uses_home_dot_claude_when_env_unset(
        self, tmp_path: Path
    ) -> None:
        """When DEPLOY_TARGET is unset, script falls back to $HOME/.claude/hooks."""
        opc_root, script, _target = _build_fixture(tmp_path)
        fake_home = tmp_path / "fake-home"
        (fake_home / ".claude").mkdir(parents=True)

        result = subprocess.run(
            ["bash", str(script)],
            env={
                "HOME": str(fake_home),
                "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert (fake_home / ".claude" / "hooks" / "src" / "sample.ts").exists()
        assert (fake_home / ".claude" / "hooks" / "dist" / "sample.mjs").exists()
