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

    def test_auto_skips_from_real_git_worktree_outside_dot_worktrees(
        self, tmp_path: Path
    ) -> None:
        """Round-2 regression: a real `git worktree add` outside /.worktrees/
        must still be detected via git-dir != git-common-dir."""
        main_repo = tmp_path / "main-repo"
        main_repo.mkdir()
        git_env = {
            "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(tmp_path / "fake-home"),
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        }
        (tmp_path / "fake-home").mkdir()
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(main_repo)],
            env=git_env,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(main_repo), "config", "user.email", "test@test"],
            env=git_env,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(main_repo), "config", "user.name", "test"],
            env=git_env,
            check=True,
        )
        (main_repo / "README").write_text("test\n")
        subprocess.run(
            ["git", "-C", str(main_repo), "add", "."], env=git_env, check=True
        )
        subprocess.run(
            ["git", "-C", str(main_repo), "commit", "-q", "-m", "init"],
            env=git_env,
            check=True,
        )
        # Custom worktree path that is NOT under /.worktrees/ or /.claude/worktrees/
        feature = tmp_path / "feature-branch"
        subprocess.run(
            [
                "git",
                "-C",
                str(main_repo),
                "worktree",
                "add",
                "-q",
                str(feature),
                "-b",
                "feature",
            ],
            env=git_env,
            check=True,
        )

        # Build the fixture inside the worktree
        scripts_dir = feature / "scripts"
        hooks_src = feature / "hooks" / "src"
        hooks_dist = feature / "hooks" / "dist"
        scripts_dir.mkdir(parents=True)
        hooks_src.mkdir(parents=True)
        hooks_dist.mkdir(parents=True)
        script = scripts_dir / "deploy_hooks.sh"
        shutil.copy(REAL_SCRIPT, script)
        script.chmod(0o755)
        (hooks_src / "sample.ts").write_text("x\n")
        (hooks_dist / "sample.mjs").write_text("y\n")
        target_parent = tmp_path / "claude-home"
        target_parent.mkdir()
        target = target_parent / "hooks"

        result = subprocess.run(
            ["bash", str(script), "--auto"],
            env={
                "DEPLOY_TARGET": str(target),
                "HOME": "/tmp/nonexistent-home",
                "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "git worktree" in result.stdout
        assert not (target / "src" / "sample.ts").exists()


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

    def test_rejects_symlinked_target(self, tmp_path: Path) -> None:
        """Round-2 regression: a symlink named 'hooks' pointing elsewhere
        must be refused, not followed, to prevent rsync --delete from
        deleting files in the symlink destination."""
        opc_root, script, _target = _build_fixture(tmp_path)
        # Real directory with a non-'hooks' name
        real_dir = tmp_path / "claude-home" / "not-hooks-real"
        real_dir.mkdir()
        (real_dir / "precious.txt").write_text("must not be deleted\n")
        # Symlink 'hooks' -> 'not-hooks-real'
        link = tmp_path / "claude-home" / "hooks"
        link.symlink_to(real_dir)

        result = _run(script, link)

        assert result.returncode == 4
        assert "symlink" in result.stderr
        # Critical: the pre-existing file inside the symlink target must
        # not have been touched by rsync.
        assert (real_dir / "precious.txt").read_text() == "must not be deleted\n"


# --- Finding #3: lock serialization ------------------------------------------


class TestLockSerialization:
    def test_held_lock_causes_skip(self, tmp_path: Path) -> None:
        """With a live owner PID and a fresh mtime, contention is real and
        the script must exit 5 without reclaiming the lock."""
        import os

        opc_root, script, target = _build_fixture(tmp_path)
        lock_parent = tmp_path / "lock-test"
        lock_parent.mkdir()
        lock_dir = lock_parent / "opc-deploy-hooks.lock.d"
        lock_dir.mkdir()
        # Use the current test process's PID as the "owner" so kill -0
        # reports the process alive. The lock mtime is fresh (just now),
        # so the age-based fallback also treats this as real contention.
        (lock_dir / "pid").write_text(f"{os.getpid()}\n")

        result = _run(
            script,
            target,
            extra_env={"TMPDIR": str(lock_parent)},
        )

        assert result.returncode == 5
        assert "another deploy is in progress" in result.stderr
        # Target should not have been touched.
        assert not (target / "src" / "sample.ts").exists()
        # The live lock should NOT have been reclaimed.
        assert lock_dir.exists()
        assert (lock_dir / "pid").read_text() == f"{os.getpid()}\n"

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

    def test_reclaims_stale_lock_with_dead_pid(self, tmp_path: Path) -> None:
        """Round-2 regression: a lock left behind by a crashed deploy (dead
        PID) must be reclaimed automatically, not left wedging all future
        runs."""
        opc_root, script, target = _build_fixture(tmp_path)
        lock_parent = tmp_path / "lock-test"
        lock_parent.mkdir()
        lock_dir = lock_parent / "opc-deploy-hooks.lock.d"
        lock_dir.mkdir()
        # Write a PID that is guaranteed dead. PID 1 is always init/launchd
        # on modern systems, so we use a high PID that is extremely unlikely
        # to exist on any reasonable machine.
        (lock_dir / "pid").write_text("99999999\n")

        result = _run(
            script,
            target,
            extra_env={"TMPDIR": str(lock_parent)},
        )

        assert result.returncode == 0, result.stderr
        assert "stale lock" in result.stderr
        assert (target / "src" / "sample.ts").exists()
        # Lock should be cleaned up on exit.
        assert not lock_dir.exists()

    def test_reclaims_stale_lock_with_missing_pid_file(self, tmp_path: Path) -> None:
        """A lock dir without a PID file (e.g. a crash between mkdir and
        echo $$ > pid) must also be reclaimable."""
        opc_root, script, target = _build_fixture(tmp_path)
        lock_parent = tmp_path / "lock-test"
        lock_parent.mkdir()
        lock_dir = lock_parent / "opc-deploy-hooks.lock.d"
        lock_dir.mkdir()
        # No PID file

        result = _run(
            script,
            target,
            extra_env={"TMPDIR": str(lock_parent)},
        )

        assert result.returncode == 0, result.stderr
        assert "stale lock" in result.stderr
        assert (target / "src" / "sample.ts").exists()

    def test_live_owner_is_respected_regardless_of_age(
        self, tmp_path: Path
    ) -> None:
        """Round-3 regression: a legitimately-running deploy that has been
        running 'too long' must NOT have its lock stolen. Set the lock dir
        mtime to ~1 hour ago while the PID points at our live test process;
        the script must still exit 5."""
        import os
        import time

        opc_root, script, target = _build_fixture(tmp_path)
        lock_parent = tmp_path / "lock-test"
        lock_parent.mkdir()
        lock_dir = lock_parent / "opc-deploy-hooks.lock.d"
        lock_dir.mkdir()
        (lock_dir / "pid").write_text(f"{os.getpid()}\n")
        # Backdate the lock dir mtime by 1 hour so any age-based logic
        # would consider it "stale".
        hour_ago = time.time() - 3600
        os.utime(lock_dir, (hour_ago, hour_ago))

        result = _run(
            script,
            target,
            extra_env={"TMPDIR": str(lock_parent)},
        )

        assert result.returncode == 5, result.stderr
        assert "another deploy is in progress" in result.stderr
        # Live owner's lock must NOT have been touched.
        assert lock_dir.exists()
        assert (lock_dir / "pid").read_text() == f"{os.getpid()}\n"
        # Target must NOT have been written.
        assert not (target / "src" / "sample.ts").exists()

    def test_parallel_reclaimers_yield_single_critical_section(
        self, tmp_path: Path
    ) -> None:
        """Round-3 regression: two processes seeing the same stale lock
        must not both enter the critical section simultaneously. Launch
        them in parallel from a dead-PID stale lock; assert at least one
        succeeds and that no process returns an unexpected error."""
        opc_root, script, target = _build_fixture(tmp_path)
        lock_parent = tmp_path / "lock-test"
        lock_parent.mkdir()
        lock_dir = lock_parent / "opc-deploy-hooks.lock.d"
        lock_dir.mkdir()
        (lock_dir / "pid").write_text("99999999\n")  # dead

        env = {
            "DEPLOY_TARGET": str(target),
            "HOME": "/tmp/nonexistent-home",
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "TMPDIR": str(lock_parent),
        }

        procs = [
            subprocess.Popen(
                ["bash", str(script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        results = [p.wait() for p in procs]
        outputs = [
            (p.stdout.read() if p.stdout else "", p.stderr.read() if p.stderr else "")
            for p in procs
        ]

        # At least one process must succeed (exit 0).
        assert 0 in results, f"neither process succeeded: {results} {outputs}"
        # Every exit code must be either 0 (success) or 5 (contention).
        # Anything else indicates corruption or unexpected state.
        for rc in results:
            assert rc in (0, 5), f"unexpected exit code {rc}: {outputs}"
        # The target must be fully populated by whichever process won.
        assert (target / "src" / "sample.ts").exists()
        assert (target / "src" / "shared" / "util.ts").exists()
        assert (target / "dist" / "sample.mjs").exists()
        # The lock dir must be cleaned up after both processes exit.
        assert not lock_dir.exists()


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
