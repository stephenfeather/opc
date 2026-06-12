"""Tests for scripts/build_hooks.sh — Issue #161 staged build-and-swap.

Exercises the bash build pipeline via subprocess with an `npx` shim on PATH
so no real esbuild/node_modules is needed. The shim parses --outdir= from
its args and either writes a fake bundle there or fails, depending on
FAKE_ESBUILD_FAIL.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_BUILD = REPO_ROOT / "scripts" / "build_hooks.sh"
REAL_DEPLOY = REPO_ROOT / "scripts" / "deploy_hooks.sh"

NPX_SHIM = """#!/bin/sh
out=""
for a in "$@"; do
    case "$a" in --outdir=*) out="${a#--outdir=}" ;; esac
done
if [ -n "${FAKE_ESBUILD_FAIL:-}" ]; then
    echo "fake esbuild: forced failure" >&2
    exit 1
fi
mkdir -p "$out"
printf 'console.log(1);\\n' > "$out/sample.mjs"
"""


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake OPC root with both scripts and hooks/src.

    Returns (opc_root, build_script_path).
    """
    opc_root = tmp_path / "opc"
    scripts_dir = opc_root / "scripts"
    hooks_src = opc_root / "hooks" / "src"
    scripts_dir.mkdir(parents=True)
    hooks_src.mkdir(parents=True)

    build_script = scripts_dir / "build_hooks.sh"
    shutil.copy(REAL_BUILD, build_script)
    build_script.chmod(0o755)
    deploy_script = scripts_dir / "deploy_hooks.sh"
    shutil.copy(REAL_DEPLOY, deploy_script)
    deploy_script.chmod(0o755)

    (hooks_src / "sample.ts").write_text("export const x = 1;\n")

    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    npx = shim_dir / "npx"
    npx.write_text(NPX_SHIM)
    npx.chmod(0o755)

    return opc_root, build_script


def _run(
    tmp_path: Path,
    build_script: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env: dict[str, str] = {
        # deploy target parent does not exist -> deploy step skips cleanly
        "DEPLOY_TARGET": str(tmp_path / "no-such-parent" / "hooks"),
        "HOME": "/tmp/nonexistent-home",
        "PATH": f"{tmp_path / 'shims'}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "TMPDIR": str(tmp_path / "tmp"),
    }
    (tmp_path / "tmp").mkdir(exist_ok=True)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(build_script), "--auto"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class TestBuildAndSwap:
    def test_successful_build_publishes_and_cleans_staging(
        self, tmp_path: Path
    ) -> None:
        opc_root, build_script = _build_fixture(tmp_path)

        result = _run(tmp_path, build_script)

        assert result.returncode == 0, result.stderr
        dist = opc_root / "hooks" / "dist"
        assert (dist / "sample.mjs").exists()
        leftovers = list((opc_root / "hooks").glob("dist.tmp.*"))
        assert leftovers == [], f"staging not cleaned: {leftovers}"

    def test_failed_build_leaves_existing_dist_intact(
        self, tmp_path: Path
    ) -> None:
        """Codex #161 round-1/2 findings: a failed build must never remove
        or replace the live bundle set."""
        opc_root, build_script = _build_fixture(tmp_path)
        dist = opc_root / "hooks" / "dist"
        dist.mkdir(parents=True)
        (dist / "existing.mjs").write_text("console.log('old');\n")

        result = _run(
            tmp_path, build_script, extra_env={"FAKE_ESBUILD_FAIL": "1"}
        )

        assert result.returncode == 1, (result.stdout, result.stderr)
        assert (dist / "existing.mjs").read_text() == "console.log('old');\n"
        leftovers = list((opc_root / "hooks").glob("dist.tmp.*"))
        assert leftovers == [], f"staging not cleaned after failure: {leftovers}"

    def test_publish_prunes_stale_bundles(self, tmp_path: Path) -> None:
        """rsync --delete in the publish step removes artifacts whose source
        is gone (the #161 stale-artifact class)."""
        opc_root, build_script = _build_fixture(tmp_path)
        dist = opc_root / "hooks" / "dist"
        dist.mkdir(parents=True)
        (dist / "retired.mjs").write_text("console.log('stale');\n")

        result = _run(tmp_path, build_script)

        assert result.returncode == 0, result.stderr
        assert (dist / "sample.mjs").exists()
        assert not (dist / "retired.mjs").exists()

    def test_live_lock_owner_aborts_fast_without_touching_dist(
        self, tmp_path: Path
    ) -> None:
        """Codex #161 round-2 finding: concurrent builds must not interleave.
        A live lock owner makes the second build exit 5 untouched."""
        opc_root, build_script = _build_fixture(tmp_path)
        dist = opc_root / "hooks" / "dist"
        dist.mkdir(parents=True)
        (dist / "existing.mjs").write_text("console.log('old');\n")

        lock_dir = tmp_path / "tmp" / f"opc-build-hooks.{os.getuid()}.lock.d"
        (tmp_path / "tmp").mkdir(exist_ok=True)
        lock_dir.mkdir(parents=True)
        holder = subprocess.Popen(["sleep", "30"])
        try:
            (lock_dir / "pid").write_text(f"{holder.pid}\n")

            result = _run(tmp_path, build_script)

            assert result.returncode == 5, (result.stdout, result.stderr)
            assert (dist / "existing.mjs").exists()
        finally:
            holder.terminate()
            holder.wait()

    def test_stale_lock_from_dead_pid_is_reclaimed(self, tmp_path: Path) -> None:
        opc_root, build_script = _build_fixture(tmp_path)
        lock_dir = tmp_path / "tmp" / f"opc-build-hooks.{os.getuid()}.lock.d"
        (tmp_path / "tmp").mkdir(exist_ok=True)
        lock_dir.mkdir(parents=True)
        (lock_dir / "pid").write_text("4000000\n")  # beyond macOS pid range

        result = _run(tmp_path, build_script)

        assert result.returncode == 0, result.stderr
        assert (opc_root / "hooks" / "dist" / "sample.mjs").exists()
