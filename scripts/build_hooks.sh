#!/usr/bin/env bash
# build_hooks.sh - Build hooks/src/*.ts into hooks/dist/ without ever
# leaving the checkout (or a symlink-installed runtime) without bundles.
#
# Issue #161, Codex round-2 findings drove this design:
#   1. NO destructive window: esbuild writes to a per-process staging dir
#      (dist.tmp.$$); the result is published into dist/ with
#      `rsync --delete --delay-updates`, so dist/ always exists and
#      readers see either the old or the new bundle set - never neither,
#      never a partial mix.
#   2. NO concurrent-build race: the whole stage-build-publish-deploy
#      sequence is serialized by an mkdir lock (same primitive as
#      deploy_hooks.sh). A second build fails fast with exit 5 instead of
#      deleting the first one's staging or publishing partial output.
#
# Usage: build_hooks.sh [--auto]   (--auto is forwarded to deploy_hooks.sh)
#
# Exit codes:
#   0  success
#   1  esbuild failed or produced an empty staging dir
#   5  another build is in progress (lock contention)

set -euo pipefail

OPC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="$OPC_ROOT/hooks"
STAGING="$HOOKS_DIR/dist.tmp.$$"

BUILD_LOCK_DIR="${TMPDIR:-/tmp}/opc-build-hooks.lock.d"
_build_lock_acquired=0

_build_cleanup() {
    rm -rf "$STAGING" 2>/dev/null || true
    if [ "$_build_lock_acquired" = "1" ]; then
        rm -rf "$BUILD_LOCK_DIR" 2>/dev/null || true
        _build_lock_acquired=0
    fi
}
trap _build_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Fail fast on contention; reclaim only dead owners (PID heuristic kept
# simple here - the deploy step has its own, stricter lock).
if ! mkdir "$BUILD_LOCK_DIR" 2>/dev/null; then
    _owner="$(cat "$BUILD_LOCK_DIR/pid" 2>/dev/null || true)"
    if [ -n "$_owner" ] && kill -0 "$_owner" 2>/dev/null; then
        echo "build_hooks: another build is in progress ($BUILD_LOCK_DIR) - aborting" >&2
        exit 5
    fi
    rm -rf "$BUILD_LOCK_DIR"
    if ! mkdir "$BUILD_LOCK_DIR" 2>/dev/null; then
        echo "build_hooks: lock contention ($BUILD_LOCK_DIR) - aborting" >&2
        exit 5
    fi
fi
_build_lock_acquired=1
printf '%s\n' "$$" >"$BUILD_LOCK_DIR/pid" 2>/dev/null || true

cd "$HOOKS_DIR"

npx esbuild src/*.ts --bundle --platform=node --format=esm \
    --outdir="$STAGING" --out-extension:.js=.mjs \
    --external:better-sqlite3 --legal-comments=inline

if [ ! -d "$STAGING" ] || [ -z "$(ls -A "$STAGING" 2>/dev/null)" ]; then
    echo "build_hooks: esbuild produced no output in $STAGING - aborting" >&2
    exit 1
fi

# Publish: dist/ is updated in place and never absent. --delete prunes
# artifacts whose source no longer exists (the #161 stale-artifact class);
# --delay-updates batches the renames so readers see old-or-new.
mkdir -p dist
rsync -a --delete --delay-updates "$STAGING/" dist/
rm -rf "$STAGING"

echo "build_hooks: published $(ls dist | wc -l | tr -d ' ') bundles to hooks/dist/"

# Plain call, NOT exec: exec would replace the shell and skip the EXIT
# trap, leaving the build lock held forever.
"$OPC_ROOT/scripts/deploy_hooks.sh" "$@"
