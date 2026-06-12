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

# aegis: namespace the lock by uid so a shared /tmp on multi-user hosts
# cannot be squatted by another user to DoS builds.
BUILD_LOCK_DIR="${TMPDIR:-/tmp}/opc-build-hooks.$(id -u).lock.d"
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

# Codex #161 round-3 finding: a remove-then-recreate stale reclaim lets two
# contenders both win (the later rm -rf deletes the earlier one's fresh
# lock). Use the same atomic quarantine-rename pattern as deploy_hooks.sh:
# mv/rename is atomic, so only one racer can quarantine a given stale lock,
# and ownership counts only after our PID survives a write + re-read.
_acquire_build_lock() {
    local attempt=0 _owner _quarantine
    while [ "$attempt" -lt 5 ]; do
        attempt=$((attempt + 1))
        if mkdir "$BUILD_LOCK_DIR" 2>/dev/null; then
            if ! printf '%s\n' "$$" >"$BUILD_LOCK_DIR/pid" 2>/dev/null; then
                continue
            fi
            if [ "$(cat "$BUILD_LOCK_DIR/pid" 2>/dev/null || true)" = "$$" ]; then
                _build_lock_acquired=1
                return 0
            fi
            continue
        fi
        _owner="$(cat "$BUILD_LOCK_DIR/pid" 2>/dev/null || true)"
        if [ -z "$_owner" ]; then
            # Grace for the mkdir-to-pid-write window of a live acquirer.
            sleep 0.1
            _owner="$(cat "$BUILD_LOCK_DIR/pid" 2>/dev/null || true)"
        fi
        if [ -n "$_owner" ] && kill -0 "$_owner" 2>/dev/null; then
            return 1
        fi
        _quarantine="${BUILD_LOCK_DIR}.stale.$$.$attempt"
        if mv "$BUILD_LOCK_DIR" "$_quarantine" 2>/dev/null; then
            rm -rf "$_quarantine" 2>/dev/null || true
        fi
    done
    return 1
}

if ! _acquire_build_lock; then
    echo "build_hooks: another build is in progress ($BUILD_LOCK_DIR) - aborting" >&2
    exit 5
fi

cd "$HOOKS_DIR"

# aegis: $$ is guessable, so a pre-created dist.tmp.<pid> (or a symlink to
# elsewhere) could redirect esbuild output. Clear anything squatting on the
# staging path, then refuse if a symlink still manages to appear there.
rm -rf "$STAGING" 2>/dev/null || true
if [ -e "$STAGING" ] || [ -L "$STAGING" ]; then
    echo "build_hooks: staging path $STAGING already exists and cannot be cleared - aborting" >&2
    exit 1
fi

npx esbuild src/*.ts --bundle --platform=node --format=esm \
    --outdir="$STAGING" --out-extension:.js=.mjs \
    --external:better-sqlite3 --legal-comments=inline

if [ -L "$STAGING" ]; then
    echo "build_hooks: staging path $STAGING is a symlink - refusing to publish" >&2
    exit 1
fi

if [ ! -d "$STAGING" ] || [ -z "$(ls -A "$STAGING" 2>/dev/null)" ]; then
    echo "build_hooks: esbuild produced no output in $STAGING - aborting" >&2
    exit 1
fi

# Publish: dist/ is updated in place and never absent. --delete-delay
# (Codex #161 round-3 finding) defers pruning until AFTER --delay-updates
# has renamed every new file into place, so a reader never sees a
# deletion-only intermediate tree: old bundles stay visible until their
# replacements exist, and stale artifacts (the #161 class) are pruned last.
# This is per-file, not a directory-atomic swap - the residual window is
# new-and-old coexisting briefly, never neither.
mkdir -p dist
rsync -a --delete-delay --delay-updates "$STAGING/" dist/
rm -rf "$STAGING"

echo "build_hooks: published $(ls dist | wc -l | tr -d ' ') bundles to hooks/dist/"

# Plain call, NOT exec: exec would replace the shell and skip the EXIT
# trap, leaving the build lock held forever.
"$OPC_ROOT/scripts/deploy_hooks.sh" "$@"
