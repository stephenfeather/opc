#!/usr/bin/env bash
# deploy_hooks.sh - Mirror hooks/src and hooks/dist into $HOME/.claude/hooks/
#
# Issue #105: ~/opc/hooks is the source of truth; ~/.claude/hooks is the
# Claude Code runtime location referenced by settings.json. This script keeps
# the runtime tree in sync via rsync --delete, then verifies with diff -rq.
#
# Usage:
#   deploy_hooks.sh         # unconditional deploy (manual `npm run deploy`)
#   deploy_hooks.sh --auto  # skip when running from a .worktrees/ checkout
#                           # (wired into hooks/package.json postbuild)
#
# Environment:
#   DEPLOY_TARGET  override destination (default: $HOME/.claude/hooks).
#                  Must resolve to an absolute path whose basename is 'hooks'
#                  and whose parent directory already exists.
#
# Exit codes:
#   0  success (or skipped because Claude Code is not installed / worktree)
#   1  hooks/src or hooks/dist is empty (forgot to build)
#   2  src/ verification mismatch
#   3  dist/ verification mismatch
#   4  refusing to deploy to unsafe target
#   5  another deploy is already in progress (lock contention)

set -euo pipefail

AUTO_MODE=0
if [ "${1:-}" = "--auto" ]; then
    AUTO_MODE=1
    shift
fi

OPC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_SRC="$OPC_ROOT/hooks"

# Finding #1: Skip auto-mode deploys when running from a git worktree.
# Worktrees exist for isolated experiments and must not stomp the live
# ~/.claude/hooks runtime shared by other Claude Code sessions.
if [ "$AUTO_MODE" = "1" ]; then
    case "$OPC_ROOT" in
        */.worktrees/*|*/.claude/worktrees/*)
            echo "deploy_hooks: running from a worktree ($OPC_ROOT) - skipping auto-deploy. Run 'npm run deploy' to override."
            exit 0
            ;;
    esac
fi

TARGET="${DEPLOY_TARGET:-$HOME/.claude/hooks}"

# Finding #2: Validate DEPLOY_TARGET before any rsync --delete call.
# Early reject obviously-unsafe values before we touch dirname/cd, which
# would otherwise resolve '/' to '//' and slip past the later checks.
case "$TARGET" in
    "" | "/" )
        echo "deploy_hooks: refusing to deploy to unsafe target '$TARGET'" >&2
        exit 4
        ;;
esac

# Resolve the parent to an absolute path; macOS ships without `realpath -m`,
# so we build the path manually from the parent directory.
TARGET_PARENT_RAW="$(dirname "$TARGET")"
if [ ! -d "$TARGET_PARENT_RAW" ]; then
    echo "deploy_hooks: $TARGET_PARENT_RAW does not exist - skipping (not a Claude Code install)"
    exit 0
fi
TARGET_PARENT_ABS="$(cd "$TARGET_PARENT_RAW" && pwd)"
TARGET_ABS="$TARGET_PARENT_ABS/$(basename "$TARGET")"

# Reject dangerous targets before any mutation.
case "$TARGET_ABS" in
    "" | "/" )
        echo "deploy_hooks: refusing to deploy to unsafe target '$TARGET_ABS'" >&2
        exit 4
        ;;
esac
if [ "$TARGET_ABS" = "$HOME" ] || [ "$TARGET_ABS" = "${HOME%/}/.claude" ]; then
    echo "deploy_hooks: refusing to deploy to unsafe target '$TARGET_ABS'" >&2
    exit 4
fi
if [ "$(basename "$TARGET_ABS")" != "hooks" ]; then
    echo "deploy_hooks: refusing to deploy to '$TARGET_ABS' - target basename must be 'hooks'" >&2
    exit 4
fi

if [ ! -d "$HOOKS_SRC/src" ] || [ -z "$(ls -A "$HOOKS_SRC/src" 2>/dev/null)" ]; then
    echo "deploy_hooks: $HOOKS_SRC/src is empty or missing - nothing to deploy" >&2
    exit 1
fi

if [ ! -d "$HOOKS_SRC/dist" ] || [ -z "$(ls -A "$HOOKS_SRC/dist" 2>/dev/null)" ]; then
    echo "deploy_hooks: $HOOKS_SRC/dist is empty - run 'cd hooks && npm run build' first" >&2
    exit 1
fi

# Finding #3: Acquire an atomic lock via mkdir so that concurrent deploys
# from different worktrees can't stomp each other mid-sync. mkdir is atomic
# on POSIX filesystems and portable across macOS/Linux (unlike flock).
LOCK_DIR="${TMPDIR:-/tmp}/opc-deploy-hooks.lock.d"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "deploy_hooks: another deploy is in progress ($LOCK_DIR exists) - skipping" >&2
    exit 5
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM

mkdir -p "$TARGET_ABS/src" "$TARGET_ABS/dist"

# Finding #3 (continued): rsync --delay-updates stages new files into a
# hidden holding directory and renames them into place at the end of the
# batch, so readers see the old or new tree - not a half-updated mix.
echo "deploy_hooks: syncing src/  -> $TARGET_ABS/src/"
rsync -a --delete --delay-updates "$HOOKS_SRC/src/" "$TARGET_ABS/src/"

echo "deploy_hooks: syncing dist/ -> $TARGET_ABS/dist/"
rsync -a --delete --delay-updates "$HOOKS_SRC/dist/" "$TARGET_ABS/dist/"

if ! diff -rq "$HOOKS_SRC/src/" "$TARGET_ABS/src/" >/dev/null 2>&1; then
    echo "deploy_hooks: src/ mismatch after sync:" >&2
    diff -rq "$HOOKS_SRC/src/" "$TARGET_ABS/src/" >&2 || true
    exit 2
fi

if ! diff -rq "$HOOKS_SRC/dist/" "$TARGET_ABS/dist/" >/dev/null 2>&1; then
    echo "deploy_hooks: dist/ mismatch after sync:" >&2
    diff -rq "$HOOKS_SRC/dist/" "$TARGET_ABS/dist/" >&2 || true
    exit 3
fi

echo "deploy_hooks: mirrored src/ and dist/ from $HOOKS_SRC/ to $TARGET_ABS/"
