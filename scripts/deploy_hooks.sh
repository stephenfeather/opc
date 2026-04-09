#!/usr/bin/env bash
# deploy_hooks.sh - Mirror hooks/src and hooks/dist into $HOME/.claude/hooks/
#
# Issue #105: ~/opc/hooks is the source of truth; ~/.claude/hooks is the
# Claude Code runtime location referenced by settings.json. This script keeps
# the runtime tree in sync via rsync --delete, then verifies with diff -rq.
#
# Usage:
#   deploy_hooks.sh         # unconditional deploy (manual `npm run deploy`)
#   deploy_hooks.sh --auto  # skip when running from a git worktree
#                           # (wired into hooks/package.json postbuild)
#
# Environment:
#   DEPLOY_TARGET  override destination (default: $HOME/.claude/hooks).
#                  Must resolve to an absolute path whose basename is 'hooks',
#                  must not be a symlink, and must not equal $HOME or
#                  $HOME/.claude (logical or physical).
#   TMPDIR         location for the mkdir-based deploy lock
#                  (default: /tmp). The lock dir is
#                  "$TMPDIR/opc-deploy-hooks.lock.d".
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

# Finding #1 (round 1+2): Skip auto-mode deploys when running from a git
# worktree. Prefer git's own worktree detection over pathname heuristics:
# when --git-dir differs from --git-common-dir, we're inside a linked
# worktree. The pathname case statement remains as a belt-and-suspenders
# fallback for environments where git isn't installed or the repo isn't a
# git checkout.
_is_git_worktree() {
    local repo_path="$1"
    command -v git >/dev/null 2>&1 || return 1
    local git_dir git_common_dir
    git_dir="$(git -C "$repo_path" rev-parse --git-dir 2>/dev/null)" || return 1
    git_common_dir="$(git -C "$repo_path" rev-parse --git-common-dir 2>/dev/null)" || return 1
    [ -n "$git_dir" ] && [ -n "$git_common_dir" ] || return 1
    # Git may return relative paths (e.g. ".git" in the main checkout).
    # Relative paths are relative to repo_path, NOT the script's CWD, so
    # canonicalize explicitly against repo_path before comparing. Without
    # this, running the script from a subdirectory like hooks/ (as npm
    # postbuild does) would make `cd "$git_dir"` resolve against hooks/
    # and mis-detect worktree status.
    case "$git_dir" in
        /*) ;;
        *) git_dir="$repo_path/$git_dir" ;;
    esac
    case "$git_common_dir" in
        /*) ;;
        *) git_common_dir="$repo_path/$git_common_dir" ;;
    esac
    if [ -d "$git_dir" ]; then
        git_dir="$(cd -P "$git_dir" && pwd -P)"
    fi
    if [ -d "$git_common_dir" ]; then
        git_common_dir="$(cd -P "$git_common_dir" && pwd -P)"
    fi
    [ "$git_dir" != "$git_common_dir" ]
}

if [ "$AUTO_MODE" = "1" ]; then
    if _is_git_worktree "$OPC_ROOT"; then
        echo "deploy_hooks: running from a git worktree ($OPC_ROOT) - skipping auto-deploy. Run 'npm run deploy' to override."
        exit 0
    fi
    case "$OPC_ROOT" in
        */.worktrees/*|*/.claude/worktrees/*)
            echo "deploy_hooks: running from a worktree ($OPC_ROOT) - skipping auto-deploy. Run 'npm run deploy' to override."
            exit 0
            ;;
    esac
fi

TARGET="${DEPLOY_TARGET:-$HOME/.claude/hooks}"

# Finding #2 (round 1): Validate DEPLOY_TARGET before any rsync --delete call.
# Early reject obviously-unsafe values before we touch dirname/cd, which
# would otherwise resolve '/' to '//' and slip past the later checks.
case "$TARGET" in
    "" | "/" )
        echo "deploy_hooks: refusing to deploy to unsafe target '$TARGET'" >&2
        exit 4
        ;;
esac

# Finding #2 (round 2): Physically resolve the target parent so that
# validation sees what rsync will actually write to. `cd -P && pwd -P`
# follows symlinks in the parent chain; macOS ships without
# `realpath -m` so we build TARGET_ABS from the physical parent plus
# the literal basename.
TARGET_PARENT_RAW="$(dirname "$TARGET")"
if [ ! -d "$TARGET_PARENT_RAW" ]; then
    echo "deploy_hooks: $TARGET_PARENT_RAW does not exist - skipping (not a Claude Code install)"
    exit 0
fi
TARGET_PARENT_PHYS="$(cd -P "$TARGET_PARENT_RAW" 2>/dev/null && pwd -P || true)"
if [ -z "$TARGET_PARENT_PHYS" ]; then
    echo "deploy_hooks: could not resolve $TARGET_PARENT_RAW physically" >&2
    exit 4
fi
TARGET_BASENAME="$(basename "$TARGET")"
TARGET_ABS="$TARGET_PARENT_PHYS/$TARGET_BASENAME"

# Resolve HOME physically too so the "unsafe target" checks work even when
# HOME itself is a symlink (e.g. ~/.claude -> ~/.dotfiles/claude is common).
_home_phys=""
if [ -n "${HOME:-}" ] && [ -d "$HOME" ]; then
    _home_phys="$(cd -P "$HOME" 2>/dev/null && pwd -P || true)"
fi

for _forbidden in "/" "${HOME:-}" "${HOME%/}/.claude" "$_home_phys" "${_home_phys%/}/.claude"; do
    if [ -n "$_forbidden" ] && [ "$TARGET_ABS" = "$_forbidden" ]; then
        echo "deploy_hooks: refusing to deploy to unsafe target '$TARGET_ABS'" >&2
        exit 4
    fi
done

if [ "$TARGET_BASENAME" != "hooks" ]; then
    echo "deploy_hooks: refusing to deploy to '$TARGET_ABS' - target basename must be 'hooks'" >&2
    exit 4
fi

# Finding #2 (round 2): Refuse if TARGET itself is a symlink. Following a
# symlink into an unrelated tree would let an attacker or a stale config
# aim rsync --delete at the wrong place.
#
# Security audit follow-up: defense-in-depth against a TOCTOU window
# between this check and the later mkdir/rsync calls. `_assert_target_not_symlink`
# is called again immediately before each filesystem mutation so a process
# that swaps TARGET_ABS into a symlink after validation still trips the
# guard before rsync --delete runs.
#
# PR #107 Copilot follow-up: $TARGET_ABS itself can be a real directory
# while $TARGET_ABS/src or $TARGET_ABS/dist is a pre-existing symlink.
# rsync with trailing slashes follows those subdirectory symlinks and
# `rsync --delete` would then prune files inside the link destination —
# reintroducing the blast radius the other symlink guards close. Check
# the subdirs alongside the parent at every mutation checkpoint.
_assert_target_not_symlink() {
    if [ -L "$TARGET_ABS" ]; then
        echo "deploy_hooks: refusing to deploy to '$TARGET_ABS' - target is a symlink" >&2
        exit 4
    fi
}

_assert_target_subdirs_not_symlinks() {
    if [ -L "$TARGET_ABS/src" ]; then
        echo "deploy_hooks: refusing - '$TARGET_ABS/src' is a symlink" >&2
        exit 4
    fi
    if [ -L "$TARGET_ABS/dist" ]; then
        echo "deploy_hooks: refusing - '$TARGET_ABS/dist' is a symlink" >&2
        exit 4
    fi
}

_assert_target_not_symlink
_assert_target_subdirs_not_symlinks

if [ ! -d "$HOOKS_SRC/src" ] || [ -z "$(ls -A "$HOOKS_SRC/src" 2>/dev/null)" ]; then
    echo "deploy_hooks: $HOOKS_SRC/src is empty or missing - nothing to deploy" >&2
    exit 1
fi

if [ ! -d "$HOOKS_SRC/dist" ] || [ -z "$(ls -A "$HOOKS_SRC/dist" 2>/dev/null)" ]; then
    echo "deploy_hooks: $HOOKS_SRC/dist is empty - run 'cd hooks && npm run build' first" >&2
    exit 1
fi

# Finding #3 (round 1+2+3): Acquire an atomic lock via mkdir so that
# concurrent deploys from different worktrees can't stomp each other
# mid-sync. mkdir is atomic on POSIX filesystems and portable across
# macOS/Linux (unlike flock). Round 2 added PID-based stale-lock recovery
# so SIGKILL, crashes, or host reboots do not wedge future deploys forever.
#
# Round 3 removes the age-based reclaim of LIVE owners: stealing a lock
# from a still-running process violates mutual exclusion, even if that
# process has been running "too long" by some arbitrary threshold. We also
# replace the rm -rf + mkdir reclaim path with an atomic `mv` to a unique
# quarantine name so two concurrent reclaimers cannot both win. `mv` /
# rename() is atomic at the POSIX level — only one racer can successfully
# rename a given source.
#
# If a legitimate deploy genuinely hangs (rsync stuck on a dead NFS mount,
# etc.), the lock stays until the user manually investigates and removes
# it: `rm -rf $TMPDIR/opc-deploy-hooks.lock.d`. That is the correct
# tradeoff — we prefer a recoverable wedge over silent concurrent writes
# to the shared runtime tree.

LOCK_DIR="${TMPDIR:-/tmp}/opc-deploy-hooks.lock.d"
LOCK_PID_FILE="$LOCK_DIR/pid"
LOCK_MAX_ATTEMPTS=5
LOCK_GRACE_MS=100

# PR #107 Copilot follow-up: track whether WE actually acquired the lock
# before the trap fires. Previously the trap unconditionally rm -rf'd
# $LOCK_DIR on exit, which meant an early-exit failure (before or during
# acquisition) could delete a *different* process's freshly-acquired lock.
# Gate the cleanup on _lock_acquired so we only release what we own.
_lock_acquired=0

_acquire_lock() {
    local attempt=0
    while [ "$attempt" -lt "$LOCK_MAX_ATTEMPTS" ]; do
        attempt=$((attempt + 1))

        if mkdir "$LOCK_DIR" 2>/dev/null; then
            # Write the PID, tolerating failure. Between our mkdir and
            # this write, another process could have quarantined our
            # fresh lock (via the stale-reclaim path below). If the PID
            # write fails — or the file no longer matches our PID after
            # writing — loop back and try again. Without this, `set -e`
            # would kill the script on a write failure and the trap
            # would try to rm -rf a directory that now belongs to someone
            # else.
            if ! printf '%s\n' "$$" >"$LOCK_PID_FILE" 2>/dev/null; then
                continue
            fi
            if [ "$(cat "$LOCK_PID_FILE" 2>/dev/null || true)" = "$$" ]; then
                _lock_acquired=1
                return 0
            fi
            continue
        fi

        # Lock dir already exists. Read owner PID.
        local owner=""
        if [ -f "$LOCK_PID_FILE" ]; then
            owner="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
        fi

        # PR #107 Copilot follow-up: grace period for in-flight
        # acquisitions. Another process may be in the narrow window
        # between `mkdir "$LOCK_DIR"` and writing the PID file.
        # Immediately treating a missing PID file as stale would let us
        # quarantine a lock that's currently being created. Sleep briefly
        # and re-read; if the PID file is still missing after the grace
        # period, the creator almost certainly crashed and reclamation
        # is justified.
        if [ -z "$owner" ]; then
            sleep "0.$(printf '%03d' "$LOCK_GRACE_MS")"
            if [ -f "$LOCK_PID_FILE" ]; then
                owner="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
            fi
        fi

        if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
            # Live owner - real contention, regardless of age. Never
            # steal a lock from a running process.
            return 1
        fi

        # Stale: owner PID is dead or the PID file is still missing
        # after the grace period. Atomically quarantine the stale lock
        # dir. `mv` on a directory is a rename() syscall, which is
        # atomic at the POSIX level - only one concurrent racer can win.
        # The losing racer loops back and re-checks.
        local quarantine="${LOCK_DIR}.stale.$$.$attempt"
        if mv "$LOCK_DIR" "$quarantine" 2>/dev/null; then
            rm -rf "$quarantine" 2>/dev/null || true
            echo "deploy_hooks: reclaimed stale lock (owner=${owner:-unknown})" >&2
            continue
        fi
    done
    return 1
}

_release_lock() {
    # Only clean up if we actually acquired the lock. Prevents an
    # early-exit trap from wiping another process's lock.
    if [ "$_lock_acquired" = "1" ]; then
        rm -rf "$LOCK_DIR" 2>/dev/null || true
        _lock_acquired=0
    fi
}

if ! _acquire_lock; then
    echo "deploy_hooks: another deploy is in progress ($LOCK_DIR) - skipping" >&2
    exit 5
fi
trap _release_lock EXIT INT TERM

# Defense-in-depth: re-check the target (and its subdirs) is still not a
# symlink right before each filesystem mutation. Closes the TOCTOU window
# flagged by the security audit.
_assert_target_not_symlink
_assert_target_subdirs_not_symlinks
mkdir -p "$TARGET_ABS/src" "$TARGET_ABS/dist"

# Finding #3 (round 1): rsync --delay-updates stages new files into a
# hidden holding directory and renames them into place at the end of the
# batch, so readers see the old or new tree - not a half-updated mix.
_assert_target_not_symlink
_assert_target_subdirs_not_symlinks
echo "deploy_hooks: syncing src/  -> $TARGET_ABS/src/"
rsync -a --delete --delay-updates "$HOOKS_SRC/src/" "$TARGET_ABS/src/"

_assert_target_not_symlink
_assert_target_subdirs_not_symlinks
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
