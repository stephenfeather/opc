#!/usr/bin/env bash
# deploy_hooks.sh - Mirror hooks/src and hooks/dist into $HOME/.claude/hooks/
#
# Issue #105: ~/opc/hooks is the source of truth; ~/.claude/hooks is the
# Claude Code runtime location referenced by settings.json. This script keeps
# the runtime tree in sync via rsync --delete, then verifies with diff -rq.
#
# Environment:
#   DEPLOY_TARGET  override destination (default: $HOME/.claude/hooks)
#
# Exit codes:
#   0  success (or skipped because Claude Code is not installed)
#   1  hooks/dist or hooks/src is empty (forgot to build)
#   2  src/ verification mismatch
#   3  dist/ verification mismatch

set -euo pipefail

OPC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_SRC="$OPC_ROOT/hooks"
TARGET="${DEPLOY_TARGET:-$HOME/.claude/hooks}"
CLAUDE_ROOT="$(dirname "$TARGET")"

if [ ! -d "$CLAUDE_ROOT" ]; then
    echo "deploy_hooks: $CLAUDE_ROOT does not exist - skipping (not a Claude Code install)"
    exit 0
fi

if [ ! -d "$HOOKS_SRC/src" ] || [ -z "$(ls -A "$HOOKS_SRC/src" 2>/dev/null)" ]; then
    echo "deploy_hooks: $HOOKS_SRC/src is empty or missing - nothing to deploy" >&2
    exit 1
fi

if [ ! -d "$HOOKS_SRC/dist" ] || [ -z "$(ls -A "$HOOKS_SRC/dist" 2>/dev/null)" ]; then
    echo "deploy_hooks: $HOOKS_SRC/dist is empty - run 'cd hooks && npm run build' first" >&2
    exit 1
fi

mkdir -p "$TARGET/src" "$TARGET/dist"

echo "deploy_hooks: syncing src/  -> $TARGET/src/"
rsync -a --delete "$HOOKS_SRC/src/" "$TARGET/src/"

echo "deploy_hooks: syncing dist/ -> $TARGET/dist/"
rsync -a --delete "$HOOKS_SRC/dist/" "$TARGET/dist/"

if ! diff -rq "$HOOKS_SRC/src/" "$TARGET/src/" >/dev/null 2>&1; then
    echo "deploy_hooks: src/ mismatch after sync:" >&2
    diff -rq "$HOOKS_SRC/src/" "$TARGET/src/" >&2 || true
    exit 2
fi

if ! diff -rq "$HOOKS_SRC/dist/" "$TARGET/dist/" >/dev/null 2>&1; then
    echo "deploy_hooks: dist/ mismatch after sync:" >&2
    diff -rq "$HOOKS_SRC/dist/" "$TARGET/dist/" >&2 || true
    exit 3
fi

echo "deploy_hooks: mirrored src/ and dist/ from $HOOKS_SRC/ to $TARGET/"
