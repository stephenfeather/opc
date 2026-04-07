#!/bin/bash
# PreToolUse Hook: Architecture Context Injection
# Injects tldr arch output for planning tasks

set -e
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR" || exit 0

cat | node dist/arch-context-inject.mjs
