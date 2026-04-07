#!/bin/bash
# PreToolUse Hook: TLDR Context Injection
# Injects TLDR context into Task prompts based on intent

set -e
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR" || exit 0

cat | node dist/tldr-context-inject.mjs
