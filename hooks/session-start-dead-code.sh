#!/bin/bash
# SessionStart Hook: Dead Code Detection
# Runs tldr dead on startup and warns about unused functions

set -e
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR" || exit 0

cat | node dist/session-start-dead-code.mjs
