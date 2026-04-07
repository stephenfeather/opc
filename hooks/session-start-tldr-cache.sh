#!/bin/bash
# SessionStart Hook: TLDR Cache Awareness
# Checks if TLDR caches exist and notifies availability

set -e
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR" || exit 0

cat | node dist/session-start-tldr-cache.mjs
