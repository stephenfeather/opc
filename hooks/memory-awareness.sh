#!/bin/bash
# Memory Awareness Hook - checks if user prompt matches stored learnings
set -e
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"
cat | node dist/memory-awareness.mjs
