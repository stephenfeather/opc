#!/bin/bash
# TLDR Read Enforcer Hook - BLOCKING VERSION
# Intercepts code file reads and returns TLDR context (L1 AST + L2 Call Graph)
# Result: 95% token savings

set -e
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR" || exit 0

# Run the bundled hook
cat | node dist/tldr-read-enforcer.mjs
