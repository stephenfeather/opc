#!/bin/bash
# Smart Search Router Hook
# Routes Grep to AST-grep (structural) or allows through (literal/semantic)
# Stores context for tldr-read-enforcer cross-file lookup

set -e
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR" || exit 0

# Run the bundled hook
cat | node dist/smart-search-router.mjs
