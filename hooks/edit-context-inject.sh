#!/bin/bash
set -e
cd "$(dirname "$0")"
cat | node dist/edit-context-inject.mjs
