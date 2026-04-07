#!/bin/bash
set -e
cd "$(dirname "$0")"
cat | node dist/import-validator.mjs
