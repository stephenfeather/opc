#!/bin/bash
set -e
cd "$(dirname "$0")"
cat | node dist/signature-helper.mjs
