#!/usr/bin/env bash
# Cron wrapper: incremental rescan of every registered document collection.
# Install with a crontab line such as:
#   0 6 * * * /Users/stephenfeather/opc/scripts/core/documents/rescan_cron.sh
#
# Logs to ~/.claude/logs/opc-docs-rescan.log. Safe to run repeatedly:
# ingest is hash-incremental and idempotent.
set -euo pipefail

OPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${HOME}/.claude/logs"
mkdir -p "${LOG_DIR}"

cd "${OPC_DIR}"
{
    echo "=== opc-docs rescan $(date -Iseconds) ==="
    uv run python scripts/core/documents/cli.py scan --all
} >> "${LOG_DIR}/opc-docs-rescan.log" 2>&1
