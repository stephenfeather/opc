#!/usr/bin/env bash
# Cron wrapper: incremental rescan of every registered document collection.
# Install with a crontab line such as:
#   0 6 * * * /Users/stephenfeather/opc/scripts/core/documents/rescan_cron.sh
#
# Logs to ~/.claude/logs/opc-docs-rescan.log and writes a one-line status to
# ~/.claude/logs/opc-docs-rescan.status. Safe to run repeatedly: ingest is
# hash-incremental and idempotent.
#
# Re-entrancy: an atomic mkdir lock prevents overlapping runs (cron firing
# again before a slow scan finishes, or a manual scan racing the cron one).
# mkdir is used rather than flock because flock is absent on macOS by default.
set -uo pipefail

# cron runs with a minimal PATH (often just /usr/bin:/bin), so user-installed
# binaries like `uv` (in ~/.local/bin or /opt/homebrew/bin) are not found and
# the job silently fails. Prepend the common install dirs.
PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export PATH

# Portable ISO-8601 timestamp: `date -Iseconds` is a GNU extension that BSD/macOS
# `date` rejects, so format explicitly.
ts() { date "+%Y-%m-%dT%H:%M:%S%z"; }

OPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${HOME}/.claude/logs"
LOG_FILE="${LOG_DIR}/opc-docs-rescan.log"
STATUS_FILE="${LOG_DIR}/opc-docs-rescan.status"
LOCK_DIR="${TMPDIR:-/tmp}/opc-docs-rescan.lock"
mkdir -p "${LOG_DIR}"

# Atomic lock: mkdir fails if the directory already exists (another run holds it).
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "$(ts) opc-docs rescan: another run holds ${LOCK_DIR}; skipping" >> "${LOG_FILE}"
    echo "last_skip=$(ts) reason=locked" > "${STATUS_FILE}"
    exit 0
fi
# Release the lock on any exit (normal, error, or signal).
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT INT TERM

cd "${OPC_DIR}"
start="$(ts)"
{
    echo "=== opc-docs rescan ${start} ==="
    uv run python scripts/core/documents/cli.py scan --all
} >> "${LOG_FILE}" 2>&1
rc=$?

echo "last_run=${start} finished=$(ts) exit=${rc}" > "${STATUS_FILE}"
exit "${rc}"
