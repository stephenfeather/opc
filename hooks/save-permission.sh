#!/bin/bash
# Save Permission — adds a permission rule to project settings.local.json
# Usage: bash save-permission.sh "Bash(npm test:*)" [/path/to/project]
#
# Idempotent: skips if permission already exists

PERMISSION="$1"
PROJECT_DIR="${2:-${CLAUDE_PROJECT_DIR:-$(pwd)}}"
SETTINGS_FILE="${PROJECT_DIR}/.claude/settings.local.json"

if [ -z "$PERMISSION" ]; then
    echo "Usage: save-permission.sh <permission-string> [project-dir]" >&2
    exit 1
fi

mkdir -p "$(dirname "$SETTINGS_FILE")" 2>/dev/null

# Create settings file if it doesn't exist
if [ ! -f "$SETTINGS_FILE" ]; then
    echo '{"permissions":{"allow":[]}}' > "$SETTINGS_FILE"
fi

# Check if permission already exists
if jq -e --arg perm "$PERMISSION" \
    '.permissions.allow // [] | map(select(. == $perm)) | length > 0' \
    "$SETTINGS_FILE" >/dev/null 2>&1; then
    echo "Already exists: ${PERMISSION}"
    exit 0
fi

# Add permission to allow array (create structure if needed)
TEMP_FILE=$(mktemp)
jq --arg perm "$PERMISSION" '
    .permissions //= {} |
    .permissions.allow //= [] |
    .permissions.allow += [$perm]
' "$SETTINGS_FILE" > "$TEMP_FILE" && mv "$TEMP_FILE" "$SETTINGS_FILE"

echo "Added to ${SETTINGS_FILE}: ${PERMISSION}"
