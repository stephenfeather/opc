#!/bin/bash
# Permission Request Handler — fires on PermissionRequest hook event
#
# Logs permission requests (the ones that actually prompt) to JSONL,
# then passes through to the native Claude Code dialog.

INPUT=$(</dev/stdin)

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
LOG_DIR="${PROJECT_DIR}/.claude"
LOG_FILE="${LOG_DIR}/permission-requests.jsonl"

mkdir -p "$LOG_DIR" 2>/dev/null

# Log with prompted=true to distinguish from PreToolUse entries
echo "$INPUT" | jq -c --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg proj "$(basename "$PROJECT_DIR")" '{
  timestamp: $ts,
  project: $proj,
  tool: .tool_name,
  input: .tool_input,
  permission_mode: .permission_mode,
  session_id: .session_id,
  suggestions: .permission_suggestions,
  prompted: true
}' >> "$LOG_FILE" 2>/dev/null

# Pass through to native dialog
echo '{"result":"continue"}'
