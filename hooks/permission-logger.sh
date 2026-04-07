#!/bin/bash
# Permission Logger Hook — logs ALL tool use requests
# Runs as PreToolUse hook on all tools (matcher: ".*")
# Always returns continue — purely passive logging
#
# Output: {project_root}/.claude/permission-requests.jsonl

INPUT=$(</dev/stdin)

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
LOG_DIR="${PROJECT_DIR}/.claude"
LOG_FILE="${LOG_DIR}/permission-requests.jsonl"

mkdir -p "$LOG_DIR" 2>/dev/null

echo "$INPUT" | jq -c --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg proj "$(basename "$PROJECT_DIR")" '{
  timestamp: $ts,
  project: $proj,
  tool: .tool_name,
  input: .tool_input,
  permission_mode: .permission_mode,
  session_id: .session_id,
  suggested_permission: (
    if .tool_name == "Bash" then
      "Bash(" + ((.tool_input.command // "") | split(" ")[0]) + ":*)"
    elif (.tool_name // "" | IN("Read","Write","Edit")) then
      .tool_name + "(/" + ((.tool_input.file_path // "") | gsub("/[^/]+$"; "/**")) + ")"
    elif .tool_name == "Agent" then
      "Agent(" + (.tool_input.subagent_type // "general-purpose") + ")"
    elif .tool_name == "Skill" then
      "Skill(" + (.tool_input.skill // "*") + ")"
    elif ((.tool_name // "") | startswith("mcp__")) then
      .tool_name
    elif (.tool_name // "" | IN("Glob","Grep")) then
      .tool_name
    else
      .tool_name + "(*)"
    end
  )
}' >> "$LOG_FILE" 2>/dev/null

echo '{"result":"continue"}'
