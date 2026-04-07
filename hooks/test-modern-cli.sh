#!/bin/bash
# Test script for modern-cli-enforcer hook
HOOK="node /Users/stephenfeather/.claude/hooks/dist/modern-cli-enforcer.mjs"

run_test() {
  local label="$1"
  local json="$2"
  local result
  result=$(echo "$json" | $HOOK 2>&1)
  local decision=$(echo "$result" | jq -r '.hookSpecificOutput.permissionDecision // "allow"' 2>/dev/null)
  printf "%-45s -> %s\n" "$label" "$decision"
}

echo "=== Modern CLI Enforcer Tests ==="
echo ""

# Should DENY (first command, legacy)
run_test "grep as first cmd" '{"tool_name":"Bash","tool_input":{"command":"grep -r TODO src/"}}'
run_test "ls as first cmd" '{"tool_name":"Bash","tool_input":{"command":"ls -la /tmp"}}'
run_test "head as first cmd" '{"tool_name":"Bash","tool_input":{"command":"head -5 /tmp/file.txt"}}'
run_test "python3 as first cmd" '{"tool_name":"Bash","tool_input":{"command":"python3 script.py"}}'
run_test "cat as first cmd" '{"tool_name":"Bash","tool_input":{"command":"cat foo.txt"}}'
run_test "pgrep as first cmd" '{"tool_name":"Bash","tool_input":{"command":"pgrep -f node"}}'
run_test "find as first cmd" '{"tool_name":"Bash","tool_input":{"command":"find . -name foo"}}'

echo ""

# Should ALLOW (piped, pipeOk=true)
run_test "grep after pipe (pipeOk)" '{"tool_name":"Bash","tool_input":{"command":"docker ps | grep node"}}'
run_test "head after pipe (pipeOk)" '{"tool_name":"Bash","tool_input":{"command":"rg TODO -l | head -10"}}'
run_test "tail after pipe (pipeOk)" '{"tool_name":"Bash","tool_input":{"command":"rg TODO -l | tail -5"}}'
run_test "sed after pipe (pipeOk)" '{"tool_name":"Bash","tool_input":{"command":"rg TODO -l | sed s/foo/bar/"}}'
run_test "awk after pipe (pipeOk)" '{"tool_name":"Bash","tool_input":{"command":"procs | awk \"{print \\$1}\""}}'

echo ""

# Should DENY (piped, pipeOk=false)
run_test "pgrep after pipe (NOT pipeOk)" '{"tool_name":"Bash","tool_input":{"command":"echo test | pgrep node"}}'
run_test "python3 after pipe (NOT pipeOk)" '{"tool_name":"Bash","tool_input":{"command":"echo code | python3"}}'

echo ""

# Should ALLOW (modern tools)
run_test "eza (modern)" '{"tool_name":"Bash","tool_input":{"command":"eza -la /tmp"}}'
run_test "rg (modern)" '{"tool_name":"Bash","tool_input":{"command":"rg TODO src/"}}'
run_test "fd (modern)" '{"tool_name":"Bash","tool_input":{"command":"fd pattern src/"}}'
run_test "uv run python (modern)" '{"tool_name":"Bash","tool_input":{"command":"uv run python script.py"}}'
run_test "bat (modern)" '{"tool_name":"Bash","tool_input":{"command":"bat foo.txt"}}'
run_test "procs (modern)" '{"tool_name":"Bash","tool_input":{"command":"procs node"}}'
