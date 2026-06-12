#!/bin/bash
# Credential-leak hook dry-run harness.
#
# Pipes a battery of synthetic Claude Code hook payloads through each of
# the three layers and prints expected-vs-actual decisions. No side effects;
# safe to run at any time.
#
# Usage:
#   ./credential-leak-dryrun.sh           # run all suites
#   ./credential-leak-dryrun.sh bash      # just the Bash hook
#   ./credential-leak-dryrun.sh read      # just the Read hook
#   ./credential-leak-dryrun.sh prompt    # just the prompt hook

set -u

HOOKS_DIR="$(cd "$(dirname "$0")" && pwd)"
BASH_HOOK="$HOOKS_DIR/credential-leak-guard.sh"
READ_HOOK="node $HOOKS_DIR/dist/read-secret-block.mjs"
PROMPT_HOOK="node $HOOKS_DIR/dist/prompt-secret-block.mjs"

PASS=0
FAIL=0
FAILED_CASES=()

# decide_kind <output> -> echoes "ask"|"deny"|"block"|"allow"
decide_kind() {
    local out="$1"
    if [ -z "$out" ]; then echo "allow"; return; fi
    local d
    d=$(printf '%s' "$out" | jq -r '.hookSpecificOutput.permissionDecision // .decision // empty' 2>/dev/null)
    case "$d" in
        ask|deny|block|allow) echo "$d" ;;
        *) echo "malformed" ;;
    esac
}

run_case() {
    local hook_cmd="$1"
    local expected="$2"
    local label="$3"
    local payload="$4"

    local out
    out=$(printf '%s' "$payload" | eval "$hook_cmd")
    local actual
    actual=$(decide_kind "$out")

    if [ "$actual" = "$expected" ]; then
        PASS=$((PASS + 1))
        printf "  \033[32m✓\033[0m %-25s %s\n" "$label" "(decision=$actual)"
    else
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("$label: expected=$expected actual=$actual")
        printf "  \033[31m✗\033[0m %-25s expected=%s actual=%s\n" "$label" "$expected" "$actual"
        if [ -n "$out" ] && [ "$actual" = "malformed" ]; then
            printf "    raw: %s\n" "$out"
        fi
    fi
}

suite_bash() {
    echo
    echo "Layer 1: credential-leak-guard.sh (PreToolUse Bash, ask mode, pipe-aware)"

    # ASK cases — bare stdout-emitting credential commands
    run_case "$BASH_HOOK" ask  "bare gh auth token"        '{"tool_input":{"command":"gh auth token"}}'
    run_case "$BASH_HOOK" ask  "bare cat .env"             '{"tool_input":{"command":"cat .env"}}'
    run_case "$BASH_HOOK" ask  'echo $SECRET-shaped var'   '{"tool_input":{"command":"echo $OPENAI_API_KEY"}}'
    run_case "$BASH_HOOK" ask  "infisical --plain"         '{"tool_input":{"command":"infisical secrets get FOO --plain"}}'
    run_case "$BASH_HOOK" ask  "security find-generic -w"  '{"tool_input":{"command":"security find-generic-password -a foo -w"}}'
    run_case "$BASH_HOOK" ask  "1Password op read"         '{"tool_input":{"command":"op read op://vault/item/field"}}'
    run_case "$BASH_HOOK" ask  "aws configure get"         '{"tool_input":{"command":"aws configure get aws_secret_access_key"}}'
    run_case "$BASH_HOOK" ask  "logical-OR not pipe"       '{"tool_input":{"command":"gh auth token || true"}}'
    run_case "$BASH_HOOK" ask  "multi-segment 2nd bare"    '{"tool_input":{"command":"echo hi; gh auth token"}}'
    run_case "$BASH_HOOK" ask  "stderr-only redirect"     '{"tool_input":{"command":"gh auth token 2>/dev/null"}}'
    run_case "$BASH_HOOK" ask  "stderr append redirect"   '{"tool_input":{"command":"gh auth token 2>>err.log"}}'
    run_case "$BASH_HOOK" ask  "2>&1 merges to stdout"    '{"tool_input":{"command":"gh auth token 2>&1"}}'
    run_case "$BASH_HOOK" allow "stdout+stderr to file"   '{"tool_input":{"command":"gh auth token >/tmp/t 2>&1"}}'

    # ALLOW cases — captured / piped / redirected, or unrelated
    run_case "$BASH_HOOK" allow "captured \$( )"           '{"tool_input":{"command":"TOKEN=$(gh auth token)"}}'
    run_case "$BASH_HOOK" allow "captured backticks"       '{"tool_input":{"command":"TOKEN=`gh auth token`"}}'
    run_case "$BASH_HOOK" allow "piped to pbcopy"          '{"tool_input":{"command":"gh auth token | pbcopy"}}'
    run_case "$BASH_HOOK" allow "redirected to file"       '{"tool_input":{"command":"gh auth token > /tmp/t"}}'
    run_case "$BASH_HOOK" allow "cat .env piped to grep"   '{"tool_input":{"command":"cat .env.local | rg API"}}'
    run_case "$BASH_HOOK" allow "printenv piped grep"      '{"tool_input":{"command":"printenv | rg API"}}'
    run_case "$BASH_HOOK" allow "unrelated ls"             '{"tool_input":{"command":"ls"}}'
    run_case "$BASH_HOOK" allow "unrelated git status"     '{"tool_input":{"command":"git status"}}'
    run_case "$BASH_HOOK" allow "missing command"          '{}'
    run_case "$BASH_HOOK" ask  "cat doctl config"          '{"tool_input":{"command":"cat ~/.config/doctl/config.yaml"}}'
    run_case "$BASH_HOOK" ask  "bat gh hosts.yml"          '{"tool_input":{"command":"bat ~/.config/gh/hosts.yml"}}'
    run_case "$BASH_HOOK" allow "doctl config piped"       '{"tool_input":{"command":"cat ~/.config/doctl/config.yaml | rg token"}}'
}

suite_read() {
    echo
    echo "Layer 2: read-secret-block.ts (PreToolUse Read, deny mode)"

    run_case "$READ_HOOK" deny  ".env"                     '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/p/.env"}}'
    run_case "$READ_HOOK" deny  ".env.local"               '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/p/.env.local"}}'
    run_case "$READ_HOOK" deny  ".envrc"                   '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/p/.envrc"}}'
    run_case "$READ_HOOK" deny  ".netrc"                   '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/.netrc"}}'
    run_case "$READ_HOOK" deny  ".npmrc"                   '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/.npmrc"}}'
    run_case "$READ_HOOK" deny  "GCP ADC"                  '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/.config/gcloud/application_default_credentials.json"}}'
    run_case "$READ_HOOK" deny  "AWS credentials"          '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/.aws/credentials"}}'
    run_case "$READ_HOOK" deny  "SSH ed25519 key"          '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/.ssh/id_ed25519"}}'
    run_case "$READ_HOOK" deny  ".pem"                     '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/priv.pem"}}'
    run_case "$READ_HOOK" deny  ".docker/config.json"      '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/.docker/config.json"}}'

    run_case "$READ_HOOK" allow "tsconfig.json"            '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/p/tsconfig.json"}}'
    run_case "$READ_HOOK" allow "package.json"             '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/p/package.json"}}'
    run_case "$READ_HOOK" allow "generic app config.json"  '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/app/config.json"}}'
    run_case "$READ_HOOK" allow "README.md"                '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/README.md"}}'
    run_case "$READ_HOOK" allow "Edit tool (not Read)"     '{"tool_name":"Edit","tool_input":{"file_path":"/Users/x/p/.env"}}'
    run_case "$READ_HOOK" deny  "doctl config.yaml"        '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/.config/doctl/config.yaml"}}'
    run_case "$READ_HOOK" deny  "gh hosts.yml"             '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/.config/gh/hosts.yml"}}'
    run_case "$READ_HOOK" deny  "1Password op state"       '{"tool_name":"Read","tool_input":{"file_path":"/Users/x/.config/op/op.sqlite"}}'
}

suite_prompt() {
    local hex64="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    local resend_tail="AaBbCcDdEeFfGgHhIiJjKk1234567890"
    echo
    echo "Layer 3: prompt-secret-block.ts (UserPromptSubmit, block mode)"

    # 64-char hex token shape — vendor RULES generic-token rule
    run_case "$PROMPT_HOOK" block "OpenRouter key"         "{\"prompt\":\"key sk-or-v1-${hex64:0:52}\"}"
    run_case "$PROMPT_HOOK" block "Resend key"             "{\"prompt\":\"resend re_${resend_tail}\"}"
    run_case "$PROMPT_HOOK" block "AWS access key"         '{"prompt":"AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"}'
    run_case "$PROMPT_HOOK" block "DigitalOcean PAT"       "{\"prompt\":\"my DO token is dop_v1_${hex64} and works\"}"

    run_case "$PROMPT_HOOK" allow "plain English"          '{"prompt":"please refactor the login form to use a hook"}'
    run_case "$PROMPT_HOOK" allow "empty prompt"           '{"prompt":""}'
    run_case "$PROMPT_HOOK" allow "no prompt key"          '{}'
}

WHICH="${1:-all}"
case "$WHICH" in
    bash) suite_bash ;;
    read) suite_read ;;
    prompt) suite_prompt ;;
    all) suite_bash; suite_read; suite_prompt ;;
    *) echo "usage: $0 [bash|read|prompt|all]" >&2; exit 2 ;;
esac

echo
echo "─────────────────────────────────────────"
echo "Total: $((PASS + FAIL)) cases — $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    echo
    echo "Failures:"
    for line in "${FAILED_CASES[@]}"; do echo "  - $line"; done
    exit 1
fi
exit 0
