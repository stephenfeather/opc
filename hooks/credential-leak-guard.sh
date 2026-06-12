#!/bin/bash
# Credential Leak Guard — PreToolUse on Bash
#
# Threat model: NOT exfiltration (that's security-guard.sh). This guards against
# the user/agent invoking stdout-emitting credential commands whose output
# Claude Code records verbatim in its JSONL transcript at
# ~/.claude/projects/.../<session>.jsonl.
#
# Decision: `permissionDecision: ask` — strict deny would get this hook disabled
# within a week. The ask prompt forces a conscious choice without breaking flow.
#
# Pipe-aware: flag ONLY when the risky command's stdout would land in the
# transcript. If it's captured ($(...) / `...`), piped (| cmd), or redirected
# (> file), the output never appears in the transcript, so don't flag.
#
# Pattern matching and capture detection live in Python to avoid the regex
# composition fragility of nested bash grep -E expressions.

set -u

# Read the hook payload from stdin and pass it to Python via env, so the
# heredoc below is free to feed the Python source on stdin.
PAYLOAD=$(cat -)
export PAYLOAD

exec python3 - <<'PY'
import json
import os
import re
import sys

# Each tuple: (pattern, description). Patterns are Python regex.
RISKY = [
    (r'\bgh\s+auth\s+token\b', 'gh auth token'),
    (r'\binfisical\s+secrets\s+get\b[^;&|]*--plain\b', 'infisical secrets get --plain'),
    (r'\binfisical\s+export\b', 'infisical export'),
    (r'\bcat\s+[^|>;&]*\.env(\.[A-Za-z0-9_.-]+)?\b', 'cat .env file'),
    (r'\bbat\s+[^|>;&]*\.env(\.[A-Za-z0-9_.-]+)?\b', 'bat .env file'),
    (r'\becho\s+["\']?\$\{?[A-Z][A-Z0-9_]*(TOKEN|KEY|SECRET|PASSWORD|PASS|API|CREDENTIAL)[A-Z0-9_]*\}?',
     'echo of credential-shaped env var'),
    (r'\bprintenv\b[^;&]*\|[^|]*\bgrep\b', 'printenv | grep'),
    (r'\bsecurity\s+find-generic-password\b[^;&|]*-w\b', 'security find-generic-password -w'),
    (r'\bsecurity\s+find-internet-password\b[^;&|]*-w\b', 'security find-internet-password -w'),
    (r'\bop\s+read\s+op://', '1Password op read op://'),
    (r'\bop\s+item\s+get\b[^;&|]*--fields\b', '1Password op item get --fields'),
    (r'\baws\s+configure\s+get\b', 'aws configure get'),
    (r'\bdoppler\s+secrets\s+get\b[^;&|]*--plain\b', 'doppler secrets get --plain'),
    (r'\bvault\s+kv\s+get\b', 'vault kv get'),
    # CLI credential-config reads: doctl / gh / 1Password op state directories.
    # Mirrors read-secret-block.ts for the Bash side. No `doctl auth token`
    # equivalent exists; the leak surface is the config file itself.
    (r'\b(cat|bat|less|more)\s+[^|>;&]*\.config/(doctl|gh|op)/',
     'read of CLI credential config (doctl/gh/op)'),
]

try:
    payload = json.loads(os.environ.get('PAYLOAD', '') or '{}')
except Exception:
    sys.exit(0)

cmd = (payload.get('tool_input') or {}).get('command') or ''
if not cmd:
    sys.exit(0)


def is_captured(match: re.Match, full_cmd: str) -> bool:
    """Return True if the matched region's stdout will NOT reach the transcript.

    Captured means: inside $(...) / `...`, piped to another command, or
    redirected to a file. We inspect the structure around the match by
    walking the string — no regex-on-regex composition.
    """
    start, end = match.span()

    # --- inside $(...) ? walk left counting parens, see if we hit unmatched '$('
    depth = 0
    i = start - 1
    while i >= 0:
        ch = full_cmd[i]
        if ch == ')':
            depth += 1
        elif ch == '(':
            if depth == 0:
                # unmatched '(' to our left — is it preceded by '$'?
                if i > 0 and full_cmd[i - 1] == '$':
                    return True
                break
            depth -= 1
        i -= 1

    # --- inside backticks ? count backticks to the left of start
    # An odd count means we're inside an open backtick span.
    if full_cmd[:start].count('`') % 2 == 1:
        return True

    # --- piped or redirected after the match, within same segment ---
    # Segment ends at unquoted ; && || or end-of-string. We walk char by char
    # tracking quote state, and look for a real pipe ('|' not part of '||')
    # or a redirect ('>' not part of '>=' or '2>&1' style — any '>' counts).
    in_single = False
    in_double = False
    i = end
    n = len(full_cmd)
    while i < n:
        ch = full_cmd[i]
        nxt = full_cmd[i + 1] if i + 1 < n else ''
        prv = full_cmd[i - 1] if i > 0 else ''
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == ';':
                return False
            if ch == '&' and nxt == '&':
                return False
            if ch == '|' and nxt == '|':
                i += 2
                continue
            if ch == '|' and prv != '|':
                return True
            if ch == '>':
                return True
        i += 1
    return False


matched = []
for pattern, description in RISKY:
    for m in re.finditer(pattern, cmd):
        if not is_captured(m, cmd):
            matched.append(description)
            break  # one hit per rule is enough

if not matched:
    sys.exit(0)

joined = '; '.join(matched)
reason = (
    "This command emits a credential to stdout, which Claude Code records "
    "verbatim in its session transcript (~/.claude/projects/.../<session>.jsonl). "
    f"Detected: {joined}. To run safely, capture the output (TOKEN=$(...)), "
    "pipe it (| pbcopy), or redirect it (> file). Approve only if you accept "
    "the leak risk."
)

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "ask",
        "permissionDecisionReason": reason,
    }
}))
PY
