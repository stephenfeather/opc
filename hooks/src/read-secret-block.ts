/*!
 * Read Secret Block — PreToolUse on Read
 *
 * Hard-blocks the Read tool from ingesting credential files. Reading them
 * lands their contents in the Claude Code JSONL transcript verbatim; once
 * there, the only mitigation is the Layer-3 scrubber. Prevention is cheaper.
 *
 * Targets the Read tool specifically (not Bash) — more precise than the
 * upstream blog's bash-only approach for `cat .env`. The Bash side is
 * covered by credential-leak-guard.sh (ask mode) and security-guard.sh
 * (deny on exfil patterns); Read-tool reads bypass both.
 *
 * Decision: `deny`. Unlike the Bash credential commands, there is no
 * legitimate workflow for the Read tool to ingest .env / .netrc / GCP ADC
 * during a Claude Code session — by the time those files matter, the user
 * is already aware of them and can paste a redacted excerpt manually.
 */

import { readFileSync } from 'fs';
import { basename } from 'path';

interface HookInput {
  session_id?: string;
  hook_event_name?: string;
  tool_name?: string;
  tool_input?: {
    file_path?: string;
  };
}

interface HookOutput {
  hookSpecificOutput: {
    hookEventName: 'PreToolUse';
    permissionDecision: 'deny';
    permissionDecisionReason: string;
  };
}

// Each rule: regex matched against both basename and full path.
// Order matters only for the human-readable description picked for the
// reason string — the first match wins.
const SECRET_FILE_RULES: { pattern: RegExp; description: string }[] = [
  { pattern: /(^|\/)\.env(\.[A-Za-z0-9_.-]+)?$/, description: '.env file' },
  { pattern: /(^|\/)\.envrc$/, description: '.envrc (direnv)' },
  { pattern: /(^|\/)\.netrc$/, description: '.netrc' },
  { pattern: /(^|\/)\.npmrc$/, description: '.npmrc (may contain auth tokens)' },
  { pattern: /(^|\/)\.pypirc$/, description: '.pypirc (PyPI credentials)' },
  { pattern: /(^|\/)\.dockercfg$/, description: 'docker config (legacy auth)' },
  { pattern: /(^|\/)config\.json$/i, description: 'docker/k8s config.json (auth blob)' },
  { pattern: /(^|\/)application_default_credentials\.json$/, description: 'GCP Application Default Credentials' },
  { pattern: /(^|\/)credentials(\.[A-Za-z0-9_.-]+)?$/, description: 'credentials file (AWS / generic)' },
  { pattern: /(^|\/)id_(rsa|ecdsa|ed25519|dsa)$/, description: 'SSH private key' },
  { pattern: /\.pem$/, description: 'PEM key/cert' },
  { pattern: /\.p12$/, description: 'PKCS#12 keystore' },
  { pattern: /\.pfx$/, description: 'PFX keystore' },
  { pattern: /\.key$/, description: 'private key file' },
  { pattern: /(^|\/)service-account.*\.json$/i, description: 'GCP service-account key' },
  { pattern: /(^|\/)gcloud\/(legacy_credentials|access_tokens\.db)/, description: 'gcloud credential store' },
  { pattern: /(^|\/)\.aws\/credentials$/, description: 'AWS credentials' },
  { pattern: /(^|\/)\.kube\/config$/, description: 'Kubernetes kubeconfig' },
  { pattern: /(^|\/)\.config\/doctl\/config\.yaml$/, description: 'doctl config (DigitalOcean access token)' },
  { pattern: /(^|\/)\.config\/gh\/hosts\.yml$/, description: 'gh CLI hosts.yml (GitHub OAuth token)' },
  { pattern: /(^|\/)\.config\/op\//, description: '1Password CLI state' },
];

// docker/k8s config.json is denylisted, but countless harmless project
// config.json files exist (tsconfig, package, etc.). Only block when the
// path actually points at a credential-store location.
const CONFIG_JSON_DIRS = [
  /\.docker\//,
  /\.kube\//,
];

function classify(filePath: string): { description: string } | null {
  const name = basename(filePath);
  for (const rule of SECRET_FILE_RULES) {
    if (rule.pattern.test(filePath) || rule.pattern.test(name)) {
      // Special-case: config.json — only credential-store paths
      if (rule.description.startsWith('docker/k8s config.json')) {
        if (!CONFIG_JSON_DIRS.some((d) => d.test(filePath))) continue;
      }
      return { description: rule.description };
    }
  }
  return null;
}

function readStdin(): string {
  return readFileSync(0, 'utf-8');
}

function main(): void {
  let input: HookInput;
  try {
    input = JSON.parse(readStdin()) as HookInput;
  } catch {
    process.exit(0);
  }

  if (input.tool_name !== 'Read') process.exit(0);
  const filePath = input.tool_input?.file_path;
  if (!filePath) process.exit(0);

  const hit = classify(filePath);
  if (!hit) process.exit(0);

  const output: HookOutput = {
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      permissionDecision: 'deny',
      permissionDecisionReason:
        `Refusing to read ${hit.description} (${filePath}). Reading credential ` +
        `files writes their contents verbatim into the Claude Code session ` +
        `transcript (~/.claude/projects/.../<session>.jsonl). If you need a ` +
        `specific value, paste a redacted excerpt manually, or use ` +
        `\`grep <KEY> ${filePath}\` from Bash to extract just one line.`,
    },
  };

  process.stdout.write(JSON.stringify(output) + '\n');
  process.exit(0);
}

main();
