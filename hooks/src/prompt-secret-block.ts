/*!
 * Prompt Secret Block — UserPromptSubmit
 *
 * Scans the user's prompt with scanAll() (vendor RULES + LOCAL_RULES) and
 * hard-blocks submission if any credential-category finding is detected.
 * The prompt never enters the Claude Code transcript when blocked.
 *
 * Scope: blocks only `category: "secret"` findings. PII findings (names,
 * emails, phone numbers) from the vendor PII rules would produce too many
 * false positives on legitimate prompts — that's a separate guardrail with
 * different ergonomics. Add a `prompt-pii-block.ts` if PII blocking is
 * later desired.
 *
 * Upstream sensitive-canary supports inline allow tags like [allow-secret];
 * we omit that machinery here. If a legitimate secret-shaped value needs
 * to be sent (rare), the user can rephrase or paste it from a tool call
 * after the secret has been used. Keeping the surface small reduces the
 * chance the bypass is misused.
 */

import { readFileSync } from 'fs';
import { scanAll } from './credential-scanner/scan-all.ts';
import type { Finding } from './credential-scanner/rules.ts';

interface HookInput {
  session_id?: string;
  hook_event_name?: string;
  prompt?: string;
}

interface BlockOutput {
  decision: 'block';
  reason: string;
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

  const prompt = input.prompt ?? '';
  if (!prompt) process.exit(0);

  const findings: Finding[] = scanAll(prompt).filter(
    (f: Finding) => f.category === 'secret',
  );
  if (findings.length === 0) process.exit(0);

  // Dedupe by ruleId+redacted form so a repeated key doesn't spam the reason.
  const seen = new Set<string>();
  const unique = findings.filter((f: Finding) => {
    const key = `${f.ruleId}:${f.matchRedacted}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  const lines = unique.map(
    (f: Finding) => `  • ${f.description} (${f.ruleId}): ${f.matchRedacted}`,
  );

  const reason =
    'Your prompt contains values that look like live credentials. Blocking ' +
    'submission to keep them out of the Claude Code transcript.\n\n' +
    'Detected:\n' +
    lines.join('\n') +
    '\n\nWhat to do:\n' +
    '  1. Remove or redact the values above and resubmit.\n' +
    '  2. If a redacted excerpt is enough, replace the secret with ' +
    '[REDACTED] and describe its role.\n' +
    '  3. If this is a false positive (e.g. an example value from public docs), ' +
    'rephrase so the value is wrapped in obvious markers and resubmit.';

  const output: BlockOutput = { decision: 'block', reason };
  process.stdout.write(JSON.stringify(output) + '\n');
  process.exit(0);
}

main();
