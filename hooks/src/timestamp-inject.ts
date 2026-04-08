/*!
 * Timestamp Injection Hook (UserPromptSubmit)
 *
 * Injects the current local time into every prompt as additionalContext,
 * enabling time-aware capabilities:
 * - Session pacing alerts
 * - Elapsed-time diagnostics
 * - Calendar awareness
 * - Rate-of-progress tracking
 */

import { readFileSync } from 'fs';

interface UserPromptSubmitInput {
  session_id: string;
  hook_event_name: string;
  prompt: string;
  cwd: string;
}

function readStdin(): string {
  return readFileSync(0, 'utf-8');
}

/**
 * Format the current time for Claude's context.
 * Includes ISO timestamp, local human-readable time, day of week, and timezone.
 */
function formatTimestamp(now: Date): string {
  const iso = now.toISOString();
  const local = now.toLocaleString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;

  return `Current time: ${local} (${tz}) | ISO: ${iso}`;
}

function main(): void {
  let input: UserPromptSubmitInput;
  try {
    input = JSON.parse(readStdin());
  } catch {
    // Malformed or empty stdin — no-op to avoid killing the hook
    return;
  }

  if (!input || typeof input !== 'object' || !input.hook_event_name) {
    return;
  }

  // Skip for subagents — they don't need timestamps
  if (process.env.CLAUDE_AGENT_ID) {
    return;
  }

  const now = new Date();
  const timestamp = formatTimestamp(now);

  console.log(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: 'UserPromptSubmit',
      additionalContext: timestamp,
    },
  }));
}

main();
