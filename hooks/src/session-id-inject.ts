/*!
 * SessionID Injection Hook (SessionStart)
 *
 * Injects the current session ID into every prompt as additionalContext,
 * enabling session-aware capabilities
 * in your hooks. This is especially useful for tracking and debugging
 * conversations across multiple interactions.
 */

import { readFileSync } from 'fs';


interface SessionStartInput {
  session_id: string;
  hook_event_name: string;
  prompt: string;
  cwd: string;
}

function readStdin(): string {
  return readFileSync(0, 'utf-8');
}

function main(): void {
  let input: SessionStartInput;
  try {
    input = JSON.parse(readStdin());
  } catch {
    // Malformed or empty stdin — no-op to avoid killing the hook
    return;
  }

  if (!input || typeof input !== 'object' || !input.session_id) {
    return;
  }

  // Skip for subagents — they don't need session IDs
  if (process.env.CLAUDE_AGENT_ID) {
    return;
  }

  let SessionId: string = `SessionId: ${input.session_id}`;
  console.log(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: 'SessionStart',
      additionalContext: SessionId,
    },
  }));
}

main();
