/**
 * PostToolUse Hook - Refreshes session heartbeat on every tool use.
 *
 * Keeps the session alive in the coordination layer by updating
 * last_heartbeat in PostgreSQL. Without this, sessions age out
 * after 5 minutes (the getActiveSessions cutoff).
 *
 * Always outputs { result: "continue" } — never blocks tool execution.
 */

import { readFileSync } from 'fs';
import { updateHeartbeatDetached, isValidId } from './shared/db-utils-pg.js';
import { getProject } from './shared/session-id.js';

export function main(): void {
  // Try to get session ID from stdin (PostToolUse input)
  let sessionId: string | null = null;
  try {
    const stdinContent = readFileSync(0, 'utf-8');
    const input = JSON.parse(stdinContent);
    if (input && typeof input.session_id === 'string' && isValidId(input.session_id)) {
      sessionId = input.session_id;
    }
  } catch {
    // stdin parse failure — continue silently below
  }

  // If no session ID, just continue silently
  if (!sessionId) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  const project = getProject();

  // Truly fire-and-forget — spawns a detached process and unrefs it
  // so the parent exits immediately without waiting for the DB update.
  updateHeartbeatDetached(sessionId, project);

  console.log(JSON.stringify({ result: 'continue' }));
}

// Run if executed directly
if (typeof process !== 'undefined' && process.argv[1] && (process.argv[1].endsWith('heartbeat.ts') || process.argv[1].endsWith('heartbeat.js') || process.argv[1].endsWith('heartbeat.mjs'))) {
  main();
}
