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
import { updateHeartbeat, isValidId } from './shared/db-utils-pg.js';
import { readSessionId, getProject } from './shared/session-id.js';

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
    // stdin parse failure — fall back to persisted file
  }

  // Fall back to persisted session ID
  if (!sessionId) {
    const persisted = readSessionId();
    if (persisted && isValidId(persisted)) {
      sessionId = persisted;
    }
  }

  // If no session ID, just continue silently
  if (!sessionId) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  const project = getProject();

  // Fire and forget — never block on heartbeat failure
  updateHeartbeat(sessionId, project);

  console.log(JSON.stringify({ result: 'continue' }));
}

// Run if executed directly
main();
