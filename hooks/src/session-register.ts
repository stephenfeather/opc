/*!
 * SessionStart Hook - Registers session in coordination layer.
 *
 * This hook:
 * 1. Registers the session in PostgreSQL for cross-session awareness
 * 2. Injects session ID, memory system health, and pending tasks summary
 *
 * Peer session awareness is handled by peer-awareness.ts (UserPromptSubmit).
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { registerSession, isValidId } from './shared/db-utils-pg.js';
import { pgCoordinationStatus, backendExplicitlySet } from './shared/backend-resolution.js';
import { getProject } from './shared/session-id.js';
import { checkMemoryHealth, formatHealthWarnings, getPendingTasksSummary } from './session-context.js';
import type { SessionStartInput, HookOutput } from './shared/types.js';

/**
 * Main entry point for the SessionStart hook.
 * Registers the session and injects awareness message.
 */
export function main(): void {
  // Skip registration for daemon-spawned extraction sessions
  if (process.env.CLAUDE_MEMORY_EXTRACTION) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  // Read hook input from stdin
  let input: SessionStartInput;
  try {
    const stdinContent = readFileSync(0, 'utf-8');
    input = JSON.parse(stdinContent) as SessionStartInput;
  } catch {
    // If we can't read input, just continue silently
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  const sessionId = input.session_id;
  if (typeof sessionId !== 'string' || !isValidId(sessionId)) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  const project = getProject();
  const projectName = project.split('/').pop() || 'unknown';

  // Store session ID in environment for other hooks within this process
  process.env.COORDINATION_SESSION_ID = sessionId;

  // Backend gate (#265): only register in Postgres when PG is the active
  // backend. SessionStart is the right place to surface the configuration state
  // to the user (once per session, in the awareness message — not per-event
  // stderr). Four cases:
  //   - active (postgres)         -> register normally; normal health logic.
  //   - misconfig                 -> skip; user-facing "misconfigured" warning.
  //   - explicit sqlite           -> skip; silent (unambiguous operator intent).
  //   - no URL + no backend var   -> skip; user-facing note so a lost DB URL is
  //                                  distinguishable from intentional sqlite.
  const pgStatus = pgCoordinationStatus();
  let pgMessage: string | undefined;
  if (pgStatus.misconfig) {
    pgMessage = `PostgreSQL: misconfigured (${pgStatus.misconfig})`;
  } else if (!pgStatus.active && !backendExplicitlySet()) {
    pgMessage =
      'PostgreSQL: no connection URL set — cross-session coordination disabled. ' +
      'Set CONTINUOUS_CLAUDE_DB_URL to enable it, or AGENTICA_MEMORY_BACKEND=sqlite to silence this note.';
  }

  // Register session in PostgreSQL (include Claude's session ID, transcript path, and PID for crash recovery)
  // process.ppid is the Claude CLI process that spawned this hook
  const registerResult = pgStatus.active
    ? registerSession(sessionId, project, '', input.session_id, input.transcript_path, process.ppid)
    : { success: false };

  // Check memory system health. pgApplicable=false under a non-postgres backend
  // so a skipped registration is not reported as "PostgreSQL: unreachable";
  // pgMessage (when set) carries the misconfig/no-config line instead.
  const daemonPidPath = join(process.env.HOME || '/tmp', '.claude', 'memory-daemon.pid');
  const health = checkMemoryHealth(registerResult.success, daemonPidPath, pgStatus.active, pgMessage);
  const healthWarnings = formatHealthWarnings(health);

  // Check for pending tasks
  const tasksPath = join(project, 'thoughts', 'shared', 'Tasks.md');
  const tasksSummary = getPendingTasksSummary(tasksPath);

  // Build awareness message
  let awarenessMessage = `
<system-reminder>
Session: ${sessionId}
Project: ${projectName}`;

  if (healthWarnings) {
    awarenessMessage += `\n\n${healthWarnings}`;
  }

  if (tasksSummary) {
    awarenessMessage += `\n\n${tasksSummary}`;
  }

  awarenessMessage += `\n</system-reminder>`;

  // Output hook result with awareness injection
  const output: HookOutput = {
    result: 'continue',
    message: awarenessMessage,
  };

  console.log(JSON.stringify(output));
}

// Run if executed directly
main();
