//! @hook UserPromptSubmit @preserve
/**
 * UserPromptSubmit Hook - Peer session awareness with 60s file-based cache.
 *
 * Injects active peer sessions into context so Claude knows who else is working.
 * Silent when solo (no peers = no output). Uses file-based caching since each
 * hook invocation is a fresh Node.js process.
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { getActiveSessions } from './shared/db-utils-pg.js';
import { readSessionId, getProject } from './shared/session-id.js';
import { readPeerCache, writePeerCache, formatPeerMessage } from './session-context.js';

const CACHE_TTL_SECONDS = 60;

export function main(): void {
  // Skip for subagents
  if (process.env.CLAUDE_AGENT_ID) {
    console.log(JSON.stringify({}));
    return;
  }

  // Read stdin to get current session_id (authoritative for self-filtering)
  let ownSessionId: string | null = null;
  try {
    const stdinContent = readFileSync(0, 'utf-8');
    const input = JSON.parse(stdinContent);
    if (input && typeof input.session_id === 'string') {
      ownSessionId = input.session_id;
    }
  } catch {
    // stdin parse failure — fall back to persisted file
  }

  // Fall back to persisted session ID only if stdin didn't provide one
  if (!ownSessionId) {
    ownSessionId = readSessionId();
  }

  const project = getProject();
  const cachePath = join(process.env.HOME || '/tmp', '.claude', 'cache', 'peer-sessions.json');

  // Use full project path as cache key to prevent cross-project/worktree bleed
  // (basename alone would collide between repos or worktrees with the same name)
  let sessions = readPeerCache(cachePath, project, CACHE_TTL_SECONDS);

  if (sessions === null) {
    // Cache miss or stale — query DB
    const result = getActiveSessions(project);
    if (result.success) {
      sessions = result.sessions;
      writePeerCache(cachePath, project, sessions);
    } else {
      // DB unreachable — degrade gracefully, no output
      console.log(JSON.stringify({}));
      return;
    }
  }

  // Filter out own session
  const peers = sessions.filter(s => s.id !== ownSessionId);
  const message = formatPeerMessage(peers);

  if (!message) {
    console.log(JSON.stringify({}));
    return;
  }

  console.log(JSON.stringify({
    result: 'continue',
    hookSpecificOutput: {
      hookEventName: 'UserPromptSubmit',
      additionalContext: message,
    },
  }));
}

// Run if executed directly
main();
