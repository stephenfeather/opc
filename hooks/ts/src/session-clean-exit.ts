//! @hook Stop @preserve
/**
 * Session Clean Exit Hook (SessionEnd)
 *
 * Marks the session as cleanly exited in PostgreSQL.
 * If this hook doesn't fire (crash/hang), the session remains without
 * an exited_at timestamp and session-crash-recovery.ts will detect it
 * on next startup.
 */

import { readFileSync } from 'fs';
import { markSessionExited } from './shared/db-utils-pg.js';

interface SessionEndInput {
  session_id: string;
  transcript_path: string;
  reason: 'clear' | 'logout' | 'prompt_input_exit' | 'other';
}

async function main() {
  const input: SessionEndInput = JSON.parse(readFileSync(0, 'utf-8'));

  // Mark session as cleanly exited in DB using Claude's session UUID
  markSessionExited(input.session_id);

  console.log(JSON.stringify({ result: 'continue' }));
}

main().catch(() => console.log(JSON.stringify({ result: 'continue' })));
