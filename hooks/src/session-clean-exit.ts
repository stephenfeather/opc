/*!
 * Session Clean Exit Hook (SessionEnd)
 *
 * Marks the session as cleanly exited in PostgreSQL.
 * If this hook doesn't fire (crash/hang), the session remains without
 * an exited_at timestamp and session-crash-recovery.ts will detect it
 * on next startup.
 */

import { readStdinSync } from './shared/stdin.js';
import { markSessionExited } from './shared/db-utils-pg.js';

interface SessionEndInput {
  session_id: string;
  transcript_path: string;
  reason: 'clear' | 'logout' | 'prompt_input_exit' | 'other';
}

async function main() {
  let input: SessionEndInput;
  try {
    input = JSON.parse(readStdinSync());
  } catch {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  // Mark session as cleanly exited in DB using Claude's session UUID
  markSessionExited(input.session_id);

  console.log(JSON.stringify({ result: 'continue' }));
}

main().catch(() => console.log(JSON.stringify({ result: 'continue' })));
