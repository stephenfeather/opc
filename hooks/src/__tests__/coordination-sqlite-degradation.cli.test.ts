/**
 * End-to-end degradation tests (#265): coordination hooks that inherit the
 * db-utils-pg backend gate must no-op gracefully under a sqlite backend — emit
 * a valid {result:'continue'}, never crash, never block a tool. Exercises the
 * built dist bundles with the DB env stripped (no ambient-URL dependency).
 */

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'child_process';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(__dirname, '..', '..', 'dist');
const SESSION = 'abcdef12-3456-7890-abcd-ef1234567890';

function run(bin: string, payload: object) {
  return spawnSync('node', [path.join(DIST, bin)], {
    encoding: 'utf-8',
    input: JSON.stringify(payload),
    env: {
      ...process.env,
      CONTINUOUS_CLAUDE_DB_URL: '',
      DATABASE_URL: '',
      OPC_POSTGRES_URL: '',
      AGENTICA_MEMORY_BACKEND: 'sqlite',
    },
  });
}

describe('coordination hooks degrade cleanly under sqlite (#265)', () => {
  it('heartbeat: exits 0 and continues without spawning a doomed PG write', () => {
    const res = run('heartbeat.mjs', { session_id: SESSION });
    expect(res.status).toBe(0);
    expect(JSON.parse(res.stdout.trim()).result).toBe('continue');
  });

  it('file-claims (PreToolUse:Edit): exits 0 and continues — never blocks the edit', () => {
    const res = run('file-claims.mjs', {
      session_id: SESSION,
      tool_name: 'Edit',
      tool_input: { file_path: '/tmp/example.ts' },
    });
    expect(res.status).toBe(0);
    const out = JSON.parse(res.stdout.trim());
    expect(out.result).toBe('continue');
    // A PreToolUse hook must not deny/block under sqlite
    expect(out.result).not.toBe('block');
    expect(out.decision).not.toBe('block');
  });
});
