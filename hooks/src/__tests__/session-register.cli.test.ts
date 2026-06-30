/**
 * CLI behavior tests for the session-register SessionStart hook under the #265
 * backend gate. Exercises the built dist bundle end-to-end with the DB env
 * stripped, so the gate resolves to sqlite / misconfig deterministically (no
 * ambient DATABASE_URL leakage).
 */

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'child_process';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BIN = path.resolve(__dirname, '..', '..', 'dist', 'session-register.mjs');

const SESSION = 'abcdef12-3456-7890-abcd-ef1234567890';

function run(extraEnv: Record<string, string>) {
  return spawnSync('node', [BIN], {
    encoding: 'utf-8',
    input: JSON.stringify({ session_id: SESSION, transcript_path: '/tmp/none.jsonl' }),
    env: {
      ...process.env,
      CONTINUOUS_CLAUDE_DB_URL: '',
      DATABASE_URL: '',
      OPC_POSTGRES_URL: '',
      AGENTICA_MEMORY_BACKEND: '',
      ...extraEnv,
    },
  });
}

describe('session-register CLI under the #265 backend gate', () => {
  it('no-config (no URL, no backend var): continues and surfaces a user-facing no-URL note', () => {
    const res = run({});
    expect(res.status).toBe(0);
    const out = JSON.parse(res.stdout.trim());
    expect(out.result).toBe('continue');
    // Ambiguous case (maybe a lost DB URL) is surfaced in the awareness message.
    expect(out.message ?? '').toContain('no connection URL set');
    expect(out.message ?? '').not.toContain('PostgreSQL: unreachable');
    expect(out.message ?? '').not.toContain('misconfigured');
  });

  it('explicit sqlite: continues and stays fully silent about PostgreSQL', () => {
    const res = run({ AGENTICA_MEMORY_BACKEND: 'sqlite' });
    expect(res.status).toBe(0);
    const out = JSON.parse(res.stdout.trim());
    expect(out.result).toBe('continue');
    expect(out.message ?? '').not.toContain('PostgreSQL');
  });

  it('misconfig (postgres w/o URL): continues and surfaces a user-facing "misconfigured" warning', () => {
    const res = run({ AGENTICA_MEMORY_BACKEND: 'postgres' });
    expect(res.status).toBe(0);
    const out = JSON.parse(res.stdout.trim());
    expect(out.result).toBe('continue');
    expect(out.message ?? '').toContain('PostgreSQL: misconfigured');
    expect(out.message ?? '').not.toContain('PostgreSQL: unreachable');
  });
});
