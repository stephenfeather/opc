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
  it('sqlite default: exits 0, continues, and does NOT warn "PostgreSQL: unreachable"', () => {
    const res = run({});
    expect(res.status).toBe(0);
    const out = JSON.parse(res.stdout.trim());
    expect(out.result).toBe('continue');
    expect(out.message ?? '').not.toContain('PostgreSQL: unreachable');
  });

  it('explicit sqlite: continues, no PG warning', () => {
    const res = run({ AGENTICA_MEMORY_BACKEND: 'sqlite' });
    expect(res.status).toBe(0);
    const out = JSON.parse(res.stdout.trim());
    expect(out.result).toBe('continue');
    expect(out.message ?? '').not.toContain('PostgreSQL: unreachable');
  });

  it('misconfig (postgres w/o URL): continues, emits a loud stderr diagnostic, no PG warning', () => {
    const res = run({ AGENTICA_MEMORY_BACKEND: 'postgres' });
    expect(res.status).toBe(0);
    const out = JSON.parse(res.stdout.trim());
    expect(out.result).toBe('continue');
    expect(out.message ?? '').not.toContain('PostgreSQL: unreachable');
    expect(res.stderr).toMatch(/no PostgreSQL connection URL/);
  });
});
