/**
 * Tests for the backend gate at the db-utils-pg chokepoints (issue #265).
 *
 * `runPgQuery` / `runPgQueryDetached` consult the shared backend resolver before
 * touching Postgres. When the backend is not postgres they no-op gracefully so
 * every consumer inherits the AGENTICA_MEMORY_BACKEND decision in one place.
 *
 * Env-leakage guard (#214 / pre-mortem tiger #1): the live env sets
 * DATABASE_URL, so each test strips ALL URL vars + AGENTICA_MEMORY_BACKEND and
 * sets only what it asserts. `vi.resetModules()` per test gives a fresh module
 * instance so the once-per-process stderr guard starts clean.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const ENV_KEYS = [
  'CONTINUOUS_CLAUDE_DB_URL',
  'DATABASE_URL',
  'OPC_POSTGRES_URL',
  'AGENTICA_MEMORY_BACKEND',
];

describe('db-utils-pg backend gate (#265)', () => {
  let saved: Record<string, string | undefined>;
  let stderrSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.resetModules();
    saved = {};
    for (const k of ENV_KEYS) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
    stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
  });

  afterEach(() => {
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
    vi.restoreAllMocks();
  });

  it('runPgQuery no-ops (success:false, no stderr) under the sqlite default', async () => {
    const { runPgQuery } = await import('../shared/db-utils-pg.js');
    const res = runPgQuery('print("x")');
    expect(res.success).toBe(false);
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it('runPgQuery no-ops (success:false, no stderr) under explicit sqlite', async () => {
    process.env.AGENTICA_MEMORY_BACKEND = 'sqlite';
    const { runPgQuery } = await import('../shared/db-utils-pg.js');
    const res = runPgQuery('print("x")');
    expect(res.success).toBe(false);
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it('runPgQuery returns success:false WITH a loud stderr on misconfig (postgres w/o URL)', async () => {
    process.env.AGENTICA_MEMORY_BACKEND = 'postgres';
    const { runPgQuery } = await import('../shared/db-utils-pg.js');
    const res = runPgQuery('print("x")');
    expect(res.success).toBe(false);
    expect(res.stderr).toMatch(/no PostgreSQL connection URL/);
    expect(stderrSpy).toHaveBeenCalledTimes(1);
  });

  it('redacts credentials in the misconfig diagnostic (return + stderr)', async () => {
    process.env.AGENTICA_MEMORY_BACKEND = 'postgres://user:supersecret@h/db';
    const { runPgQuery } = await import('../shared/db-utils-pg.js');
    const res = runPgQuery('print("x")');
    expect(res.success).toBe(false);
    expect(res.stderr).not.toContain('supersecret');
    const written = stderrSpy.mock.calls.map((c: unknown[]) => String(c[0])).join('');
    expect(written).not.toContain('supersecret');
    expect(written).toContain('://***@');
  });

  it('emits the misconfig diagnostic at most once per process', async () => {
    process.env.AGENTICA_MEMORY_BACKEND = 'postgres';
    const { runPgQuery } = await import('../shared/db-utils-pg.js');
    runPgQuery('print("x")');
    runPgQuery('print("x")');
    runPgQuery('print("x")');
    expect(stderrSpy).toHaveBeenCalledTimes(1);
  });

  it('runPgQueryDetached no-ops without throwing or writing stderr under sqlite', async () => {
    const { runPgQueryDetached } = await import('../shared/db-utils-pg.js');
    expect(() => runPgQueryDetached('print("x")')).not.toThrow();
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it('runPgQueryDetached emits the misconfig diagnostic once', async () => {
    process.env.AGENTICA_MEMORY_BACKEND = 'postgres';
    const { runPgQueryDetached } = await import('../shared/db-utils-pg.js');
    runPgQueryDetached('print("x")');
    runPgQueryDetached('print("x")');
    expect(stderrSpy).toHaveBeenCalledTimes(1);
  });
});
