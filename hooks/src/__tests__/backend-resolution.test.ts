/**
 * Tests for the TS backend/URL resolver (issue #265).
 *
 * Faithful port of the Python resolver test surface
 * (scripts/core/db/backend_resolution.py). The pure functions take an explicit
 * `env` mapping and read nothing from process.env, so every case below passes
 * its own env object — NEVER relying on ambient DATABASE_URL (the #214
 * false-green trap).
 */

import { describe, it, expect } from 'vitest';

const MOD = '../shared/backend-resolution.js';

describe('resolveUrl', () => {
  it('returns null when no URL var is set', async () => {
    const { resolveUrl } = await import(MOD);
    expect(resolveUrl({})).toBeNull();
  });

  it('honors precedence: CONTINUOUS_CLAUDE_DB_URL > DATABASE_URL > OPC_POSTGRES_URL', async () => {
    const { resolveUrl } = await import(MOD);
    expect(
      resolveUrl({
        CONTINUOUS_CLAUDE_DB_URL: 'postgres://canonical',
        DATABASE_URL: 'postgres://compat',
        OPC_POSTGRES_URL: 'postgres://legacy',
      }),
    ).toBe('postgres://canonical');
    expect(
      resolveUrl({ DATABASE_URL: 'postgres://compat', OPC_POSTGRES_URL: 'postgres://legacy' }),
    ).toBe('postgres://compat');
    expect(resolveUrl({ OPC_POSTGRES_URL: 'postgres://legacy' })).toBe('postgres://legacy');
  });

  it('treats empty and whitespace-only values as unset', async () => {
    const { resolveUrl } = await import(MOD);
    expect(resolveUrl({ CONTINUOUS_CLAUDE_DB_URL: '' })).toBeNull();
    expect(resolveUrl({ CONTINUOUS_CLAUDE_DB_URL: '   ' })).toBeNull();
    // a blank canonical falls through to the next var
    expect(
      resolveUrl({ CONTINUOUS_CLAUDE_DB_URL: '  ', DATABASE_URL: 'postgres://compat' }),
    ).toBe('postgres://compat');
  });

  it('strips surrounding whitespace from the returned URL', async () => {
    const { resolveUrl } = await import(MOD);
    expect(resolveUrl({ CONTINUOUS_CLAUDE_DB_URL: '  postgres://x  ' })).toBe('postgres://x');
  });
});

describe('resolveBackend', () => {
  it('defaults to sqlite when nothing is set', async () => {
    const { resolveBackend } = await import(MOD);
    expect(resolveBackend({})).toBe('sqlite');
  });

  it('honors an explicit default override', async () => {
    const { resolveBackend } = await import(MOD);
    expect(resolveBackend({}, null)).toBeNull();
    expect(resolveBackend({}, 'postgres')).toBe('postgres');
  });

  it('implies postgres when any URL is present and no backend var is set', async () => {
    const { resolveBackend } = await import(MOD);
    expect(resolveBackend({ DATABASE_URL: 'postgres://x' })).toBe('postgres');
  });

  it('honors a valid explicit override (case-insensitive)', async () => {
    const { resolveBackend } = await import(MOD);
    expect(resolveBackend({ AGENTICA_MEMORY_BACKEND: 'sqlite' })).toBe('sqlite');
    expect(resolveBackend({ AGENTICA_MEMORY_BACKEND: 'POSTGRES', DATABASE_URL: 'postgres://x' })).toBe('postgres');
    expect(resolveBackend({ AGENTICA_MEMORY_BACKEND: '  SqLiTe  ' })).toBe('sqlite');
  });

  it('explicit sqlite override wins even when a URL is present', async () => {
    const { resolveBackend } = await import(MOD);
    expect(
      resolveBackend({ AGENTICA_MEMORY_BACKEND: 'sqlite', DATABASE_URL: 'postgres://x' }),
    ).toBe('sqlite');
  });

  it('treats blank/whitespace-only backend var as unset (not invalid)', async () => {
    const { resolveBackend } = await import(MOD);
    expect(resolveBackend({ AGENTICA_MEMORY_BACKEND: '' })).toBe('sqlite');
    expect(resolveBackend({ AGENTICA_MEMORY_BACKEND: '   ' })).toBe('sqlite');
    expect(resolveBackend({ AGENTICA_MEMORY_BACKEND: '  ', DATABASE_URL: 'postgres://x' })).toBe('postgres');
  });

  it('throws on a non-empty invalid override (Finding 1)', async () => {
    const { resolveBackend } = await import(MOD);
    expect(() => resolveBackend({ AGENTICA_MEMORY_BACKEND: 'sqllite' })).toThrow(/Invalid AGENTICA_MEMORY_BACKEND/);
    expect(() => resolveBackend({ AGENTICA_MEMORY_BACKEND: 'sqllite' })).toThrow(/sqlite.*postgres/);
  });

  it('throws on postgres-without-URL (Finding 3)', async () => {
    const { resolveBackend } = await import(MOD);
    expect(() => resolveBackend({ AGENTICA_MEMORY_BACKEND: 'postgres' })).toThrow(/no PostgreSQL connection URL/);
    // a whitespace-only URL does NOT satisfy the requirement
    expect(() =>
      resolveBackend({ AGENTICA_MEMORY_BACKEND: 'postgres', DATABASE_URL: '   ' }),
    ).toThrow(/no PostgreSQL connection URL/);
  });

  it('echoes a safe-shaped typo to aid debugging', async () => {
    const { resolveBackend } = await import(MOD);
    let msg = '';
    try {
      resolveBackend({ AGENTICA_MEMORY_BACKEND: 'sqllite' });
    } catch (e) {
      msg = e instanceof Error ? e.message : String(e);
    }
    expect(msg).toContain("'sqllite'"); // short alnum token is safe to reflect
  });

  it('never reflects credential-shaped values verbatim — DSN with a scheme (#265 r3)', async () => {
    const { resolveBackend } = await import(MOD);
    let msg = '';
    try {
      resolveBackend({ AGENTICA_MEMORY_BACKEND: 'postgres://user:supersecret@host.example.com:5432/mydb' });
    } catch (e) {
      msg = e instanceof Error ? e.message : String(e);
    }
    expect(msg).toMatch(/Invalid AGENTICA_MEMORY_BACKEND/);
    expect(msg).toContain('<redacted non-token value>');
    expect(msg).not.toContain('supersecret');
    expect(msg).not.toContain('user');
    expect(msg).not.toContain('host');
  });

  it('does not leak a password containing @ (the first-@ redaction blind spot) (#265 r3)', async () => {
    const { resolveBackend } = await import(MOD);
    let msg = '';
    try {
      resolveBackend({ AGENTICA_MEMORY_BACKEND: 'postgres://user:p@ssword@host/db' });
    } catch (e) {
      msg = e instanceof Error ? e.message : String(e);
    }
    expect(msg).toContain('<redacted non-token value>');
    expect(msg).not.toContain('ssword'); // the fragment the old regex leaked
    expect(msg).not.toContain('p@ssword');
  });

  it('does not leak a no-scheme credential string (#265 r3)', async () => {
    const { resolveBackend } = await import(MOD);
    let msg = '';
    try {
      resolveBackend({ AGENTICA_MEMORY_BACKEND: 'user:supersecret@host/db' });
    } catch (e) {
      msg = e instanceof Error ? e.message : String(e);
    }
    expect(msg).toContain('<redacted non-token value>');
    expect(msg).not.toContain('supersecret');
  });
});

describe('backendExplicitlySet', () => {
  it('is true only when AGENTICA_MEMORY_BACKEND is a non-blank value', async () => {
    const { backendExplicitlySet } = await import(MOD);
    expect(backendExplicitlySet({})).toBe(false);
    expect(backendExplicitlySet({ AGENTICA_MEMORY_BACKEND: '' })).toBe(false);
    expect(backendExplicitlySet({ AGENTICA_MEMORY_BACKEND: '   ' })).toBe(false);
    expect(backendExplicitlySet({ AGENTICA_MEMORY_BACKEND: 'sqlite' })).toBe(true);
    // even an invalid value counts as "explicitly set" (it's an operator statement)
    expect(backendExplicitlySet({ AGENTICA_MEMORY_BACKEND: 'sqllite' })).toBe(true);
  });
});

describe('pgCoordinationStatus', () => {
  it('active=true only when backend resolves to postgres', async () => {
    const { pgCoordinationStatus } = await import(MOD);
    expect(pgCoordinationStatus({ DATABASE_URL: 'postgres://x' })).toEqual({ active: true });
    expect(pgCoordinationStatus({ AGENTICA_MEMORY_BACKEND: 'postgres', DATABASE_URL: 'postgres://x' })).toEqual({ active: true });
  });

  it('active=false with no misconfig for the sqlite default / explicit sqlite', async () => {
    const { pgCoordinationStatus } = await import(MOD);
    expect(pgCoordinationStatus({})).toEqual({ active: false });
    expect(pgCoordinationStatus({ AGENTICA_MEMORY_BACKEND: 'sqlite' })).toEqual({ active: false });
  });

  it('active=false WITH a misconfig message on invalid value or postgres-without-URL', async () => {
    const { pgCoordinationStatus } = await import(MOD);
    const invalid = pgCoordinationStatus({ AGENTICA_MEMORY_BACKEND: 'sqllite' });
    expect(invalid.active).toBe(false);
    expect(invalid.misconfig).toMatch(/Invalid AGENTICA_MEMORY_BACKEND/);

    const noUrl = pgCoordinationStatus({ AGENTICA_MEMORY_BACKEND: 'postgres' });
    expect(noUrl.active).toBe(false);
    expect(noUrl.misconfig).toMatch(/no PostgreSQL connection URL/);
  });

  it('redacts credentials in the misconfig message', async () => {
    const { pgCoordinationStatus } = await import(MOD);
    const r = pgCoordinationStatus({ AGENTICA_MEMORY_BACKEND: 'postgres://user:supersecret@h/db' });
    expect(r.active).toBe(false);
    expect(r.misconfig).not.toContain('supersecret');
  });
});
