/**
 * Unified backend / connection-URL resolution for the TypeScript hooks
 * (issue #265).
 *
 * Faithful port of `scripts/core/db/backend_resolution.py` so the hooks make
 * the SAME backend decision as the Python memory pipeline. Before this module,
 * `db-utils-pg.ts::getPgConnectionString()` resolved only the three URL vars and
 * ignored `AGENTICA_MEMORY_BACKEND` — so an operator who pinned
 * `AGENTICA_MEMORY_BACKEND=sqlite` got a daemon-vs-hooks split-brain (daemon
 * reads the SQLite `sessions` table while the hooks keep writing to Postgres).
 *
 * Precedence (documented once, here — mirrors the Python module):
 *
 *   - URL (`resolveUrl`): CONTINUOUS_CLAUDE_DB_URL (canonical) > DATABASE_URL
 *     (compat) > OPC_POSTGRES_URL (legacy). Empty / whitespace-only values are
 *     ignored; the returned URL is stripped. Returns null when none are set.
 *   - Backend (`resolveBackend`):
 *       1. An explicit, valid AGENTICA_MEMORY_BACKEND (case-insensitive) wins —
 *          an operator override. A non-empty invalid value, or `postgres` with
 *          no URL, THROWS (fail-fast, issue #214 parity) rather than silently
 *          falling through.
 *       2. Presence of any URL implies `postgres`.
 *       3. Otherwise the supplied `default` (`sqlite` unless overridden).
 *
 * The pure functions (`resolveUrl`, `resolveBackend`) read only their `env`
 * argument; `getConnectionUrl` / `getActiveBackend` / `pgCoordinationStatus`
 * bind them to `process.env`.
 */

/** Connection-URL env vars in priority order (canonical, compat, legacy). */
export const URL_VARS: readonly string[] = [
  'CONTINUOUS_CLAUDE_DB_URL',
  'DATABASE_URL',
  'OPC_POSTGRES_URL',
];

/** Backends an explicit AGENTICA_MEMORY_BACKEND override may name. */
export const VALID_BACKENDS: ReadonlySet<string> = new Set(['sqlite', 'postgres']);

/** Env var that lets an operator pin the backend regardless of URL presence. */
export const BACKEND_VAR = 'AGENTICA_MEMORY_BACKEND';

type Env = Record<string, string | undefined>;

/**
 * Return the connection URL by precedence, or null if unset.
 *
 * Empty or whitespace-only values are treated as unset, and the returned URL is
 * stripped of surrounding whitespace so a blank DSN never reaches the backend
 * (a connection string never carries meaningful leading/trailing whitespace).
 * This keeps the postgres-without-URL fail-fast in `resolveBackend` from being
 * bypassed by a templated "   " value (issue #214).
 *
 * Pure: reads only `env`.
 */
export function resolveUrl(env: Env): string | null {
  for (const varName of URL_VARS) {
    const value = env[varName];
    if (value && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

/**
 * Return the backend name ('sqlite' / 'postgres') for `env`.
 *
 * Precedence:
 *   1. An explicit, valid AGENTICA_MEMORY_BACKEND (case-insensitive).
 *   2. Presence of any connection URL implies 'postgres'.
 *   3. `default` ('sqlite' by default; pass null to signal "undetermined").
 *
 * Fail-fast on misconfiguration (issue #214). An explicit override is an
 * operator statement, so a broken one is a hard error rather than a silent
 * fall-through — regardless of `default`:
 *   - Finding 1: a non-empty AGENTICA_MEMORY_BACKEND that does not name a valid
 *     backend (e.g. the typo "sqllite") throws.
 *   - Finding 3: AGENTICA_MEMORY_BACKEND=postgres with no connection URL throws.
 *     Blank/whitespace-only values are treated as unset, not invalid.
 *
 * Pure: reads only `env`.
 *
 * @throws Error on an invalid override (Finding 1) or explicit postgres with no
 *   connection URL (Finding 3).
 */
export function resolveBackend(env: Env, defaultBackend: string | null = 'sqlite'): string | null {
  const raw = env[BACKEND_VAR] ?? '';
  const explicit = raw.trim().toLowerCase();
  if (explicit) {
    if (!VALID_BACKENDS.has(explicit)) {
      // Reflect the bad value to aid debugging, but harden it first: a backend
      // selector is a short token, so a long value is a misconfiguration (e.g.
      // a DSN pasted into the wrong var). Redact any `://user:pass@` credential
      // segment (mirroring artifact_index's URL redaction) and cap the length,
      // so a credential-bearing paste is not reflected back into stderr/logs
      // (issue #214 defense-in-depth).
      const redacted = raw.replace(/:\/\/[^@]+@/g, '://***@');
      const shown = redacted.length <= 32 ? redacted : redacted.slice(0, 32) + '…';
      throw new Error(
        `Invalid ${BACKEND_VAR}='${shown}': expected 'sqlite' or 'postgres' (case-insensitive).`,
      );
    }
    if (explicit === 'postgres' && resolveUrl(env) === null) {
      throw new Error(
        `${BACKEND_VAR}=postgres but no PostgreSQL connection URL is set; ` +
          `set one of ${URL_VARS.join(', ')}.`,
      );
    }
    return explicit;
  }
  if (resolveUrl(env) !== null) {
    return 'postgres';
  }
  return defaultBackend;
}

/** Resolve the connection URL from the live environment. */
export function getConnectionUrl(): string | null {
  return resolveUrl(process.env);
}

/** Resolve the active backend from the live environment. */
export function getActiveBackend(defaultBackend: string | null = 'sqlite'): string | null {
  return resolveBackend(process.env, defaultBackend);
}

/**
 * Status of the Postgres coordination layer for the given env — the single
 * function the hook chokepoints and session-register consume.
 *
 *   - active: true   → backend is postgres; PG reads/writes should proceed.
 *   - active: false (no misconfig) → backend is sqlite (explicit or the
 *       no-config default); PG operations should gracefully no-op.
 *   - active: false WITH misconfig → an operator misconfiguration
 *       (invalid value, or postgres-without-URL). The message is already
 *       credential-redacted by `resolveBackend`, so it is safe to log. Callers
 *       should surface it loudly (stderr) but never block — fail-loud, not
 *       fail-closed.
 */
export function pgCoordinationStatus(env: Env = process.env): { active: boolean; misconfig?: string } {
  try {
    return { active: resolveBackend(env) === 'postgres' };
  } catch (err) {
    return { active: false, misconfig: err instanceof Error ? err.message : String(err) };
  }
}
