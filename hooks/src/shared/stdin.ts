/**
 * Synchronous, EAGAIN-tolerant stdin reader for hooks (issue #291).
 *
 * `fs.readFileSync(0)` treats fd 0 as a regular file. Claude Code hands each
 * hook a pipe and writes the JSON payload into it *after* spawning; if the hook
 * starts reading before that write lands, a non-blocking pipe on Node 26
 * yields EAGAIN and readFileSync throws instead of waiting. Reproduced with a
 * parent that writes 200 ms after spawn: rc=1 `node:fs:539 EAGAIN`.
 *
 * This reader loops on readSync, sleeping briefly on EAGAIN until data or EOF
 * arrives, so every hook sees the complete payload regardless of scheduling.
 *
 * Semantics relative to readFileSync(0) are otherwise unchanged: the read
 * completes at EOF, which Claude Code produces by closing the hook's stdin
 * after writing the payload. On a *blocking* pipe readSync simply waits for
 * that EOF (as readFileSync did); the idle cap below only bounds the
 * non-blocking EAGAIN spin and is not a wall-clock deadline for the hook —
 * that remains Claude Code's per-hook timeout.
 */

import { fstatSync, readSync } from 'fs';

const CHUNK_SIZE = 64 * 1024;
const EAGAIN_SLEEP_MS = 5;
/**
 * Stop retrying EAGAIN after this long with no data at all (non-blocking pipes
 * only). Well under Claude Code's default 60s hook timeout; a reader that has
 * seen no byte in 10s is not going to get a payload.
 */
const DEFAULT_MAX_IDLE_MS = 10_000;
/**
 * Override for the idle cap (ms). The vitest config sets it to 0 so a hook
 * module that reads stdin at import time fails instantly under test, exactly
 * as readFileSync(0) used to, instead of waiting out the cap per module.
 *
 * Honoured only under a test runner (VITEST or NODE_ENV=test): several hooks
 * that read stdin are security guards which fail open when the read throws,
 * so an env-controlled way to force that throw must not exist in production
 * (aegis review of #291).
 */
export const MAX_IDLE_ENV = 'HOOK_STDIN_MAX_IDLE_MS';

export function isTestRunner(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.VITEST) || env.NODE_ENV === 'test';
}

function resolveMaxIdleMs(explicit: number | undefined): number {
  if (explicit !== undefined) return explicit;
  const fromEnv = process.env[MAX_IDLE_ENV];
  if (
    isTestRunner() &&
    fromEnv !== undefined &&
    fromEnv !== '' &&
    Number.isFinite(Number(fromEnv))
  ) {
    return Math.max(0, Number(fromEnv));
  }
  return DEFAULT_MAX_IDLE_MS;
}

function isTty(fd: number): boolean {
  try {
    return fstatSync(fd).isCharacterDevice();
  } catch {
    return false;
  }
}

const sleepCell = new Int32Array(new SharedArrayBuffer(4));

function sleepMs(ms: number): void {
  Atomics.wait(sleepCell, 0, 0, ms);
}

export interface ReadStdinOptions {
  /**
   * Max time to keep retrying EAGAIN with no data before giving up (default
   * 10 s). Applies to non-blocking pipes only; a blocking read waits for EOF.
   */
  maxIdleMs?: number;
  /** File descriptor to read (default 0). Exposed for tests. */
  fd?: number;
}

/**
 * Read stdin to EOF and return it as UTF-8.
 *
 * Returns "" if stdin is closed with no data. Throws only for non-EAGAIN
 * errors, or — on a non-blocking pipe — if no data arrives within `maxIdleMs`
 * (RangeError: stdin idle).
 */
export function readStdinSync(options: ReadStdinOptions = {}): string {
  const fd = options.fd ?? 0;
  const maxIdleMs = resolveMaxIdleMs(options.maxIdleMs);
  // A terminal is never a hook payload source; don't sit waiting on a human.
  if (isTty(fd)) {
    return '';
  }
  const chunks: Buffer[] = [];
  const buf = Buffer.alloc(CHUNK_SIZE);
  let idleMs = 0;

  for (;;) {
    let n: number;
    try {
      n = readSync(fd, buf, 0, buf.length, null);
    } catch (err) {
      const code = (err as NodeJS.ErrnoException)?.code;
      if (code === 'EAGAIN' || code === 'EWOULDBLOCK') {
        if (idleMs >= maxIdleMs) {
          throw new RangeError(`stdin idle for ${maxIdleMs} ms with no data`);
        }
        sleepMs(EAGAIN_SLEEP_MS);
        idleMs += EAGAIN_SLEEP_MS;
        continue;
      }
      if (code === 'EOF') {
        break; // Windows console quirk
      }
      throw err;
    }
    if (n === 0) {
      break;
    }
    idleMs = 0;
    chunks.push(Buffer.from(buf.subarray(0, n)));
  }

  return Buffer.concat(chunks).toString('utf-8');
}

/** Read stdin and JSON-parse it. Empty input parses as `{}`. */
export function readStdinJson<T = unknown>(options: ReadStdinOptions = {}): T {
  const raw = readStdinSync(options).trim();
  return (raw ? JSON.parse(raw) : {}) as T;
}
