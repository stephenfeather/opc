/**
 * Tests for shared/stdin.ts (issue #291).
 *
 * The EAGAIN race only shows up across a real process boundary, so the helper
 * is bundled with esbuild into a temp fixture and spawned under the three
 * conditions from the original repro: payload already in the pipe, payload
 * written after the child starts (the failing case for readFileSync(0)), and
 * an inherited stdin. (Whether readFileSync(0) fails on the late write depends
 * on how the parent created the pipe — it does under Claude Code and a plain
 * Node spawn, not under vitest's worker — so that is deliberately not asserted.)
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { execSync, spawn, spawnSync } from 'child_process';
import { buildSync } from 'esbuild';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { MAX_IDLE_ENV, readStdinJson, readStdinSync } from '../shared/stdin.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const HELPER_SRC = path.join(HERE, '..', 'shared', 'stdin.ts');

let tmp: string;
let fixtureHelper: string; // echoes stdin via readStdinSync

function runLateWrite(script: string, payload: string, delayMs = 200): Promise<{ code: number | null; out: string; err: string }> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [script], { stdio: ['pipe', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => (out += d));
    child.stderr.on('data', (d) => (err += d));
    setTimeout(() => {
      child.stdin.write(payload);
      child.stdin.end();
    }, delayMs);
    child.on('close', (code) => resolve({ code, out, err }));
  });
}

beforeAll(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'stdin-291-'));
  const helperEntry = path.join(tmp, 'helper-entry.ts');
  fs.writeFileSync(
    helperEntry,
    `import { readStdinSync } from ${JSON.stringify(HELPER_SRC)};\nprocess.stdout.write(readStdinSync());\n`,
  );
  fixtureHelper = path.join(tmp, 'helper.mjs');
  buildSync({ entryPoints: [helperEntry], bundle: true, platform: 'node', format: 'esm', outfile: fixtureHelper, logLevel: 'silent' });
});

afterAll(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

describe('readStdinSync across a process boundary', () => {
  const payload = JSON.stringify({ session_id: 'abc', tool_name: 'Write', n: 1 });

  it('reads a payload that is already in the pipe', () => {
    const r = spawnSync(process.execPath, [fixtureHelper], { input: payload, encoding: 'utf-8' });
    expect(r.status).toBe(0);
    expect(r.stdout).toBe(payload);
  });

  it('waits for a payload written after the child started (the #291 race)', async () => {
    const r = await runLateWrite(fixtureHelper, payload);
    expect(r.err).toBe('');
    expect(r.code).toBe(0);
    expect(r.out).toBe(payload);
  });

  it('still waits for a late payload when the cap is generous', async () => {
    const r = await new Promise<{ code: number | null; out: string }>((resolve) => {
      const child = spawn(process.execPath, [fixtureHelper], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, HOOK_STDIN_MAX_IDLE_MS: '5000' },
      });
      let out = '';
      child.stdout.on('data', (d) => (out += d));
      setTimeout(() => {
        child.stdin.write(payload);
        child.stdin.end();
      }, 300);
      child.on('close', (code) => resolve({ code, out }));
    });
    expect(r.code).toBe(0);
    expect(r.out).toBe(payload);
  });

  it('returns "" on an immediately closed stdin', () => {
    const r = spawnSync(process.execPath, [fixtureHelper], { input: '', encoding: 'utf-8' });
    expect(r.status).toBe(0);
    expect(r.stdout).toBe('');
  });

  it('handles payloads larger than one read chunk', () => {
    const big = 'x'.repeat(200 * 1024);
    const r = spawnSync(process.execPath, [fixtureHelper], { input: big, encoding: 'utf-8' });
    expect(r.status).toBe(0);
    expect(r.stdout.length).toBe(big.length);
  });
});

describe('readStdinSync / readStdinJson in-process (fd injected)', () => {
  function withFile(content: string, fn: (fd: number) => void) {
    const p = path.join(tmp, `in-${Math.random().toString(36).slice(2)}.txt`);
    fs.writeFileSync(p, content);
    const fd = fs.openSync(p, 'r');
    try {
      fn(fd);
    } finally {
      fs.closeSync(fd);
    }
  }

  it('reads a regular file to EOF', () => {
    withFile('hello', (fd) => expect(readStdinSync({ fd })).toBe('hello'));
  });

  it('parses JSON and treats empty input as {}', () => {
    withFile('{"a":1}', (fd) => expect(readStdinJson<{ a: number }>({ fd })).toEqual({ a: 1 }));
    withFile('  \n', (fd) => expect(readStdinJson({ fd })).toEqual({}));
  });

  it('rejects malformed JSON loudly', () => {
    withFile('{nope', (fd) => expect(() => readStdinJson({ fd })).toThrow(SyntaxError));
  });

  it('propagates non-EAGAIN errors', () => {
    expect(() => readStdinSync({ fd: 987654 })).toThrow(/EBADF/);
  });
});

describe('EAGAIN handling on a non-blocking pipe (deterministic via FIFO)', () => {
  // A FIFO opened O_NONBLOCK with a writer attached but silent yields EAGAIN on
  // read — exactly the state a hook is in when Claude Code has spawned it but
  // not yet written the payload. No process boundary, no scheduling luck.
  let fifo: string;
  let readerFd: number;
  let writerFd: number;

  beforeAll(() => {
    fifo = path.join(tmp, 'pipe.fifo');
    execSync(`mkfifo ${JSON.stringify(fifo)}`);
    readerFd = fs.openSync(fifo, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
    writerFd = fs.openSync(fifo, fs.constants.O_WRONLY | fs.constants.O_NONBLOCK);
  });

  afterAll(() => {
    for (const fd of [writerFd, readerFd]) {
      try {
        fs.closeSync(fd);
      } catch {
        /* already closed */
      }
    }
  });

  it('the raw read really is EAGAIN while the writer is silent', () => {
    expect(() => fs.readSync(readerFd, Buffer.alloc(8), 0, 8, null)).toThrow(/EAGAIN/);
  });

  it('times out with a clear error when the cap elapses with no data', () => {
    const started = Date.now();
    expect(() => readStdinSync({ fd: readerFd, maxIdleMs: 40 })).toThrow(/stdin idle for 40 ms/);
    expect(Date.now() - started).toBeLessThan(1000);
  });

  it('honours HOOK_STDIN_MAX_IDLE_MS when no explicit cap is given', () => {
    const prev = process.env[MAX_IDLE_ENV];
    process.env[MAX_IDLE_ENV] = '0';
    try {
      expect(() => readStdinSync({ fd: readerFd })).toThrow(/stdin idle for 0 ms/);
    } finally {
      if (prev === undefined) delete process.env[MAX_IDLE_ENV];
      else process.env[MAX_IDLE_ENV] = prev;
    }
  });

  it('returns the payload once the writer delivers it and closes', () => {
    // A timer cannot fire while readStdinSync blocks, so the retry path is
    // covered by the two tests above; this proves data followed by EOF ends
    // the loop with the full payload. Must run last: it closes the writer.
    const payload = '{"late":true}';
    fs.writeSync(writerFd, payload);
    fs.closeSync(writerFd);
    expect(readStdinSync({ fd: readerFd, maxIdleMs: 500 })).toBe(payload);
  });
});

describe('source guard', () => {
  it('no hook reads stdin with readFileSync(0) any more', () => {
    const srcDir = path.join(HERE, '..');
    const offenders: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          if (entry.name !== '__tests__') walk(p);
        } else if (p.endsWith('.ts') && !p.endsWith('.d.ts')) {
          if (/readFileSync\(\s*0\s*,/.test(fs.readFileSync(p, 'utf-8'))) offenders.push(path.relative(srcDir, p));
        }
      }
    };
    walk(srcDir);
    expect(offenders).toEqual([]);
  });
});
