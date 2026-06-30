/**
 * Tests for opcDir injection vulnerability in runPgQuery().
 *
 * Issue #88: Python code injection via opcDir interpolation.
 * The opcDir path was interpolated directly into a Python code string
 * using template literals. A path containing a single quote would break
 * out of the Python string literal, enabling arbitrary code execution.
 *
 * Fix: Pass opcDir via environment variable instead of string interpolation.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { spawnSync, spawn } from 'child_process';

// We test runPgQuery indirectly by mocking its dependencies and
// verifying the generated Python code is safe.

// Mock child_process spawnSync (runPgQuery) and spawn (runPgQueryDetached) to
// capture what gets passed to Python.
vi.mock('child_process', () => ({
  spawnSync: vi.fn(() => ({
    status: 0,
    stdout: 'ok',
    stderr: '',
  })),
  spawn: vi.fn(() => ({ unref: vi.fn() })),
}));

// Mock opc-path to return controlled opcDir values
vi.mock('../shared/opc-path.js', () => ({
  requireOpcDir: vi.fn(() => '/tmp/safe-path'),
}));

import { runPgQuery, runPgQueryDetached } from '../shared/db-utils-pg.js';
import { requireOpcDir } from '../shared/opc-path.js';

const mockedSpawnSync = vi.mocked(spawnSync);
const mockedSpawn = vi.mocked(spawn);
const mockedRequireOpcDir = vi.mocked(requireOpcDir);

describe('runPgQuery opcDir injection protection (Issue #88)', () => {
  // #265 backend gate: runPgQuery now no-ops unless the backend resolves to
  // postgres. Pin a URL so the gate proceeds and spawnSync is reached —
  // deterministic regardless of ambient env (no #214-style leakage dependency).
  let savedDbUrl: string | undefined;
  let savedBackend: string | undefined;

  beforeEach(() => {
    savedDbUrl = process.env.CONTINUOUS_CLAUDE_DB_URL;
    savedBackend = process.env.AGENTICA_MEMORY_BACKEND;
    process.env.CONTINUOUS_CLAUDE_DB_URL = 'postgres://test:test@localhost:5432/test';
    delete process.env.AGENTICA_MEMORY_BACKEND;
    vi.clearAllMocks();
    mockedSpawnSync.mockReturnValue({
      status: 0,
      stdout: 'ok',
      stderr: '',
      pid: 1234,
      signal: null,
      output: ['', 'ok', ''],
    });
  });

  afterEach(() => {
    if (savedDbUrl === undefined) delete process.env.CONTINUOUS_CLAUDE_DB_URL;
    else process.env.CONTINUOUS_CLAUDE_DB_URL = savedDbUrl;
    if (savedBackend === undefined) delete process.env.AGENTICA_MEMORY_BACKEND;
    else process.env.AGENTICA_MEMORY_BACKEND = savedBackend;
  });

  it('should not interpolate opcDir directly into Python code string', () => {
    // A path with a single quote that would break Python string interpolation
    const maliciousPath = "/tmp/it's-a-trap";
    mockedRequireOpcDir.mockReturnValue(maliciousPath);

    runPgQuery('print("hello")');

    // Get the Python code that was passed to spawnSync
    expect(mockedSpawnSync).toHaveBeenCalledTimes(1);
    const callArgs = mockedSpawnSync.mock.calls[0];
    const pythonArgs = callArgs[1] as string[];

    // Find the -c argument (the Python code)
    const cIndex = pythonArgs.indexOf('-c');
    expect(cIndex).toBeGreaterThan(-1);
    const pythonCode = pythonArgs[cIndex + 1];

    // The Python code must NOT contain the raw malicious path
    // If it does, the single quote would break the Python string literal
    expect(pythonCode).not.toContain(maliciousPath);
    expect(pythonCode).not.toContain("'${opcDir}'");
  });

  it('should pass opcDir safely via environment variable or sys.argv', () => {
    const pathWithQuote = "/tmp/path'with\"quotes";
    mockedRequireOpcDir.mockReturnValue(pathWithQuote);

    runPgQuery('print("hello")');

    const callArgs = mockedSpawnSync.mock.calls[0];
    const spawnOptions = callArgs[2] as Record<string, unknown>;

    // opcDir should be passed via environment variable
    const env = spawnOptions.env as Record<string, string>;
    expect(env._OPC_DIR).toBe(pathWithQuote);

    // The Python code should read from os.environ, not from interpolation
    const pythonArgs = callArgs[1] as string[];
    const cIndex = pythonArgs.indexOf('-c');
    const pythonCode = pythonArgs[cIndex + 1];
    expect(pythonCode).toContain("os.environ.get('_OPC_DIR')");
  });

  it('should handle opcDir with backticks and dollar signs safely', () => {
    const dangerousPath = '/tmp/$HOME/`whoami`/test';
    mockedRequireOpcDir.mockReturnValue(dangerousPath);

    runPgQuery('print("hello")');

    const callArgs = mockedSpawnSync.mock.calls[0];
    const pythonArgs = callArgs[1] as string[];
    const cIndex = pythonArgs.indexOf('-c');
    const pythonCode = pythonArgs[cIndex + 1];

    // The dangerous path should NOT appear in the code string
    expect(pythonCode).not.toContain(dangerousPath);
    expect(pythonCode).not.toContain('$HOME');
    expect(pythonCode).not.toContain('`whoami`');
  });

  it('should set cwd to opcDir in spawn options', () => {
    const testPath = '/tmp/test-opc';
    mockedRequireOpcDir.mockReturnValue(testPath);

    runPgQuery('print("hello")');

    const callArgs = mockedSpawnSync.mock.calls[0];
    const spawnOptions = callArgs[2] as Record<string, unknown>;
    expect(spawnOptions.cwd).toBe(testPath);
  });

  it('should pass user args after the Python code', () => {
    mockedRequireOpcDir.mockReturnValue('/tmp/safe');

    runPgQuery('print(sys.argv[1])', ['arg1', 'arg2']);

    const callArgs = mockedSpawnSync.mock.calls[0];
    const pythonArgs = callArgs[1] as string[];

    // args should be at the end: ['run', 'python', '-c', code, 'arg1', 'arg2']
    expect(pythonArgs[pythonArgs.length - 2]).toBe('arg1');
    expect(pythonArgs[pythonArgs.length - 1]).toBe('arg2');
  });
});

// ---------------------------------------------------------------------------
// #265 round-1 regression: the URL injected into the subprocess MUST match the
// resolver-selected URL the gate approved (precedence + blank-skip + trim),
// not the old raw `||` lookup. Otherwise a blank canonical var shadows a valid
// fallback and the subprocess connects with whitespace.
// ---------------------------------------------------------------------------

describe('runPgQuery URL resolution parity with the backend gate (#265)', () => {
  const ENV_KEYS = [
    'CONTINUOUS_CLAUDE_DB_URL',
    'DATABASE_URL',
    'OPC_POSTGRES_URL',
    'AGENTICA_MEMORY_BACKEND',
  ];
  let saved: Record<string, string | undefined>;

  beforeEach(() => {
    saved = {};
    for (const k of ENV_KEYS) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
    vi.clearAllMocks();
    mockedRequireOpcDir.mockReturnValue('/tmp/safe-path');
    mockedSpawnSync.mockReturnValue({
      status: 0,
      stdout: 'ok',
      stderr: '',
      pid: 1234,
      signal: null,
      output: ['', 'ok', ''],
    });
  });

  afterEach(() => {
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  function injectedDbUrl(): string {
    const spawnOptions = mockedSpawnSync.mock.calls[0][2] as Record<string, unknown>;
    const env = spawnOptions.env as Record<string, string>;
    return env.CONTINUOUS_CLAUDE_DB_URL;
  }

  it('injects the resolver fallback when the canonical var is blank but DATABASE_URL is valid', () => {
    process.env.CONTINUOUS_CLAUDE_DB_URL = '   ';
    process.env.DATABASE_URL = 'postgres://valid@h/db';

    const res = runPgQuery('print("x")');

    expect(res.success).toBe(true);
    expect(mockedSpawnSync).toHaveBeenCalledTimes(1);
    expect(injectedDbUrl()).toBe('postgres://valid@h/db');
  });

  it('injects the trimmed URL (surrounding whitespace stripped)', () => {
    process.env.CONTINUOUS_CLAUDE_DB_URL = '  postgres://valid@h/db  ';

    runPgQuery('print("x")');

    expect(injectedDbUrl()).toBe('postgres://valid@h/db');
  });
});

// ---------------------------------------------------------------------------
// runPgQueryDetached opcDir injection protection (gemini PR #266 review).
// The detached twin of runPgQuery had the same Issue #88 hole: it interpolated
// opcDir directly into the Python string. It must pass opcDir via _OPC_DIR env
// and read it with os.environ.get, exactly like runPgQuery.
// ---------------------------------------------------------------------------

describe('runPgQueryDetached opcDir injection protection (#88 parity)', () => {
  const ENV_KEYS = [
    'CONTINUOUS_CLAUDE_DB_URL',
    'DATABASE_URL',
    'OPC_POSTGRES_URL',
    'AGENTICA_MEMORY_BACKEND',
  ];
  let saved: Record<string, string | undefined>;

  beforeEach(() => {
    saved = {};
    for (const k of ENV_KEYS) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
    // Pin a URL so the backend gate proceeds to the spawn path deterministically.
    process.env.CONTINUOUS_CLAUDE_DB_URL = 'postgres://test@localhost/test';
    vi.clearAllMocks();
    mockedRequireOpcDir.mockReturnValue('/tmp/safe-path');
    mockedSpawn.mockReturnValue({ unref: vi.fn() } as unknown as ReturnType<typeof spawn>);
  });

  afterEach(() => {
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  function detachedPythonCode(): string {
    const callArgs = mockedSpawn.mock.calls[0];
    const argv = callArgs[1] as string[];
    const cIndex = argv.indexOf('-c');
    return argv[cIndex + 1];
  }

  it('does not interpolate opcDir into the Python code string', () => {
    const maliciousPath = "/tmp/it's-a-trap";
    mockedRequireOpcDir.mockReturnValue(maliciousPath);

    runPgQueryDetached('print("hello")');

    expect(mockedSpawn).toHaveBeenCalledTimes(1);
    const code = detachedPythonCode();
    expect(code).not.toContain(maliciousPath);
    expect(code).not.toContain("'${opcDir}'");
  });

  it('passes opcDir via the _OPC_DIR environment variable and reads it from os.environ', () => {
    const pathWithQuote = "/tmp/path'with\"quotes";
    mockedRequireOpcDir.mockReturnValue(pathWithQuote);

    runPgQueryDetached('print("hello")');

    const spawnOptions = mockedSpawn.mock.calls[0][2] as Record<string, unknown>;
    const env = spawnOptions.env as Record<string, string>;
    expect(env._OPC_DIR).toBe(pathWithQuote);
    expect(detachedPythonCode()).toContain("os.environ.get('_OPC_DIR')");
  });
});
