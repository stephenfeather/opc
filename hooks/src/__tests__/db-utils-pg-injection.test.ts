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
import { spawnSync } from 'child_process';

// We test runPgQuery indirectly by mocking its dependencies and
// verifying the generated Python code is safe.

// Mock child_process.spawnSync to capture what gets passed to Python
vi.mock('child_process', () => ({
  spawnSync: vi.fn(() => ({
    status: 0,
    stdout: 'ok',
    stderr: '',
  })),
}));

// Mock opc-path to return controlled opcDir values
vi.mock('../shared/opc-path.js', () => ({
  requireOpcDir: vi.fn(() => '/tmp/safe-path'),
}));

import { runPgQuery } from '../shared/db-utils-pg.js';
import { requireOpcDir } from '../shared/opc-path.js';

const mockedSpawnSync = vi.mocked(spawnSync);
const mockedRequireOpcDir = vi.mocked(requireOpcDir);

describe('runPgQuery opcDir injection protection (Issue #88)', () => {
  beforeEach(() => {
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
    expect(pythonCode).toContain("os.environ['_OPC_DIR']");
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
