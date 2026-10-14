/**
 * Issue #301: the local-dev tldr daemon start (`uv run tldr daemon start`)
 * must never re-resolve or rewrite a uv.lock as a side effect, and must not
 * walk up from an empty packages/tldr-code into the enclosing opc project.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

import { localTldrStartSpec } from '../daemon-client.js';

describe('localTldrStartSpec', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'tldr-local-start-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('returns null when the tldr package dir does not exist', () => {
    expect(localTldrStartSpec(join(root, 'missing'), '/proj', {})).toBeNull();
  });

  it('returns null when the tldr package dir has no pyproject.toml', () => {
    // An empty dir would make `uv run` walk up into the parent project.
    const tldrPath = join(root, 'tldr-code');
    mkdirSync(tldrPath);
    expect(localTldrStartSpec(tldrPath, '/proj', {})).toBeNull();
  });

  function makeCheckout(files: string[]): string {
    const tldrPath = join(root, 'tldr-code');
    mkdirSync(join(tldrPath, '.venv', 'bin'), { recursive: true });
    files.forEach((f) => writeFileSync(join(tldrPath, f), ''));
    return tldrPath;
  }

  it('returns null for a pyproject-only checkout with no lock and no venv tldr', () => {
    // Frozen `uv run` exits when uv.lock is missing; unfrozen would create one.
    const tldrPath = makeCheckout(['pyproject.toml']);
    expect(localTldrStartSpec(tldrPath, '/proj', {})).toBeNull();
  });

  it('runs the checkout venv tldr directly when installed (no uv, no lock touch)', () => {
    const tldrPath = makeCheckout(['pyproject.toml', join('.venv', 'bin', 'tldr')]);

    const spec = localTldrStartSpec(tldrPath, '/proj', { PATH: '/bin' });

    expect(spec).not.toBeNull();
    expect(spec!.command).toBe(join(tldrPath, '.venv', 'bin', 'tldr'));
    expect(spec!.args).toEqual(['daemon', 'start', '--project', '/proj']);
    expect(spec!.cwd).toBe(tldrPath);
    expect(spec!.env.PATH).toBe('/bin');
  });

  it('uses frozen uv run when the checkout has a uv.lock but no venv tldr', () => {
    const tldrPath = makeCheckout(['pyproject.toml', 'uv.lock']);

    const spec = localTldrStartSpec(tldrPath, '/proj', { PATH: '/bin' });

    expect(spec).not.toBeNull();
    expect(spec!.command).toBe('uv');
    expect(spec!.env.UV_FROZEN).toBe('1');
    expect(spec!.env.PATH).toBe('/bin');
    expect(spec!.cwd).toBe(tldrPath);
    expect(spec!.args).toEqual(['run', 'tldr', 'daemon', 'start', '--project', '/proj']);
  });
});
