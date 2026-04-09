/**
 * Unit tests for `resolveWorkstreamName` in session-start-continuity.ts.
 *
 * The resolver derives the current workstream name from git state so the
 * SessionStart hook can look up handoffs for the correct stream (issue #86).
 * Fallback chain: git branch → worktree top basename → null.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { execSync } from 'child_process';

import { resolveWorkstreamName } from '../session-start-continuity.js';

describe('resolveWorkstreamName', () => {
  let testDir: string;

  beforeEach(() => {
    testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'resolveWorkstreamName-test-'));
  });

  afterEach(() => {
    fs.rmSync(testDir, { recursive: true, force: true });
  });

  it('should return the default branch name for a freshly initialized repo', () => {
    execSync('git init -b main', { cwd: testDir, stdio: 'ignore' });
    expect(resolveWorkstreamName(testDir)).toBe('main');
  });

  it('should return a renamed branch name', () => {
    execSync('git init -b main', { cwd: testDir, stdio: 'ignore' });
    execSync('git branch -M refactor-tdd-fp-memory-daemon', { cwd: testDir, stdio: 'ignore' });
    expect(resolveWorkstreamName(testDir)).toBe('refactor-tdd-fp-memory-daemon');
  });

  it('should sanitize slashes in branch names (feature/auth → feature-auth)', () => {
    execSync('git init -b main', { cwd: testDir, stdio: 'ignore' });
    execSync('git branch -M feature/auth', { cwd: testDir, stdio: 'ignore' });
    expect(resolveWorkstreamName(testDir)).toBe('feature-auth');
  });

  it('should return null for a non-git directory', () => {
    // testDir exists but has no .git
    expect(resolveWorkstreamName(testDir)).toBeNull();
  });

  it('should fall back to worktree basename on detached HEAD', () => {
    execSync('git init -b main', { cwd: testDir, stdio: 'ignore' });
    // Create an initial commit so we have something to detach at
    execSync('git config user.email test@example.com', { cwd: testDir, stdio: 'ignore' });
    execSync('git config user.name test', { cwd: testDir, stdio: 'ignore' });
    execSync('git config commit.gpgsign false', { cwd: testDir, stdio: 'ignore' });
    fs.writeFileSync(path.join(testDir, 'README.md'), 'test');
    execSync('git add README.md', { cwd: testDir, stdio: 'ignore' });
    execSync('git commit -m init', { cwd: testDir, stdio: 'ignore' });
    // Detach HEAD
    execSync('git checkout --detach HEAD', { cwd: testDir, stdio: 'ignore' });

    const result = resolveWorkstreamName(testDir);
    // fs.realpathSync resolves symlinks that can appear in $TMPDIR on macOS.
    const expected = path.basename(fs.realpathSync(testDir));
    expect(result).toBe(expected);
  });

  it('should fall back to worktree basename when the branch name fails isValidId (after sanitization)', () => {
    execSync('git init -b main', { cwd: testDir, stdio: 'ignore' });
    // Branch names with dots are legal in git but fail SAFE_ID_PATTERN
    // (/^[a-zA-Z0-9_-]{1,64}$/). The resolver should reject them and then
    // fall back to the worktree basename, which does pass isValidId.
    execSync('git branch -M feature.v1', { cwd: testDir, stdio: 'ignore' });

    const result = resolveWorkstreamName(testDir);
    const expected = path.basename(fs.realpathSync(testDir));
    expect(result).toBe(expected);
  });
});
