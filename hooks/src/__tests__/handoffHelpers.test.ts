/**
 * Tests for handoff directory helpers in session-start-continuity.ts:
 * buildHandoffDirName, parseHandoffDirName, findSessionHandoffWithUUID.
 *
 * The `findSessionHandoff` function and its tests were removed as part of
 * the issue #86 fix — the hook now uses workstream-scoped lookup via
 * findSessionHandoffWithUUID instead of the unscoped mtime-based legacy path.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

import {
  buildHandoffDirName,
  parseHandoffDirName,
  findSessionHandoffWithUUID
} from '../session-start-continuity.js';

describe('buildHandoffDirName', () => {
  it('should append 8-char UUID suffix to session name', () => {
    const result = buildHandoffDirName('auth-refactor', '550e8400-e29b-41d4-a716-446655440000');
    expect(result).toBe('auth-refactor-550e8400');
  });

  it('should handle UUID without dashes', () => {
    const result = buildHandoffDirName('my-feature', '550e8400e29b41d4a716446655440000');
    expect(result).toBe('my-feature-550e8400');
  });

  it('should handle short session names', () => {
    const result = buildHandoffDirName('fix', 'abcd1234-0000-0000-0000-000000000000');
    expect(result).toBe('fix-abcd1234');
  });
});

describe('parseHandoffDirName', () => {
  it('should extract session name and UUID from suffixed directory', () => {
    const result = parseHandoffDirName('auth-refactor-550e8400');
    expect(result).toEqual({
      sessionName: 'auth-refactor',
      uuidShort: '550e8400'
    });
  });

  it('should handle legacy directory without UUID suffix', () => {
    const result = parseHandoffDirName('auth-refactor');
    expect(result).toEqual({
      sessionName: 'auth-refactor',
      uuidShort: null
    });
  });

  it('should handle session names with multiple hyphens', () => {
    const result = parseHandoffDirName('my-cool-feature-v2-abcd1234');
    expect(result).toEqual({
      sessionName: 'my-cool-feature-v2',
      uuidShort: 'abcd1234'
    });
  });

  it('should not parse non-hex suffix as UUID', () => {
    // "v2" is not 8 hex chars, so treat as part of session name
    const result = parseHandoffDirName('my-feature-v2');
    expect(result).toEqual({
      sessionName: 'my-feature-v2',
      uuidShort: null
    });
  });

  it('should require exactly 8 hex chars for UUID', () => {
    // "abc123" is only 6 chars
    const result = parseHandoffDirName('my-feature-abc123');
    expect(result).toEqual({
      sessionName: 'my-feature-abc123',
      uuidShort: null
    });
  });
});

describe('findSessionHandoffWithUUID', () => {
  let testDir: string;
  let originalProjectDir: string | undefined;

  beforeEach(() => {
    testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'uuid-handoff-test-'));
    originalProjectDir = process.env.CLAUDE_PROJECT_DIR;
    process.env.CLAUDE_PROJECT_DIR = testDir;
  });

  afterEach(() => {
    if (originalProjectDir !== undefined) {
      process.env.CLAUDE_PROJECT_DIR = originalProjectDir;
    } else {
      delete process.env.CLAUDE_PROJECT_DIR;
    }
    fs.rmSync(testDir, { recursive: true, force: true });
  });

  it('should find handoff with exact UUID match', () => {
    const sessionId = '550e8400-e29b-41d4-a716-446655440000';
    const dirName = 'auth-refactor-550e8400';
    const handoffDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', dirName);
    fs.mkdirSync(handoffDir, { recursive: true });
    fs.writeFileSync(path.join(handoffDir, 'current.md'), '# Handoff');

    const result = findSessionHandoffWithUUID('auth-refactor', sessionId);

    expect(result).not.toBeNull();
    expect(result!.includes('auth-refactor-550e8400')).toBe(true);
  });

  it('should fall back to legacy path without UUID', () => {
    const sessionId = '550e8400-e29b-41d4-a716-446655440000';
    // Create legacy directory (no UUID suffix)
    const handoffDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', 'auth-refactor');
    fs.mkdirSync(handoffDir, { recursive: true });
    fs.writeFileSync(path.join(handoffDir, 'current.md'), '# Legacy handoff');

    const result = findSessionHandoffWithUUID('auth-refactor', sessionId);

    expect(result).not.toBeNull();
    expect(result!.includes('auth-refactor')).toBe(true);
    expect(result!.includes('550e8400')).toBe(false);
  });

  it('should prefer UUID-suffixed directory over legacy', async () => {
    const sessionId = '550e8400-e29b-41d4-a716-446655440000';

    // Create legacy directory first
    const legacyDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', 'auth-refactor');
    fs.mkdirSync(legacyDir, { recursive: true });
    fs.writeFileSync(path.join(legacyDir, 'current.md'), '# Legacy');

    await new Promise(resolve => setTimeout(resolve, 50));

    // Create UUID-suffixed directory
    const uuidDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', 'auth-refactor-550e8400');
    fs.mkdirSync(uuidDir, { recursive: true });
    fs.writeFileSync(path.join(uuidDir, 'current.md'), '# UUID handoff');

    const result = findSessionHandoffWithUUID('auth-refactor', sessionId);

    expect(result).not.toBeNull();
    expect(result!.includes('550e8400')).toBe(true);
  });

  it('should find other sessions UUID dirs when no exact match', () => {
    const sessionId = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
    // Create a different UUID's directory for same session name
    const otherDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', 'auth-refactor-11111111');
    fs.mkdirSync(otherDir, { recursive: true });
    fs.writeFileSync(path.join(otherDir, 'current.md'), '# Other session');

    const result = findSessionHandoffWithUUID('auth-refactor', sessionId);

    // Should find the other session's handoff as fallback
    expect(result).not.toBeNull();
    expect(result!.includes('auth-refactor')).toBe(true);
  });

  it('should return null when no matching session exists', () => {
    const sessionId = '550e8400-e29b-41d4-a716-446655440000';
    // Create handoffs directory but no matching session
    const handoffsBase = path.join(testDir, 'thoughts', 'shared', 'handoffs');
    fs.mkdirSync(handoffsBase, { recursive: true });

    const result = findSessionHandoffWithUUID('nonexistent', sessionId);

    expect(result).toBeNull();
  });
});
