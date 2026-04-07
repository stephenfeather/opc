import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { execFileSync } from 'child_process';
import { join } from 'path';

const HOOK_PATH = join(__dirname, '..', '..', 'dist', 'timestamp-inject.mjs');

/**
 * Helper: run the hook with given stdin JSON, return parsed stdout.
 */
function runHook(input: Record<string, unknown>, env: Record<string, string> = {}): unknown | null {
  const stdinData = JSON.stringify(input);
  try {
    const stdout = execFileSync('node', [HOOK_PATH], {
      input: stdinData,
      encoding: 'utf-8',
      env: { ...process.env, ...env },
      timeout: 5000,
    });
    return stdout.trim() ? JSON.parse(stdout.trim()) : null;
  } catch (e: any) {
    // If the hook exits with code 0 but no output, that's a skip
    if (e.status === 0) return null;
    throw e;
  }
}

const BASE_INPUT = {
  session_id: 'test-session-123',
  hook_event_name: 'UserPromptSubmit',
  prompt: 'Help me fix the auth bug',
  cwd: '/tmp/test',
};

describe('timestamp-inject hook', () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    // Restore env
    delete process.env.CLAUDE_AGENT_ID;
  });

  it('should inject a timestamp into additionalContext', () => {
    const result = runHook(BASE_INPUT) as any;

    expect(result).not.toBeNull();
    expect(result.hookSpecificOutput).toBeDefined();
    expect(result.hookSpecificOutput.hookEventName).toBe('UserPromptSubmit');
    expect(result.hookSpecificOutput.additionalContext).toMatch(/Current time:/);
    expect(result.hookSpecificOutput.additionalContext).toMatch(/ISO:/);
  });

  it('should include day of week in the timestamp', () => {
    const result = runHook(BASE_INPUT) as any;
    const ctx = result.hookSpecificOutput.additionalContext;

    // Should contain a day name
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    const hasDay = days.some(day => ctx.includes(day));
    expect(hasDay).toBe(true);
  });

  it('should include timezone info', () => {
    const result = runHook(BASE_INPUT) as any;
    const ctx = result.hookSpecificOutput.additionalContext;

    // Should contain a timezone identifier (e.g., America/Chicago)
    expect(ctx).toMatch(/\(/);
    expect(ctx).toMatch(/\)/);
  });

  it('should include a valid ISO 8601 timestamp', () => {
    const result = runHook(BASE_INPUT) as any;
    const ctx: string = result.hookSpecificOutput.additionalContext;

    // Extract ISO portion
    const isoMatch = ctx.match(/ISO:\s*(\S+)/);
    expect(isoMatch).not.toBeNull();

    const isoDate = new Date(isoMatch![1]);
    expect(isoDate.getTime()).not.toBeNaN();

    // Should be within the last 5 seconds
    const now = Date.now();
    expect(Math.abs(now - isoDate.getTime())).toBeLessThan(5000);
  });

  it('should skip for subagents (CLAUDE_AGENT_ID set)', () => {
    const result = runHook(BASE_INPUT, { CLAUDE_AGENT_ID: 'agent-abc' });
    expect(result).toBeNull();
  });
});
