/**
 * Tests for session ID utilities.
 *
 * After #85, the singleton file (.coordination-session-id) is removed.
 * Session IDs come from stdin (provided by Claude Code) — not from files.
 *
 * Retained utilities: generateSessionId(), getSessionId() (env-only), getProject()
 * Removed: writeSessionId(), readSessionId(), getSessionIdFile()
 */

import {
  generateSessionId,
  getSessionId,
  getProject,
} from '../shared/session-id.js';

describe('generateSessionId', () => {
  it('returns a string starting with "s-" when no BRAINTRUST_SPAN_ID', () => {
    const originalSpanId = process.env.BRAINTRUST_SPAN_ID;
    delete process.env.BRAINTRUST_SPAN_ID;

    try {
      const result = generateSessionId();
      expect(result).toMatch(/^s-[a-z0-9]+$/);
    } finally {
      if (originalSpanId) {
        process.env.BRAINTRUST_SPAN_ID = originalSpanId;
      }
    }
  });

  it('uses BRAINTRUST_SPAN_ID when available', () => {
    const originalSpanId = process.env.BRAINTRUST_SPAN_ID;
    process.env.BRAINTRUST_SPAN_ID = 'test1234-5678-abcd-efgh';

    try {
      const result = generateSessionId();
      expect(result).toBe('test1234');
    } finally {
      if (originalSpanId) {
        process.env.BRAINTRUST_SPAN_ID = originalSpanId;
      } else {
        delete process.env.BRAINTRUST_SPAN_ID;
      }
    }
  });
});

describe('getSessionId', () => {
  let originalCoordId: string | undefined;
  let originalSpanId: string | undefined;

  beforeEach(() => {
    originalCoordId = process.env.COORDINATION_SESSION_ID;
    originalSpanId = process.env.BRAINTRUST_SPAN_ID;
    delete process.env.COORDINATION_SESSION_ID;
    delete process.env.BRAINTRUST_SPAN_ID;
  });

  afterEach(() => {
    if (originalCoordId) {
      process.env.COORDINATION_SESSION_ID = originalCoordId;
    } else {
      delete process.env.COORDINATION_SESSION_ID;
    }
    if (originalSpanId) {
      process.env.BRAINTRUST_SPAN_ID = originalSpanId;
    } else {
      delete process.env.BRAINTRUST_SPAN_ID;
    }
  });

  it('prefers COORDINATION_SESSION_ID env var', () => {
    process.env.COORDINATION_SESSION_ID = 's-from-env';

    const result = getSessionId();
    expect(result).toBe('s-from-env');
  });

  it('does NOT read from singleton file (no file fallback)', () => {
    // With no env var set, getSessionId should fall through to generate —
    // it must NOT attempt to read from ~/.claude/.coordination-session-id
    const result = getSessionId();
    // Should be a generated ID, not a file-read ID
    expect(result).toMatch(/^s-[a-z0-9]+$/);
  });

  it('generates new ID when no env var available', () => {
    const result = getSessionId();
    expect(result).toMatch(/^s-[a-z0-9]+$/);
  });

  it('uses BRAINTRUST_SPAN_ID as fallback', () => {
    process.env.BRAINTRUST_SPAN_ID = 'fb123456-7890-abcd';

    const result = getSessionId();
    expect(result).toBe('fb123456');
  });
});

describe('getProject', () => {
  let originalProjectDir: string | undefined;

  beforeEach(() => {
    originalProjectDir = process.env.CLAUDE_PROJECT_DIR;
  });

  afterEach(() => {
    if (originalProjectDir) {
      process.env.CLAUDE_PROJECT_DIR = originalProjectDir;
    } else {
      delete process.env.CLAUDE_PROJECT_DIR;
    }
  });

  it('returns CLAUDE_PROJECT_DIR when set', () => {
    process.env.CLAUDE_PROJECT_DIR = '/test/project';

    const result = getProject();
    expect(result).toBe('/test/project');
  });

  it('falls back to cwd when CLAUDE_PROJECT_DIR not set', () => {
    delete process.env.CLAUDE_PROJECT_DIR;

    const result = getProject();
    expect(result).toBe(process.cwd());
  });
});

describe('removed exports', () => {
  it('writeSessionId is not exported', async () => {
    const mod = await import('../shared/session-id.js');
    expect('writeSessionId' in mod).toBe(false);
  });

  it('readSessionId is not exported', async () => {
    const mod = await import('../shared/session-id.js');
    expect('readSessionId' in mod).toBe(false);
  });

  it('getSessionIdFile is not exported', async () => {
    const mod = await import('../shared/session-id.js');
    expect('getSessionIdFile' in mod).toBe(false);
  });
});
