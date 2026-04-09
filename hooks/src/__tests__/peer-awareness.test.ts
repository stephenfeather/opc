/**
 * Tests for peer-awareness hook — stdin-only session ID (#85).
 *
 * After #85, peer-awareness reads session_id exclusively from stdin.
 * The readSessionId() file fallback is removed.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('peer-awareness hook', () => {
  let consoleSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.resetModules();
    consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('uses session_id from stdin to filter self from peers', async () => {
    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) {
            return JSON.stringify({ session_id: 's-my-session' });
          }
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      getActiveSessions: vi.fn().mockReturnValue({
        success: true,
        sessions: [
          { id: 's-my-session', working_on: 'task A' },
          { id: 's-peer-1', working_on: 'task B' },
        ],
      }),
      isValidId: (id: string) => /^[a-zA-Z0-9_-]+$/.test(id),
    }));
    vi.doMock('../shared/session-id.js', () => ({
      getProject: () => '/project',
    }));
    vi.doMock('../session-context.js', () => ({
      readPeerCache: () => null, // force DB query
      writePeerCache: vi.fn(),
      formatPeerMessage: (peers: Array<{ id: string }>) =>
        peers.length > 0 ? `Peers: ${peers.map(p => p.id).join(', ')}` : null,
    }));

    const { main } = await import('../peer-awareness.js');
    main();

    const output = consoleSpy.mock.calls.map(c => c[0]).join('');
    // Should show peer but not self
    expect(output).toContain('s-peer-1');
    expect(output).not.toContain('s-my-session');
  });

  it('outputs empty when stdin has no session_id (no file fallback)', async () => {
    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) return '{}';
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      getActiveSessions: vi.fn(),
      isValidId: (id: string) => /^[a-zA-Z0-9_-]+$/.test(id),
    }));
    vi.doMock('../shared/session-id.js', () => ({
      getProject: () => '/project',
    }));
    vi.doMock('../session-context.js', () => ({
      readPeerCache: () => null,
      writePeerCache: vi.fn(),
      formatPeerMessage: () => null,
    }));

    const { main } = await import('../peer-awareness.js');
    main();

    // Module auto-executes main() on import, so we check the last call
    const lastCall = consoleSpy.mock.calls[consoleSpy.mock.calls.length - 1][0];
    expect(lastCall).toBe('{}');
  });

  it('does not import readSessionId from session-id module', async () => {
    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) return JSON.stringify({ session_id: 's-test' });
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      getActiveSessions: vi.fn().mockReturnValue({ success: true, sessions: [] }),
      isValidId: (id: string) => /^[a-zA-Z0-9_-]+$/.test(id),
    }));
    // Only getProject — readSessionId should not be needed
    vi.doMock('../shared/session-id.js', () => ({
      getProject: () => '/project',
    }));
    vi.doMock('../session-context.js', () => ({
      readPeerCache: () => null,
      writePeerCache: vi.fn(),
      formatPeerMessage: () => null,
    }));

    vi.resetModules();
    const { main } = await import('../peer-awareness.js');
    expect(() => main()).not.toThrow();
  });

  it('skips for subagents', async () => {
    const origAgentId = process.env.CLAUDE_AGENT_ID;
    process.env.CLAUDE_AGENT_ID = 'sub-1';

    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) return JSON.stringify({ session_id: 's-test' });
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      getActiveSessions: vi.fn(),
      isValidId: () => true,
    }));
    vi.doMock('../shared/session-id.js', () => ({
      getProject: () => '/project',
    }));
    vi.doMock('../session-context.js', () => ({
      readPeerCache: () => null,
      writePeerCache: vi.fn(),
      formatPeerMessage: () => null,
    }));

    const { main } = await import('../peer-awareness.js');
    main();

    // Module auto-executes main() on import, so we check the last call
    const lastCall = consoleSpy.mock.calls[consoleSpy.mock.calls.length - 1][0];
    expect(lastCall).toBe('{}');

    if (origAgentId) {
      process.env.CLAUDE_AGENT_ID = origAgentId;
    } else {
      delete process.env.CLAUDE_AGENT_ID;
    }
  });
});
