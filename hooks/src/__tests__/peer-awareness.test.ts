/**
 * Tests for peer-awareness hook — stdin-only session ID (#85).
 *
 * After #85, peer-awareness reads session_id exclusively from stdin.
 * The readSessionId() file fallback is removed.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('peer-awareness hook', () => {
  let consoleSpy: ReturnType<typeof vi.spyOn>;
  const BACKEND_ENV_KEYS = [
    'CONTINUOUS_CLAUDE_DB_URL',
    'DATABASE_URL',
    'OPC_POSTGRES_URL',
    'AGENTICA_MEMORY_BACKEND',
  ];
  let savedBackendEnv: Record<string, string | undefined>;

  beforeEach(() => {
    vi.resetModules();
    consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    // Pin a URL so the #265 backend gate is deterministically active (postgres)
    // by default — independent of ambient DATABASE_URL. The sqlite-gate test
    // below overrides this. (Avoids the #214 ambient-env false-green/false-red.)
    savedBackendEnv = {};
    for (const k of BACKEND_ENV_KEYS) {
      savedBackendEnv[k] = process.env[k];
      delete process.env[k];
    }
    process.env.CONTINUOUS_CLAUDE_DB_URL = 'postgres://test@localhost/test';
  });

  afterEach(() => {
    vi.restoreAllMocks();
    for (const k of BACKEND_ENV_KEYS) {
      if (savedBackendEnv[k] === undefined) delete process.env[k];
      else process.env[k] = savedBackendEnv[k];
    }
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
          return (actual.readFileSync as (...args: unknown[]) => string | Buffer)(fd, ...rest);
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

    const output = consoleSpy.mock.calls.map((c: unknown[]) => c[0]).join('');
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
          return (actual.readFileSync as (...args: unknown[]) => string | Buffer)(fd, ...rest);
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
          return (actual.readFileSync as (...args: unknown[]) => string | Buffer)(fd, ...rest);
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

  it('stays silent under a sqlite backend even when the peer cache is fresh (#265 / PR266)', async () => {
    // The peer cache (60s TTL) is a read path that does NOT go through the
    // db-utils-pg gate. A cache entry written while PG was active must not be
    // served after switching to sqlite — peer-awareness must gate on the
    // backend before reading the cache.
    const savedEnv = {
      CONTINUOUS_CLAUDE_DB_URL: process.env.CONTINUOUS_CLAUDE_DB_URL,
      DATABASE_URL: process.env.DATABASE_URL,
      OPC_POSTGRES_URL: process.env.OPC_POSTGRES_URL,
      AGENTICA_MEMORY_BACKEND: process.env.AGENTICA_MEMORY_BACKEND,
    };
    delete process.env.CONTINUOUS_CLAUDE_DB_URL;
    delete process.env.DATABASE_URL;
    delete process.env.OPC_POSTGRES_URL;
    process.env.AGENTICA_MEMORY_BACKEND = 'sqlite';

    try {
      vi.doMock('fs', async () => {
        const actual = await vi.importActual<typeof import('fs')>('fs');
        return {
          ...actual,
          readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
            if (fd === 0) return JSON.stringify({ session_id: 's-my-session' });
            return (actual.readFileSync as (...args: unknown[]) => string | Buffer)(fd, ...rest);
          }),
        };
      });
      const getActiveSessions = vi.fn();
      vi.doMock('../shared/db-utils-pg.js', () => ({
        getActiveSessions,
        isValidId: (id: string) => /^[a-zA-Z0-9_-]+$/.test(id),
      }));
      vi.doMock('../shared/session-id.js', () => ({ getProject: () => '/project' }));
      vi.doMock('../session-context.js', () => ({
        // Fresh cache HIT — would inject a stale PG peer if the gate is missing.
        readPeerCache: () => [{ id: 's-peer-1', working_on: 'task B' }],
        writePeerCache: vi.fn(),
        formatPeerMessage: (peers: Array<{ id: string }>) =>
          peers.length > 0 ? `Peers: ${peers.map(p => p.id).join(', ')}` : null,
      }));

      const { main } = await import('../peer-awareness.js');
      main();

      const output = consoleSpy.mock.calls.map((c: unknown[]) => c[0]).join('');
      expect(output).not.toContain('s-peer-1');
      const lastCall = consoleSpy.mock.calls[consoleSpy.mock.calls.length - 1][0];
      expect(lastCall).toBe('{}');
      expect(getActiveSessions).not.toHaveBeenCalled();
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v === undefined) delete process.env[k];
        else process.env[k] = v;
      }
    }
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
          return (actual.readFileSync as (...args: unknown[]) => string | Buffer)(fd, ...rest);
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
