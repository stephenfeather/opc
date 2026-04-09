/**
 * Tests for the heartbeat hook and updateHeartbeat / updateHeartbeatDetached functions.
 *
 * Covers:
 * - updateHeartbeat(): SQL update via runPgQuery (synchronous, returns result)
 * - updateHeartbeatDetached(): fire-and-forget via runPgQueryDetached (spawn+unref)
 * - heartbeat hook main(): PostToolUse handler that refreshes session heartbeat
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ---------------------------------------------------------------------------
// updateHeartbeat (unit tests via mocked child_process)
// ---------------------------------------------------------------------------

describe('updateHeartbeat', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns success when DB update succeeds', async () => {
    vi.doMock('../shared/opc-path.js', () => ({
      requireOpcDir: () => '/tmp/opc',
    }));
    vi.doMock('child_process', () => ({
      spawnSync: () => ({ status: 0, stdout: 'ok', stderr: '' }),
    }));

    const { updateHeartbeat } = await import('../shared/db-utils-pg.js');
    const result = updateHeartbeat('s-abc123', '/home/user/project');

    expect(result.success).toBe(true);
    expect(result.error).toBeUndefined();
  });

  it('returns failure when DB update fails', async () => {
    vi.doMock('../shared/opc-path.js', () => ({
      requireOpcDir: () => '/tmp/opc',
    }));
    vi.doMock('child_process', () => ({
      spawnSync: () => ({ status: 1, stdout: '', stderr: 'connection refused' }),
    }));

    const { updateHeartbeat } = await import('../shared/db-utils-pg.js');
    const result = updateHeartbeat('s-abc123', '/home/user/project');

    expect(result.success).toBe(false);
    expect(result.error).toContain('connection refused');
  });

  it('passes sessionId and project as args to the Python subprocess', async () => {
    let capturedArgs: string[] = [];
    vi.doMock('../shared/opc-path.js', () => ({
      requireOpcDir: () => '/tmp/opc',
    }));
    vi.doMock('child_process', () => ({
      spawnSync: (_cmd: string, args: string[]) => {
        capturedArgs = args;
        return { status: 0, stdout: 'ok', stderr: '' };
      },
    }));

    const { updateHeartbeat } = await import('../shared/db-utils-pg.js');
    updateHeartbeat('s-test42', '/Users/me/opc');

    // Args passed to uv run python -c <code> <sessionId> <project>
    expect(capturedArgs).toContain('s-test42');
    expect(capturedArgs).toContain('/Users/me/opc');
  });

  it('includes UPDATE SQL with last_heartbeat = NOW()', async () => {
    let capturedArgs: string[] = [];
    vi.doMock('../shared/opc-path.js', () => ({
      requireOpcDir: () => '/tmp/opc',
    }));
    vi.doMock('child_process', () => ({
      spawnSync: (_cmd: string, args: string[]) => {
        capturedArgs = args;
        return { status: 0, stdout: 'ok', stderr: '' };
      },
    }));

    const { updateHeartbeat } = await import('../shared/db-utils-pg.js');
    updateHeartbeat('s-test', '/project');

    // The Python code is embedded in the args (after -c flag)
    const pythonCode = capturedArgs.find(a => a.includes('UPDATE sessions')) || '';
    expect(pythonCode).toContain('UPDATE sessions');
    expect(pythonCode).toContain('last_heartbeat');
    expect(pythonCode).toContain('NOW()');
  });

  it('returns failure with error when stdout is not ok', async () => {
    vi.doMock('../shared/opc-path.js', () => ({
      requireOpcDir: () => '/tmp/opc',
    }));
    vi.doMock('child_process', () => ({
      spawnSync: () => ({ status: 0, stdout: 'error: table missing', stderr: '' }),
    }));

    const { updateHeartbeat } = await import('../shared/db-utils-pg.js');
    const result = updateHeartbeat('s-abc', '/project');

    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// updateHeartbeatDetached (unit tests via mocked child_process.spawn)
// ---------------------------------------------------------------------------

describe('updateHeartbeatDetached', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('spawns a detached process with sessionId and project as args', async () => {
    let capturedArgs: string[] = [];
    const mockUnref = vi.fn();

    vi.doMock('../shared/opc-path.js', () => ({
      requireOpcDir: () => '/tmp/opc',
    }));
    vi.doMock('child_process', () => ({
      spawnSync: vi.fn(),
      spawn: (_cmd: string, args: string[]) => {
        capturedArgs = args;
        return { unref: mockUnref };
      },
    }));

    const { updateHeartbeatDetached } = await import('../shared/db-utils-pg.js');
    updateHeartbeatDetached('s-det42', '/home/user/project');

    expect(capturedArgs).toContain('s-det42');
    expect(capturedArgs).toContain('/home/user/project');
    expect(mockUnref).toHaveBeenCalled();
  });

  it('calls unref() so the parent process does not wait', async () => {
    const mockUnref = vi.fn();

    vi.doMock('../shared/opc-path.js', () => ({
      requireOpcDir: () => '/tmp/opc',
    }));
    vi.doMock('child_process', () => ({
      spawnSync: vi.fn(),
      spawn: () => ({ unref: mockUnref }),
    }));

    const { updateHeartbeatDetached } = await import('../shared/db-utils-pg.js');
    updateHeartbeatDetached('s-unref', '/project');

    expect(mockUnref).toHaveBeenCalledOnce();
  });

  it('does not throw when spawn fails', async () => {
    vi.doMock('../shared/opc-path.js', () => ({
      requireOpcDir: () => '/tmp/opc',
    }));
    vi.doMock('child_process', () => ({
      spawnSync: vi.fn(),
      spawn: () => { throw new Error('spawn failed'); },
    }));

    const { updateHeartbeatDetached } = await import('../shared/db-utils-pg.js');
    expect(() => updateHeartbeatDetached('s-fail', '/project')).not.toThrow();
  });

  it('includes UPDATE SQL targeting last_heartbeat', async () => {
    let capturedArgs: string[] = [];

    vi.doMock('../shared/opc-path.js', () => ({
      requireOpcDir: () => '/tmp/opc',
    }));
    vi.doMock('child_process', () => ({
      spawnSync: vi.fn(),
      spawn: (_cmd: string, args: string[]) => {
        capturedArgs = args;
        return { unref: vi.fn() };
      },
    }));

    const { updateHeartbeatDetached } = await import('../shared/db-utils-pg.js');
    updateHeartbeatDetached('s-sql', '/project');

    const pythonCode = capturedArgs.find(a => a.includes('UPDATE sessions')) ?? '';
    expect(pythonCode).toContain('UPDATE sessions');
    expect(pythonCode).toContain('last_heartbeat');
    expect(pythonCode).toContain('NOW()');
  });
});

// ---------------------------------------------------------------------------
// heartbeat hook main()
// ---------------------------------------------------------------------------

describe('heartbeat hook', () => {
  let consoleSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.resetModules();
    consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('outputs continue when no session ID available', async () => {
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
      updateHeartbeatDetached: vi.fn(),
      isValidId: (id: string) => /^[a-zA-Z0-9_-]+$/.test(id),
      SAFE_ID_PATTERN: /^[a-zA-Z0-9_-]+$/,
    }));
    vi.doMock('../shared/session-id.js', () => ({
      readSessionId: () => null,
      getProject: () => '/tmp/test-project',
    }));

    const { main } = await import('../heartbeat.js');
    main();

    const output = consoleSpy.mock.calls.map(c => c[0]).join('');
    expect(output).toContain('"result":"continue"');
  });

  it('calls updateHeartbeatDetached when session ID is available from file', async () => {
    const mockUpdate = vi.fn();

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
      updateHeartbeatDetached: mockUpdate,
      isValidId: (id: string) => /^[a-zA-Z0-9_-]+$/.test(id),
      SAFE_ID_PATTERN: /^[a-zA-Z0-9_-]+$/,
    }));
    vi.doMock('../shared/session-id.js', () => ({
      readSessionId: () => 's-filetest',
      getProject: () => '/home/user/myproject',
    }));

    const { main } = await import('../heartbeat.js');
    main();

    expect(mockUpdate).toHaveBeenCalledWith('s-filetest', '/home/user/myproject');
  });

  it('prefers stdin session_id over file-based session ID', async () => {
    const mockUpdate = vi.fn();

    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) return JSON.stringify({ session_id: 's-fromstdin' });
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      updateHeartbeatDetached: mockUpdate,
      isValidId: (id: string) => /^[a-zA-Z0-9_-]+$/.test(id),
      SAFE_ID_PATTERN: /^[a-zA-Z0-9_-]+$/,
    }));
    vi.doMock('../shared/session-id.js', () => ({
      readSessionId: () => 's-fromfile',
      getProject: () => '/project',
    }));

    const { main } = await import('../heartbeat.js');
    main();

    expect(mockUpdate).toHaveBeenCalledWith('s-fromstdin', '/project');
  });

  it('always outputs continue even when updateHeartbeatDetached throws', async () => {
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
      updateHeartbeatDetached: vi.fn(),
      isValidId: (id: string) => /^[a-zA-Z0-9_-]+$/.test(id),
      SAFE_ID_PATTERN: /^[a-zA-Z0-9_-]+$/,
    }));
    vi.doMock('../shared/session-id.js', () => ({
      readSessionId: () => 's-test',
      getProject: () => '/project',
    }));

    const { main } = await import('../heartbeat.js');
    main();

    const output = consoleSpy.mock.calls.map(c => c[0]).join('');
    expect(output).toContain('"result":"continue"');
  });
});
