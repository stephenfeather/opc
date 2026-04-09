/**
 * Tests for file-claims hook — stdin-based session ID (#85).
 *
 * After #85, file-claims reads session_id from stdin (input.session_id)
 * instead of calling getSessionId() which used the singleton file.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('file-claims hook', () => {
  let consoleSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.resetModules();
    consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('uses input.session_id from stdin for file claims', async () => {
    const mockClaimFile = vi.fn();
    const mockCheckClaim = vi.fn().mockReturnValue({ claimed: false });

    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) {
            return JSON.stringify({
              session_id: 's-stdin-id',
              tool_name: 'Edit',
              tool_input: { file_path: '/project/src/main.ts' },
            });
          }
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      checkFileClaim: mockCheckClaim,
      claimFile: mockClaimFile,
    }));
    vi.doMock('../shared/session-id.js', () => ({
      getProject: () => '/project',
    }));

    const { main } = await import('../file-claims.js');
    main();

    // Should use the stdin session_id, not getSessionId()
    expect(mockClaimFile).toHaveBeenCalledWith(
      '/project/src/main.ts',
      '/project',
      's-stdin-id',
    );
  });

  it('returns continue early when stdin has no session_id', async () => {
    const mockClaimFile = vi.fn();
    const mockCheckClaim = vi.fn();

    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) {
            return JSON.stringify({
              tool_name: 'Edit',
              tool_input: { file_path: '/project/src/main.ts' },
            });
          }
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      checkFileClaim: mockCheckClaim,
      claimFile: mockClaimFile,
    }));
    vi.doMock('../shared/session-id.js', () => ({
      getProject: () => '/project',
    }));

    const { main } = await import('../file-claims.js');
    main();

    // Should NOT call any DB functions — early return
    expect(mockCheckClaim).not.toHaveBeenCalled();
    expect(mockClaimFile).not.toHaveBeenCalled();

    const output = consoleSpy.mock.calls.map(c => c[0]).join('');
    expect(output).toContain('"result":"continue"');
  });

  it('does not import getSessionId from session-id module', async () => {
    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) {
            return JSON.stringify({
              session_id: 's-test',
              tool_name: 'Edit',
              tool_input: { file_path: '/project/file.ts' },
            });
          }
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      checkFileClaim: vi.fn().mockReturnValue({ claimed: false }),
      claimFile: vi.fn(),
    }));
    // Only export getProject — getSessionId should not be needed
    vi.doMock('../shared/session-id.js', () => ({
      getProject: () => '/project',
    }));

    vi.resetModules();
    const { main } = await import('../file-claims.js');
    expect(() => main()).not.toThrow();
  });

  it('warns when file is claimed by another session', async () => {
    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) {
            return JSON.stringify({
              session_id: 's-my-session',
              tool_name: 'Edit',
              tool_input: { file_path: '/project/src/shared.ts' },
            });
          }
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      checkFileClaim: vi.fn().mockReturnValue({ claimed: true, claimedBy: 's-other' }),
      claimFile: vi.fn(),
    }));
    vi.doMock('../shared/session-id.js', () => ({
      getProject: () => '/project',
    }));

    const { main } = await import('../file-claims.js');
    main();

    const output = consoleSpy.mock.calls.map(c => c[0]).join('');
    expect(output).toContain('File Conflict Warning');
    expect(output).toContain('s-other');
  });

  it('continues silently for non-Edit tools', async () => {
    vi.doMock('fs', async () => {
      const actual = await vi.importActual<typeof import('fs')>('fs');
      return {
        ...actual,
        readFileSync: vi.fn((fd: unknown, ...rest: unknown[]) => {
          if (fd === 0) {
            return JSON.stringify({
              session_id: 's-test',
              tool_name: 'Read',
              tool_input: { file_path: '/project/file.ts' },
            });
          }
          return actual.readFileSync(fd as string, ...rest as [string]);
        }),
      };
    });
    vi.doMock('../shared/db-utils-pg.js', () => ({
      checkFileClaim: vi.fn(),
      claimFile: vi.fn(),
    }));
    vi.doMock('../shared/session-id.js', () => ({
      getProject: () => '/project',
    }));

    const { main } = await import('../file-claims.js');
    main();

    const output = consoleSpy.mock.calls.map(c => c[0]).join('');
    expect(output).toContain('"result":"continue"');
  });
});
