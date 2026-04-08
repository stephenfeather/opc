/**
 * Tests for session context injection functions.
 *
 * Covers:
 * - checkMemoryHealth(): PG + daemon liveness checks
 * - getPendingTasksSummary(): Tasks.md header extraction
 * - formatPeerMessage(): peer session formatting
 * - readPeerCache() / writePeerCache(): file-based TTL cache
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { existsSync, readFileSync, writeFileSync, mkdirSync, unlinkSync, rmSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

// ---------------------------------------------------------------------------
// checkMemoryHealth
// ---------------------------------------------------------------------------

describe('checkMemoryHealth', () => {
  let originalKill: typeof process.kill;

  beforeEach(() => {
    originalKill = process.kill;
  });

  afterEach(() => {
    process.kill = originalKill;
  });

  it('returns healthy when PG succeeded and daemon PID is alive', async () => {
    const { checkMemoryHealth } = await import('../session-context.js');

    // Mock process.kill to succeed (signal 0 = existence check)
    process.kill = vi.fn(() => true) as unknown as typeof process.kill;

    const tmpPid = join(tmpdir(), `test-daemon-${Date.now()}.pid`);
    writeFileSync(tmpPid, '12345');

    try {
      const result = checkMemoryHealth(true, tmpPid);
      expect(result.pgHealthy).toBe(true);
      expect(result.daemonRunning).toBe(true);
    } finally {
      unlinkSync(tmpPid);
    }
  });

  it('returns pgHealthy=false when registerResult is false', async () => {
    const { checkMemoryHealth } = await import('../session-context.js');

    const result = checkMemoryHealth(false, '/nonexistent/path.pid');
    expect(result.pgHealthy).toBe(false);
  });

  it('returns daemonRunning=false when PID file is missing', async () => {
    const { checkMemoryHealth } = await import('../session-context.js');

    const result = checkMemoryHealth(true, '/nonexistent/path.pid');
    expect(result.daemonRunning).toBe(false);
  });

  it('returns daemonRunning=false when PID process is dead', async () => {
    const { checkMemoryHealth } = await import('../session-context.js');

    // Mock process.kill to throw (process not found)
    process.kill = vi.fn(() => {
      throw new Error('ESRCH');
    }) as unknown as typeof process.kill;

    const tmpPid = join(tmpdir(), `test-daemon-dead-${Date.now()}.pid`);
    writeFileSync(tmpPid, '99999');

    try {
      const result = checkMemoryHealth(true, tmpPid);
      expect(result.daemonRunning).toBe(false);
    } finally {
      unlinkSync(tmpPid);
    }
  });

  it('returns daemonRunning=false when PID file contains non-numeric content', async () => {
    const { checkMemoryHealth } = await import('../session-context.js');

    const tmpPid = join(tmpdir(), `test-daemon-bad-${Date.now()}.pid`);
    writeFileSync(tmpPid, 'not-a-number');

    try {
      const result = checkMemoryHealth(true, tmpPid);
      expect(result.daemonRunning).toBe(false);
    } finally {
      unlinkSync(tmpPid);
    }
  });
});

// ---------------------------------------------------------------------------
// getPendingTasksSummary
// ---------------------------------------------------------------------------

describe('getPendingTasksSummary', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = join(tmpdir(), `test-tasks-${Date.now()}`);
    mkdirSync(tmpDir, { recursive: true });
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it('returns null when Tasks.md does not exist', async () => {
    const { getPendingTasksSummary } = await import('../session-context.js');
    const result = getPendingTasksSummary(join(tmpDir, 'nonexistent.md'));
    expect(result).toBeNull();
  });

  it('returns null when Tasks.md is empty', async () => {
    const { getPendingTasksSummary } = await import('../session-context.js');
    const tasksPath = join(tmpDir, 'Tasks.md');
    writeFileSync(tasksPath, '');
    const result = getPendingTasksSummary(tasksPath);
    expect(result).toBeNull();
  });

  it('returns null when Tasks.md has only the top-level heading', async () => {
    const { getPendingTasksSummary } = await import('../session-context.js');
    const tasksPath = join(tmpDir, 'Tasks.md');
    writeFileSync(tasksPath, '# Tasks\n');
    const result = getPendingTasksSummary(tasksPath);
    expect(result).toBeNull();
  });

  it('extracts task titles from ## headers', async () => {
    const { getPendingTasksSummary } = await import('../session-context.js');
    const tasksPath = join(tmpDir, 'Tasks.md');
    writeFileSync(tasksPath, `# Tasks

## Fix the login bug

- Details here

## Add caching layer

- More details
`);
    const result = getPendingTasksSummary(tasksPath);
    expect(result).toBe('Pending tasks (2): Fix the login bug, Add caching layer');
  });

  it('truncates to first 3 titles with ellipsis when more exist', async () => {
    const { getPendingTasksSummary } = await import('../session-context.js');
    const tasksPath = join(tmpDir, 'Tasks.md');
    writeFileSync(tasksPath, `# Tasks

## Task Alpha
## Task Beta
## Task Gamma
## Task Delta
## Task Epsilon
`);
    const result = getPendingTasksSummary(tasksPath);
    expect(result).toBe('Pending tasks (5): Task Alpha, Task Beta, Task Gamma, ...');
  });

  it('handles exactly 3 tasks without ellipsis', async () => {
    const { getPendingTasksSummary } = await import('../session-context.js');
    const tasksPath = join(tmpDir, 'Tasks.md');
    writeFileSync(tasksPath, `# Tasks

## One
## Two
## Three
`);
    const result = getPendingTasksSummary(tasksPath);
    expect(result).toBe('Pending tasks (3): One, Two, Three');
  });
});

// ---------------------------------------------------------------------------
// formatPeerMessage
// ---------------------------------------------------------------------------

describe('formatPeerMessage', () => {
  it('returns null when no peers', async () => {
    const { formatPeerMessage } = await import('../session-context.js');
    const result = formatPeerMessage([]);
    expect(result).toBeNull();
  });

  it('formats a single peer', async () => {
    const { formatPeerMessage } = await import('../session-context.js');
    const result = formatPeerMessage([
      { id: 's-abc123', project: 'opc', working_on: 'reranker tuning', started_at: null, last_heartbeat: null },
    ]);
    expect(result).toContain('Active peer sessions (1)');
    expect(result).toContain('s-abc123: reranker tuning');
  });

  it('formats multiple peers', async () => {
    const { formatPeerMessage } = await import('../session-context.js');
    const result = formatPeerMessage([
      { id: 's-abc', project: 'opc', working_on: 'task A', started_at: null, last_heartbeat: null },
      { id: 's-def', project: 'opc', working_on: '', started_at: null, last_heartbeat: null },
      { id: 's-ghi', project: 'opc', working_on: 'task C', started_at: null, last_heartbeat: null },
    ]);
    expect(result).toContain('Active peer sessions (3)');
    expect(result).toContain('s-abc: task A');
    expect(result).toContain('s-def: working...');
    expect(result).toContain('s-ghi: task C');
  });

  it('shows "working..." for peers with empty working_on', async () => {
    const { formatPeerMessage } = await import('../session-context.js');
    const result = formatPeerMessage([
      { id: 's-xyz', project: 'opc', working_on: '', started_at: null, last_heartbeat: null },
    ]);
    expect(result).toContain('s-xyz: working...');
  });
});

// ---------------------------------------------------------------------------
// readPeerCache / writePeerCache
// ---------------------------------------------------------------------------

describe('peer cache', () => {
  let cacheDir: string;
  let cachePath: string;

  beforeEach(() => {
    cacheDir = join(tmpdir(), `test-cache-${Date.now()}`);
    mkdirSync(cacheDir, { recursive: true });
    cachePath = join(cacheDir, 'peer-sessions.json');
  });

  afterEach(() => {
    rmSync(cacheDir, { recursive: true, force: true });
  });

  it('returns null when cache file does not exist', async () => {
    const { readPeerCache } = await import('../session-context.js');
    const result = readPeerCache(cachePath, 'opc', 60);
    expect(result).toBeNull();
  });

  it('returns null when cache is corrupt JSON', async () => {
    const { readPeerCache } = await import('../session-context.js');
    writeFileSync(cachePath, 'not json');
    const result = readPeerCache(cachePath, 'opc', 60);
    expect(result).toBeNull();
  });

  it('returns null when cache is for a different project', async () => {
    const { readPeerCache, writePeerCache } = await import('../session-context.js');
    writePeerCache(cachePath, 'other-project', []);
    const result = readPeerCache(cachePath, 'opc', 60);
    expect(result).toBeNull();
  });

  it('returns null when cache is stale (older than TTL)', async () => {
    const { readPeerCache } = await import('../session-context.js');
    const staleData = {
      cached_at: new Date(Date.now() - 61000).toISOString(),
      project: 'opc',
      sessions: [],
    };
    writeFileSync(cachePath, JSON.stringify(staleData));
    const result = readPeerCache(cachePath, 'opc', 60);
    expect(result).toBeNull();
  });

  it('returns sessions when cache is fresh', async () => {
    const { readPeerCache, writePeerCache } = await import('../session-context.js');
    const sessions = [{ id: 's-abc', project: 'opc', working_on: 'test', started_at: null, last_heartbeat: null }];
    writePeerCache(cachePath, 'opc', sessions);
    const result = readPeerCache(cachePath, 'opc', 60);
    expect(result).toEqual(sessions);
  });

  it('treats exactly 60s old cache as stale', async () => {
    const { readPeerCache } = await import('../session-context.js');
    const borderlineData = {
      cached_at: new Date(Date.now() - 60000).toISOString(),
      project: 'opc',
      sessions: [],
    };
    writeFileSync(cachePath, JSON.stringify(borderlineData));
    const result = readPeerCache(cachePath, 'opc', 60);
    expect(result).toBeNull();
  });

  it('treats 59s old cache as fresh', async () => {
    const { readPeerCache } = await import('../session-context.js');
    // Use a large margin (30s) to avoid flakiness from test execution delay
    const freshData = {
      cached_at: new Date(Date.now() - 30000).toISOString(),
      project: 'opc',
      sessions: [{ id: 's-x', project: 'opc', working_on: '', started_at: null, last_heartbeat: null }],
    };
    writeFileSync(cachePath, JSON.stringify(freshData));
    const result = readPeerCache(cachePath, 'opc', 60);
    expect(result).toHaveLength(1);
  });

  it('writePeerCache creates valid JSON that readPeerCache can read', async () => {
    const { readPeerCache, writePeerCache } = await import('../session-context.js');
    const sessions = [
      { id: 's-1', project: 'opc', working_on: 'alpha', started_at: null, last_heartbeat: null },
      { id: 's-2', project: 'opc', working_on: 'beta', started_at: null, last_heartbeat: null },
    ];
    writePeerCache(cachePath, 'opc', sessions);

    // Verify raw JSON structure
    const raw = JSON.parse(readFileSync(cachePath, 'utf-8'));
    expect(raw).toHaveProperty('cached_at');
    expect(raw).toHaveProperty('project', 'opc');
    expect(raw).toHaveProperty('sessions');
    expect(raw.sessions).toHaveLength(2);

    // Verify readPeerCache roundtrip
    const result = readPeerCache(cachePath, 'opc', 60);
    expect(result).toEqual(sessions);
  });
});

// ---------------------------------------------------------------------------
// formatHealthWarnings
// ---------------------------------------------------------------------------

describe('formatHealthWarnings', () => {
  it('returns null when everything is healthy', async () => {
    const { formatHealthWarnings } = await import('../session-context.js');
    const result = formatHealthWarnings({ pgHealthy: true, daemonRunning: true });
    expect(result).toBeNull();
  });

  it('returns PG warning when PG is down', async () => {
    const { formatHealthWarnings } = await import('../session-context.js');
    const result = formatHealthWarnings({ pgHealthy: false, daemonRunning: true });
    expect(result).toContain('PostgreSQL');
    expect(result).not.toContain('daemon');
  });

  it('returns daemon warning when daemon is down', async () => {
    const { formatHealthWarnings } = await import('../session-context.js');
    const result = formatHealthWarnings({ pgHealthy: true, daemonRunning: false });
    expect(result).toContain('daemon');
    expect(result).not.toContain('PostgreSQL');
  });

  it('returns both warnings when both are down', async () => {
    const { formatHealthWarnings } = await import('../session-context.js');
    const result = formatHealthWarnings({ pgHealthy: false, daemonRunning: false });
    expect(result).toContain('PostgreSQL');
    expect(result).toContain('daemon');
  });
});
