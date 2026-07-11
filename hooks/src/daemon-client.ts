/*!
 * Shared TypeScript client for TLDR daemon.
 *
 * Used by all TypeScript hooks to query the TLDR daemon instead of
 * spawning individual `tldr` processes. This provides:
 * - Faster queries (daemon holds indexes in memory)
 * - Reduced process overhead
 * - Consistent timeout handling
 * - Auto-start capability
 * - Graceful degradation when indexing
 */

import { existsSync, readFileSync, writeFileSync, unlinkSync } from 'fs';
import { execSync, spawnSync } from 'child_process';
import { join, resolve } from 'path';
import { tmpdir } from 'os';
import * as net from 'net';
import * as crypto from 'crypto';

/**
 * Resolve project directory to absolute path.
 * Handles relative paths like "." which would otherwise hash differently.
 */
function resolveProjectDir(projectDir: string): string {
  return resolve(projectDir);
}

/**
 * Get lock file path for daemon startup.
 * Prevents race condition when multiple hooks try to start daemon.
 */
function getLockPath(projectDir: string): string {
  const resolvedPath = resolveProjectDir(projectDir);
  const hash = crypto.createHash('md5').update(resolvedPath).digest('hex').substring(0, 8);
  return `${tmpdir()}/tldr-${hash}.lock`;
}

/**
 * Get PID file path for daemon.
 */
function getPidPath(projectDir: string): string {
  const resolvedPath = resolveProjectDir(projectDir);
  const hash = crypto.createHash('md5').update(resolvedPath).digest('hex').substring(0, 8);
  return `${tmpdir()}/tldr-${hash}.pid`;
}

/**
 * Check if daemon process is running by checking PID file and process existence.
 * This is more reliable than socket ping which can timeout when daemon is busy.
 */
function isDaemonProcessRunning(projectDir: string): boolean {
  const pidPath = getPidPath(projectDir);
  if (!existsSync(pidPath)) return false;

  try {
    const pid = parseInt(readFileSync(pidPath, 'utf-8').trim(), 10);
    if (isNaN(pid) || pid <= 0) return false;

    // kill(pid, 0) checks if process exists without sending signal
    process.kill(pid, 0);
    return true;  // Process exists
  } catch {
    return false;  // Process doesn't exist or permission denied
  }
}

/**
 * Try to acquire startup lock (non-blocking).
 * Returns true if lock acquired, false if another process holds it.
 */
function tryAcquireLock(projectDir: string): boolean {
  const lockPath = getLockPath(projectDir);
  try {
    // Check if lock exists and is recent (within 30s)
    if (existsSync(lockPath)) {
      const lockContent = readFileSync(lockPath, 'utf-8');
      const lockTime = parseInt(lockContent, 10);
      if (!isNaN(lockTime) && Date.now() - lockTime < 30000) {
        return false;  // Lock is held by another process
      }
      // Stale lock - remove it
      try { unlinkSync(lockPath); } catch { /* ignore */ }
    }
    // Create lock atomically with O_EXCL - fails if file exists (race-safe)
    writeFileSync(lockPath, Date.now().toString(), { flag: 'wx' });
    return true;
  } catch {
    // Either lock exists (race lost) or write failed - either way, don't acquire
    return false;
  }
}

/**
 * Release startup lock.
 */
function releaseLock(projectDir: string): void {
  try {
    unlinkSync(getLockPath(projectDir));
  } catch { /* ignore */ }
}

/** Query timeout in milliseconds (3 seconds) */
const QUERY_TIMEOUT = 3000;

/**
 * Per-invocation query deadline.
 *
 * QUERY_TIMEOUT bounds a single daemon round-trip, but hooks make several
 * sequential calls, so their worst case is N x QUERY_TIMEOUT — which blows
 * through Claude Code's per-hook timeout and gets the whole hook killed with
 * its output discarded (impact-refactor: 7 calls vs a 10s budget;
 * signature-helper: 10 calls vs 5s). A hook sets one overall budget up front;
 * every subsequent queryDaemonSync call is clamped to the time remaining and
 * short-circuits once the budget is spent. Module-level state is safe because
 * each hook invocation is its own short-lived Node process.
 */
let queryDeadlineAt: number | null = null;

/** Below this remaining budget a query cannot usefully complete — skip it. */
const MIN_QUERY_BUDGET_MS = 100;

/**
 * Hard bound on the fire-and-forget stats write in trackHookActivitySync.
 * The socket stays ref'd until the payload flushes, so this is also the
 * longest a finished hook process can be held open by tracking. Unix-socket
 * connect+write normally completes in single-digit milliseconds.
 */
const TRACK_WRITE_TIMEOUT_MS = 250;

/**
 * Set the shared time budget for all subsequent daemon queries in this
 * process. Call once at hook startup with a value safely under the hook's
 * Claude Code timeout.
 */
export function setQueryDeadline(budgetMs: number): void {
  queryDeadlineAt = Date.now() + budgetMs;
}

/** Remove the deadline (subsequent queries use QUERY_TIMEOUT only). */
export function clearQueryDeadline(): void {
  queryDeadlineAt = null;
}

/**
 * Milliseconds left in the current budget, or null when no deadline is set.
 * May be negative once the deadline has passed.
 */
export function remainingQueryBudget(): number | null {
  if (queryDeadlineAt === null) return null;
  return queryDeadlineAt - Date.now();
}

/** True when a deadline is set and effectively spent. */
function budgetExhausted(): boolean {
  const remaining = remainingQueryBudget();
  return remaining !== null && remaining < MIN_QUERY_BUDGET_MS;
}

/** Clamp a wait/timeout to the remaining budget (no-op without a deadline). */
function budgetClamp(defaultMs: number): number {
  const remaining = remainingQueryBudget();
  if (remaining === null) return defaultMs;
  return Math.max(0, Math.min(defaultMs, remaining));
}

/**
 * Classify an execSync/spawnSync error as a timeout kill. Node has not been
 * consistent here across versions: some set `killed: true`, Node 26 sets
 * `code: 'ETIMEDOUT'` with `killed` undefined; the SIGTERM signal is what
 * the timeout kill actually sends.
 */
function isTimeoutError(err: any): boolean {
  return err?.killed === true || err?.code === 'ETIMEDOUT' || err?.signal === 'SIGTERM';
}

/**
 * Synchronous sleep that yields the CPU (unlike a Date.now() spin, which
 * pegs a core per waiting hook process — under startup lock contention every
 * concurrent hook used to spin, amplifying the very load that causes
 * timeouts).
 */
function sleepSync(ms: number): void {
  if (ms <= 0) return;
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

/**
 * Query structure for daemon commands.
 */
export interface DaemonQuery {
  cmd: 'ping' | 'search' | 'impact' | 'extract' | 'status' | 'dead' | 'arch' | 'cfg' | 'dfg' | 'slice' | 'calls' | 'warm' | 'semantic' | 'tree' | 'structure' | 'context' | 'imports' | 'importers' | 'notify' | 'diagnostics' | 'track';
  pattern?: string;
  func?: string;
  file?: string;
  function?: string;
  project?: string;
  max_results?: number;
  language?: string;
  entry_points?: string[];
  line?: number;
  direction?: 'backward' | 'forward';
  variable?: string;
  action?: 'index' | 'search';
  query?: string;
  k?: number;
  // New fields for tree, structure, context, imports, importers
  extensions?: string[];
  exclude_hidden?: boolean;
  entry?: string;
  depth?: number;
  module?: string;
  // Session tracking for token stats (P7)
  session?: string;
  // Hook activity tracking (P8)
  hook?: string;
  success?: boolean;
  metrics?: Record<string, number>;
}

/**
 * Response structure from daemon.
 */
export interface DaemonResponse {
  status?: string;
  results?: any[];
  result?: any;
  callers?: any[];
  // Import/importer lookup responses (issue #156: consumed by importsDaemon,
  // importersDaemon, edit-context-inject, impact-refactor)
  imports?: any[];
  importers?: any[];
  error?: string;
  indexing?: boolean;
  message?: string;
  uptime?: number;
  files?: number;
  // Notify response fields
  reindex_triggered?: boolean;
  dirty_count?: number;
  threshold?: number;
  // Diagnostics response fields
  type_errors?: number;
  lint_issues?: number;
  errors?: Array<{
    file: string;
    line: number;
    column?: number;
    message: string;
    severity: 'error' | 'warning';
    source: 'pyright' | 'ruff' | 'tsc' | 'eslint';
  }>;
  // Session stats (P7)
  session_stats?: {
    session_id: string;
    raw_tokens: number;
    tldr_tokens: number;
    requests: number;
    savings_percent: number;
  };
  all_sessions?: {
    active_sessions: number;
    total_raw_tokens: number;
    total_tldr_tokens: number;
    total_requests: number;
  };
  // Hook activity stats (P8)
  hook_stats?: Record<string, {
    hook_name: string;
    invocations: number;
    successes: number;
    failures: number;
    success_rate: number;
    metrics: Record<string, number>;
  }>;
  // Track response fields
  hook?: string;
  total_invocations?: number;
}

/**
 * Connection info for daemon communication.
 */
export interface ConnectionInfo {
  type: 'unix' | 'tcp';
  path?: string;
  host?: string;
  port?: number;
}

/**
 * Get connection info based on platform.
 * Mirrors the Python daemon's logic.
 *
 * @param projectDir - Absolute path to project directory
 * @returns Connection info for Unix socket or TCP
 */
export function getConnectionInfo(projectDir: string): ConnectionInfo {
  const resolvedPath = resolveProjectDir(projectDir);
  const hash = crypto.createHash('md5').update(resolvedPath).digest('hex').substring(0, 8);

  if (process.platform === 'win32') {
    // TCP on localhost with deterministic port
    const port = 49152 + (parseInt(hash, 16) % 10000);
    return { type: 'tcp', host: '127.0.0.1', port };
  } else {
    // Unix socket
    return { type: 'unix', path: `${tmpdir()}/tldr-${hash}.sock` };
  }
}

/**
 * Compute deterministic socket path from project path.
 * Mirrors the Python daemon's logic: /tmp/tldr-{md5(path)[:8]}.sock
 *
 * @param projectDir - Absolute path to project directory
 * @returns Socket path string (Unix only, use getConnectionInfo for cross-platform)
 */
export function getSocketPath(projectDir: string): string {
  const resolvedPath = resolveProjectDir(projectDir);
  const hash = crypto.createHash('md5').update(resolvedPath).digest('hex').substring(0, 8);
  return `${tmpdir()}/tldr-${hash}.sock`;
}

/**
 * Read daemon status from .tldr/status file.
 *
 * @param projectDir - Project directory path
 * @returns Status string ('ready', 'indexing', 'stopped') or null if no status file
 */
export function getStatusFile(projectDir: string): string | null {
  const statusPath = join(projectDir, '.tldr', 'status');
  if (existsSync(statusPath)) {
    try {
      return readFileSync(statusPath, 'utf-8').trim();
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Check if daemon is currently indexing.
 *
 * @param projectDir - Project directory path
 * @returns true if daemon is indexing
 */
export function isIndexing(projectDir: string): boolean {
  return getStatusFile(projectDir) === 'indexing';
}

/**
 * Check if daemon is reachable (platform-aware).
 *
 * @param projectDir - Project directory path
 * @returns true if daemon is reachable
 */
function isDaemonReachable(projectDir: string): boolean {
  const connInfo = getConnectionInfo(projectDir);

  if (connInfo.type === 'tcp') {
    // On Windows, try to connect to TCP port
    try {
      const testSocket = new net.Socket();
      testSocket.setTimeout(100);
      let connected = false;

      testSocket.on('connect', () => {
        connected = true;
        testSocket.destroy();
      });

      testSocket.on('error', () => {
        testSocket.destroy();
      });

      testSocket.connect(connInfo.port!, connInfo.host!);
      // Give it a moment
      const end = Date.now() + 200;
      while (Date.now() < end && !connected) {
        // spin
      }
      return connected;
    } catch {
      return false;
    }
  } else {
    // Unix socket - check file exists AND daemon is actually listening
    if (!existsSync(connInfo.path!)) {
      return false;
    }

    // First check if daemon process is running via PID file
    // This is more reliable than socket ping which can timeout when busy
    if (isDaemonProcessRunning(projectDir)) {
      // Process exists - socket might just be busy, don't delete it.
      // The ping is only advisory here (both outcomes return true), so
      // never spend more of the hook's budget on it than is left.
      try {
        execSync(`echo '{"cmd":"ping"}' | nc -U "${connInfo.path}"`, {
          encoding: 'utf-8',
          timeout: Math.max(50, budgetClamp(1000)),
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return true;
      } catch {
        // Ping failed but process exists - daemon is starting or busy
        // Return true to prevent spawning duplicates
        return true;
      }
    }

    // No daemon process running - try ping to verify socket isn't stale
    try {
      execSync(`echo '{"cmd":"ping"}' | nc -U "${connInfo.path}"`, {
        encoding: 'utf-8',
        timeout: Math.max(50, budgetClamp(500)),
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      return true;
    } catch (err: any) {
      if (isTimeoutError(err)) {
        // Timed out: the connect was ACCEPTED, so a listener is alive but
        // busy (a truly stale socket refuses instantly). Treat as reachable
        // — unlinking here would let a concurrent hook delete a live
        // daemon's socket and autostart a duplicate. This matters more now
        // that the deadline can clamp this probe well below 500ms.
        return true;
      }
      // Connection refused / ENOENT AND no daemon process - socket is stale
      try {
        unlinkSync(connInfo.path!);
      } catch {
        // Ignore unlink errors
      }
      return false;
    }
  }
}

/**
 * Try to start the daemon for a project.
 * Uses a lock file to prevent race conditions when multiple hooks fire in parallel.
 *
 * @param projectDir - Project directory path
 * @returns true if start was attempted successfully
 */
export function tryStartDaemon(projectDir: string): boolean {
  // Test seam (issue #156): allow disabling the blocking spawnSync auto-start
  // so missing-socket paths resolve 'unavailable' immediately instead of
  // blocking on a 'uv run tldr daemon start' subprocess. Production default
  // (env unset) is unchanged.
  if (process.env.TLDR_NO_AUTOSTART === '1') {
    return false;
  }
  // Starting the daemon can cost tens of seconds (spawn + readiness wait);
  // under a spent hook budget, report unavailable instead.
  if (budgetExhausted()) {
    return false;
  }
  try {
    // FAST CHECK: Is daemon process running? (checks PID file + kill -0)
    // This is faster and more reliable than socket ping
    if (isDaemonProcessRunning(projectDir)) {
      return true;  // Process exists, even if socket not ready yet
    }

    // SLOW CHECK: Is daemon reachable via socket?
    // Only needed if PID file doesn't exist (first start or cleaned up)
    if (isDaemonReachable(projectDir)) {
      return true;  // Already running, no need to spawn
    }

    // Try to acquire lock - if another process is starting daemon, wait for it
    if (!tryAcquireLock(projectDir)) {
      // Another process is starting the daemon - wait and check if it succeeds
      const lockWaitMs = budgetClamp(5000);
      const start = Date.now();
      while (Date.now() - start < lockWaitMs) {
        // Check process first (faster), then socket
        if (isDaemonProcessRunning(projectDir) || isDaemonReachable(projectDir)) {
          return true;
        }
        sleepSync(Math.min(100, lockWaitMs - (Date.now() - start)));
      }
      return isDaemonProcessRunning(projectDir) || isDaemonReachable(projectDir);
    }

    try {
      // The reachability checks above may have drained the budget — starting
      // a daemon we can no longer wait for would only overrun the hook.
      if (budgetExhausted()) {
        return false;
      }

      // We hold the lock - start the daemon
      const opcDir = process.env.CLAUDE_OPC_DIR || join(projectDir, 'opc');
      const tldrPath = join(opcDir, 'packages', 'tldr-code');
      let started = false;

      // Try local dev installation first (only if it exists)
      if (existsSync(tldrPath)) {
        const result = spawnSync('uv', ['run', 'tldr', 'daemon', 'start', '--project', projectDir], {
          timeout: Math.max(MIN_QUERY_BUDGET_MS, budgetClamp(10000)),
          stdio: 'ignore',
          cwd: tldrPath,
        });
        started = result.status === 0;
      }

      // Fallback to global tldr if local didn't work
      // Skip fallback in dev mode (TLDR_DEV=1) to prevent duplicate daemons
      if (!started && !process.env.TLDR_DEV) {
        spawnSync('tldr', ['daemon', 'start', '--project', projectDir], {
          timeout: Math.max(MIN_QUERY_BUDGET_MS, budgetClamp(5000)),
          stdio: 'ignore',
        });
      }

      // Wait for daemon to become reachable (up to 10s for slow starts,
      // clamped to whatever is left of the hook's budget)
      const reachWaitMs = budgetClamp(10000);
      const start = Date.now();
      while (Date.now() - start < reachWaitMs) {
        if (isDaemonReachable(projectDir)) {
          // Daemon is ready - keep lock for a bit longer to prevent races
          // (clamped: the race-prevention hold is worth less than the hook)
          sleepSync(budgetClamp(1000));
          return true;
        }
        sleepSync(Math.min(100, reachWaitMs - (Date.now() - start)));
      }

      return isDaemonReachable(projectDir);
    } finally {
      // Always release lock
      releaseLock(projectDir);
    }
  } catch {
    return false;
  }
}

/**
 * Query the daemon asynchronously using net.Socket.
 *
 * @param query - Query to send to daemon
 * @param projectDir - Project directory path
 * @returns Promise resolving to daemon response
 */
export function queryDaemon(query: DaemonQuery, projectDir: string): Promise<DaemonResponse> {
  return new Promise((resolve, reject) => {
    // Check if indexing - return early with indexing flag
    if (isIndexing(projectDir)) {
      resolve({
        indexing: true,
        status: 'indexing',
        message: 'Daemon is still indexing, results may be incomplete',
      });
      return;
    }

    const connInfo = getConnectionInfo(projectDir);

    // Check if daemon is reachable
    if (!isDaemonReachable(projectDir)) {
      // Try to start daemon
      if (!tryStartDaemon(projectDir)) {
        resolve({ status: 'unavailable', error: 'Daemon not running and could not start' });
        return;
      }
    }

    const client = new net.Socket();
    let data = '';
    let resolved = false;

    // Timeout handling
    const timer = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        client.destroy();
        resolve({ status: 'error', error: 'timeout' });
      }
    }, QUERY_TIMEOUT);

    // Connect based on platform
    if (connInfo.type === 'tcp') {
      client.connect(connInfo.port!, connInfo.host!, () => {
        client.write(JSON.stringify(query) + '\n');
      });
    } else {
      client.connect(connInfo.path!, () => {
        client.write(JSON.stringify(query) + '\n');
      });
    }

    client.on('data', (chunk) => {
      data += chunk.toString();
      if (data.includes('\n')) {
        if (!resolved) {
          resolved = true;
          clearTimeout(timer);
          client.end();
          try {
            resolve(JSON.parse(data.trim()));
          } catch {
            resolve({ status: 'error', error: 'Invalid JSON response from daemon' });
          }
        }
      }
    });

    client.on('error', (err) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);
        if (err.message.includes('ECONNREFUSED') || err.message.includes('ENOENT')) {
          resolve({ status: 'unavailable', error: 'Daemon not running' });
        } else {
          resolve({ status: 'error', error: err.message });
        }
      }
    });

    client.on('close', () => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);
        if (data) {
          try {
            resolve(JSON.parse(data.trim()));
          } catch {
            resolve({ status: 'error', error: 'Incomplete response' });
          }
        } else {
          resolve({ status: 'error', error: 'Connection closed without response' });
        }
      }
    });
  });
}

/**
 * Query the daemon synchronously using nc (netcat) or PowerShell (Windows).
 * Fallback for contexts where async is not available.
 *
 * @param query - Query to send to daemon
 * @param projectDir - Project directory path
 * @returns Daemon response
 */
export function queryDaemonSync(query: DaemonQuery, projectDir: string): DaemonResponse {
  // Deadline spent — skip the query (and especially the auto-start path)
  // rather than letting Claude Code kill the whole hook.
  if (budgetExhausted()) {
    return { status: 'unavailable', error: 'query deadline exceeded' };
  }

  // Check if indexing - return early with indexing flag
  if (isIndexing(projectDir)) {
    return {
      indexing: true,
      status: 'indexing',
      message: 'Daemon is still indexing, results may be incomplete',
    };
  }

  const connInfo = getConnectionInfo(projectDir);

  // Check if daemon is reachable
  if (!isDaemonReachable(projectDir)) {
    // Try to start daemon
    if (!tryStartDaemon(projectDir)) {
      return { status: 'unavailable', error: 'Daemon not running and could not start' };
    }
  }

  // The reachability probe / auto-start above may have consumed the rest of
  // the budget — don't begin a query that can no longer usefully complete.
  if (budgetExhausted()) {
    return { status: 'unavailable', error: 'query deadline exceeded' };
  }

  // Each execSync below must never wait longer than what is left of the budget.
  const queryTimeout = Math.max(MIN_QUERY_BUDGET_MS, budgetClamp(QUERY_TIMEOUT));

  try {
    const input = JSON.stringify(query);
    let result: string;

    if (connInfo.type === 'tcp') {
      // Windows: Use PowerShell to communicate with TCP socket
      const psCommand = `
        $client = New-Object System.Net.Sockets.TcpClient('${connInfo.host}', ${connInfo.port})
        $stream = $client.GetStream()
        $writer = New-Object System.IO.StreamWriter($stream)
        $reader = New-Object System.IO.StreamReader($stream)
        $writer.WriteLine('${input.replace(/'/g, "''")}')
        $writer.Flush()
        $response = $reader.ReadLine()
        $client.Close()
        Write-Output $response
      `.trim();

      result = execSync(`powershell -Command "${psCommand.replace(/"/g, '\\"')}"`, {
        encoding: 'utf-8',
        timeout: queryTimeout,
      });
    } else {
      // Unix: Use nc (netcat) to communicate with Unix socket
      // echo '{"cmd":"ping"}' | nc -U /tmp/tldr-xxx.sock
      result = execSync(`echo '${input}' | nc -U "${connInfo.path}"`, {
        encoding: 'utf-8',
        timeout: queryTimeout,
      });
    }

    return JSON.parse(result.trim());
  } catch (err: any) {
    if (isTimeoutError(err)) {
      return { status: 'error', error: 'timeout' };
    }
    if (err.message?.includes('ECONNREFUSED') || err.message?.includes('ENOENT')) {
      return { status: 'unavailable', error: 'Daemon not running' };
    }
    return { status: 'error', error: err.message || 'Unknown error' };
  }
}

/**
 * Convenience function to ping the daemon.
 *
 * @param projectDir - Project directory path
 * @returns true if daemon responds to ping
 */
export async function pingDaemon(projectDir: string): Promise<boolean> {
  const response = await queryDaemon({ cmd: 'ping' }, projectDir);
  return response.status === 'ok';
}

/**
 * Convenience function to search using the daemon.
 *
 * @param pattern - Search pattern
 * @param projectDir - Project directory path
 * @param maxResults - Maximum results to return
 * @returns Search results or empty array
 */
export async function searchDaemon(
  pattern: string,
  projectDir: string,
  maxResults: number = 100
): Promise<any[]> {
  const response = await queryDaemon(
    { cmd: 'search', pattern, max_results: maxResults },
    projectDir
  );
  return response.results || [];
}

/**
 * Convenience function to get impact analysis (callers of a function).
 *
 * @param funcName - Function name to analyze
 * @param projectDir - Project directory path
 * @returns Array of callers or empty array
 */
export async function impactDaemon(funcName: string, projectDir: string): Promise<any[]> {
  const response = await queryDaemon({ cmd: 'impact', func: funcName }, projectDir);
  return response.callers || [];
}

/**
 * Convenience function to extract file info.
 *
 * @param filePath - Path to file to extract
 * @param projectDir - Project directory path
 * @param sessionId - Optional session ID for token tracking
 * @returns Extraction result or null
 */
export async function extractDaemon(filePath: string, projectDir: string, sessionId?: string): Promise<any | null> {
  const response = await queryDaemon({ cmd: 'extract', file: filePath, session: sessionId }, projectDir);
  return response.result || null;
}

/**
 * Get daemon status.
 *
 * @param projectDir - Project directory path
 * @returns Status response
 */
export async function statusDaemon(projectDir: string): Promise<DaemonResponse> {
  return queryDaemon({ cmd: 'status' }, projectDir);
}

/**
 * Convenience function for dead code analysis.
 *
 * @param projectDir - Project directory path
 * @param entryPoints - Optional list of entry point patterns to exclude
 * @param language - Language to analyze (default: python)
 * @returns Dead code analysis result
 */
export async function deadCodeDaemon(
  projectDir: string,
  entryPoints?: string[],
  language: string = 'python'
): Promise<any> {
  const response = await queryDaemon(
    { cmd: 'dead', entry_points: entryPoints, language },
    projectDir
  );
  return response.result || response;
}

/**
 * Convenience function for architecture analysis.
 *
 * @param projectDir - Project directory path
 * @param language - Language to analyze (default: python)
 * @returns Architecture analysis result
 */
export async function archDaemon(projectDir: string, language: string = 'python'): Promise<any> {
  const response = await queryDaemon({ cmd: 'arch', language }, projectDir);
  return response.result || response;
}

/**
 * Convenience function for CFG extraction.
 *
 * @param filePath - Path to source file
 * @param funcName - Function name to analyze
 * @param projectDir - Project directory path
 * @param language - Language (default: python)
 * @returns CFG result
 */
export async function cfgDaemon(
  filePath: string,
  funcName: string,
  projectDir: string,
  language: string = 'python'
): Promise<any> {
  const response = await queryDaemon(
    { cmd: 'cfg', file: filePath, function: funcName, language },
    projectDir
  );
  return response.result || response;
}

/**
 * Convenience function for DFG extraction.
 *
 * @param filePath - Path to source file
 * @param funcName - Function name to analyze
 * @param projectDir - Project directory path
 * @param language - Language (default: python)
 * @returns DFG result
 */
export async function dfgDaemon(
  filePath: string,
  funcName: string,
  projectDir: string,
  language: string = 'python'
): Promise<any> {
  const response = await queryDaemon(
    { cmd: 'dfg', file: filePath, function: funcName, language },
    projectDir
  );
  return response.result || response;
}

/**
 * Convenience function for program slicing.
 *
 * @param filePath - Path to source file
 * @param funcName - Function name
 * @param line - Line number to slice from
 * @param projectDir - Project directory path
 * @param direction - backward or forward (default: backward)
 * @param variable - Optional variable to track
 * @returns Slice result with lines array
 */
export async function sliceDaemon(
  filePath: string,
  funcName: string,
  line: number,
  projectDir: string,
  direction: 'backward' | 'forward' = 'backward',
  variable?: string
): Promise<any> {
  const response = await queryDaemon(
    { cmd: 'slice', file: filePath, function: funcName, line, direction, variable },
    projectDir
  );
  return response;
}

/**
 * Convenience function for building call graph.
 *
 * @param projectDir - Project directory path
 * @param language - Language (default: python)
 * @returns Call graph result
 */
export async function callsDaemon(projectDir: string, language: string = 'python'): Promise<any> {
  const response = await queryDaemon({ cmd: 'calls', language }, projectDir);
  return response.result || response;
}

/**
 * Convenience function for cache warming.
 *
 * @param projectDir - Project directory path
 * @param language - Language (default: python)
 * @returns Warm result with file/edge counts
 */
export async function warmDaemon(projectDir: string, language: string = 'python'): Promise<any> {
  return queryDaemon({ cmd: 'warm', language }, projectDir);
}

/**
 * Convenience function for semantic search.
 *
 * @param projectDir - Project directory path
 * @param query - Search query
 * @param k - Number of results (default: 10)
 * @returns Search results
 */
export async function semanticSearchDaemon(
  projectDir: string,
  query: string,
  k: number = 10
): Promise<any[]> {
  const response = await queryDaemon(
    { cmd: 'semantic', action: 'search', query, k },
    projectDir
  );
  return response.results || [];
}

/**
 * Convenience function for semantic indexing.
 *
 * @param projectDir - Project directory path
 * @param language - Language (default: python)
 * @returns Index result with count
 */
export async function semanticIndexDaemon(
  projectDir: string,
  language: string = 'python'
): Promise<any> {
  return queryDaemon({ cmd: 'semantic', action: 'index', language }, projectDir);
}

/**
 * Convenience function for file tree.
 *
 * @param projectDir - Project directory path
 * @param extensions - File extensions to filter (optional)
 * @param excludeHidden - Exclude hidden files (default: true)
 * @returns File tree result
 */
export async function treeDaemon(
  projectDir: string,
  extensions?: string[],
  excludeHidden: boolean = true
): Promise<any> {
  const response = await queryDaemon(
    { cmd: 'tree', extensions, exclude_hidden: excludeHidden },
    projectDir
  );
  return response.result || response;
}

/**
 * Convenience function for code structure.
 *
 * @param projectDir - Project directory path
 * @param language - Language (default: python)
 * @param maxResults - Max files to analyze (default: 100)
 * @returns Code structure result
 */
export async function structureDaemon(
  projectDir: string,
  language: string = 'python',
  maxResults: number = 100
): Promise<any> {
  const response = await queryDaemon(
    { cmd: 'structure', language, max_results: maxResults },
    projectDir
  );
  return response.result || response;
}

/**
 * Convenience function for relevant context.
 *
 * @param projectDir - Project directory path
 * @param entry - Entry point (function name or Class.method)
 * @param language - Language (default: python)
 * @param depth - Call depth (default: 2)
 * @returns Relevant context result
 */
export async function contextDaemon(
  projectDir: string,
  entry: string,
  language: string = 'python',
  depth: number = 2
): Promise<any> {
  const response = await queryDaemon(
    { cmd: 'context', entry, language, depth },
    projectDir
  );
  return response.result || response;
}

/**
 * Convenience function for extracting imports from a file.
 *
 * @param projectDir - Project directory path
 * @param filePath - File path to analyze
 * @param language - Language (default: python)
 * @returns Imports array
 */
export async function importsDaemon(
  projectDir: string,
  filePath: string,
  language: string = 'python'
): Promise<any[]> {
  const response = await queryDaemon(
    { cmd: 'imports', file: filePath, language },
    projectDir
  );
  return response.imports || [];
}

/**
 * Convenience function for reverse import lookup.
 *
 * @param projectDir - Project directory path
 * @param module - Module name to search for importers
 * @param language - Language (default: python)
 * @returns Importers result with files that import the module
 */
export async function importersDaemon(
  projectDir: string,
  module: string,
  language: string = 'python'
): Promise<any> {
  return queryDaemon({ cmd: 'importers', module, language }, projectDir);
}

/**
 * Track hook activity via the daemon (P8).
 *
 * Fire-and-forget: this function catches errors silently so hooks
 * don't fail just because stats tracking is unavailable.
 *
 * @param hookName - Name of the hook (e.g., 'post-edit-diagnostics')
 * @param projectDir - Project directory path
 * @param success - Whether the hook succeeded (default: true)
 * @param metrics - Hook-specific metrics to track
 */
export function trackHookActivity(
  hookName: string,
  projectDir: string,
  success: boolean = true,
  metrics: Record<string, number> = {}
): void {
  // Fire-and-forget - don't await, don't block
  queryDaemon(
    { cmd: 'track', hook: hookName, success, metrics },
    projectDir
  ).catch(() => {
    // Silently ignore errors - stats tracking is best-effort
  });
}

/**
 * Track hook activity synchronously via the daemon (P8).
 *
 * Use this version in hooks that run synchronously and can't await.
 * Errors are silently ignored for graceful degradation.
 *
 * @param hookName - Name of the hook (e.g., 'post-edit-diagnostics')
 * @param projectDir - Project directory path
 * @param success - Whether the hook succeeded (default: true)
 * @param metrics - Hook-specific metrics to track
 */
export function trackHookActivitySync(
  hookName: string,
  projectDir: string,
  success: boolean = true,
  metrics: Record<string, number> = {}
): void {
  try {
    // Stats are best-effort: never charge them against the hook's time
    // budget and never leave anything running after the hook exits. The
    // write happens on an in-process socket that is destroyed as soon as
    // the payload flushes (no response wait), and a short timeout bounds
    // the pathological accept-but-hang case. The socket stays ref'd until
    // then — an unref'd socket lets the hook process exit before connect
    // even runs, silently dropping every sample — but the hold is at most
    // TRACK_WRITE_TIMEOUT_MS and typically single-digit ms. If the daemon
    // isn't up, drop the sample instead of auto-starting it.
    const connInfo = getConnectionInfo(projectDir);
    const payload = JSON.stringify({ cmd: 'track', hook: hookName, success, metrics }) + '\n';

    let sock: net.Socket;
    if (connInfo.type === 'tcp') {
      sock = net.createConnection({ host: connInfo.host!, port: connInfo.port! });
    } else {
      if (!connInfo.path || !existsSync(connInfo.path)) {
        return;
      }
      sock = net.createConnection(connInfo.path);
    }

    sock.setTimeout(TRACK_WRITE_TIMEOUT_MS);
    sock.on('timeout', () => sock.destroy());
    sock.on('error', () => sock.destroy());
    sock.on('connect', () => {
      sock.write(payload, () => sock.destroy());
    });
  } catch {
    // Silently ignore errors - stats tracking is best-effort
  }
}
