/**
 * Pure functions for session context injection.
 *
 * Provides health checks, task summary extraction, peer session formatting,
 * and file-based peer cache management. Used by session-register.ts (SessionStart)
 * and peer-awareness.ts (UserPromptSubmit).
 *
 * All functions are pure or have isolated I/O — no DB access, no stdin parsing.
 */

import { readFileSync, writeFileSync, mkdirSync, renameSync } from 'fs';
import { dirname, join } from 'path';
import type { SessionInfo } from './shared/db-utils-pg.js';

// ---------------------------------------------------------------------------
// Health checks
// ---------------------------------------------------------------------------

export interface HealthStatus {
  pgHealthy: boolean;
  daemonRunning: boolean;
}

/**
 * Check memory system health: PG connectivity and daemon liveness.
 *
 * @param pgRegistrationSucceeded - Whether the session registration DB call succeeded
 * @param pidFilePath - Path to the daemon PID file (~/.claude/memory-daemon.pid)
 * @returns Health status for PG and daemon
 */
export function checkMemoryHealth(
  pgRegistrationSucceeded: boolean,
  pidFilePath: string,
): HealthStatus {
  const pgHealthy = pgRegistrationSucceeded;
  let daemonRunning = false;

  try {
    const pidContent = readFileSync(pidFilePath, 'utf-8').trim();
    const pid = parseInt(pidContent, 10);
    if (!isNaN(pid) && pid > 0) {
      process.kill(pid, 0); // signal 0 = existence check
      daemonRunning = true;
    }
  } catch {
    // Process not found or permission error — daemon is not running
    daemonRunning = false;
  }

  return { pgHealthy, daemonRunning };
}

/**
 * Format health warnings for injection. Returns null if everything is healthy.
 */
export function formatHealthWarnings(health: HealthStatus): string | null {
  const warnings: string[] = [];

  if (!health.pgHealthy) {
    warnings.push('- PostgreSQL: unreachable');
  }
  if (!health.daemonRunning) {
    warnings.push('- Memory daemon: not running');
  }

  if (warnings.length === 0) return null;

  return `Health warnings:\n${warnings.join('\n')}`;
}

// ---------------------------------------------------------------------------
// Tasks summary
// ---------------------------------------------------------------------------

/**
 * Extract a compact summary of pending tasks from Tasks.md.
 *
 * @param tasksFilePath - Absolute path to Tasks.md
 * @returns Summary string like "Pending tasks (5): Title1, Title2, Title3, ..."
 *          or null if file is missing/empty/has no tasks
 */
export function getPendingTasksSummary(tasksFilePath: string): string | null {
  try {
    const content = readFileSync(tasksFilePath, 'utf-8');
    if (!content.trim()) return null;

    const titles = content
      .split('\n')
      .filter(line => line.startsWith('## '))
      .map(line => line.slice(3).trim());

    if (titles.length === 0) return null;

    const MAX_SHOWN = 3;
    const shown = titles.slice(0, MAX_SHOWN);
    const suffix = titles.length > MAX_SHOWN ? ', ...' : '';

    return `Pending tasks (${titles.length}): ${shown.join(', ')}${suffix}`;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Peer session formatting
// ---------------------------------------------------------------------------

/**
 * Format peer sessions for context injection.
 * Returns null if no peers (silent when solo).
 */
export function formatPeerMessage(peers: SessionInfo[]): string | null {
  if (peers.length === 0) return null;

  const lines = peers.map(
    s => `- ${s.id}: ${s.working_on || 'working...'}`,
  );

  return `Active peer sessions (${peers.length}):\n${lines.join('\n')}`;
}

// ---------------------------------------------------------------------------
// Peer cache (file-based TTL)
// ---------------------------------------------------------------------------

interface PeerCache {
  cached_at: string;
  project: string;
  sessions: SessionInfo[];
}

/**
 * Read peer sessions from file cache.
 * Returns null if cache is missing, corrupt, stale, or for a different project.
 *
 * @param cachePath - Path to peer-sessions.json
 * @param project - Cache key for project isolation (use full canonical path to prevent cross-project collisions)
 * @param ttlSeconds - Cache TTL in seconds (entries >= this age are stale)
 */
export function readPeerCache(
  cachePath: string,
  project: string,
  ttlSeconds: number,
): SessionInfo[] | null {
  try {
    const raw = readFileSync(cachePath, 'utf-8');
    const data = JSON.parse(raw) as Record<string, unknown>;

    // Validate cache shape before trusting
    if (typeof data.cached_at !== 'string' || typeof data.project !== 'string' || !Array.isArray(data.sessions)) {
      return null;
    }

    if (data.project !== project) return null;

    const cachedTime = new Date(data.cached_at).getTime();
    if (!isFinite(cachedTime)) return null;

    const age = (Date.now() - cachedTime) / 1000;
    if (age >= ttlSeconds) return null;

    return data.sessions as SessionInfo[];
  } catch {
    return null;
  }
}

/**
 * Write peer sessions to file cache.
 * Creates parent directory if needed.
 */
export function writePeerCache(
  cachePath: string,
  project: string,
  sessions: SessionInfo[],
): void {
  try {
    const dir = dirname(cachePath);
    mkdirSync(dir, { recursive: true });

    const data: PeerCache = {
      cached_at: new Date().toISOString(),
      project,
      sessions,
    };

    // Atomic write: write to temp file then rename to avoid corruption from concurrent hooks
    const tmpPath = cachePath + '.tmp.' + process.pid;
    writeFileSync(tmpPath, JSON.stringify(data), { encoding: 'utf-8' });
    renameSync(tmpPath, cachePath);
  } catch {
    // Cache write failure is non-fatal — next call will query DB
  }
}
