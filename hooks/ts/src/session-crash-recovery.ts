/**
 * Session Crash Recovery Hook (SessionStart)
 *
 * Detects when a previous session crashed (no clean exit recorded in DB)
 * and creates a recovery handoff from the old transcript.
 *
 * Flow:
 * 1. Query PostgreSQL sessions table for crashed sessions on this project
 *    (exited_at IS NULL AND last_heartbeat is stale)
 * 2. If found → parse the old transcript using transcript-parser
 * 3. Create a recovery handoff YAML in the standard format
 * 4. Mark crashed sessions as acknowledged in DB
 * 5. Inform the user via system message
 */

import { existsSync, writeFileSync, mkdirSync } from 'fs';
import { readFileSync } from 'fs';
import { execSync } from 'child_process';
import * as path from 'path';
import { parseTranscript, generateAutoHandoff } from './transcript-parser.js';
import { getCrashedSessions, markSessionsAcknowledged } from './shared/db-utils-pg.js';
import type { CrashedSession } from './shared/db-utils-pg.js';

interface SessionStartInput {
  session_id: string;
  transcript_path: string;
  type?: string; // startup | resume | compact | clear
}

interface HookOutput {
  systemMessage?: string;
  hookSpecificOutput?: {
    hookEventName: string;
    additionalContext?: string;
  };
}

/**
 * Determine session name using fallback chain:
 * 1. Existing handoffs folder
 * 2. Git worktree name
 * 3. Git branch name
 * 4. Directory name
 */
function getSessionName(projectDir: string): string {
  // 1. Existing handoffs
  try {
    const handoffsDir = path.join(projectDir, 'thoughts', 'shared', 'handoffs');
    if (existsSync(handoffsDir)) {
      const result = execSync(
        `ls -td "${handoffsDir}"/*/ 2>/dev/null | head -1 | xargs basename`,
        { encoding: 'utf-8', timeout: 5000, stdio: ['pipe', 'pipe', 'pipe'] }
      ).trim();
      if (result) return result;
    }
  } catch { /* continue */ }

  // 2. Git worktree name
  try {
    const result = execSync(
      'basename "$(git worktree list --porcelain 2>/dev/null | head -1 | sed \'s/^worktree //\')" 2>/dev/null',
      { cwd: projectDir, encoding: 'utf-8', timeout: 5000, stdio: ['pipe', 'pipe', 'pipe'] }
    ).trim();
    if (result) return result;
  } catch { /* continue */ }

  // 3. Git branch name
  try {
    const result = execSync('git branch --show-current 2>/dev/null', {
      cwd: projectDir, encoding: 'utf-8', timeout: 5000, stdio: ['pipe', 'pipe', 'pipe']
    }).trim();
    if (result) return result;
  } catch { /* continue */ }

  // 4. Directory name
  return path.basename(projectDir);
}

function createRecoveryHandoff(
  crashed: CrashedSession,
  projectDir: string
): string | null {
  if (!crashed.transcript_path || !existsSync(crashed.transcript_path)) {
    return null;
  }

  const summary = parseTranscript(crashed.transcript_path);

  // Skip if the session did nothing (no tool calls, no files modified)
  if (summary.recentToolCalls.length === 0 && summary.filesModified.length === 0) {
    return null;
  }

  const sessionName = getSessionName(projectDir);
  let handoffContent = generateAutoHandoff(summary, sessionName);

  // Patch the auto-handoff to indicate crash recovery
  handoffContent = handoffContent
    .replace('outcome: PARTIAL_PLUS', 'outcome: PARTIAL_MINUS')
    .replace('status: partial', 'status: crashed')
    .replace(
      'auto_compact: "Context limit reached, auto-compacted"',
      'crash_recovery: "Previous session ended unexpectedly (CLI crash/hang)"'
    );

  // Write the recovery handoff
  const handoffDir = path.join(projectDir, 'thoughts', 'shared', 'handoffs', sessionName);
  mkdirSync(handoffDir, { recursive: true });

  const now = new Date();
  const dateStr = now.toISOString().split('T')[0];
  const timeStr = `${String(now.getHours()).padStart(2, '0')}-${String(now.getMinutes()).padStart(2, '0')}`;
  const filename = `${dateStr}_${timeStr}_crash-recovery.yaml`;
  const handoffPath = path.join(handoffDir, filename);

  writeFileSync(handoffPath, handoffContent);

  return `thoughts/shared/handoffs/${sessionName}/${filename}`;
}

/**
 * Check if a process is alive.
 * Sends signal 0 which checks existence without actually signaling.
 */
function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/**
 * Determine if an unexited session actually crashed.
 *
 * Strategy:
 * 1. If PID is stored and dead → definite crash (immediate detection)
 * 2. If PID is stored and alive → still running, not crashed
 * 3. If no PID → fall back to stale heartbeat (>5 min)
 */
function isSessionCrashed(session: CrashedSession): boolean {
  if (session.pid) {
    return !isProcessAlive(session.pid);
  }
  // No PID stored - fall back to stale heartbeat
  if (!session.last_heartbeat) return true;
  const heartbeat = new Date(session.last_heartbeat).getTime();
  const staleThreshold = 5 * 60 * 1000; // 5 minutes
  return Date.now() - heartbeat > staleThreshold;
}

async function main() {
  const input: SessionStartInput = JSON.parse(readFileSync(0, 'utf-8'));
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();

  // Only run on startup (not resume/compact/clear)
  if (input.type && input.type !== 'startup') {
    console.log('{}');
    return;
  }

  // Query DB for all unexited sessions on this project
  const result = getCrashedSessions(projectDir);

  if (!result.success || result.sessions.length === 0) {
    console.log('{}');
    return;
  }

  // Filter to actually crashed sessions (dead PID or stale heartbeat)
  const crashedSessions = result.sessions.filter(isSessionCrashed);

  if (crashedSessions.length === 0) {
    console.log('{}');
    return;
  }

  // Process the most recent crashed session
  const crashed = crashedSessions[0];
  const handoffPath = createRecoveryHandoff(crashed, projectDir);

  // Mark all crashed sessions as acknowledged so they aren't detected again
  const crashedIds = crashedSessions.map(s => s.id);
  markSessionsAcknowledged(crashedIds);

  if (handoffPath) {
    const contextMsg = [
      'Previous session ended unexpectedly (crash/hang).',
      `Recovery handoff created: ${handoffPath}`,
      `Resume with: /resume_handoff ${handoffPath}`,
    ].join('\n');

    const output: HookOutput = {
      systemMessage: `⚠️ Crash detected! Recovery handoff: ${handoffPath}`,
      hookSpecificOutput: {
        hookEventName: 'SessionStart',
        additionalContext: contextMsg,
      },
    };
    console.log(JSON.stringify(output));
  } else {
    console.log('{}');
  }
}

main().catch(() => console.log('{}'));
