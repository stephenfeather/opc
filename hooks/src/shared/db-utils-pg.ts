/**
 * PostgreSQL database utilities for Claude Code hooks.
 *
 * Migrated from SQLite (db-utils.ts) to PostgreSQL.
 * Uses coordination_pg.py via Python subprocess for queries.
 *
 * Exports:
 * - getPgConnectionString(): Returns PostgreSQL connection string
 * - runPgQuery(): Executes async Python query via coordination_pg
 * - getActiveAgentCountPg(): Returns count of running agents from PostgreSQL
 * - queryBroadcasts(): Query blackboard messages for swarm coordination
 * - queryPipelineArtifacts(): Query pipeline artifacts for upstream context
 */

import { spawn, spawnSync } from 'child_process';
import type { QueryResult } from './types.js';
import { requireOpcDir } from './opc-path.js';
import { pgCoordinationStatus, getConnectionUrl } from './backend-resolution.js';

// Re-export SAFE_ID_PATTERN and isValidId from pattern-router for convenience
export { SAFE_ID_PATTERN, isValidId } from './pattern-router.js';

// ---------------------------------------------------------------------------
// Backend gate (issue #265)
// ---------------------------------------------------------------------------
//
// Every Postgres operation routes through runPgQuery / runPgQueryDetached, so
// gating here makes ALL coordination consumers (session-register, heartbeat,
// file-claims, peer-awareness, working-on-sync, crash-recovery, broadcasts)
// honor AGENTICA_MEMORY_BACKEND in one place — mirroring the Python design
// where the pure resolver decides and consumers inherit.
//
// #62 RELAXATION (intentional, user-approved): historically a missing DB URL
// threw loudly here. Per #265 and to match Python resolve_backend's
// default='sqlite', "no URL + no explicit backend" now resolves to sqlite and
// the chokepoint no-ops gracefully instead of throwing. Loud failure is
// preserved exactly where intent is explicit: an invalid backend value, or
// AGENTICA_MEMORY_BACKEND=postgres with no URL, still surfaces a (credential-
// redacted) diagnostic. Hooks never block — fail-loud, not fail-closed.

// Once-per-process guard so a sustained misconfig logs once per hook invocation
// rather than on every PG call within it (a single hook like file-claims makes
// multiple PG calls). The diagnostic only reaches stderr/debug logs, never the
// user-facing channel; cross-process throttling is out of scope.
let misconfigLogged = false;

/**
 * Decide whether a Postgres operation should proceed, emitting a one-time
 * redacted diagnostic on operator misconfiguration.
 *
 * @returns proceed=true when the backend is postgres. proceed=false when the
 *   backend is sqlite or misconfigured — callers must no-op gracefully.
 *   `reason` carries the (already credential-redacted) misconfig message when
 *   applicable.
 */
function pgGate(): { proceed: boolean; reason?: string } {
  const status = pgCoordinationStatus();
  if (status.active) {
    return { proceed: true };
  }
  if (status.misconfig) {
    if (!misconfigLogged) {
      misconfigLogged = true;
      process.stderr.write(`[db-utils-pg] ${status.misconfig}\n`);
    }
    return { proceed: false, reason: status.misconfig };
  }
  return { proceed: false };
}

/**
 * Get the PostgreSQL connection string.
 *
 * Checks environment variables in priority order:
 * 1. CONTINUOUS_CLAUDE_DB_URL (canonical)
 * 2. DATABASE_URL (backwards compat)
 * 3. OPC_POSTGRES_URL (legacy)
 *
 * Issue #62: no hardcoded development fallback. Throws if none set.
 *
 * Delegates to the shared resolver (getConnectionUrl -> resolveUrl) so URL
 * selection here is byte-identical to the backend gate's decision (#265): same
 * precedence, blank/whitespace-only values skipped, and the selected URL
 * trimmed. Without this, the gate could approve Postgres via a valid fallback
 * (e.g. DATABASE_URL) while this function returned a blank canonical
 * CONTINUOUS_CLAUDE_DB_URL and fed whitespace to the subprocess.
 *
 * @returns PostgreSQL connection string
 * @throws Error when no DB env var is set
 */
export function getPgConnectionString(): string {
  const url = getConnectionUrl();
  if (!url) {
    throw new Error(
      "Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL (preferred), " +
        "DATABASE_URL, or OPC_POSTGRES_URL. For local Docker dev, run " +
        "`docker compose -f docker/docker-compose.yml up -d` and export the " +
        "credentials from docker/.env before invoking this hook."
    );
  }
  return url;
}

/**
 * Execute a PostgreSQL query via coordination_pg.py.
 *
 * Uses spawnSync with uv run to execute async Python code.
 * The Python code receives arguments via sys.argv.
 *
 * @param pythonCode - Python code to execute (receives args via sys.argv)
 * @param args - Arguments passed to Python (sys.argv[1], sys.argv[2], ...)
 * @returns QueryResult with success, stdout, and stderr
 */
export function runPgQuery(pythonCode: string, args: string[] = []): QueryResult {
  // Backend gate (#265): when the backend is not postgres, no-op gracefully via
  // the {success:false} path every consumer already handles. See pgGate() above
  // for the #62 relaxation rationale.
  const gate = pgGate();
  if (!gate.proceed) {
    return { success: false, stdout: '', stderr: gate.reason ?? 'postgres backend inactive' };
  }

  const opcDir = requireOpcDir();

  // Resolve the DB URL up-front — BEFORE the try/catch. After the gate, the
  // backend is postgres, which guarantees a URL is present (resolveBackend
  // throws postgres-without-URL), so this no longer throws in practice; it
  // remains as a defensive resolution of the canonical URL for the subprocess.
  const resolvedDbUrl = getPgConnectionString();

  // Wrap the Python code to use asyncio.run() for async queries.
  // SECURITY: opcDir is passed via _OPC_DIR environment variable to avoid
  // code injection through paths containing quotes or special characters.
  // See: https://github.com/stephenfeather/opc/issues/88
  const wrappedCode = `
import sys
import os
import asyncio
import json

# Add opc to path for imports (read from env to avoid code injection)
_opc_dir = os.environ.get('_OPC_DIR')
if not _opc_dir:
    raise RuntimeError('_OPC_DIR environment variable not set - must be called via runPgQuery()')
sys.path.insert(0, _opc_dir)
os.chdir(_opc_dir)

${pythonCode}
`;

  try {
    const result = spawnSync('uv', ['run', 'python', '-c', wrappedCode, ...args], {
      encoding: 'utf-8',
      maxBuffer: 1024 * 1024,
      timeout: 5000,  // 5 second timeout - fail gracefully if DB unreachable
      cwd: opcDir,
      env: {
        ...process.env,
        // Never rewrite opc's uv.lock from a hook-triggered uv run (issue #71
        // follow-up); use the lock as-is. Intentional updates use `uv lock`.
        UV_FROZEN: '1',
        CONTINUOUS_CLAUDE_DB_URL: resolvedDbUrl,
        _OPC_DIR: opcDir,
      },
    });

    return {
      success: result.status === 0,
      stdout: result.stdout?.trim() || '',
      stderr: result.stderr || '',
    };
  } catch (err) {
    return {
      success: false,
      stdout: '',
      stderr: String(err),
    };
  }
}

/**
 * Query broadcasts/blackboard messages from PostgreSQL.
 *
 * Queries the blackboard table for messages in a swarm that
 * the current agent hasn't read yet.
 *
 * @param swarmId - Swarm identifier
 * @param agentId - Current agent identifier (to exclude from sender)
 * @param limit - Maximum number of messages to return
 * @returns Array of broadcast messages
 */
export function queryBroadcasts(
  swarmId: string,
  agentId: string,
  limit: number = 10
): { success: boolean; broadcasts: BroadcastMessage[] } {
  const pythonCode = `
from scripts.agentica_patterns.coordination_pg import CoordinationDBPg
import json

swarm_id = sys.argv[1]
agent_id = sys.argv[2]
limit = int(sys.argv[3])

async def main():
    async with CoordinationDBPg() as db:
        # Query blackboard for messages this agent hasn't read
        messages = await db.read_from_blackboard(swarm_id, agent_id)

        # Limit results
        messages = messages[:limit]

        # Convert to JSON-serializable format
        result = []
        for msg in messages:
            result.append({
                'sender': msg.sender_agent,
                'type': msg.message_type,
                'payload': msg.payload,
                'time': msg.created_at.isoformat() if msg.created_at else None
            })

        print(json.dumps(result))

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [swarmId, agentId, String(limit)]);

  if (!result.success) {
    return { success: false, broadcasts: [] };
  }

  try {
    const broadcasts = JSON.parse(result.stdout || '[]') as BroadcastMessage[];
    return { success: true, broadcasts };
  } catch {
    return { success: false, broadcasts: [] };
  }
}

/**
 * Query pipeline artifacts from PostgreSQL.
 *
 * Queries the pipeline_artifacts table for artifacts from upstream stages.
 *
 * @param pipelineId - Pipeline identifier
 * @param currentStage - Current stage index (will get artifacts from earlier stages)
 * @returns Array of pipeline artifacts
 */
export function queryPipelineArtifacts(
  pipelineId: string,
  currentStage: number
): { success: boolean; artifacts: PipelineArtifact[] } {
  const pythonCode = `
import asyncpg
import json
import os

pipeline_id = sys.argv[1]
current_stage = int(sys.argv[2])
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        # Query pipeline artifacts from upstream stages
        rows = await conn.fetch('''
            SELECT stage_index, artifact_type, artifact_path, artifact_content, created_at
            FROM pipeline_artifacts
            WHERE pipeline_id = $1 AND stage_index < $2
            ORDER BY stage_index ASC, created_at DESC
        ''', pipeline_id, current_stage)

        artifacts = []
        for row in rows:
            artifacts.append({
                'stage': row['stage_index'],
                'type': row['artifact_type'],
                'path': row['artifact_path'],
                'content': row['artifact_content'],
                'time': row['created_at'].isoformat() if row['created_at'] else None
            })

        print(json.dumps(artifacts))
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [pipelineId, String(currentStage)]);

  if (!result.success) {
    return { success: false, artifacts: [] };
  }

  try {
    const artifacts = JSON.parse(result.stdout || '[]') as PipelineArtifact[];
    return { success: true, artifacts };
  } catch {
    return { success: false, artifacts: [] };
  }
}

/**
 * Get count of active (running) agents from PostgreSQL.
 *
 * Queries the agents table for agents with status='running'.
 *
 * @returns Number of running agents, or 0 on any error
 */
export function getActiveAgentCountPg(): number {
  const pythonCode = `
from scripts.agentica_patterns.coordination_pg import CoordinationDBPg
import json

async def main():
    async with CoordinationDBPg() as db:
        agents = await db.get_running_agents()
        print(len(agents))

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode);

  if (!result.success) {
    return 0;
  }

  const count = parseInt(result.stdout, 10);
  return isNaN(count) ? 0 : count;
}

/**
 * Register a new agent in PostgreSQL.
 *
 * @param agentId - Unique agent identifier
 * @param sessionId - Session that spawned the agent
 * @param pattern - Coordination pattern (swarm, hierarchical, etc.)
 * @param pid - Process ID for orphan detection
 * @returns Object with success boolean and any error message
 */
export function registerAgentPg(
  agentId: string,
  sessionId: string,
  pattern: string | null = null,
  pid: number | null = null
): { success: boolean; error?: string } {
  const pythonCode = `
from scripts.agentica_patterns.coordination_pg import CoordinationDBPg
import json

agent_id = sys.argv[1]
session_id = sys.argv[2]
pattern = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != 'null' else None
pid = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != 'null' else None

async def main():
    try:
        async with CoordinationDBPg() as db:
            await db.register_agent(
                agent_id=agent_id,
                session_id=session_id,
                pattern=pattern,
                pid=pid
            )
        print('ok')
    except Exception as e:
        print(f'error: {e}')

asyncio.run(main())
`;

  const args = [
    agentId,
    sessionId,
    pattern || 'null',
    pid !== null ? String(pid) : 'null',
  ];

  const result = runPgQuery(pythonCode, args);

  if (!result.success || result.stdout !== 'ok') {
    return {
      success: false,
      error: result.stderr || result.stdout || 'Unknown error',
    };
  }

  return { success: true };
}

/**
 * Mark an agent as completed in PostgreSQL.
 *
 * @param agentId - Agent identifier to complete
 * @param status - Final status ('completed' or 'failed')
 * @param errorMessage - Optional error message for failed status
 * @returns Object with success boolean and any error message
 */
export function completeAgentPg(
  agentId: string,
  status: string = 'completed',
  errorMessage: string | null = null
): { success: boolean; error?: string } {
  const pythonCode = `
from scripts.agentica_patterns.coordination_pg import CoordinationDBPg
import json

agent_id = sys.argv[1]
status = sys.argv[2]
error_message = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != 'null' else None

async def main():
    try:
        async with CoordinationDBPg() as db:
            await db.complete_agent(
                agent_id=agent_id,
                status=status,
                result_summary=error_message
            )
        print('ok')
    except Exception as e:
        print(f'error: {e}')

asyncio.run(main())
`;

  const args = [
    agentId,
    status,
    errorMessage || 'null',
  ];

  const result = runPgQuery(pythonCode, args);

  if (!result.success || result.stdout !== 'ok') {
    return {
      success: false,
      error: result.stderr || result.stdout || 'Unknown error',
    };
  }

  return { success: true };
}

// Type definitions for broadcast messages
export interface BroadcastMessage {
  sender: string;
  type: string;
  payload: Record<string, unknown>;
  time: string | null;
}

// Type definitions for pipeline artifacts
export interface PipelineArtifact {
  stage: number;
  type: string;
  path: string | null;
  content: string | null;
  time: string | null;
}

// =============================================================================
// COORDINATION LAYER: Session Registration
// =============================================================================

/**
 * Register a session in the coordination layer.
 *
 * @param sessionId - Unique session identifier
 * @param project - Project directory path
 * @param workingOn - Description of current task
 * @returns Object with success boolean and any error message
 */
export function registerSession(
  sessionId: string,
  project: string,
  workingOn: string = '',
  claudeSessionId?: string,
  transcriptPath?: string,
  pid?: number
): { success: boolean; error?: string } {
  const pythonCode = `
import asyncpg
import os
from datetime import datetime

session_id = sys.argv[1]
project = sys.argv[2]
working_on = sys.argv[3] if len(sys.argv) > 3 else ''
claude_session_id = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != 'null' else None
transcript_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != 'null' else None
pid = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != 'null' else None
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        # Create table if not exists
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                working_on TEXT,
                started_at TIMESTAMP DEFAULT NOW(),
                last_heartbeat TIMESTAMP DEFAULT NOW()
            )
        ''')

        # Migrate schema: add columns for crash recovery
        for col in [
            ('claude_session_id', 'TEXT'),
            ('transcript_path', 'TEXT'),
            ('exited_at', 'TIMESTAMP'),
            ('pid', 'INTEGER'),
            # Issue #228 item 2: already-surfaced filtering. Self-heal a fresh
            # DB the hook touches before the migration runs.
            ('surfaced_learning_ids', 'UUID[]'),
        ]:
            await conn.execute(f'ALTER TABLE sessions ADD COLUMN IF NOT EXISTS {col[0]} {col[1]}')

        # Upsert session (clear exited_at on re-register, e.g. resume)
        await conn.execute('''
            INSERT INTO sessions (id, project, working_on, claude_session_id, transcript_path, pid, started_at, last_heartbeat, exited_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW(), NULL)
            ON CONFLICT (id) DO UPDATE SET
                -- Issue #65: SessionStart/resume re-registers with working_on=''.
                -- COALESCE+NULLIF preserves an existing label when the new value
                -- is blank, so the working-on-sync hook's value survives a resume;
                -- a non-empty value still updates.
                working_on = COALESCE(NULLIF(EXCLUDED.working_on, ''), sessions.working_on),
                claude_session_id = EXCLUDED.claude_session_id,
                transcript_path = EXCLUDED.transcript_path,
                pid = EXCLUDED.pid,
                last_heartbeat = NOW(),
                exited_at = NULL
        ''', session_id, project, working_on, claude_session_id, transcript_path, pid)

        print('ok')
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [
    sessionId,
    project,
    workingOn,
    claudeSessionId || 'null',
    transcriptPath || 'null',
    pid !== undefined ? String(pid) : 'null',
  ]);

  if (!result.success || result.stdout !== 'ok') {
    return {
      success: false,
      error: result.stderr || result.stdout || 'Unknown error',
    };
  }

  return { success: true };
}

/**
 * Get active sessions from the coordination layer.
 *
 * @param project - Optional project filter
 * @returns Array of active sessions
 */
export function getActiveSessions(project?: string): {
  success: boolean;
  sessions: SessionInfo[];
} {
  const pythonCode = `
import asyncpg
import os
import json
from datetime import datetime, timedelta

project_filter = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != 'null' else None
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        # Get sessions active in last 5 minutes
        cutoff = datetime.utcnow() - timedelta(minutes=5)

        if project_filter:
            rows = await conn.fetch('''
                SELECT id, project, working_on, started_at, last_heartbeat
                FROM sessions
                WHERE project = $1 AND last_heartbeat > $2
                ORDER BY started_at DESC
            ''', project_filter, cutoff)
        else:
            rows = await conn.fetch('''
                SELECT id, project, working_on, started_at, last_heartbeat
                FROM sessions
                WHERE last_heartbeat > $1
                ORDER BY started_at DESC
            ''', cutoff)

        sessions = []
        for row in rows:
            sessions.append({
                'id': row['id'],
                'project': row['project'],
                'working_on': row['working_on'],
                'started_at': row['started_at'].isoformat() if row['started_at'] else None,
                'last_heartbeat': row['last_heartbeat'].isoformat() if row['last_heartbeat'] else None
            })

        print(json.dumps(sessions))
    except Exception as e:
        print(json.dumps([]))
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [project || 'null']);

  if (!result.success) {
    return { success: false, sessions: [] };
  }

  try {
    const sessions = JSON.parse(result.stdout || '[]') as SessionInfo[];
    return { success: true, sessions };
  } catch {
    return { success: false, sessions: [] };
  }
}

// =============================================================================
// COORDINATION LAYER: Session Exit & Crash Detection
// =============================================================================

/**
 * Mark a session as cleanly exited.
 *
 * @param claudeSessionId - Claude's session UUID (from SessionEnd input)
 * @returns Object with success boolean and any error message
 */
export function markSessionExited(
  claudeSessionId: string
): { success: boolean; error?: string } {
  const pythonCode = `
import asyncpg
import os

claude_session_id = sys.argv[1]
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        result = await conn.execute('''
            UPDATE sessions SET exited_at = NOW()
            WHERE claude_session_id = $1 AND exited_at IS NULL
        ''', claude_session_id)
        print('ok')
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [claudeSessionId]);

  if (!result.success || result.stdout !== 'ok') {
    return {
      success: false,
      error: result.stderr || result.stdout || 'Unknown error',
    };
  }

  return { success: true };
}

/**
 * Find sessions that may have crashed (no clean exit recorded).
 *
 * Returns ALL sessions with exited_at IS NULL for this project.
 * The caller is responsible for determining if a session actually crashed
 * (via PID liveness check or stale heartbeat fallback).
 *
 * @param project - Project directory to check
 * @returns Array of unexited sessions with transcript paths and PIDs
 */
export function getCrashedSessions(
  project: string,
): { success: boolean; sessions: CrashedSession[] } {
  const pythonCode = `
import asyncpg
import os
import json

project = sys.argv[1]
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        rows = await conn.fetch('''
            SELECT id, project, claude_session_id, transcript_path, pid, started_at, last_heartbeat
            FROM sessions
            WHERE project = $1
              AND exited_at IS NULL
            ORDER BY started_at DESC
        ''', project)

        sessions = []
        for row in rows:
            sessions.append({
                'id': row['id'],
                'project': row['project'],
                'claude_session_id': row['claude_session_id'],
                'transcript_path': row['transcript_path'],
                'pid': row['pid'],
                'started_at': row['started_at'].isoformat() if row['started_at'] else None,
                'last_heartbeat': row['last_heartbeat'].isoformat() if row['last_heartbeat'] else None
            })

        print(json.dumps(sessions))
    except Exception as e:
        print(json.dumps([]))
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [project]);

  if (!result.success) {
    return { success: false, sessions: [] };
  }

  try {
    const sessions = JSON.parse(result.stdout || '[]') as CrashedSession[];
    return { success: true, sessions };
  } catch {
    return { success: false, sessions: [] };
  }
}

/**
 * Mark crashed sessions as acknowledged (set exited_at so they aren't detected again).
 *
 * @param sessionIds - Coordination session IDs to mark
 */
export function markSessionsAcknowledged(
  sessionIds: string[]
): { success: boolean } {
  if (sessionIds.length === 0) return { success: true };

  const pythonCode = `
import asyncpg
import os
import json

session_ids = json.loads(sys.argv[1])
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('''
            UPDATE sessions SET exited_at = NOW()
            WHERE id = ANY($1::text[])
        ''', session_ids)
        print('ok')
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [JSON.stringify(sessionIds)]);
  return { success: result.success && result.stdout === 'ok' };
}

// =============================================================================
// COORDINATION LAYER: File Claims
// =============================================================================

/**
 * Check if a file is claimed by another session.
 *
 * @param filePath - Path to the file
 * @param project - Project directory
 * @param mySessionId - Current session ID
 * @returns Claim info if claimed by another session
 */
export function checkFileClaim(
  filePath: string,
  project: string,
  mySessionId: string
): { claimed: boolean; claimedBy?: string; claimedAt?: string } {
  const pythonCode = `
import asyncpg
import os
import json

file_path = sys.argv[1]
project = sys.argv[2]
my_session_id = sys.argv[3]
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        # Create table if not exists
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS file_claims (
                file_path TEXT,
                project TEXT,
                session_id TEXT,
                claimed_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (file_path, project)
            )
        ''')

        row = await conn.fetchrow('''
            SELECT session_id, claimed_at FROM file_claims
            WHERE file_path = $1 AND project = $2 AND session_id != $3
        ''', file_path, project, my_session_id)

        if row:
            print(json.dumps({
                'claimed': True,
                'claimedBy': row['session_id'],
                'claimedAt': row['claimed_at'].isoformat() if row['claimed_at'] else None
            }))
        else:
            print(json.dumps({'claimed': False}))
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [filePath, project, mySessionId]);

  if (!result.success) {
    return { claimed: false };
  }

  try {
    return JSON.parse(result.stdout || '{"claimed": false}');
  } catch {
    return { claimed: false };
  }
}

/**
 * Claim a file for the current session.
 *
 * @param filePath - Path to the file
 * @param project - Project directory
 * @param sessionId - Session claiming the file
 */
export function claimFile(
  filePath: string,
  project: string,
  sessionId: string
): { success: boolean } {
  const pythonCode = `
import asyncpg
import os

file_path = sys.argv[1]
project = sys.argv[2]
session_id = sys.argv[3]
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('''
            INSERT INTO file_claims (file_path, project, session_id, claimed_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (file_path, project) DO UPDATE SET
                session_id = EXCLUDED.session_id,
                claimed_at = NOW()
        ''', file_path, project, session_id)
        print('ok')
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [filePath, project, sessionId]);
  return { success: result.success && result.stdout === 'ok' };
}

// =============================================================================
// COORDINATION LAYER: Findings
// =============================================================================

/**
 * Broadcast a finding to the coordination layer.
 *
 * @param sessionId - Session that discovered the finding
 * @param topic - Topic/category of the finding
 * @param finding - The finding content
 * @param relevantTo - Array of files/topics this is relevant to
 */
export function broadcastFinding(
  sessionId: string,
  topic: string,
  finding: string,
  relevantTo: string[] = []
): { success: boolean } {
  const pythonCode = `
import asyncpg
import os
import json

session_id = sys.argv[1]
topic = sys.argv[2]
finding = sys.argv[3]
relevant_to = json.loads(sys.argv[4]) if len(sys.argv) > 4 else []
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        # Create table if not exists
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS findings (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                finding TEXT NOT NULL,
                relevant_to TEXT[],
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')

        await conn.execute('''
            INSERT INTO findings (session_id, topic, finding, relevant_to)
            VALUES ($1, $2, $3, $4)
        ''', session_id, topic, finding, relevant_to)
        print('ok')
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [
    sessionId,
    topic,
    finding,
    JSON.stringify(relevantTo),
  ]);
  return { success: result.success && result.stdout === 'ok' };
}

/**
 * Get relevant findings for a topic or file.
 *
 * @param query - Topic or file path to search for
 * @param excludeSessionId - Session to exclude (usually current session)
 * @param limit - Maximum findings to return
 */
export function getRelevantFindings(
  query: string,
  excludeSessionId: string,
  limit: number = 5
): { success: boolean; findings: FindingInfo[] } {
  const pythonCode = `
import asyncpg
import os
import json

query = sys.argv[1]
exclude_session = sys.argv[2]
limit = int(sys.argv[3])
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        # Search by topic match or relevance
        rows = await conn.fetch('''
            SELECT session_id, topic, finding, relevant_to, created_at
            FROM findings
            WHERE session_id != $1
              AND (topic ILIKE '%' || $2 || '%'
                   OR $2 = ANY(relevant_to)
                   OR finding ILIKE '%' || $2 || '%')
            ORDER BY created_at DESC
            LIMIT $3
        ''', exclude_session, query, limit)

        findings = []
        for row in rows:
            findings.append({
                'session_id': row['session_id'],
                'topic': row['topic'],
                'finding': row['finding'],
                'relevant_to': row['relevant_to'],
                'created_at': row['created_at'].isoformat() if row['created_at'] else None
            })

        print(json.dumps(findings))
    except Exception as e:
        print(json.dumps([]))
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [query, excludeSessionId, String(limit)]);

  if (!result.success) {
    return { success: false, findings: [] };
  }

  try {
    const findings = JSON.parse(result.stdout || '[]') as FindingInfo[];
    return { success: true, findings };
  } catch {
    return { success: false, findings: [] };
  }
}

/**
 * Update the heartbeat timestamp for an active session.
 *
 * @param sessionId - Session identifier to refresh
 * @param project - Project directory path
 * @returns Object with success boolean and any error message
 */
export function updateHeartbeat(
  sessionId: string,
  project: string,
): { success: boolean; error?: string } {
  const pythonCode = `
import asyncpg
import os

session_id = sys.argv[1]
project = sys.argv[2]
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        status = await conn.execute('''
            UPDATE sessions SET last_heartbeat = NOW()
            WHERE id = $1 AND project = $2
        ''', session_id, project)
        if status == 'UPDATE 1':
            print('ok')
        else:
            print('not found')
    finally:
        await conn.close()

asyncio.run(main())
`;

  const result = runPgQuery(pythonCode, [sessionId, project]);

  if (!result.success || result.stdout !== 'ok') {
    return {
      success: false,
      error: result.stderr || result.stdout || 'Unknown error',
    };
  }

  return { success: true };
}

/**
 * Execute a PostgreSQL query in a detached background process.
 *
 * Spawns the uv/Python subprocess as detached and calls unref() so the
 * parent process never waits for it. Errors are silently swallowed because
 * callers that use this path have explicitly opted into fire-and-forget
 * semantics (e.g., heartbeat refreshes).
 *
 * @param pythonCode - Python code to execute (receives args via sys.argv)
 * @param args - Arguments passed to Python (sys.argv[1], sys.argv[2], ...)
 */
export function runPgQueryDetached(pythonCode: string, args: string[] = []): void {
  // Backend gate (#265): when the backend is not postgres, no-op (no spawn). The
  // fire-and-forget detached path is the high-frequency heartbeat route, so a
  // sqlite override must not spawn a doomed subprocess. See pgGate() above for
  // the #62 relaxation rationale.
  if (!pgGate().proceed) {
    return;
  }

  // Resolve DB URL up-front, BEFORE the try/catch. After the gate the backend is
  // postgres (which guarantees a URL is present), so this resolves the canonical
  // URL for the subprocess without throwing in practice.
  const resolvedDbUrl = getPgConnectionString();
  const opcDir = requireOpcDir();
  try {
    const wrappedCode = `
import sys
import os
import asyncio
import json

# Add opc to path for imports
sys.path.insert(0, '${opcDir}')
os.chdir('${opcDir}')

${pythonCode}
`;

    const child = spawn('uv', ['run', 'python', '-c', wrappedCode, ...args], {
      detached: true,
      stdio: 'ignore',
      cwd: opcDir,
      env: {
        ...process.env,
        // Never rewrite opc's uv.lock from a hook-triggered uv run (issue #71
        // follow-up); the frequent heartbeat path runs through here.
        UV_FROZEN: '1',
        CONTINUOUS_CLAUDE_DB_URL: resolvedDbUrl,
      },
    });

    child.unref();
  } catch {
    // Fire-and-forget: swallow spawn-path errors so the hook never blocks.
    // Configuration errors (missing DB URL) have already been raised above
    // before reaching this try, so they remain loud.
  }
}

/**
 * Update the heartbeat timestamp in a detached background process.
 *
 * Unlike updateHeartbeat(), this function returns immediately and never
 * blocks the caller. Use this in PostToolUse hooks where adding latency
 * on every tool call is unacceptable.
 *
 * @param sessionId - Session identifier to refresh
 * @param project - Project directory path
 */
export function updateHeartbeatDetached(sessionId: string, project: string): void {
  const pythonCode = `
import asyncpg
import os

session_id = sys.argv[1]
project = sys.argv[2]
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('''
            UPDATE sessions SET last_heartbeat = NOW()
            WHERE id = $1 AND project = $2
        ''', session_id, project)
    finally:
        await conn.close()

asyncio.run(main())
`;

  runPgQueryDetached(pythonCode, [sessionId, project]);
}

/**
 * Update sessions.working_on in a detached background process.
 *
 * Issue #65: populated by the working-on-sync PostToolUse hook so peer
 * sessions can see what each session is doing. Fire-and-forget — never
 * adds latency to the tool call that triggered it.
 *
 * @param sessionId - Session identifier to update
 * @param project - Project directory path
 * @param workingOn - Human-readable label of current work ('' clears it)
 */
export function updateWorkingOnDetached(
  sessionId: string,
  project: string,
  workingOn: string,
): void {
  const pythonCode = `
import asyncpg
import os

session_id = sys.argv[1]
project = sys.argv[2]
working_on = sys.argv[3]
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL') or os.environ.get('OPC_POSTGRES_URL')
if not pg_url:
    sys.exit('ERROR: Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL, DATABASE_URL, or OPC_POSTGRES_URL.')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('''
            UPDATE sessions SET working_on = $3, last_heartbeat = NOW()
            WHERE id = $1 AND project = $2
        ''', session_id, project, working_on)
    finally:
        await conn.close()

asyncio.run(main())
`;

  runPgQueryDetached(pythonCode, [sessionId, project, workingOn]);
}

// Type definitions for sessions

export interface SessionInfo {
  id: string;
  project: string;
  working_on: string;
  started_at: string | null;
  last_heartbeat: string | null;
}

export interface CrashedSession {
  id: string;
  project: string;
  claude_session_id: string | null;
  transcript_path: string | null;
  pid: number | null;
  started_at: string | null;
  last_heartbeat: string | null;
}

// Type definitions for findings
export interface FindingInfo {
  session_id: string;
  topic: string;
  finding: string;
  relevant_to: string[];
  created_at: string | null;
}
