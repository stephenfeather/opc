// src/session-register.ts
import { readFileSync as readFileSync3 } from "fs";
import { join as join2 } from "path";

// src/shared/db-utils-pg.ts
import { spawn, spawnSync } from "child_process";

// src/shared/opc-path.ts
import { existsSync, readFileSync } from "fs";
import { join } from "path";
function getOpcDirFromConfig() {
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  if (!homeDir) return null;
  const configPath = join(homeDir, ".claude", "opc.json");
  if (!existsSync(configPath)) return null;
  try {
    const content = readFileSync(configPath, "utf-8");
    const config = JSON.parse(content);
    const opcDir = config.opc_dir;
    if (opcDir && typeof opcDir === "string" && existsSync(opcDir)) {
      return opcDir;
    }
  } catch {
  }
  return null;
}
function getOpcDir() {
  const envOpcDir = process.env.CLAUDE_OPC_DIR;
  if (envOpcDir && existsSync(envOpcDir)) {
    return envOpcDir;
  }
  const configOpcDir = getOpcDirFromConfig();
  if (configOpcDir) {
    return configOpcDir;
  }
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const localOpc = join(projectDir, "opc");
  if (existsSync(localOpc)) {
    return localOpc;
  }
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  if (homeDir) {
    const globalClaude = join(homeDir, ".claude");
    const globalScripts = join(globalClaude, "scripts", "core");
    if (existsSync(globalScripts)) {
      return globalClaude;
    }
  }
  return null;
}
function requireOpcDir() {
  const opcDir = getOpcDir();
  if (!opcDir) {
    console.log(JSON.stringify({ result: "continue" }));
    process.exit(0);
  }
  return opcDir;
}

// src/shared/backend-resolution.ts
var URL_VARS = [
  "CONTINUOUS_CLAUDE_DB_URL",
  "DATABASE_URL",
  "OPC_POSTGRES_URL"
];
var VALID_BACKENDS = /* @__PURE__ */ new Set(["sqlite", "postgres"]);
var BACKEND_VAR = "AGENTICA_MEMORY_BACKEND";
function resolveUrl(env) {
  for (const varName of URL_VARS) {
    const value = env[varName];
    if (value && value.trim()) {
      return value.trim();
    }
  }
  return null;
}
function resolveBackend(env, defaultBackend = "sqlite") {
  const raw = env[BACKEND_VAR] ?? "";
  const explicit = raw.trim().toLowerCase();
  if (explicit) {
    if (!VALID_BACKENDS.has(explicit)) {
      const redacted = raw.replace(/:\/\/[^@]+@/g, "://***@");
      const shown = redacted.length <= 32 ? redacted : redacted.slice(0, 32) + "\u2026";
      throw new Error(
        `Invalid ${BACKEND_VAR}='${shown}': expected 'sqlite' or 'postgres' (case-insensitive).`
      );
    }
    if (explicit === "postgres" && resolveUrl(env) === null) {
      throw new Error(
        `${BACKEND_VAR}=postgres but no PostgreSQL connection URL is set; set one of ${URL_VARS.join(", ")}.`
      );
    }
    return explicit;
  }
  if (resolveUrl(env) !== null) {
    return "postgres";
  }
  return defaultBackend;
}
function pgCoordinationStatus(env = process.env) {
  try {
    return { active: resolveBackend(env) === "postgres" };
  } catch (err) {
    return { active: false, misconfig: err instanceof Error ? err.message : String(err) };
  }
}

// src/shared/pattern-router.ts
var SAFE_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;
function isValidId(id) {
  return SAFE_ID_PATTERN.test(id);
}

// src/shared/db-utils-pg.ts
var misconfigLogged = false;
function pgGate() {
  const status = pgCoordinationStatus();
  if (status.active) {
    return { proceed: true };
  }
  if (status.misconfig) {
    if (!misconfigLogged) {
      misconfigLogged = true;
      process.stderr.write(`[db-utils-pg] ${status.misconfig}
`);
    }
    return { proceed: false, reason: status.misconfig };
  }
  return { proceed: false };
}
function getPgConnectionString() {
  const url = process.env.CONTINUOUS_CLAUDE_DB_URL || process.env.DATABASE_URL || process.env.OPC_POSTGRES_URL;
  if (!url) {
    throw new Error(
      "Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL (preferred), DATABASE_URL, or OPC_POSTGRES_URL. For local Docker dev, run `docker compose -f docker/docker-compose.yml up -d` and export the credentials from docker/.env before invoking this hook."
    );
  }
  return url;
}
function runPgQuery(pythonCode, args = []) {
  const gate = pgGate();
  if (!gate.proceed) {
    return { success: false, stdout: "", stderr: gate.reason ?? "postgres backend inactive" };
  }
  const opcDir = requireOpcDir();
  const resolvedDbUrl = getPgConnectionString();
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
    const result = spawnSync("uv", ["run", "python", "-c", wrappedCode, ...args], {
      encoding: "utf-8",
      maxBuffer: 1024 * 1024,
      timeout: 5e3,
      // 5 second timeout - fail gracefully if DB unreachable
      cwd: opcDir,
      env: {
        ...process.env,
        // Never rewrite opc's uv.lock from a hook-triggered uv run (issue #71
        // follow-up); use the lock as-is. Intentional updates use `uv lock`.
        UV_FROZEN: "1",
        CONTINUOUS_CLAUDE_DB_URL: resolvedDbUrl,
        _OPC_DIR: opcDir
      }
    });
    return {
      success: result.status === 0,
      stdout: result.stdout?.trim() || "",
      stderr: result.stderr || ""
    };
  } catch (err) {
    return {
      success: false,
      stdout: "",
      stderr: String(err)
    };
  }
}
function registerSession(sessionId, project, workingOn = "", claudeSessionId, transcriptPath, pid) {
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
    claudeSessionId || "null",
    transcriptPath || "null",
    pid !== void 0 ? String(pid) : "null"
  ]);
  if (!result.success || result.stdout !== "ok") {
    return {
      success: false,
      error: result.stderr || result.stdout || "Unknown error"
    };
  }
  return { success: true };
}

// src/shared/session-id.ts
function getProject() {
  return process.env.CLAUDE_PROJECT_DIR || process.cwd();
}

// src/session-context.ts
import { readFileSync as readFileSync2, writeFileSync, mkdirSync, renameSync } from "fs";
function checkMemoryHealth(pgRegistrationSucceeded, pidFilePath, pgApplicable = true) {
  const pgHealthy = pgApplicable ? pgRegistrationSucceeded : true;
  let daemonRunning = false;
  try {
    const pidContent = readFileSync2(pidFilePath, "utf-8").trim();
    const pid = parseInt(pidContent, 10);
    if (!isNaN(pid) && pid > 0) {
      process.kill(pid, 0);
      daemonRunning = true;
    }
  } catch {
    daemonRunning = false;
  }
  return { pgHealthy, daemonRunning };
}
function formatHealthWarnings(health) {
  const warnings = [];
  if (!health.pgHealthy) {
    warnings.push("- PostgreSQL: unreachable");
  }
  if (!health.daemonRunning) {
    warnings.push("- Memory daemon: not running");
  }
  if (warnings.length === 0) return null;
  return `Health warnings:
${warnings.join("\n")}`;
}
function getPendingTasksSummary(tasksFilePath) {
  try {
    const content = readFileSync2(tasksFilePath, "utf-8");
    if (!content.trim()) return null;
    const titles = content.split("\n").filter((line) => line.startsWith("## ")).map((line) => line.slice(3).trim());
    if (titles.length === 0) return null;
    const MAX_SHOWN = 3;
    const shown = titles.slice(0, MAX_SHOWN);
    const suffix = titles.length > MAX_SHOWN ? ", ..." : "";
    return `Pending tasks (${titles.length}): ${shown.join(", ")}${suffix}`;
  } catch {
    return null;
  }
}

// src/session-register.ts
/*!
 * SessionStart Hook - Registers session in coordination layer.
 *
 * This hook:
 * 1. Registers the session in PostgreSQL for cross-session awareness
 * 2. Injects session ID, memory system health, and pending tasks summary
 *
 * Peer session awareness is handled by peer-awareness.ts (UserPromptSubmit).
 */
function main() {
  if (process.env.CLAUDE_MEMORY_EXTRACTION) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  let input;
  try {
    const stdinContent = readFileSync3(0, "utf-8");
    input = JSON.parse(stdinContent);
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const sessionId = input.session_id;
  if (typeof sessionId !== "string" || !isValidId(sessionId)) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const project = getProject();
  const projectName = project.split("/").pop() || "unknown";
  process.env.COORDINATION_SESSION_ID = sessionId;
  const pgStatus = pgCoordinationStatus();
  if (pgStatus.misconfig) {
    process.stderr.write(`[session-register] ${pgStatus.misconfig}
`);
  }
  const registerResult = pgStatus.active ? registerSession(sessionId, project, "", input.session_id, input.transcript_path, process.ppid) : { success: false };
  const daemonPidPath = join2(process.env.HOME || "/tmp", ".claude", "memory-daemon.pid");
  const health = checkMemoryHealth(registerResult.success, daemonPidPath, pgStatus.active);
  const healthWarnings = formatHealthWarnings(health);
  const tasksPath = join2(project, "thoughts", "shared", "Tasks.md");
  const tasksSummary = getPendingTasksSummary(tasksPath);
  let awarenessMessage = `
<system-reminder>
Session: ${sessionId}
Project: ${projectName}`;
  if (healthWarnings) {
    awarenessMessage += `

${healthWarnings}`;
  }
  if (tasksSummary) {
    awarenessMessage += `

${tasksSummary}`;
  }
  awarenessMessage += `
</system-reminder>`;
  const output = {
    result: "continue",
    message: awarenessMessage
  };
  console.log(JSON.stringify(output));
}
main();
export {
  main
};
