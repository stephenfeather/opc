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

// src/shared/pattern-router.ts
var SAFE_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;
function isValidId(id) {
  return SAFE_ID_PATTERN.test(id);
}

// src/shared/db-utils-pg.ts
function getPgConnectionString() {
  return process.env.CONTINUOUS_CLAUDE_DB_URL || process.env.DATABASE_URL || process.env.OPC_POSTGRES_URL || "postgresql://claude:claude_dev@localhost:5432/continuous_claude";
}
function runPgQuery(pythonCode, args = []) {
  const opcDir = requireOpcDir();
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
        CONTINUOUS_CLAUDE_DB_URL: getPgConnectionString(),
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
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL', 'postgresql://claude:claude_dev@localhost:5432/continuous_claude')

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
        ]:
            await conn.execute(f'ALTER TABLE sessions ADD COLUMN IF NOT EXISTS {col[0]} {col[1]}')

        # Upsert session (clear exited_at on re-register, e.g. resume)
        await conn.execute('''
            INSERT INTO sessions (id, project, working_on, claude_session_id, transcript_path, pid, started_at, last_heartbeat, exited_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW(), NULL)
            ON CONFLICT (id) DO UPDATE SET
                working_on = EXCLUDED.working_on,
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
function checkMemoryHealth(pgRegistrationSucceeded, pidFilePath) {
  const pgHealthy = pgRegistrationSucceeded;
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
  const registerResult = registerSession(sessionId, project, "", input.session_id, input.transcript_path, process.ppid);
  const daemonPidPath = join2(process.env.HOME || "/tmp", ".claude", "memory-daemon.pid");
  const health = checkMemoryHealth(registerResult.success, daemonPidPath);
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
