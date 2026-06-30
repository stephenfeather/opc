// src/file-claims.ts
import { readFileSync as readFileSync2 } from "fs";

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
function getConnectionUrl() {
  return resolveUrl(process.env);
}
function pgCoordinationStatus(env = process.env) {
  try {
    return { active: resolveBackend(env) === "postgres" };
  } catch (err) {
    return { active: false, misconfig: err instanceof Error ? err.message : String(err) };
  }
}

// src/shared/db-utils-pg.ts
function pgGate() {
  const status = pgCoordinationStatus();
  if (status.active) {
    return { proceed: true };
  }
  return { proceed: false, reason: status.misconfig };
}
function getPgConnectionString() {
  const url = getConnectionUrl();
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
function checkFileClaim(filePath, project, mySessionId) {
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
function claimFile(filePath, project, sessionId) {
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
  return { success: result.success && result.stdout === "ok" };
}

// src/shared/session-id.ts
function getProject() {
  return process.env.CLAUDE_PROJECT_DIR || process.cwd();
}

// src/file-claims.ts
/*!
 * PreToolUse:Edit Hook - Check and claim files for conflict prevention.
 *
 * This hook:
 * 1. Checks if another session has claimed the file
 * 2. Warns if file is being edited by another session
 * 3. Claims the file for the current session
 *
 * Session ID comes from stdin (input.session_id), provided by Claude Code.
 * Part of the coordination layer architecture (Phase 1).
 */
function main() {
  let input;
  try {
    const stdinContent = readFileSync2(0, "utf-8");
    input = JSON.parse(stdinContent);
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (input.tool_name !== "Edit") {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const filePath = input.tool_input?.file_path;
  if (!filePath) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const sessionId = input.session_id;
  if (typeof sessionId !== "string" || !sessionId) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const project = getProject();
  const claimCheck = checkFileClaim(filePath, project, sessionId);
  let output;
  if (claimCheck.claimed) {
    const fileName = filePath.split("/").pop() || filePath;
    output = {
      result: "continue",
      // Allow edit, just warn
      message: `\u26A0\uFE0F **File Conflict Warning**
\`${fileName}\` is being edited by Session ${claimCheck.claimedBy}
Consider coordinating with the other session to avoid conflicts.`
    };
  } else {
    claimFile(filePath, project, sessionId);
    output = { result: "continue" };
  }
  console.log(JSON.stringify(output));
}
main();
export {
  main
};
