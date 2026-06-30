// src/session-clean-exit.ts
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
function pgCoordinationStatus(env = process.env) {
  try {
    return { active: resolveBackend(env) === "postgres" };
  } catch (err) {
    return { active: false, misconfig: err instanceof Error ? err.message : String(err) };
  }
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
function markSessionExited(claudeSessionId) {
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
  if (!result.success || result.stdout !== "ok") {
    return {
      success: false,
      error: result.stderr || result.stdout || "Unknown error"
    };
  }
  return { success: true };
}

// src/session-clean-exit.ts
/*!
 * Session Clean Exit Hook (SessionEnd)
 *
 * Marks the session as cleanly exited in PostgreSQL.
 * If this hook doesn't fire (crash/hang), the session remains without
 * an exited_at timestamp and session-crash-recovery.ts will detect it
 * on next startup.
 */
async function main() {
  let input;
  try {
    input = JSON.parse(readFileSync2(0, "utf-8"));
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  markSessionExited(input.session_id);
  console.log(JSON.stringify({ result: "continue" }));
}
main().catch(() => console.log(JSON.stringify({ result: "continue" })));
