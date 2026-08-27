// src/shared/stdin.ts
import { fstatSync, readSync } from "fs";
var CHUNK_SIZE = 64 * 1024;
var EAGAIN_SLEEP_MS = 5;
var DEFAULT_MAX_IDLE_MS = 1e4;
var MAX_IDLE_ENV = "HOOK_STDIN_MAX_IDLE_MS";
function isTestRunner(env = process.env) {
  return Boolean(env.VITEST) || env.NODE_ENV === "test";
}
function resolveMaxIdleMs(explicit) {
  if (explicit !== void 0) return explicit;
  const fromEnv = process.env[MAX_IDLE_ENV];
  if (isTestRunner() && fromEnv !== void 0 && fromEnv !== "" && Number.isFinite(Number(fromEnv))) {
    return Math.max(0, Number(fromEnv));
  }
  return DEFAULT_MAX_IDLE_MS;
}
function isTty(fd) {
  try {
    return fstatSync(fd).isCharacterDevice();
  } catch {
    return false;
  }
}
var sleepCell = new Int32Array(new SharedArrayBuffer(4));
function sleepMs(ms) {
  Atomics.wait(sleepCell, 0, 0, ms);
}
function readStdinSync(options = {}) {
  const fd = options.fd ?? 0;
  const maxIdleMs = resolveMaxIdleMs(options.maxIdleMs);
  if (isTty(fd)) {
    return "";
  }
  const chunks = [];
  const buf = Buffer.alloc(CHUNK_SIZE);
  let idleMs = 0;
  for (; ; ) {
    let n;
    try {
      n = readSync(fd, buf, 0, buf.length, null);
    } catch (err) {
      const code = err?.code;
      if (code === "EAGAIN" || code === "EWOULDBLOCK") {
        if (idleMs >= maxIdleMs) {
          throw new RangeError(`stdin idle for ${maxIdleMs} ms with no data`);
        }
        sleepMs(EAGAIN_SLEEP_MS);
        idleMs += EAGAIN_SLEEP_MS;
        continue;
      }
      if (code === "EOF") {
        break;
      }
      throw err;
    }
    if (n === 0) {
      break;
    }
    idleMs = 0;
    chunks.push(Buffer.from(buf.subarray(0, n)));
  }
  return Buffer.concat(chunks).toString("utf-8");
}

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
      const candidate = raw.trim();
      const safeToken = /^[A-Za-z0-9_-]{1,8}$/.test(candidate);
      const shown = safeToken ? `'${candidate}'` : "<redacted non-token value>";
      throw new Error(
        `Invalid ${BACKEND_VAR}=${shown}: expected 'sqlite' or 'postgres' (case-insensitive).`
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

// src/shared/pattern-router.ts
var SAFE_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;
function isValidId(id) {
  return SAFE_ID_PATTERN.test(id);
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
function runPgQueryDetached(pythonCode, args = []) {
  if (!pgGate().proceed) {
    return;
  }
  const resolvedDbUrl = getPgConnectionString();
  const opcDir = requireOpcDir();
  try {
    const wrappedCode = `
import sys
import os
import asyncio
import json

# Add opc to path for imports (read from env to avoid code injection)
_opc_dir = os.environ.get('_OPC_DIR')
if not _opc_dir:
    raise RuntimeError('_OPC_DIR environment variable not set - must be called via runPgQueryDetached()')
sys.path.insert(0, _opc_dir)
os.chdir(_opc_dir)

${pythonCode}
`;
    const child = spawn("uv", ["run", "python", "-c", wrappedCode, ...args], {
      detached: true,
      stdio: "ignore",
      cwd: opcDir,
      env: {
        ...process.env,
        // Never rewrite opc's uv.lock from a hook-triggered uv run (issue #71
        // follow-up); the frequent heartbeat path runs through here.
        UV_FROZEN: "1",
        CONTINUOUS_CLAUDE_DB_URL: resolvedDbUrl,
        _OPC_DIR: opcDir
      }
    });
    child.unref();
  } catch {
  }
}
function updateHeartbeatDetached(sessionId, project) {
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

// src/shared/session-id.ts
function getProject() {
  return process.env.CLAUDE_PROJECT_DIR || process.cwd();
}

// src/heartbeat.ts
function main() {
  let sessionId = null;
  try {
    const stdinContent = readStdinSync();
    const input = JSON.parse(stdinContent);
    if (input && typeof input.session_id === "string" && isValidId(input.session_id)) {
      sessionId = input.session_id;
    }
  } catch {
  }
  if (!sessionId) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const project = getProject();
  updateHeartbeatDetached(sessionId, project);
  console.log(JSON.stringify({ result: "continue" }));
}
if (typeof process !== "undefined" && process.argv[1] && (process.argv[1].endsWith("heartbeat.ts") || process.argv[1].endsWith("heartbeat.js") || process.argv[1].endsWith("heartbeat.mjs"))) {
  main();
}
export {
  main
};
