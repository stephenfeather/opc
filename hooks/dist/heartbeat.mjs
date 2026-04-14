// src/heartbeat.ts
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

// src/shared/pattern-router.ts
var SAFE_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;
function isValidId(id) {
  return SAFE_ID_PATTERN.test(id);
}

// src/shared/db-utils-pg.ts
function getPgConnectionString() {
  const url = process.env.CONTINUOUS_CLAUDE_DB_URL || process.env.DATABASE_URL || process.env.OPC_POSTGRES_URL;
  if (!url) {
    throw new Error(
      "Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL (preferred), DATABASE_URL, or OPC_POSTGRES_URL. For local Docker dev, run `docker compose -f docker/docker-compose.yml up -d` and export the credentials from docker/.env before invoking this hook."
    );
  }
  return url;
}
function runPgQueryDetached(pythonCode, args = []) {
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
    const child = spawn("uv", ["run", "python", "-c", wrappedCode, ...args], {
      detached: true,
      stdio: "ignore",
      cwd: opcDir,
      env: {
        ...process.env,
        CONTINUOUS_CLAUDE_DB_URL: resolvedDbUrl
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
    const stdinContent = readFileSync2(0, "utf-8");
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
