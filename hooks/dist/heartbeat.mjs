// src/heartbeat.ts
import { readFileSync as readFileSync3 } from "fs";

// src/shared/db-utils-pg.ts
import { spawnSync } from "child_process";

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

# Add opc to path for imports
sys.path.insert(0, '${opcDir}')
os.chdir('${opcDir}')

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
        CONTINUOUS_CLAUDE_DB_URL: getPgConnectionString()
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
function updateHeartbeat(sessionId, project) {
  const pythonCode = `
import asyncpg
import os

session_id = sys.argv[1]
project = sys.argv[2]
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL', 'postgresql://claude:claude_dev@localhost:5432/continuous_claude')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('''
            UPDATE sessions SET last_heartbeat = NOW()
            WHERE id = $1 AND project = $2
        ''', session_id, project)
        print('ok')
    finally:
        await conn.close()

asyncio.run(main())
`;
  const result = runPgQuery(pythonCode, [sessionId, project]);
  if (!result.success || result.stdout !== "ok") {
    return {
      success: false,
      error: result.stderr || result.stdout || "Unknown error"
    };
  }
  return { success: true };
}

// src/shared/session-id.ts
import { mkdirSync, readFileSync as readFileSync2, writeFileSync } from "fs";
import { join as join2 } from "path";
var SESSION_ID_FILENAME = ".coordination-session-id";
function getSessionIdFile(options = {}) {
  const claudeDir = join2(process.env.HOME || "/tmp", ".claude");
  if (options.createDir) {
    try {
      mkdirSync(claudeDir, { recursive: true, mode: 448 });
    } catch {
    }
  }
  return join2(claudeDir, SESSION_ID_FILENAME);
}
function readSessionId() {
  try {
    const sessionFile = getSessionIdFile();
    const id = readFileSync2(sessionFile, "utf-8").trim();
    return id || null;
  } catch {
    return null;
  }
}
function getProject() {
  return process.env.CLAUDE_PROJECT_DIR || process.cwd();
}

// src/heartbeat.ts
function main() {
  let sessionId = null;
  try {
    const stdinContent = readFileSync3(0, "utf-8");
    const input = JSON.parse(stdinContent);
    if (input && typeof input.session_id === "string" && isValidId(input.session_id)) {
      sessionId = input.session_id;
    }
  } catch {
  }
  if (!sessionId) {
    const persisted = readSessionId();
    if (persisted && isValidId(persisted)) {
      sessionId = persisted;
    }
  }
  if (!sessionId) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const project = getProject();
  updateHeartbeat(sessionId, project);
  console.log(JSON.stringify({ result: "continue" }));
}
main();
export {
  main
};
