// src/peer-awareness.ts
import { readFileSync as readFileSync4 } from "fs";
import { join as join4 } from "path";

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
function getActiveSessions(project) {
  const pythonCode = `
import asyncpg
import os
import json
from datetime import datetime, timedelta

project_filter = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != 'null' else None
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL', 'postgresql://claude:claude_dev@localhost:5432/continuous_claude')

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
  const result = runPgQuery(pythonCode, [project || "null"]);
  if (!result.success) {
    return { success: false, sessions: [] };
  }
  try {
    const sessions = JSON.parse(result.stdout || "[]");
    return { success: true, sessions };
  } catch {
    return { success: false, sessions: [] };
  }
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

// src/session-context.ts
import { readFileSync as readFileSync3, writeFileSync as writeFileSync2, mkdirSync as mkdirSync2, renameSync } from "fs";
import { dirname } from "path";
function formatPeerMessage(peers) {
  if (peers.length === 0) return null;
  const lines = peers.map(
    (s) => `- ${s.id}: ${s.working_on || "working..."}`
  );
  return `Active peer sessions (${peers.length}):
${lines.join("\n")}`;
}
function readPeerCache(cachePath, project, ttlSeconds) {
  try {
    const raw = readFileSync3(cachePath, "utf-8");
    const data = JSON.parse(raw);
    if (typeof data.cached_at !== "string" || typeof data.project !== "string" || !Array.isArray(data.sessions)) {
      return null;
    }
    if (data.project !== project) return null;
    const cachedTime = new Date(data.cached_at).getTime();
    if (!isFinite(cachedTime)) return null;
    const age = (Date.now() - cachedTime) / 1e3;
    if (age >= ttlSeconds) return null;
    return data.sessions;
  } catch {
    return null;
  }
}
function writePeerCache(cachePath, project, sessions) {
  try {
    const dir = dirname(cachePath);
    mkdirSync2(dir, { recursive: true });
    const data = {
      cached_at: (/* @__PURE__ */ new Date()).toISOString(),
      project,
      sessions
    };
    const tmpPath = cachePath + ".tmp." + process.pid;
    writeFileSync2(tmpPath, JSON.stringify(data), { encoding: "utf-8" });
    renameSync(tmpPath, cachePath);
  } catch {
  }
}

// src/peer-awareness.ts
//! @hook UserPromptSubmit @preserve
var CACHE_TTL_SECONDS = 60;
function main() {
  if (process.env.CLAUDE_AGENT_ID) {
    console.log(JSON.stringify({}));
    return;
  }
  let ownSessionId = null;
  try {
    const stdinContent = readFileSync4(0, "utf-8");
    const input = JSON.parse(stdinContent);
    if (input && typeof input.session_id === "string" && isValidId(input.session_id)) {
      ownSessionId = input.session_id;
    }
  } catch {
  }
  if (!ownSessionId) {
    ownSessionId = readSessionId();
  }
  if (!ownSessionId) {
    console.log(JSON.stringify({}));
    return;
  }
  const project = getProject();
  const cachePath = join4(process.env.HOME || "/tmp", ".claude", "cache", "peer-sessions.json");
  let sessions = readPeerCache(cachePath, project, CACHE_TTL_SECONDS);
  if (sessions === null) {
    const result = getActiveSessions(project);
    if (result.success) {
      sessions = result.sessions;
      writePeerCache(cachePath, project, sessions);
    } else {
      console.log(JSON.stringify({}));
      return;
    }
  }
  const peers = sessions.filter((s) => s.id !== ownSessionId);
  const message = formatPeerMessage(peers);
  if (!message) {
    console.log(JSON.stringify({}));
    return;
  }
  console.log(JSON.stringify({
    result: "continue",
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext: message
    }
  }));
}
main();
export {
  main
};
