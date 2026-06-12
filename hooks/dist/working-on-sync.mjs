// src/working-on-sync.ts
import { readFileSync as readFileSync2, writeFileSync, renameSync, mkdirSync, existsSync as existsSync2 } from "fs";
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
function updateWorkingOnDetached(sessionId, project, workingOn) {
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

// src/shared/session-id.ts
function getProject() {
  return process.env.CLAUDE_PROJECT_DIR || process.cwd();
}

// src/working-on-sync.ts
var EMPTY_CACHE = { tasks: {}, currentId: null };
function parseCreatedTaskId(toolResponse) {
  const text = typeof toolResponse === "string" ? toolResponse : toolResponse && typeof toolResponse === "object" ? JSON.stringify(toolResponse) : "";
  const m = text.match(/Task #(\d+)\b/);
  return m ? m[1] : null;
}
function pickTodoInProgress(todos) {
  const t = (todos || []).find((x) => x.status === "in_progress");
  if (!t) return "";
  return (t.activeForm || t.content || "").trim();
}
function deriveWorkingOn(input, cache) {
  const tool = input.tool_name;
  const ti = input.tool_input || {};
  const next = {
    tasks: { ...cache.tasks },
    currentId: cache.currentId
  };
  if (tool === "TodoWrite") {
    return { workingOn: pickTodoInProgress(ti.todos), cache: next };
  }
  if (tool === "TaskCreate") {
    const id = parseCreatedTaskId(input.tool_response);
    const label = (ti.activeForm || ti.subject || "").trim();
    if (id && label) next.tasks[id] = label;
    return { workingOn: null, cache: next };
  }
  if (tool === "TaskUpdate") {
    const id = ti.taskId;
    if (!id) return { workingOn: null, cache: next };
    if (ti.status === "in_progress") {
      const label = next.tasks[id];
      if (!label) return { workingOn: null, cache: next };
      next.currentId = id;
      return { workingOn: label, cache: next };
    }
    if ((ti.status === "completed" || ti.status === "deleted") && id === next.currentId) {
      next.currentId = null;
      return { workingOn: "", cache: next };
    }
  }
  return { workingOn: null, cache: next };
}
function cachePath(sessionId) {
  return join2(
    process.env.HOME || "",
    ".claude",
    "cache",
    "working-on",
    `${sessionId}.json`
  );
}
function readCache(sessionId) {
  try {
    const raw = readFileSync2(cachePath(sessionId), "utf-8");
    const parsed = JSON.parse(raw);
    return {
      tasks: parsed.tasks && typeof parsed.tasks === "object" ? parsed.tasks : {},
      currentId: typeof parsed.currentId === "string" ? parsed.currentId : null
    };
  } catch {
    return { ...EMPTY_CACHE, tasks: {} };
  }
}
function writeCache(sessionId, cache) {
  const p = cachePath(sessionId);
  try {
    const dir = join2(p, "..");
    if (!existsSync2(dir)) mkdirSync(dir, { recursive: true });
    const tmp = `${p}.tmp`;
    writeFileSync(tmp, JSON.stringify(cache), "utf-8");
    renameSync(tmp, p);
  } catch {
  }
}
function main() {
  let input;
  try {
    input = JSON.parse(readFileSync2(0, "utf-8"));
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const sessionId = input.session_id;
  const relevant = input.tool_name === "TodoWrite" || input.tool_name === "TaskCreate" || input.tool_name === "TaskUpdate";
  if (!sessionId || !isValidId(sessionId) || !relevant) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const cache = readCache(sessionId);
  const { workingOn, cache: nextCache } = deriveWorkingOn(input, cache);
  writeCache(sessionId, nextCache);
  if (workingOn !== null) {
    updateWorkingOnDetached(sessionId, getProject(), workingOn);
  }
  console.log(JSON.stringify({ result: "continue" }));
}
if (typeof process !== "undefined" && process.argv[1] && (process.argv[1].endsWith("working-on-sync.ts") || process.argv[1].endsWith("working-on-sync.js") || process.argv[1].endsWith("working-on-sync.mjs"))) {
  main();
}
export {
  deriveWorkingOn,
  main,
  parseCreatedTaskId,
  pickTodoInProgress
};
