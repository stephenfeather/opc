// src/session-start-memory-push.ts
import { existsSync as existsSync2 } from "fs";

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

// src/session-start-memory-push.ts
import { spawnSync } from "child_process";
import { join as join2 } from "path";

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

// src/shared/memory-opt-out.ts
/*!
 * Memory injection opt-out.
 *
 * Setting OPC_MEMORY_LOSS=1 in the environment disables the hooks that inject
 * stored learnings into Claude's context (memory-awareness on UserPromptSubmit,
 * session-start-memory-push on SessionStart). Useful for sessions where recall
 * noise is unwanted — demos, benchmarks, or debugging the hooks themselves.
 *
 * Pure: reads only the env object it is handed.
 */
var MEMORY_LOSS_ENV = "OPC_MEMORY_LOSS";
var TRUTHY = /* @__PURE__ */ new Set(["1", "true", "yes", "on"]);
function isMemoryInjectionDisabled(env = process.env) {
  const raw = env[MEMORY_LOSS_ENV];
  if (raw === void 0) return false;
  return TRUTHY.has(raw.trim().toLowerCase());
}

// src/session-start-memory-push.ts
/*!
 * Session Start Memory Push Hook (SessionStart, type=startup)
 *
 * Proactively surfaces high-value, never-recalled learnings at session start.
 * Targets two pools:
 *   1. Stale high-confidence learnings for the current project
 *   2. Pattern representatives from anti_pattern / problem_solution clusters
 *
 * Calls push_learnings.py via subprocess (mirrors memory-awareness.ts pattern).
 * Injects results via hookSpecificOutput.additionalContext.
 */
function normalizeProjectName(projectDir) {
  const cleaned = projectDir.replace(/[\\/]+$/, "");
  const parts = cleaned.split(/[\\/]/);
  const worktreeIdx = parts.indexOf(".worktrees");
  if (worktreeIdx > 0) {
    return parts[worktreeIdx - 1] ?? "";
  }
  return parts.pop() ?? "";
}
function main() {
  let input;
  try {
    const stdinContent = readStdinSync();
    input = JSON.parse(stdinContent);
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const eventType = input.type || input.source || "startup";
  if (eventType !== "startup") {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (process.env.CLAUDE_AGENT_ID) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (isMemoryInjectionDisabled()) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (process.env.CLAUDE_MEMORY_EXTRACTION) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const sentinel = join2(projectDir, ".claude", "no-memory-push");
  if (existsSync2(sentinel)) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const opcDir = getOpcDir();
  if (!opcDir) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const projectName = normalizeProjectName(projectDir);
  if (!projectName || projectName.startsWith("-")) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const result = spawnSync("uv", [
    "run",
    "python",
    "scripts/core/push_learnings.py",
    "--project",
    projectName,
    "--k",
    "5",
    "--json",
    "--max-chars",
    "150"
  ], {
    encoding: "utf-8",
    cwd: opcDir,
    env: {
      ...process.env,
      // Never rewrite opc's uv.lock from a hook-triggered uv run (issue #71).
      UV_FROZEN: "1",
      PYTHONPATH: opcDir
    },
    timeout: 8e3
  });
  if (result.status !== 0 || !result.stdout) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  let data;
  try {
    data = JSON.parse(result.stdout);
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (!data.results || data.results.length === 0) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const resultLines = data.results.map((r, i) => {
    const base = `${i + 1}. [${r.learning_type}|${r.confidence}] ${r.content} (id: ${r.id})`;
    const label = r.pattern_label ? `
   \u21B3 Pattern: "${r.pattern_label}"` : "";
    return base + label;
  }).join("\n");
  const context = [
    `PROACTIVE MEMORY (${data.results.length} learnings for "${projectName}"):`,
    resultLines,
    "These were surfaced proactively. Use /recall for full content.",
    'If any learning helps or misleads you, submit feedback: mcp__opc-memory__store_feedback(learning_id="<id>", helpful=true/false)'
  ].join("\n");
  console.log(JSON.stringify({
    result: "continue",
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext: context
    }
  }));
}
main();
