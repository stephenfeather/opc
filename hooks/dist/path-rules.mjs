// src/path-rules.ts
import { readFileSync, existsSync } from "fs";

// src/shared/stdin.ts
import { fstatSync, readSync } from "fs";
var CHUNK_SIZE = 64 * 1024;
var EAGAIN_SLEEP_MS = 5;
var DEFAULT_MAX_IDLE_MS = 1e4;
var MAX_IDLE_ENV = "HOOK_STDIN_MAX_IDLE_MS";
function resolveMaxIdleMs(explicit) {
  if (explicit !== void 0) return explicit;
  const fromEnv = process.env[MAX_IDLE_ENV];
  if (fromEnv !== void 0 && fromEnv !== "" && Number.isFinite(Number(fromEnv))) {
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

// src/path-rules.ts
import { join } from "path";
/*!
 * Path-based rule injection hook
 *
 * Fires on PreToolUse for Read/Edit/Write
 * Matches file paths against patterns and injects relevant skill content
 */
var PATH_RULES = [
  // Hook development
  { pattern: /\.claude\/hooks\//, skillName: "hooks", description: "Hook development" },
  // Skill development
  { pattern: /\.claude\/skills\//, skillName: "skill-development", description: "Skill development" },
  // Agent cache
  { pattern: /\.claude\/cache\/agents\//, skillName: "agent-context-isolation", description: "Agent context isolation" },
  // Continuity ledgers
  { pattern: /thoughts\/ledgers\/CONTINUITY_CLAUDE-/, skillName: "continuity", description: "Continuity ledger" },
  // Agentica
  { pattern: /opc\/scripts\/agentica/, skillName: "async-repl-protocol", description: "Agentica REPL protocol" },
  // MCP scripts
  { pattern: /scripts\/.*\.py$/, skillName: "mcp-scripts", description: "MCP scripts" },
  // Lean files
  { pattern: /\.lean$/, skillName: "llm-tuning-patterns", description: "LLM tuning for proofs" },
  // Skill rules config
  { pattern: /skill-rules\.json$/, skillName: "router-first-architecture", description: "Router-first architecture" },
  // Wiring/hooks infrastructure
  { pattern: /\.claude\/settings\.json$/, skillName: "wiring", description: "Wiring verification" }
];
function readStdin() {
  return readStdinSync();
}
function getProjectDir() {
  return process.env.CLAUDE_PROJECT_DIR || process.cwd();
}
function loadSkillContent(skillName) {
  const projectDir = getProjectDir();
  const skillPath = join(projectDir, ".claude", "skills", skillName, "SKILL.md");
  if (!existsSync(skillPath)) return null;
  try {
    let content = readFileSync(skillPath, "utf-8");
    if (content.startsWith("---")) {
      const end = content.indexOf("---", 3);
      if (end !== -1) content = content.slice(end + 3).trim();
    }
    return content;
  } catch {
    return null;
  }
}
function getMatchingSkills(filePath) {
  const matched = [];
  for (const rule of PATH_RULES) {
    if (rule.pattern.test(filePath)) {
      matched.push(rule.skillName);
    }
  }
  return matched;
}
async function main() {
  const input = JSON.parse(readStdin());
  const filePath = input.tool_input?.file_path;
  if (!filePath) {
    console.log("{}");
    return;
  }
  const skills = getMatchingSkills(filePath);
  if (skills.length === 0) {
    console.log("{}");
    return;
  }
  const contents = [];
  for (const skill of skills) {
    const content = loadSkillContent(skill);
    if (content) contents.push(content);
  }
  if (contents.length === 0) {
    console.log("{}");
    return;
  }
  console.log(JSON.stringify({
    continue: true,
    systemMessage: contents.join("\n\n---\n\n")
  }));
}
main().catch(() => process.exit(1));
