// src/session-start-memory-push.ts
import { readFileSync as readFileSync2, existsSync as existsSync2 } from "fs";
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

// src/session-start-memory-push.ts
function main() {
  let input;
  try {
    const stdinContent = readFileSync2(0, "utf-8");
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
  const projectName = projectDir.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
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
