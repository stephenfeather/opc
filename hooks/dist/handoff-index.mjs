// src/handoff-index.ts
import * as fs from "fs";
import * as path from "path";
import { spawn, execSync } from "child_process";
import Database from "better-sqlite3";
//! @hook PostToolUse:Write @preserve
function getPpid(pid) {
  if (process.platform === "win32") {
    try {
      const result = execSync(`wmic process where ProcessId=${pid} get ParentProcessId`, {
        encoding: "utf-8",
        timeout: 5e3
      });
      for (const line of result.split("\n")) {
        const trimmed = line.trim();
        if (/^\d+$/.test(trimmed)) {
          return parseInt(trimmed, 10);
        }
      }
    } catch {
    }
    return null;
  }
  try {
    const result = execSync(`ps -o ppid= -p ${pid}`, {
      encoding: "utf-8",
      timeout: 5e3
    });
    const ppid = parseInt(result.trim(), 10);
    return isNaN(ppid) ? null : ppid;
  } catch {
    return null;
  }
}
function getTerminalShellPid() {
  try {
    const parent = process.ppid;
    if (!parent) return null;
    const grandparent = getPpid(parent);
    if (!grandparent) return null;
    return getPpid(grandparent);
  } catch {
    return null;
  }
}
function storeSessionAffinity(projectDir, terminalPid, sessionName) {
  const dbPath = path.join(projectDir, ".claude", "cache", "artifact-index", "context.db");
  const dbDir = path.dirname(dbPath);
  try {
    if (!fs.existsSync(dbDir)) {
      fs.mkdirSync(dbDir, { recursive: true });
    }
    const db = new Database(dbPath);
    db.exec(`
      CREATE TABLE IF NOT EXISTS instance_sessions (
        terminal_pid TEXT PRIMARY KEY,
        session_name TEXT NOT NULL,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
      )
    `);
    const stmt = db.prepare(`
      INSERT OR REPLACE INTO instance_sessions (terminal_pid, session_name, updated_at)
      VALUES (?, ?, datetime('now'))
    `);
    stmt.run(terminalPid.toString(), sessionName);
    db.close();
  } catch {
  }
}
function extractSessionName(filePath) {
  const parts = filePath.split("/");
  const handoffsIdx = parts.findIndex((p) => p === "handoffs");
  if (handoffsIdx >= 0 && handoffsIdx < parts.length - 1) {
    return parts[handoffsIdx + 1];
  }
  return null;
}
var INDEXABLE_TOOLS = /* @__PURE__ */ new Set(["Write", "Edit", "MultiEdit"]);
function indexScriptCandidates(projectDir) {
  return [
    path.join(projectDir, "scripts", "core", "artifact_index.py"),
    path.join(projectDir, "scripts", "artifact_index.py")
  ];
}
function isWithinProject(fullPath, projectDir) {
  const rel = path.relative(path.resolve(projectDir), path.resolve(fullPath));
  return rel !== "" && !rel.startsWith("..") && !path.isAbsolute(rel);
}
function isIndexableToolEvent(toolName) {
  return !!toolName && INDEXABLE_TOOLS.has(toolName);
}
function classifyArtifactPath(filePath) {
  if (!filePath) return null;
  const normalized = filePath.replace(/\\/g, "/");
  const segments = normalized.split("/");
  const isMd = normalized.endsWith(".md");
  const isYaml = normalized.endsWith(".yaml") || normalized.endsWith(".yml");
  if (segments.includes("handoffs") && (isMd || isYaml)) {
    return "handoff";
  }
  if (isMd && normalized.includes("/thoughts/shared/plans/")) {
    return "plan";
  }
  if (isMd && normalized.startsWith("thoughts/shared/plans/")) {
    return "plan";
  }
  return null;
}
async function main() {
  const input = JSON.parse(await readStdin());
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  if (!isIndexableToolEvent(input.tool_name)) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const filePath = input.tool_input?.file_path || "";
  const kind = classifyArtifactPath(filePath);
  if (kind === null) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  try {
    const fullPath = path.isAbsolute(filePath) ? filePath : path.join(projectDir, filePath);
    if (!isWithinProject(fullPath, projectDir) || !fs.existsSync(fullPath)) {
      console.log(JSON.stringify({ result: "continue" }));
      return;
    }
    let content = fs.readFileSync(fullPath, "utf-8");
    let modified = false;
    const isYamlFile = fullPath.endsWith(".yaml") || fullPath.endsWith(".yml");
    const hasFrontmatter = content.startsWith("---");
    const hasRootSpanId = content.includes("root_span_id:");
    if (kind === "handoff" && !hasRootSpanId) {
      const stateFile = path.join(homeDir, ".claude", "state", "braintrust_sessions", `${input.session_id}.json`);
      if (fs.existsSync(stateFile)) {
        try {
          const stateContent = fs.readFileSync(stateFile, "utf-8");
          const state = JSON.parse(stateContent);
          const newFields = [
            `root_span_id: ${state.root_span_id}`,
            `turn_span_id: ${state.current_turn_span_id || ""}`,
            `session_id: ${input.session_id}`
          ].join("\n");
          if (isYamlFile) {
            content = `${newFields}
${content}`;
          } else if (hasFrontmatter) {
            content = content.replace(/^---\n/, `---
${newFields}
`);
          } else {
            content = `---
${newFields}
---

${content}`;
          }
          const tempPath = fullPath + ".tmp";
          fs.writeFileSync(tempPath, content);
          fs.renameSync(tempPath, fullPath);
          modified = true;
        } catch (stateErr) {
        }
      }
    }
    if (kind === "handoff") {
      const terminalPid = getTerminalShellPid();
      const sessionName = extractSessionName(fullPath);
      if (terminalPid && sessionName) {
        storeSessionAffinity(projectDir, terminalPid, sessionName);
      }
    }
    const indexScript = indexScriptCandidates(projectDir).find((p) => fs.existsSync(p));
    if (indexScript) {
      const child = spawn("uv", ["run", "python", indexScript, "--file", fullPath], {
        cwd: projectDir,
        detached: true,
        stdio: "ignore",
        // Never rewrite the project's uv.lock from a hook-triggered uv run (issue #71).
        env: { ...process.env, UV_FROZEN: "1" }
      });
      child.unref();
    }
    console.log(JSON.stringify({ result: "continue" }));
  } catch (err) {
    console.log(JSON.stringify({ result: "continue" }));
  }
}
async function readStdin() {
  return new Promise((resolve2) => {
    let data = "";
    process.stdin.on("data", (chunk) => data += chunk);
    process.stdin.on("end", () => resolve2(data));
  });
}
if (process.argv[1] && (process.argv[1].endsWith("handoff-index.ts") || process.argv[1].endsWith("handoff-index.js") || process.argv[1].endsWith("handoff-index.mjs"))) {
  main().catch(console.error);
}
export {
  classifyArtifactPath,
  indexScriptCandidates,
  isIndexableToolEvent,
  isWithinProject
};
