// src/pre-compact-continuity.ts
import * as fs from "fs";
import * as path from "path";
import { spawnSync, spawn } from "child_process";
//! @hook PreCompact @preserve
function getMemoryPushContext() {
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  const pushFile = path.join(homeDir, ".claude", "cache", "memory-push.json");
  if (!fs.existsSync(pushFile)) return "";
  try {
    const stat = fs.statSync(pushFile);
    const ageMs = Date.now() - stat.mtimeMs;
    if (ageMs > 2 * 60 * 60 * 1e3) return "";
    const pushData = JSON.parse(fs.readFileSync(pushFile, "utf-8"));
    if (!pushData.results || pushData.results.length === 0) return "";
    const currentProject = (process.env.CLAUDE_PROJECT_DIR || process.cwd()).replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
    if (pushData.project && currentProject && pushData.project !== currentProject) return "";
    const pushLines = pushData.results.map((r, i) => {
      const label = r.pattern_label ? `
   \u21B3 Pattern: "${r.pattern_label}"` : "";
      return `${i + 1}. [${r.learning_type}|${r.confidence}] ${r.content} (id: ${r.id.slice(0, 8)})${label}`;
    }).join("\n");
    return `
PROACTIVE MEMORY (carried through compaction):
${pushLines}`;
  } catch {
    return "";
  }
}
var MINI_HANDOFF_SCRIPT = path.join(
  process.env.CLAUDE_OPC_DIR || path.join(process.env.HOME || "", "opc"),
  "scripts",
  "core",
  "generate_mini_handoff.py"
);
var NARRATIVE_PROMPT_TEMPLATE = `You are running headless. Read the JSONL transcript at {TRANSCRIPT_PATH} once, then write a narrative YAML handoff to {OUTPUT_PATH} using the Write tool.

Required YAML shape (exact field names \u2014 statusline parser depends on goal: and now:):
---
session: {SESSION_NAME}
session_uuid: {SESSION_UUID}
date: {DATE}
status: partial
outcome: PARTIAL_PLUS
---

goal: <1-2 sentences: what the session accomplished>
now: <first INCOMPLETE next action \u2014 must not appear in done_this_session>
test: <command or manual step to verify>

done_this_session:
  - task: <narrative task, not a raw path>
    files: [<relevant file paths>]

blockers: []
questions: []

decisions:
  - <label>: <rationale>

findings:
  - <finding>: <details>

worked:
  - Problem \u2192 Solution \u2014 <what worked and why>
failed:
  - <what broke, why to avoid it>

next:
  - <first concrete next step>

files:
  created: [...]
  modified: [...]

Rules:
1. Read the transcript ONCE. No exploration, no grep, no other files.
2. Write the YAML ONCE via the Write tool.
3. Do not run Bash, TaskCreate, or any other tool.
4. Keep fields concise. No code blocks. Prefer file.ext:line refs.
5. If the session accomplished nothing meaningful, set status: blocked and explain in goal.
6. Exit immediately after the Write succeeds.`;
function runMechanicalHandoff(transcriptPath, sessionId, projectDir) {
  const outputPath = path.join(
    projectDir,
    "thoughts",
    "shared",
    "handoffs",
    "auto",
    `${sessionId}.yaml`
  );
  if (!fs.existsSync(MINI_HANDOFF_SCRIPT)) {
    return { ok: false, outputPath, error: `generate_mini_handoff.py not found at ${MINI_HANDOFF_SCRIPT}` };
  }
  const result = spawnSync("python3", [
    MINI_HANDOFF_SCRIPT,
    "--jsonl",
    transcriptPath,
    "--session-id",
    sessionId,
    "--project-dir",
    projectDir,
    "--output",
    outputPath,
    "--format",
    "yaml"
  ], { encoding: "utf-8", timeout: 1e4 });
  if (result.status !== 0) {
    return {
      ok: false,
      outputPath,
      error: (result.stderr || result.error?.message || `exit ${result.status}`).toString().slice(0, 300)
    };
  }
  return { ok: true, outputPath };
}
function spawnNarrativeHandoff(transcriptPath, sessionId, projectDir) {
  const outputPath = path.join(
    projectDir,
    "thoughts",
    "shared",
    "handoffs",
    "auto",
    `${sessionId}.narrative.yaml`
  );
  const today = (/* @__PURE__ */ new Date()).toISOString().slice(0, 10);
  const sessionName = path.basename(projectDir);
  const prompt = NARRATIVE_PROMPT_TEMPLATE.replace("{TRANSCRIPT_PATH}", transcriptPath).replace("{OUTPUT_PATH}", outputPath).replace("{SESSION_NAME}", sessionName).replace("{SESSION_UUID}", sessionId).replace("{DATE}", today);
  const logDir = path.join(process.env.HOME || "", ".claude", "cache", "pre-compact-narrative");
  fs.mkdirSync(logDir, { recursive: true });
  const logPath = path.join(logDir, `${sessionId}.log`);
  const logFd = fs.openSync(logPath, "a");
  const child = spawn("claude", ["-p", prompt, "--permission-mode", "auto"], {
    detached: true,
    stdio: ["ignore", logFd, logFd],
    env: { ...process.env, CLAUDE_PRECOMPACT_CHILD: "1" }
  });
  child.unref();
  return outputPath;
}
async function main() {
  const input = JSON.parse(await readStdin());
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const pushContext = getMemoryPushContext();
  const lines = [];
  const ledgerDir = path.join(projectDir, "thoughts", "ledgers");
  let ledgerPath = null;
  if (fs.existsSync(ledgerDir)) {
    const ledgerFiles = fs.readdirSync(ledgerDir).filter((f) => f.startsWith("CONTINUITY_CLAUDE-") && f.endsWith(".md"));
    if (ledgerFiles.length > 0) {
      const mostRecent = ledgerFiles.sort((a, b) => {
        const statA = fs.statSync(path.join(ledgerDir, a));
        const statB = fs.statSync(path.join(ledgerDir, b));
        return statB.mtime.getTime() - statA.mtime.getTime();
      })[0];
      ledgerPath = path.join(ledgerDir, mostRecent);
    }
  }
  if (input.trigger !== "auto") {
    const msg = ledgerPath ? `[PreCompact:manual] Consider updating ledger: ${path.basename(ledgerPath)}` : `[PreCompact:manual] Compaction starting.`;
    console.log(JSON.stringify({ continue: true, systemMessage: msg + pushContext }));
    return;
  }
  if (input.transcript_path && fs.existsSync(input.transcript_path)) {
    const mech = runMechanicalHandoff(input.transcript_path, input.session_id, projectDir);
    if (mech.ok) {
      lines.push(`[PreCompact:auto] Mechanical handoff \u2192 ${path.relative(projectDir, mech.outputPath)}`);
    } else {
      lines.push(`[PreCompact:auto] Mechanical handoff failed: ${mech.error}`);
    }
    if (process.env.CLAUDE_PRECOMPACT_NARRATIVE === "1" && !process.env.CLAUDE_PRECOMPACT_CHILD) {
      try {
        const narrativePath = spawnNarrativeHandoff(input.transcript_path, input.session_id, projectDir);
        lines.push(`[PreCompact:auto] Narrative handoff spawned \u2192 ${path.relative(projectDir, narrativePath)} (async)`);
      } catch (err) {
        lines.push(`[PreCompact:auto] Narrative spawn failed: ${err.message}`);
      }
    }
  } else {
    lines.push(`[PreCompact:auto] No transcript available \u2014 skipping handoff generation.`);
  }
  if (ledgerPath) {
    const briefSummary = generateAutoSummary(projectDir, input.session_id);
    if (briefSummary) {
      appendToLedger(ledgerPath, briefSummary);
      lines.push(`[PreCompact:auto] Appended summary to ${path.basename(ledgerPath)}`);
    }
  }
  const output = {
    continue: true,
    systemMessage: lines.join("\n") + pushContext
  };
  console.log(JSON.stringify(output));
}
function generateAutoSummary(projectDir, sessionId) {
  const timestamp = (/* @__PURE__ */ new Date()).toISOString();
  const lines = [];
  const cacheDir = path.join(projectDir, ".claude", "tsc-cache", sessionId || "default");
  const editedFilesPath = path.join(cacheDir, "edited-files.log");
  let editedFiles = [];
  if (fs.existsSync(editedFilesPath)) {
    const content = fs.readFileSync(editedFilesPath, "utf-8");
    editedFiles = [...new Set(
      content.split("\n").filter((line) => line.trim()).map((line) => {
        const parts = line.split(":");
        return parts[1]?.replace(projectDir + "/", "") || "";
      }).filter((f) => f)
    )];
  }
  const gitClaudeDir = path.join(projectDir, ".git", "claude", "branches");
  let buildAttempts = { passed: 0, failed: 0 };
  if (fs.existsSync(gitClaudeDir)) {
    try {
      const branches = fs.readdirSync(gitClaudeDir);
      for (const branch of branches) {
        const attemptsFile = path.join(gitClaudeDir, branch, "attempts.jsonl");
        if (fs.existsSync(attemptsFile)) {
          const content = fs.readFileSync(attemptsFile, "utf-8");
          content.split("\n").filter((l) => l.trim()).forEach((line) => {
            try {
              const attempt = JSON.parse(line);
              if (attempt.type === "build_pass") buildAttempts.passed++;
              if (attempt.type === "build_fail") buildAttempts.failed++;
            } catch {
            }
          });
        }
      }
    } catch {
    }
  }
  if (editedFiles.length === 0 && buildAttempts.passed === 0 && buildAttempts.failed === 0) {
    return null;
  }
  lines.push(`
## Session Auto-Summary (${timestamp})`);
  if (editedFiles.length > 0) {
    lines.push(`- Files changed: ${editedFiles.slice(0, 10).join(", ")}${editedFiles.length > 10 ? ` (+${editedFiles.length - 10} more)` : ""}`);
  }
  if (buildAttempts.passed > 0 || buildAttempts.failed > 0) {
    lines.push(`- Build/test: ${buildAttempts.passed} passed, ${buildAttempts.failed} failed`);
  }
  return lines.join("\n");
}
function appendToLedger(ledgerPath, summary) {
  try {
    let content = fs.readFileSync(ledgerPath, "utf-8");
    const stateMatch = content.match(/## State\n/);
    if (stateMatch) {
      const nowMatch = content.match(/(\n-\s*Now:)/);
      if (nowMatch && nowMatch.index) {
        content = content.slice(0, nowMatch.index) + summary + content.slice(nowMatch.index);
      } else {
        const nextSection = content.indexOf("\n## ", content.indexOf("## State") + 1);
        if (nextSection > 0) {
          content = content.slice(0, nextSection) + summary + "\n" + content.slice(nextSection);
        } else {
          content += summary;
        }
      }
    } else {
      content += summary;
    }
    fs.writeFileSync(ledgerPath, content);
  } catch (err) {
  }
}
async function readStdin() {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.on("data", (chunk) => data += chunk);
    process.stdin.on("end", () => resolve(data));
  });
}
main().catch(console.error);
