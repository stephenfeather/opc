// src/session-crash-recovery.ts
import { existsSync as existsSync3, writeFileSync, mkdirSync } from "fs";
import { readFileSync as readFileSync3 } from "fs";
import { execSync } from "child_process";
import * as path from "path";

// src/transcript-parser.ts
import * as fs from "fs";
/*!
 * Transcript Parser Module
 *
 * Parses JSONL transcript files from Claude Code sessions and extracts
 * high-signal data for use by PreCompact hooks and auto-handoff generation.
 */
function parseTranscript(transcriptPath) {
  const summary = {
    lastTodos: [],
    recentToolCalls: [],
    lastAssistantMessage: "",
    filesModified: [],
    errorsEncountered: []
  };
  if (!fs.existsSync(transcriptPath)) {
    return summary;
  }
  const content = fs.readFileSync(transcriptPath, "utf-8");
  const lines = content.split("\n").filter((line) => line.trim());
  const allToolCalls = [];
  const modifiedFiles = /* @__PURE__ */ new Set();
  const errors = [];
  let lastTodoState = [];
  let lastAssistant = "";
  for (const line of lines) {
    try {
      const raw = JSON.parse(line);
      const tl = raw;
      if (tl.type === "assistant" && tl.message?.content) {
        const blocks = Array.isArray(tl.message.content) ? tl.message.content : [];
        for (const block of blocks) {
          if (block.type === "text" && block.text) {
            lastAssistant = block.text;
          }
          if (block.type === "tool_use" && block.name) {
            const toolName = block.name;
            const toolInput = block.input || {};
            const toolCall = {
              name: toolName,
              timestamp: tl.timestamp,
              input: toolInput,
              success: true
            };
            if (toolName === "TodoWrite" || toolName === "TaskCreate") {
              const input = toolInput;
              if (input?.todos) {
                lastTodoState = input.todos.map((t, idx) => ({
                  id: t.id || `todo-${idx}`,
                  content: t.content || "",
                  status: t.status || "pending"
                }));
              }
            }
            if (toolName === "Edit" || toolName === "Write" || toolName === "NotebookEdit") {
              const input = toolInput;
              const filePath = input?.file_path || input?.path;
              if (filePath && typeof filePath === "string") {
                modifiedFiles.add(filePath);
              }
            }
            if (toolName === "Bash") {
              const input = toolInput;
              if (input?.command) {
                toolCall.input = { command: input.command };
              }
            }
            allToolCalls.push(toolCall);
          }
        }
      }
      if (tl.type === "user" && tl.message?.content) {
        const blocks = Array.isArray(tl.message.content) ? tl.message.content : [];
        for (const block of blocks) {
          if (block.type === "tool_result") {
            if (block.is_error && allToolCalls.length > 0) {
              allToolCalls[allToolCalls.length - 1].success = false;
              const errText = typeof block.content === "string" ? block.content : "Tool returned error";
              errors.push(errText.substring(0, 200));
            }
          }
        }
      }
      const entry = raw;
      if (!tl.message) {
        if (entry.role === "assistant" && typeof entry.content === "string") {
          lastAssistant = entry.content;
        } else if (entry.type === "assistant" && typeof entry.content === "string") {
          lastAssistant = entry.content;
        }
        if (entry.tool_name || entry.type === "tool_use" && !tl.message) {
          const toolName = entry.tool_name || entry.name;
          if (toolName) {
            const toolCall = {
              name: toolName,
              timestamp: entry.timestamp,
              input: entry.tool_input,
              success: true
            };
            if (toolName === "TodoWrite") {
              const input = entry.tool_input;
              if (input?.todos) {
                lastTodoState = input.todos.map((t, idx) => ({
                  id: t.id || `todo-${idx}`,
                  content: t.content || "",
                  status: t.status || "pending"
                }));
              }
            }
            if (toolName === "Edit" || toolName === "Write") {
              const input = entry.tool_input;
              const filePath = input?.file_path || input?.path;
              if (filePath && typeof filePath === "string") {
                modifiedFiles.add(filePath);
              }
            }
            if (toolName === "Bash") {
              const input = entry.tool_input;
              if (input?.command) {
                toolCall.input = { command: input.command };
              }
            }
            allToolCalls.push(toolCall);
          }
        }
        if (entry.type === "tool_result" || entry.tool_result !== void 0) {
          const result = entry.tool_result;
          if (result) {
            const exitCode = result.exit_code ?? result.exitCode;
            if (exitCode !== void 0 && exitCode !== 0) {
              if (allToolCalls.length > 0) {
                allToolCalls[allToolCalls.length - 1].success = false;
              }
              const errorMsg = result.stderr || result.error || "Command failed";
              const lastTool = allToolCalls[allToolCalls.length - 1];
              const command = lastTool?.input?.command || "unknown command";
              errors.push(`${command}: ${errorMsg.substring(0, 200)}`);
            }
          }
          if (entry.error) {
            errors.push(entry.error.substring(0, 200));
            if (allToolCalls.length > 0) {
              allToolCalls[allToolCalls.length - 1].success = false;
            }
          }
        }
      }
    } catch {
      continue;
    }
  }
  summary.lastTodos = lastTodoState;
  summary.recentToolCalls = allToolCalls.slice(-5);
  summary.lastAssistantMessage = lastAssistant.substring(0, 500);
  summary.filesModified = Array.from(modifiedFiles);
  summary.errorsEncountered = errors.slice(-5);
  return summary;
}
function generateAutoHandoff(summary, sessionName) {
  const timestamp = (/* @__PURE__ */ new Date()).toISOString();
  const dateOnly = timestamp.split("T")[0];
  const lines = [];
  const inProgress = summary.lastTodos.filter((t) => t.status === "in_progress");
  const pending = summary.lastTodos.filter((t) => t.status === "pending");
  const completed = summary.lastTodos.filter((t) => t.status === "completed");
  const currentTask = inProgress[0]?.content || pending[0]?.content || "Continue from auto-compact";
  const goalSummary = completed.length > 0 ? `Completed ${completed.length} task(s) before auto-compact` : "Session auto-compacted";
  lines.push("---");
  lines.push(`session: ${sessionName}`);
  lines.push(`date: ${dateOnly}`);
  lines.push("status: partial");
  lines.push("outcome: PARTIAL_PLUS");
  lines.push("---");
  lines.push("");
  lines.push(`goal: ${goalSummary}`);
  lines.push(`now: ${currentTask}`);
  lines.push("test: # No test command captured");
  lines.push("");
  lines.push("done_this_session:");
  if (completed.length > 0) {
    completed.forEach((t) => {
      lines.push(`  - task: "${t.content.replace(/"/g, '\\"')}"`);
      lines.push("    files: []");
    });
  } else {
    lines.push('  - task: "Session started"');
    lines.push("    files: []");
  }
  lines.push("");
  lines.push("blockers:");
  if (summary.errorsEncountered.length > 0) {
    summary.errorsEncountered.slice(0, 3).forEach((e) => {
      const safeError = e.replace(/"/g, '\\"').substring(0, 100);
      lines.push(`  - "${safeError}"`);
    });
  } else {
    lines.push("  []");
  }
  lines.push("");
  lines.push("questions:");
  if (pending.length > 0) {
    pending.slice(0, 3).forEach((t) => {
      lines.push(`  - "Resume: ${t.content.replace(/"/g, '\\"')}"`);
    });
  } else {
    lines.push("  []");
  }
  lines.push("");
  lines.push("decisions:");
  lines.push('  - auto_compact: "Context limit reached, auto-compacted"');
  lines.push("");
  lines.push("findings:");
  lines.push(`  - tool_calls: "${summary.recentToolCalls.length} recent tool calls"`);
  lines.push(`  - files_modified: "${summary.filesModified.length} files changed"`);
  lines.push("");
  lines.push("worked:");
  const successfulTools = summary.recentToolCalls.filter((t) => t.success);
  if (successfulTools.length > 0) {
    lines.push(`  - "${successfulTools.map((t) => t.name).join(", ")} completed successfully"`);
  } else {
    lines.push("  []");
  }
  lines.push("");
  lines.push("failed:");
  const failedTools = summary.recentToolCalls.filter((t) => !t.success);
  if (failedTools.length > 0) {
    lines.push(`  - "${failedTools.map((t) => t.name).join(", ")} encountered errors"`);
  } else {
    lines.push("  []");
  }
  lines.push("");
  lines.push("next:");
  if (inProgress.length > 0) {
    lines.push(`  - "Continue: ${inProgress[0].content.replace(/"/g, '\\"')}"`);
  }
  if (pending.length > 0) {
    pending.slice(0, 2).forEach((t) => {
      lines.push(`  - "${t.content.replace(/"/g, '\\"')}"`);
    });
  }
  if (inProgress.length === 0 && pending.length === 0) {
    lines.push('  - "Review session state and continue"');
  }
  lines.push("");
  lines.push("files:");
  lines.push("  created: []");
  lines.push("  modified:");
  if (summary.filesModified.length > 0) {
    summary.filesModified.slice(0, 10).forEach((f) => {
      lines.push(`    - "${f}"`);
    });
  } else {
    lines.push("    []");
  }
  return lines.join("\n");
}
var isMainModule = import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.log("Usage: npx tsx transcript-parser.ts <transcript-path> [session-name]");
    process.exit(1);
  }
  const transcriptPath = args[0];
  const sessionName = args[1] || "test-session";
  console.log(`Parsing transcript: ${transcriptPath}`);
  const summary = parseTranscript(transcriptPath);
  console.log("\n--- Summary ---");
  console.log(JSON.stringify(summary, null, 2));
  console.log("\n--- Auto-Handoff ---");
  console.log(generateAutoHandoff(summary, sessionName));
}

// src/shared/db-utils-pg.ts
import { spawnSync } from "child_process";

// src/shared/opc-path.ts
import { existsSync as existsSync2, readFileSync as readFileSync2 } from "fs";
import { join } from "path";
function getOpcDirFromConfig() {
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  if (!homeDir) return null;
  const configPath = join(homeDir, ".claude", "opc.json");
  if (!existsSync2(configPath)) return null;
  try {
    const content = readFileSync2(configPath, "utf-8");
    const config = JSON.parse(content);
    const opcDir = config.opc_dir;
    if (opcDir && typeof opcDir === "string" && existsSync2(opcDir)) {
      return opcDir;
    }
  } catch {
  }
  return null;
}
function getOpcDir() {
  const envOpcDir = process.env.CLAUDE_OPC_DIR;
  if (envOpcDir && existsSync2(envOpcDir)) {
    return envOpcDir;
  }
  const configOpcDir = getOpcDirFromConfig();
  if (configOpcDir) {
    return configOpcDir;
  }
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const localOpc = join(projectDir, "opc");
  if (existsSync2(localOpc)) {
    return localOpc;
  }
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  if (homeDir) {
    const globalClaude = join(homeDir, ".claude");
    const globalScripts = join(globalClaude, "scripts", "core");
    if (existsSync2(globalScripts)) {
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

# Add opc to path for imports (read from env to avoid code injection)
_opc_dir = os.environ.get('_OPC_DIR')
if not _opc_dir:
    raise RuntimeError('_OPC_DIR environment variable not set - must be called via runPgQuery()')
sys.path.insert(0, _opc_dir)
os.chdir(_opc_dir)

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
        CONTINUOUS_CLAUDE_DB_URL: getPgConnectionString(),
        _OPC_DIR: opcDir
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
function getCrashedSessions(project) {
  const pythonCode = `
import asyncpg
import os
import json

project = sys.argv[1]
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL', 'postgresql://claude:claude_dev@localhost:5432/continuous_claude')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        rows = await conn.fetch('''
            SELECT id, project, claude_session_id, transcript_path, pid, started_at, last_heartbeat
            FROM sessions
            WHERE project = $1
              AND exited_at IS NULL
            ORDER BY started_at DESC
        ''', project)

        sessions = []
        for row in rows:
            sessions.append({
                'id': row['id'],
                'project': row['project'],
                'claude_session_id': row['claude_session_id'],
                'transcript_path': row['transcript_path'],
                'pid': row['pid'],
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
  const result = runPgQuery(pythonCode, [project]);
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
function markSessionsAcknowledged(sessionIds) {
  if (sessionIds.length === 0) return { success: true };
  const pythonCode = `
import asyncpg
import os
import json

session_ids = json.loads(sys.argv[1])
pg_url = os.environ.get('CONTINUOUS_CLAUDE_DB_URL') or os.environ.get('DATABASE_URL', 'postgresql://claude:claude_dev@localhost:5432/continuous_claude')

async def main():
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('''
            UPDATE sessions SET exited_at = NOW()
            WHERE id = ANY($1::text[])
        ''', session_ids)
        print('ok')
    finally:
        await conn.close()

asyncio.run(main())
`;
  const result = runPgQuery(pythonCode, [JSON.stringify(sessionIds)]);
  return { success: result.success && result.stdout === "ok" };
}

// src/session-crash-recovery.ts
/*!
 * Session Crash Recovery Hook (SessionStart)
 *
 * Detects when a previous session crashed (no clean exit recorded in DB)
 * and creates a recovery handoff from the old transcript.
 *
 * Flow:
 * 1. Query PostgreSQL sessions table for crashed sessions on this project
 *    (exited_at IS NULL AND last_heartbeat is stale)
 * 2. If found → parse the old transcript using transcript-parser
 * 3. Create a recovery handoff YAML in the standard format
 * 4. Mark crashed sessions as acknowledged in DB
 * 5. Inform the user via system message
 */
function getSessionName(projectDir) {
  try {
    const handoffsDir = path.join(projectDir, "thoughts", "shared", "handoffs");
    if (existsSync3(handoffsDir)) {
      const result = execSync(
        `ls -td "${handoffsDir}"/*/ 2>/dev/null | head -1 | xargs basename`,
        { encoding: "utf-8", timeout: 5e3, stdio: ["pipe", "pipe", "pipe"] }
      ).trim();
      if (result) return result;
    }
  } catch {
  }
  try {
    const result = execSync(
      `basename "$(git worktree list --porcelain 2>/dev/null | head -1 | sed 's/^worktree //')" 2>/dev/null`,
      { cwd: projectDir, encoding: "utf-8", timeout: 5e3, stdio: ["pipe", "pipe", "pipe"] }
    ).trim();
    if (result) return result;
  } catch {
  }
  try {
    const result = execSync("git branch --show-current 2>/dev/null", {
      cwd: projectDir,
      encoding: "utf-8",
      timeout: 5e3,
      stdio: ["pipe", "pipe", "pipe"]
    }).trim();
    if (result) return result;
  } catch {
  }
  return path.basename(projectDir);
}
function createRecoveryHandoff(crashed, projectDir) {
  if (!crashed.transcript_path || !existsSync3(crashed.transcript_path)) {
    return null;
  }
  const summary = parseTranscript(crashed.transcript_path);
  if (summary.recentToolCalls.length === 0 && summary.filesModified.length === 0) {
    return null;
  }
  const sessionName = getSessionName(projectDir);
  let handoffContent = generateAutoHandoff(summary, sessionName);
  handoffContent = handoffContent.replace("outcome: PARTIAL_PLUS", "outcome: PARTIAL_MINUS").replace("status: partial", "status: crashed").replace(
    'auto_compact: "Context limit reached, auto-compacted"',
    'crash_recovery: "Previous session ended unexpectedly (CLI crash/hang)"'
  );
  const handoffDir = path.join(projectDir, "thoughts", "shared", "handoffs", sessionName);
  mkdirSync(handoffDir, { recursive: true });
  const now = /* @__PURE__ */ new Date();
  const dateStr = now.toISOString().split("T")[0];
  const timeStr = `${String(now.getHours()).padStart(2, "0")}-${String(now.getMinutes()).padStart(2, "0")}`;
  const filename = `${dateStr}_${timeStr}_crash-recovery.yaml`;
  const handoffPath = path.join(handoffDir, filename);
  writeFileSync(handoffPath, handoffContent);
  return `thoughts/shared/handoffs/${sessionName}/${filename}`;
}
function isProcessAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
function isSessionCrashed(session) {
  if (session.pid) {
    return !isProcessAlive(session.pid);
  }
  if (!session.last_heartbeat) return true;
  const heartbeat = new Date(session.last_heartbeat).getTime();
  const staleThreshold = 5 * 60 * 1e3;
  return Date.now() - heartbeat > staleThreshold;
}
async function main() {
  const input = JSON.parse(readFileSync3(0, "utf-8"));
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  if (input.type && input.type !== "startup") {
    console.log("{}");
    return;
  }
  const result = getCrashedSessions(projectDir);
  if (!result.success || result.sessions.length === 0) {
    console.log("{}");
    return;
  }
  const crashedSessions = result.sessions.filter(isSessionCrashed);
  if (crashedSessions.length === 0) {
    console.log("{}");
    return;
  }
  const crashed = crashedSessions[0];
  const handoffPath = createRecoveryHandoff(crashed, projectDir);
  const crashedIds = crashedSessions.map((s) => s.id);
  markSessionsAcknowledged(crashedIds);
  if (handoffPath) {
    const contextMsg = [
      "Previous session ended unexpectedly (crash/hang).",
      `Recovery handoff created: ${handoffPath}`,
      `Resume with: /resume_handoff ${handoffPath}`
    ].join("\n");
    const output = {
      systemMessage: `\u26A0\uFE0F Crash detected! Recovery handoff: ${handoffPath}`,
      hookSpecificOutput: {
        hookEventName: "SessionStart",
        additionalContext: contextMsg
      }
    };
    console.log(JSON.stringify(output));
  } else {
    console.log("{}");
  }
}
main().catch(() => console.log("{}"));
