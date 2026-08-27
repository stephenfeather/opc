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

// src/failure-detection.ts
/*!
 * Failure Detection Hook (PostToolUse:Bash|Task)
 *
 * Detects errors in Bash and Task tool responses and suggests
 * documentation searches for resolution.
 *
 * - Bash: checks exit_code !== 0 and extracts error from stderr/stdout
 * - Task: scans response text for error/exception/crash patterns
 * - Extracts specific Python exception context (ModuleNotFoundError, etc.)
 * - Suggests Nia documentation search with the error context
 */
var TASK_ERROR_PATTERNS = [
  /\berror\b/i,
  /\bfailed\b/i,
  /\bexception\b/i,
  /\bcrash(ed)?\b/i,
  /\btimeout\b/i,
  /\babort(ed)?\b/i,
  /\bpanic\b/i,
  /\bfatal\b/i
];
var ERROR_CONTEXT_PATTERNS = [
  /ModuleNotFoundError:\s*No module named\s*['"]?(\w+)['"]?/i,
  /ImportError:\s*(.+)/i,
  /TypeError:\s*(.+)/i,
  /ValueError:\s*(.+)/i,
  /AttributeError:\s*(.+)/i,
  /NameError:\s*(.+)/i,
  /SyntaxError:\s*(.+)/i,
  /RuntimeError:\s*(.+)/i,
  /KeyError:\s*(.+)/i,
  /FileNotFoundError:\s*(.+)/i,
  /PermissionError:\s*(.+)/i,
  /ConnectionError:\s*(.+)/i,
  /OSError:\s*(.+)/i,
  /Error:\s*(.+)/i,
  /error:\s*(.+)/i,
  /failed:\s*(.+)/i,
  /FAILED:\s*(.+)/i
];
function isBashFailure(response) {
  if (typeof response === "object" && response !== null) {
    const bashResponse = response;
    if (typeof bashResponse.exit_code === "number" && bashResponse.exit_code !== 0) {
      const stderr = bashResponse.stderr || "";
      const stdout = bashResponse.stdout || "";
      return { failed: true, errorText: stderr || stdout };
    }
  }
  return { failed: false, errorText: "" };
}
function isTaskFailure(response) {
  let text = "";
  if (typeof response === "string") {
    text = response;
  } else if (typeof response === "object" && response !== null) {
    text = JSON.stringify(response);
  }
  for (const pattern of TASK_ERROR_PATTERNS) {
    if (pattern.test(text)) {
      return { failed: true, errorText: text };
    }
  }
  return { failed: false, errorText: "" };
}
function extractErrorContext(errorText, toolInput) {
  for (const pattern of ERROR_CONTEXT_PATTERNS) {
    const match = pattern.exec(errorText);
    if (match) {
      const context = match[1] || match[0];
      return context.substring(0, 100).trim();
    }
  }
  const firstLine = errorText.split("\n")[0] || "";
  if (firstLine.length > 100) {
    return firstLine.substring(0, 100).trim();
  }
  if (toolInput.command && typeof toolInput.command === "string") {
    return `command failed: ${toolInput.command.substring(0, 50)}`;
  }
  return "execution failed";
}
function buildNiaSearchCommand(errorContext) {
  const escapedContext = errorContext.replace(/'/g, "'\\''").replace(/"/g, '\\"');
  return `uv run python -m runtime.harness scripts/nia_docs.py search universal "${escapedContext}" --limit 5`;
}
async function main() {
  let input;
  try {
    const rawInput = readStdinSync();
    input = JSON.parse(rawInput);
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (input.tool_name !== "Bash" && input.tool_name !== "Task") {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  let failed = false;
  let errorText = "";
  if (input.tool_name === "Bash") {
    const result = isBashFailure(input.tool_response);
    failed = result.failed;
    errorText = result.errorText;
  } else if (input.tool_name === "Task") {
    const result = isTaskFailure(input.tool_response);
    failed = result.failed;
    errorText = result.errorText;
  }
  if (!failed) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const errorContext = extractErrorContext(errorText, input.tool_input);
  const niaCommand = buildNiaSearchCommand(errorContext);
  const output = {
    result: "continue",
    message: `
---
**Build/Execution Failure Detected**

Consider searching documentation for help:
\`\`\`bash
${niaCommand}
\`\`\`

Error context: ${errorContext.substring(0, 200)}
---`
  };
  console.log(JSON.stringify(output));
}
main().catch(() => {
  console.log(JSON.stringify({ result: "continue" }));
});
