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

// src/import-error-detector.ts
/*!
 * Import Error Detector - PostToolUse hook that detects Python import errors
 * and suggests the /dependency-preflight skill.
 *
 * Runs on Bash tool output, matches patterns like:
 * - ModuleNotFoundError: No module named 'X'
 * - ImportError: cannot import name 'Y'
 * - No module named 'Z'
 *
 * Returns a system reminder suggesting the skill when errors are detected.
 */
var IMPORT_ERROR_PATTERNS = [
  /ModuleNotFoundError:\s*No module named\s*['"]?(\w+)['"]?/i,
  /ImportError:\s*cannot import name\s*['"]?(\w+)['"]?/i,
  /ImportError:\s*No module named\s*['"]?(\w+)['"]?/i,
  /No module named\s*['"]?(\w+)['"]?/i,
  /ModuleNotFoundError/i,
  /circular import/i
];
function detectImportError(output) {
  for (const pattern of IMPORT_ERROR_PATTERNS) {
    const match = pattern.exec(output);
    if (match) {
      return {
        detected: true,
        module: match[1] || void 0
      };
    }
  }
  return { detected: false };
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
  if (input.tool_name !== "Bash") {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const textToCheck = [input.tool_output, input.error].filter(Boolean).join("\n");
  if (!textToCheck) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const result = detectImportError(textToCheck);
  if (result.detected) {
    const moduleName = result.module ? ` (module: ${result.module})` : "";
    const output = {
      result: "continue",
      message: `
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
\u{1F527} IMPORT ERROR DETECTED${moduleName}
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501

Consider using /dependency-preflight skill to diagnose:

1. Check Python version: uv run python --version
2. Check if installed: uv pip show ${result.module || "<module>"}
3. Verify import: uv run python -c "import ${result.module || "<module>"}"

Or invoke the skill: /dependency-preflight
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501`
    };
    console.log(JSON.stringify(output));
  } else {
    console.log(JSON.stringify({ result: "continue" }));
  }
}
main().catch(() => {
  console.log(JSON.stringify({ result: "continue" }));
});
