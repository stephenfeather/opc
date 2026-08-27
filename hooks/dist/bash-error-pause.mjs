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

// src/bash-error-pause.ts
/*!
 * Bash Error Pause Hook (PostToolUse:Bash)
 *
 * Scans Bash tool responses for warnings and errors, then injects
 * a reminder to verify the root cause before explaining to the user.
 */
var WARNING_PATTERNS = [
  /\bwarn(ing)?\b/i,
  /\bdeprecated\b/i,
  /\bWARN\b/
];
var ERROR_PATTERNS = [
  /\berror\b/i,
  /\bfailed\b/i,
  /\bfailure\b/i,
  /\bexception\b/i,
  /\bfatal\b/i,
  /\bpanic\b/i,
  /\bsegfault\b/i,
  /\bsegmentation fault\b/i,
  /\baborted\b/i,
  /\btraceback\b/i,
  /\bERROR\b/,
  /\bFAILED\b/,
  /\bFATAL\b/,
  /exit code [1-9]\d*/i,
  /returned? [1-9]\d*/i
];
var FALSE_POSITIVE_PATTERNS = [
  /0 errors?\b/i,
  /no errors?\b/i,
  /error[_-]?handl/i,
  /error[_-]?messag/i,
  /error[_-]?code/i,
  /error[_-]?type/i,
  /error[_-]?class/i,
  /on_?error/i,
  /if.*error/i,
  /catch.*error/i,
  /throw.*error/i,
  /console\.(warn|error)/i,
  /stderr/i,
  /\bwarning:\s*0\b/i,
  /0 warning/i,
  /no warning/i
];
function extractResponseText(response) {
  if (response && typeof response === "object") {
    const resp = response;
    if (typeof resp.stderr === "string") return resp.stderr;
    return "";
  }
  return "";
}
function hasNonFalsePositiveMatch(text, patterns) {
  for (const pattern of patterns) {
    const match = pattern.exec(text);
    if (!match) continue;
    const lineStart = text.lastIndexOf("\n", match.index) + 1;
    const lineEnd = text.indexOf("\n", match.index);
    const line = text.slice(lineStart, lineEnd === -1 ? void 0 : lineEnd);
    const isFalsePositive = FALSE_POSITIVE_PATTERNS.some((fp) => fp.test(line));
    if (!isFalsePositive) return true;
  }
  return false;
}
function isRecordInput(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function main() {
  let input;
  try {
    const stdinContent = readStdinSync();
    const parsed = JSON.parse(stdinContent);
    if (!isRecordInput(parsed)) {
      console.log(JSON.stringify({ result: "continue" }));
      return;
    }
    input = parsed;
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (input.tool_name !== "Bash") {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const text = extractResponseText(input.tool_response);
  if (!text.trim()) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const hasError = hasNonFalsePositiveMatch(text, ERROR_PATTERNS);
  const hasWarning = !hasError && hasNonFalsePositiveMatch(text, WARNING_PATTERNS);
  if (!hasError && !hasWarning) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const severity = hasError ? "ERROR" : "WARNING";
  const output = {
    result: "continue",
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      additionalContext: `STOP: ${severity} detected in Bash output. Verify the cause before explaining it to the user. Do NOT guess \u2014 read the error, check assumptions, trace the root cause.`
    }
  };
  console.log(JSON.stringify(output));
}
main();
export {
  main
};
