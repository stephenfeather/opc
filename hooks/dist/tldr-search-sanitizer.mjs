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

// src/tldr-search-sanitizer.ts
/*!
 * PreToolUse Hook: TLDR Search Sanitizer
 *
 * Removes unsupported "--project" flag from `tldr search` Bash commands.
 * This prevents invalid argument errors like:
 *   tldr search "pattern" /path --project /path
 */
function tokenizeCommand(command) {
  const tokens = [];
  const regex = /"([^"\\]*(\\.[^"\\]*)*)"|'([^'\\]*(\\.[^'\\]*)*)'|`[^`]*`|\\\S+|\S+/g;
  let match;
  while ((match = regex.exec(command)) !== null) {
    tokens.push(match[0]);
  }
  return tokens;
}
function isTldrSearch(tokens) {
  if (tokens.length < 2) return false;
  if (tokens[0] === "tldr" && tokens[1] === "search") return true;
  if (tokens[0] === "uv" && tokens[1] === "run" && tokens[2] === "tldr" && tokens[3] === "search") {
    return true;
  }
  return false;
}
function sanitizeTldrSearch(command) {
  const tokens = tokenizeCommand(command);
  if (!isTldrSearch(tokens)) {
    return { changed: false, sanitized: command };
  }
  const sanitizedTokens = [];
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    if (token === "--project") {
      i += 1;
      continue;
    }
    if (token.startsWith("--project=")) {
      continue;
    }
    sanitizedTokens.push(token);
  }
  const sanitized = sanitizedTokens.join(" ");
  const changed = sanitized !== command;
  return { changed, sanitized };
}
async function main() {
  let input;
  try {
    input = JSON.parse(readStdinSync());
  } catch {
    console.log("{}");
    return;
  }
  if (input.tool_name !== "Bash") {
    console.log("{}");
    return;
  }
  const command = input.tool_input?.command;
  if (!command || typeof command !== "string") {
    console.log("{}");
    return;
  }
  const { changed, sanitized } = sanitizeTldrSearch(command);
  if (!changed) {
    console.log("{}");
    return;
  }
  const output = {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "allow",
      updatedInput: {
        ...input.tool_input,
        command: sanitized
      }
    },
    systemMessage: "\u26A0\uFE0F Removed unsupported `--project` from `tldr search` command."
  };
  console.log(JSON.stringify(output));
}
main().catch(() => {
  console.log("{}");
});
