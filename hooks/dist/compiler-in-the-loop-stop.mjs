// src/compiler-in-the-loop-stop.ts
import { readFileSync, existsSync, unlinkSync } from "fs";

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

// src/compiler-in-the-loop-stop.ts
import { join } from "path";
import { tmpdir } from "os";
/*!
 * Compiler-in-the-Loop Stop Hook
 *
 * Prevents Claude from stopping if there are unresolved Lean errors/sorries.
 * Implements the APOLLO recursive repair pattern.
 */
var STATE_DIR = process.env.CLAUDE_PROJECT_DIR ? join(process.env.CLAUDE_PROJECT_DIR, ".claude", "cache", "lean") : join(tmpdir(), "claude-lean");
var STATE_FILE = join(STATE_DIR, "compiler-state.json");
var MAX_STATE_AGE_MS = 5 * 60 * 1e3;
function readStdin() {
  return readStdinSync();
}
function loadState() {
  if (!existsSync(STATE_FILE)) return null;
  try {
    const state = JSON.parse(readFileSync(STATE_FILE, "utf-8"));
    if (Date.now() - state.timestamp > MAX_STATE_AGE_MS) {
      unlinkSync(STATE_FILE);
      return null;
    }
    return state;
  } catch {
    return null;
  }
}
function clearState() {
  if (existsSync(STATE_FILE)) {
    unlinkSync(STATE_FILE);
  }
}
async function main() {
  const input = JSON.parse(readStdin());
  if (input.stop_hook_active) {
    console.log("{}");
    return;
  }
  const state = loadState();
  if (!state || !state.has_errors) {
    console.log("{}");
    return;
  }
  if (state.session_id !== input.session_id) {
    clearState();
    console.log("{}");
    return;
  }
  let repairPrompt;
  if (state.sorries.length > 0) {
    repairPrompt = `
\u{1F504} APOLLO REPAIR LOOP - Unresolved 'sorry' placeholders

File: ${state.file_path}

The proof has ${state.sorries.length} incomplete part(s):

${state.sorries.join("\n")}

**Your task:**
1. Pick ONE sorry to fix (start with the simplest)
2. Replace 'sorry' with a valid proof:
   - Try tactics: simp, ring, nlinarith, norm_num, exact, apply
   - Or provide explicit proof term
3. Re-run to check if it compiles

Continue fixing until all sorries are resolved.
`;
  } else {
    repairPrompt = `
\u{1F504} APOLLO REPAIR LOOP - Lean Compiler Errors

File: ${state.file_path}

Errors:
${state.errors.slice(0, 2e3)}

**Your task:**
1. Read the error messages carefully
2. If type error: check signatures match
3. If syntax error: check Lean 4 syntax
4. If unknown identifier: check imports
5. Consider using 'sorry' to isolate the failing part, then fix incrementally

Fix the errors and re-write the file.
`;
  }
  console.log(JSON.stringify({
    decision: "block",
    reason: repairPrompt
  }));
}
main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
