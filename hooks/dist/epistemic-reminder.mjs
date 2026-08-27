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

// src/epistemic-reminder.ts
/*!
 * Epistemic Reminder Hook (PostToolUse:Grep|Read)
 *
 * Injects epistemic discipline reminders after Grep and Read results.
 * Referenced by claim-verification.md rule.
 *
 * - After Read: reminds to update ? INFERRED claims to ✓ VERIFIED
 * - After Grep (existence checks or file-list mode): warns that grep
 *   results are not proof and Read is required for verification
 * - After Grep (other): lighter reminder
 */
function main() {
  let input;
  try {
    const stdinContent = readStdinSync();
    input = JSON.parse(stdinContent);
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (input.tool_name !== "Grep" && input.tool_name !== "Read") {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  let reminder;
  if (input.tool_name === "Read") {
    const filePath = input.tool_input?.file_path || "file";
    const fileName = filePath.split("/").pop() || "file";
    reminder = `<system-reminder>
\u2713 Read ${fileName} - note findings. Update any prior ? INFERRED claims to \u2713 VERIFIED if confirmed.
</system-reminder>`;
  } else {
    const pattern = input.tool_input?.pattern || "";
    const outputMode = input.tool_input?.output_mode || "files_with_matches";
    const existencePatterns = [
      /try.*catch/i,
      /error.*handl/i,
      /exist/i,
      /missing/i,
      /lack/i,
      /without/i,
      /no.*found/i
    ];
    const isExistenceCheck = existencePatterns.some((p) => p.test(pattern));
    const isFileListMode = outputMode === "files_with_matches";
    if (isExistenceCheck || isFileListMode) {
      reminder = `<epistemic-reminder>
\u26A0\uFE0F GREP RESULTS ARE NOT PROOF

Before claiming "X exists" or "X doesn't exist":
1. READ the actual file(s) to verify
2. Grep may miss: different naming, regex mismatch, file not searched
3. Grep may false-match: substring matches, comments, strings

REQUIRED: Use Read tool on relevant files before making existence claims.
Mark claims as: \u2713 VERIFIED (read file) | ? INFERRED (grep only) | \u2717 UNCERTAIN
</epistemic-reminder>`;
    } else {
      reminder = `<epistemic-reminder>
Grep results are evidence, not proof. Verify with Read before claiming.
</epistemic-reminder>`;
    }
  }
  const output = {
    result: "continue",
    additionalContext: reminder
  };
  console.log(JSON.stringify(output));
}
main();
export {
  main
};
