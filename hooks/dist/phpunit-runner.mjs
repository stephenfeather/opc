// src/phpunit-runner.ts
import { readFileSync, existsSync } from "fs";

// src/shared/stdin.ts
import { fstatSync, readSync } from "fs";
var CHUNK_SIZE = 64 * 1024;
var EAGAIN_SLEEP_MS = 5;
var DEFAULT_MAX_IDLE_MS = 1e4;
var MAX_IDLE_ENV = "HOOK_STDIN_MAX_IDLE_MS";
function resolveMaxIdleMs(explicit) {
  if (explicit !== void 0) return explicit;
  const fromEnv = process.env[MAX_IDLE_ENV];
  if (fromEnv !== void 0 && fromEnv !== "" && Number.isFinite(Number(fromEnv))) {
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

// src/phpunit-runner.ts
import { execSync } from "child_process";
import * as path from "path";
/*!
 * PHP Test Runner Hook (PostToolUse)
 *
 * Runs composer-defined test scripts after Edit/Write of PHP source files.
 * Detects available scripts from composer.json (test, phpcs, phpstan).
 * Skips vendor, node_modules, and non-PHP files.
 * Reports test results as additional context.
 */
function getComposerScripts(projectDir) {
  const composerPath = path.join(projectDir, "composer.json");
  if (!existsSync(composerPath)) return /* @__PURE__ */ new Set();
  try {
    const composer = JSON.parse(readFileSync(composerPath, "utf-8"));
    return new Set(Object.keys(composer.scripts || {}));
  } catch {
    return /* @__PURE__ */ new Set();
  }
}
function runCommand(cmd, cwd, timeout) {
  try {
    const result = execSync(cmd, {
      cwd,
      timeout,
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"]
    });
    return { ok: true, output: result };
  } catch (err) {
    const execErr = err;
    return { ok: false, output: (execErr.stdout || "") + (execErr.stderr || "") };
  }
}
function extractPhpunitSummary(output) {
  const lines = output.trim().split("\n");
  const summaryLine = lines.find(
    (l) => l.includes("OK (") || l.includes("FAILURES!") || l.includes("ERRORS!")
  );
  const countsLine = lines.find(
    (l) => /Tests:\s+\d+/.test(l) || /\d+ tests?, \d+ assertions?/.test(l)
  );
  if (summaryLine?.includes("OK")) {
    return countsLine || summaryLine;
  }
  const failures = [];
  const failedTests = lines.filter((l) => /^\d+\)\s/.test(l.trim()));
  for (const f of failedTests.slice(0, 5)) {
    failures.push(`  ${f.trim()}`);
  }
  const parts = [countsLine || summaryLine || "Tests failed"];
  if (failures.length > 0) {
    parts.push(...failures);
  }
  return parts.join("\n");
}
function extractPhpcsSummary(output) {
  const lines = output.trim().split("\n");
  const foundLine = lines.find((l) => l.includes("FOUND"));
  if (foundLine) return foundLine.trim();
  if (output.includes("No violations")) return "No violations";
  return lines[lines.length - 1]?.trim() || "Check complete";
}
async function main() {
  const input = JSON.parse(readStdinSync());
  if (input.tool_name !== "Edit" && input.tool_name !== "Write") {
    console.log("{}");
    return;
  }
  const filePath = input.tool_input?.file_path || input.tool_response?.filePath || input.tool_response?.file_path;
  if (!filePath || typeof filePath !== "string") {
    console.log("{}");
    return;
  }
  if (!filePath.endsWith(".php")) {
    console.log("{}");
    return;
  }
  if (filePath.includes("/vendor/") || filePath.includes("/node_modules/")) {
    console.log("{}");
    return;
  }
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const scripts = getComposerScripts(projectDir);
  if (scripts.size === 0) {
    console.log("{}");
    return;
  }
  const results = [];
  if (scripts.has("phpcs")) {
    const phpcs = runCommand("composer phpcs -- --no-colors -q 2>&1", projectDir, 3e4);
    if (phpcs.ok) {
      results.push("phpcs: OK");
    } else {
      results.push(`phpcs: ${extractPhpcsSummary(phpcs.output)}`);
    }
  }
  if (scripts.has("test")) {
    const test = runCommand(
      "XDEBUG_MODE=off composer test -- --no-coverage --colors=never 2>&1",
      projectDir,
      12e4
    );
    if (test.ok) {
      results.push(`phpunit: ${extractPhpunitSummary(test.output)}`);
    } else {
      results.push(`phpunit: FAILED
${extractPhpunitSummary(test.output)}`);
    }
  }
  if (results.length === 0) {
    console.log("{}");
    return;
  }
  const output = {
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      additionalContext: results.join("\n")
    }
  };
  console.log(JSON.stringify(output));
}
main().catch(() => console.log("{}"));
