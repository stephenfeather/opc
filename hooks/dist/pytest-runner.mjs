// src/pytest-runner.ts
import { existsSync } from "fs";

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

// src/pytest-runner.ts
import { execSync } from "child_process";
import * as path from "path";
/*!
 * Pytest Runner Hook (PostToolUse)
 *
 * Runs pytest after Edit/Write of Python source files.
 * Skips venv, .venv, vendor, node_modules, and non-Python files.
 * Reports test results as additional context.
 */
function hasPytestConfig(projectDir) {
  return existsSync(path.join(projectDir, "pytest.ini")) || existsSync(path.join(projectDir, "pyproject.toml")) || existsSync(path.join(projectDir, "setup.cfg")) || existsSync(path.join(projectDir, "conftest.py")) || existsSync(path.join(projectDir, "tests", "conftest.py"));
}
function hasUv() {
  try {
    execSync("command -v uv", { encoding: "utf-8", stdio: ["pipe", "pipe", "pipe"] });
    return true;
  } catch {
    return false;
  }
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
  if (!filePath.endsWith(".py") && !filePath.endsWith(".pyx") && !filePath.endsWith(".pyi")) {
    console.log("{}");
    return;
  }
  if (filePath.includes("/venv/") || filePath.includes("/.venv/") || filePath.includes("/vendor/") || filePath.includes("/node_modules/") || filePath.includes("/__pycache__/")) {
    console.log("{}");
    return;
  }
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  if (!hasPytestConfig(projectDir)) {
    console.log("{}");
    return;
  }
  const pytestCmd = hasUv() ? "uv run pytest" : "pytest";
  try {
    const result = execSync(`${pytestCmd} --tb=short -q --no-header 2>&1`, {
      cwd: projectDir,
      timeout: 12e4,
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
      // UV_FROZEN=1: a hook-triggered `uv run pytest` must not re-resolve/rewrite
      // the project's uv.lock as a side effect (issue #71 follow-up).
      env: { ...process.env, UV_FROZEN: "1" }
    });
    const lines = result.trim().split("\n");
    const summaryLine = lines.find(
      (l) => l.includes(" passed") || l.includes("no tests ran")
    );
    const output = {
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: `pytest: ${summaryLine || "All tests passed"}`
      }
    };
    console.log(JSON.stringify(output));
  } catch (err) {
    const execErr = err;
    const combined = (execErr.stdout || "") + (execErr.stderr || "");
    const outputLines = combined.trim().split("\n");
    const failLines = [];
    failLines.push("pytest: TESTS FAILED");
    failLines.push("");
    const summaryLine = outputLines.find(
      (l) => l.includes(" failed") || l.includes(" error")
    );
    if (summaryLine) {
      failLines.push(summaryLine.trim());
    }
    const failedTests = outputLines.filter((l) => l.startsWith("FAILED "));
    for (const test of failedTests.slice(0, 5)) {
      failLines.push(`  ${test.trim()}`);
    }
    const tbLines = outputLines.filter(
      (l) => l.includes("AssertionError") || l.includes("Error:") || l.includes("assert ")
    );
    for (const tb of tbLines.slice(0, 3)) {
      failLines.push(`  ${tb.trim()}`);
    }
    const output = {
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: failLines.join("\n")
      }
    };
    console.log(JSON.stringify(output));
  }
}
main().catch(() => console.log("{}"));
