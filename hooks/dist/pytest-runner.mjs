// src/pytest-runner.ts
import { readFileSync, existsSync } from "fs";
import { execSync } from "child_process";
import * as path from "path";
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
  const input = JSON.parse(readFileSync(0, "utf-8"));
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
      stdio: ["pipe", "pipe", "pipe"]
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
