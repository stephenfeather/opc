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

// src/import-validator.ts
import { basename } from "path";

// src/daemon-client.ts
import { existsSync, readFileSync, writeFileSync, unlinkSync } from "fs";
import { execSync, spawnSync } from "child_process";
import { join, resolve } from "path";
import { tmpdir } from "os";
import * as net from "net";
import * as crypto from "crypto";
/*!
 * Shared TypeScript client for TLDR daemon.
 *
 * Used by all TypeScript hooks to query the TLDR daemon instead of
 * spawning individual `tldr` processes. This provides:
 * - Faster queries (daemon holds indexes in memory)
 * - Reduced process overhead
 * - Consistent timeout handling
 * - Auto-start capability
 * - Graceful degradation when indexing
 */
function resolveProjectDir(projectDir) {
  return resolve(projectDir);
}
function getLockPath(projectDir) {
  const resolvedPath = resolveProjectDir(projectDir);
  const hash = crypto.createHash("md5").update(resolvedPath).digest("hex").substring(0, 8);
  return `${tmpdir()}/tldr-${hash}.lock`;
}
function getPidPath(projectDir) {
  const resolvedPath = resolveProjectDir(projectDir);
  const hash = crypto.createHash("md5").update(resolvedPath).digest("hex").substring(0, 8);
  return `${tmpdir()}/tldr-${hash}.pid`;
}
function isDaemonProcessRunning(projectDir) {
  const pidPath = getPidPath(projectDir);
  if (!existsSync(pidPath)) return false;
  try {
    const pid = parseInt(readFileSync(pidPath, "utf-8").trim(), 10);
    if (isNaN(pid) || pid <= 0) return false;
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
function tryAcquireLock(projectDir) {
  const lockPath = getLockPath(projectDir);
  try {
    if (existsSync(lockPath)) {
      const lockContent = readFileSync(lockPath, "utf-8");
      const lockTime = parseInt(lockContent, 10);
      if (!isNaN(lockTime) && Date.now() - lockTime < 3e4) {
        return false;
      }
      try {
        unlinkSync(lockPath);
      } catch {
      }
    }
    writeFileSync(lockPath, Date.now().toString(), { flag: "wx" });
    return true;
  } catch {
    return false;
  }
}
function releaseLock(projectDir) {
  try {
    unlinkSync(getLockPath(projectDir));
  } catch {
  }
}
var QUERY_TIMEOUT = 3e3;
var queryDeadlineAt = null;
var MIN_QUERY_BUDGET_MS = 100;
var TRACK_WRITE_TIMEOUT_MS = 250;
function remainingQueryBudget() {
  if (queryDeadlineAt === null) return null;
  return queryDeadlineAt - Date.now();
}
function budgetExhausted() {
  const remaining = remainingQueryBudget();
  return remaining !== null && remaining < MIN_QUERY_BUDGET_MS;
}
function budgetClamp(defaultMs) {
  const remaining = remainingQueryBudget();
  if (remaining === null) return defaultMs;
  return Math.max(0, Math.min(defaultMs, remaining));
}
function isTimeoutError(err) {
  return err?.killed === true || err?.code === "ETIMEDOUT" || err?.signal === "SIGTERM";
}
function sleepSync(ms) {
  if (ms <= 0) return;
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}
function getConnectionInfo(projectDir) {
  const resolvedPath = resolveProjectDir(projectDir);
  const hash = crypto.createHash("md5").update(resolvedPath).digest("hex").substring(0, 8);
  if (process.platform === "win32") {
    const port = 49152 + parseInt(hash, 16) % 1e4;
    return { type: "tcp", host: "127.0.0.1", port };
  } else {
    return { type: "unix", path: `${tmpdir()}/tldr-${hash}.sock` };
  }
}
function getStatusFile(projectDir) {
  const statusPath = join(projectDir, ".tldr", "status");
  if (existsSync(statusPath)) {
    try {
      return readFileSync(statusPath, "utf-8").trim();
    } catch {
      return null;
    }
  }
  return null;
}
function isIndexing(projectDir) {
  return getStatusFile(projectDir) === "indexing";
}
function isDaemonReachable(projectDir) {
  const connInfo = getConnectionInfo(projectDir);
  if (connInfo.type === "tcp") {
    try {
      const testSocket = new net.Socket();
      testSocket.setTimeout(100);
      let connected = false;
      testSocket.on("connect", () => {
        connected = true;
        testSocket.destroy();
      });
      testSocket.on("error", () => {
        testSocket.destroy();
      });
      testSocket.connect(connInfo.port, connInfo.host);
      const end = Date.now() + 200;
      while (Date.now() < end && !connected) {
      }
      return connected;
    } catch {
      return false;
    }
  } else {
    if (!existsSync(connInfo.path)) {
      return false;
    }
    if (isDaemonProcessRunning(projectDir)) {
      try {
        execSync(`echo '{"cmd":"ping"}' | nc -U "${connInfo.path}"`, {
          encoding: "utf-8",
          timeout: Math.max(50, budgetClamp(1e3)),
          stdio: ["pipe", "pipe", "pipe"]
        });
        return true;
      } catch {
        return true;
      }
    }
    try {
      execSync(`echo '{"cmd":"ping"}' | nc -U "${connInfo.path}"`, {
        encoding: "utf-8",
        timeout: Math.max(50, budgetClamp(500)),
        stdio: ["pipe", "pipe", "pipe"]
      });
      return true;
    } catch (err) {
      if (isTimeoutError(err)) {
        return true;
      }
      try {
        unlinkSync(connInfo.path);
      } catch {
      }
      return false;
    }
  }
}
function tryStartDaemon(projectDir) {
  if (process.env.TLDR_NO_AUTOSTART === "1") {
    return false;
  }
  if (budgetExhausted()) {
    return false;
  }
  try {
    if (isDaemonProcessRunning(projectDir)) {
      return true;
    }
    if (isDaemonReachable(projectDir)) {
      return true;
    }
    if (!tryAcquireLock(projectDir)) {
      const lockWaitMs = budgetClamp(5e3);
      const start = Date.now();
      while (Date.now() - start < lockWaitMs) {
        if (isDaemonProcessRunning(projectDir) || isDaemonReachable(projectDir)) {
          return true;
        }
        sleepSync(Math.min(100, lockWaitMs - (Date.now() - start)));
      }
      return isDaemonProcessRunning(projectDir) || isDaemonReachable(projectDir);
    }
    try {
      if (budgetExhausted()) {
        return false;
      }
      const opcDir = process.env.CLAUDE_OPC_DIR || join(projectDir, "opc");
      const tldrPath = join(opcDir, "packages", "tldr-code");
      let started = false;
      if (existsSync(tldrPath)) {
        const result = spawnSync("uv", ["run", "tldr", "daemon", "start", "--project", projectDir], {
          timeout: Math.max(MIN_QUERY_BUDGET_MS, budgetClamp(1e4)),
          stdio: "ignore",
          cwd: tldrPath
        });
        started = result.status === 0;
      }
      if (!started && !process.env.TLDR_DEV) {
        spawnSync("tldr", ["daemon", "start", "--project", projectDir], {
          timeout: Math.max(MIN_QUERY_BUDGET_MS, budgetClamp(5e3)),
          stdio: "ignore"
        });
      }
      const reachWaitMs = budgetClamp(1e4);
      const start = Date.now();
      while (Date.now() - start < reachWaitMs) {
        if (isDaemonReachable(projectDir)) {
          sleepSync(budgetClamp(1e3));
          return true;
        }
        sleepSync(Math.min(100, reachWaitMs - (Date.now() - start)));
      }
      return isDaemonReachable(projectDir);
    } finally {
      releaseLock(projectDir);
    }
  } catch {
    return false;
  }
}
function queryDaemonSync(query, projectDir) {
  if (budgetExhausted()) {
    return { status: "unavailable", error: "query deadline exceeded" };
  }
  if (isIndexing(projectDir)) {
    return {
      indexing: true,
      status: "indexing",
      message: "Daemon is still indexing, results may be incomplete"
    };
  }
  const connInfo = getConnectionInfo(projectDir);
  if (!isDaemonReachable(projectDir)) {
    if (!tryStartDaemon(projectDir)) {
      return { status: "unavailable", error: "Daemon not running and could not start" };
    }
  }
  if (budgetExhausted()) {
    return { status: "unavailable", error: "query deadline exceeded" };
  }
  const queryTimeout = Math.max(MIN_QUERY_BUDGET_MS, budgetClamp(QUERY_TIMEOUT));
  try {
    const input = JSON.stringify(query);
    let result;
    if (connInfo.type === "tcp") {
      const psCommand = `
        $client = New-Object System.Net.Sockets.TcpClient('${connInfo.host}', ${connInfo.port})
        $stream = $client.GetStream()
        $writer = New-Object System.IO.StreamWriter($stream)
        $reader = New-Object System.IO.StreamReader($stream)
        $writer.WriteLine('${input.replace(/'/g, "''")}')
        $writer.Flush()
        $response = $reader.ReadLine()
        $client.Close()
        Write-Output $response
      `.trim();
      result = execSync(`powershell -Command "${psCommand.replace(/"/g, '\\"')}"`, {
        encoding: "utf-8",
        timeout: queryTimeout
      });
    } else {
      result = execSync(`echo '${input}' | nc -U "${connInfo.path}"`, {
        encoding: "utf-8",
        timeout: queryTimeout
      });
    }
    return JSON.parse(result.trim());
  } catch (err) {
    if (isTimeoutError(err)) {
      return { status: "error", error: "timeout" };
    }
    if (err.message?.includes("ECONNREFUSED") || err.message?.includes("ENOENT")) {
      return { status: "unavailable", error: "Daemon not running" };
    }
    return { status: "error", error: err.message || "Unknown error" };
  }
}
function trackHookActivitySync(hookName, projectDir, success = true, metrics = {}) {
  try {
    const connInfo = getConnectionInfo(projectDir);
    const payload = JSON.stringify({ cmd: "track", hook: hookName, success, metrics }) + "\n";
    let sock;
    if (connInfo.type === "tcp") {
      sock = net.createConnection({ host: connInfo.host, port: connInfo.port });
    } else {
      if (!connInfo.path || !existsSync(connInfo.path)) {
        return;
      }
      sock = net.createConnection(connInfo.path);
    }
    sock.setTimeout(TRACK_WRITE_TIMEOUT_MS);
    sock.on("timeout", () => sock.destroy());
    sock.on("error", () => sock.destroy());
    sock.on("connect", () => {
      sock.write(payload, () => sock.destroy());
    });
  } catch {
  }
}

// src/import-validator.ts
/*!
 * Import Validator Hook (PostToolUse)
 *
 * After Write/Edit, checks if imports reference symbols that exist.
 * Uses TLDR daemon for fast symbol lookup (replaces CLI spawning).
 */
function tldrSearch(pattern, projectDir = ".") {
  try {
    const response = queryDaemonSync({ cmd: "search", pattern }, projectDir);
    if (response.indexing || response.status === "unavailable" || response.status === "error") {
      return [];
    }
    if (response.results) {
      return response.results;
    }
    return [];
  } catch {
    return [];
  }
}
function extractPythonImports(code) {
  const imports = [];
  const fromImportRe = /from\s+([\w.]+)\s+import\s+([^#\n]+)/g;
  let match;
  while ((match = fromImportRe.exec(code)) !== null) {
    const module = match[1];
    const symbols = match[2].split(",").map((s) => s.trim().split(" as ")[0].trim()).filter((s) => s && s !== "*");
    if (symbols.length > 0) {
      imports.push({ module, symbols });
    }
  }
  return imports;
}
function checkSymbolExists(symbol) {
  const projectDir = process.env.CLAUDE_PROJECT_DIR || ".";
  const funcResults = tldrSearch(`def ${symbol}`, projectDir);
  if (funcResults.length > 0) {
    return { exists: true, location: `${funcResults[0].file}:${funcResults[0].line}` };
  }
  const classResults = tldrSearch(`class ${symbol}`, projectDir);
  if (classResults.length > 0) {
    return { exists: true, location: `${classResults[0].file}:${classResults[0].line}` };
  }
  return { exists: false };
}
async function main() {
  const input = JSON.parse(readStdinSync());
  if (input.tool_name !== "Write" && input.tool_name !== "Edit") {
    console.log("{}");
    return;
  }
  const code = input.tool_input.content || input.tool_input.new_string || "";
  if (!code) {
    console.log("{}");
    return;
  }
  const filePath = input.tool_input.file_path || "";
  if (!filePath.endsWith(".py")) {
    console.log("{}");
    return;
  }
  const imports = extractPythonImports(code);
  const warnings = [];
  for (const imp of imports) {
    for (const symbol of imp.symbols) {
      const check = checkSymbolExists(symbol);
      if (check.exists && check.location) {
        const expectedModule = imp.module.replace(/\./g, "/");
        if (!check.location.includes(expectedModule)) {
          const actualFile = basename(check.location.split(":")[0]);
          warnings.push(`${symbol}: imported from ${imp.module} but defined in ${actualFile}`);
        }
      }
    }
  }
  if (warnings.length === 0) {
    console.log("{}");
    return;
  }
  const output = {
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      additionalContext: `[Import check]
${warnings.join("\n")}`
    }
  };
  const projectDir = process.env.CLAUDE_PROJECT_DIR || ".";
  trackHookActivitySync("import-validator", projectDir, true, {
    writes_validated: 1,
    warnings_found: warnings.length
  });
  console.log(JSON.stringify(output));
}
main().catch(() => console.log("{}"));
