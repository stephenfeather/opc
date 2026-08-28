// src/session-start-tldr-cache.ts
import { readFileSync, existsSync } from "fs";

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

// src/session-start-tldr-cache.ts
import { join } from "path";
import { spawn } from "child_process";
/*!
 * SessionStart Hook: TLDR Cache Warming (Async)
 *
 * On session startup, triggers cache warming in a detached background process.
 * Returns immediately to avoid blocking startup.
 *
 * The daemon will warm the cache asynchronously. First TLDR command
 * will use the warmed cache or trigger on-demand indexing.
 */
function readStdin() {
  return readStdinSync();
}
function getCacheAge(projectDir) {
  const metaPath = join(projectDir, ".claude", "cache", "tldr", "meta.json");
  if (!existsSync(metaPath)) return void 0;
  try {
    const meta = JSON.parse(readFileSync(metaPath, "utf-8"));
    const cachedAt = new Date(meta.cached_at);
    return Math.round((Date.now() - cachedAt.getTime()) / (1e3 * 60 * 60));
  } catch {
    return void 0;
  }
}
function isCacheStale(projectDir) {
  const cacheDir = join(projectDir, ".claude", "cache", "tldr");
  if (!existsSync(cacheDir)) return true;
  const age = getCacheAge(projectDir);
  return age === void 0 || age > 24;
}
function main() {
  let input;
  try {
    input = JSON.parse(readStdin());
  } catch {
    console.log("{}");
    return;
  }
  if (!["startup", "resume"].includes(input.source)) {
    console.log("{}");
    return;
  }
  const projectDir = process.env.CLAUDE_PROJECT_DIR || input.cwd;
  if (isCacheStale(projectDir)) {
    const child = spawn("tldr", ["daemon", "warm", "--project", projectDir], {
      detached: true,
      stdio: "ignore",
      shell: process.platform === "win32"
      // Shell needed on Windows
    });
    child.unref();
  }
  console.log("{}");
}
main();
