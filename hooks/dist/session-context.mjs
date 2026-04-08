// src/session-context.ts
import { readFileSync, writeFileSync, mkdirSync, renameSync } from "fs";
import { dirname } from "path";
function checkMemoryHealth(pgRegistrationSucceeded, pidFilePath) {
  const pgHealthy = pgRegistrationSucceeded;
  let daemonRunning = false;
  try {
    const pidContent = readFileSync(pidFilePath, "utf-8").trim();
    const pid = parseInt(pidContent, 10);
    if (!isNaN(pid) && pid > 0) {
      process.kill(pid, 0);
      daemonRunning = true;
    }
  } catch {
    daemonRunning = false;
  }
  return { pgHealthy, daemonRunning };
}
function formatHealthWarnings(health) {
  const warnings = [];
  if (!health.pgHealthy) {
    warnings.push("- PostgreSQL: unreachable");
  }
  if (!health.daemonRunning) {
    warnings.push("- Memory daemon: not running");
  }
  if (warnings.length === 0) return null;
  return `Health warnings:
${warnings.join("\n")}`;
}
function getPendingTasksSummary(tasksFilePath) {
  try {
    const content = readFileSync(tasksFilePath, "utf-8");
    if (!content.trim()) return null;
    const titles = content.split("\n").filter((line) => line.startsWith("## ")).map((line) => line.slice(3).trim());
    if (titles.length === 0) return null;
    const MAX_SHOWN = 3;
    const shown = titles.slice(0, MAX_SHOWN);
    const suffix = titles.length > MAX_SHOWN ? ", ..." : "";
    return `Pending tasks (${titles.length}): ${shown.join(", ")}${suffix}`;
  } catch {
    return null;
  }
}
function formatPeerMessage(peers) {
  if (peers.length === 0) return null;
  const lines = peers.map(
    (s) => `- ${s.id}: ${s.working_on || "working..."}`
  );
  return `Active peer sessions (${peers.length}):
${lines.join("\n")}`;
}
function readPeerCache(cachePath, project, ttlSeconds) {
  try {
    const raw = readFileSync(cachePath, "utf-8");
    const data = JSON.parse(raw);
    if (data.project !== project) return null;
    const age = (Date.now() - new Date(data.cached_at).getTime()) / 1e3;
    if (age >= ttlSeconds) return null;
    return data.sessions;
  } catch {
    return null;
  }
}
function writePeerCache(cachePath, project, sessions) {
  try {
    const dir = dirname(cachePath);
    mkdirSync(dir, { recursive: true });
    const data = {
      cached_at: (/* @__PURE__ */ new Date()).toISOString(),
      project,
      sessions
    };
    const tmpPath = cachePath + ".tmp." + process.pid;
    writeFileSync(tmpPath, JSON.stringify(data), { encoding: "utf-8" });
    renameSync(tmpPath, cachePath);
  } catch {
  }
}
export {
  checkMemoryHealth,
  formatHealthWarnings,
  formatPeerMessage,
  getPendingTasksSummary,
  readPeerCache,
  writePeerCache
};
