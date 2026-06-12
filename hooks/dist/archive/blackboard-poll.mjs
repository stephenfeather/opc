// src/blackboard-poll.ts
import * as fs from "fs";
import * as path from "path";
var BLACKBOARD_CACHE_DIR = process.env.BLACKBOARD_CACHE_DIR || "/tmp/claude-blackboard";
var POLL_STATE_DIR = process.env.POLL_STATE_DIR || "/tmp/claude-blackboard/poll-state";
var MAX_ENTRIES_IN_REMINDER = 5;
function getAgentContext() {
  return {
    agentId: process.env.CLAUDE_AGENT_ID || null,
    patternId: process.env.CLAUDE_PATTERN_ID || null,
    projectId: process.env.CLAUDE_PROJECT_ID || null
  };
}
function getLastPollTimestamp(sessionId) {
  const stateFile = path.join(POLL_STATE_DIR, `${sessionId}.json`);
  try {
    if (fs.existsSync(stateFile)) {
      const state = JSON.parse(fs.readFileSync(stateFile, "utf-8"));
      return state.last_poll_timestamp || null;
    }
  } catch {
  }
  return null;
}
function saveLastPollTimestamp(sessionId, timestamp) {
  fs.mkdirSync(POLL_STATE_DIR, { recursive: true });
  const stateFile = path.join(POLL_STATE_DIR, `${sessionId}.json`);
  fs.writeFileSync(
    stateFile,
    JSON.stringify({ last_poll_timestamp: timestamp }, null, 2)
  );
}
function readBlackboardCache(projectId) {
  const cacheFile = path.join(BLACKBOARD_CACHE_DIR, `${projectId}.json`);
  try {
    if (fs.existsSync(cacheFile)) {
      return JSON.parse(fs.readFileSync(cacheFile, "utf-8"));
    }
  } catch {
  }
  return [];
}
function filterEntries(entries, agentId, patternId, sinceTimestamp) {
  const now = Date.now();
  return entries.filter((e) => {
    if (e.ttl_seconds) {
      const entryTime = new Date(e.timestamp).getTime();
      const age = (now - entryTime) / 1e3;
      if (age > e.ttl_seconds) {
        return false;
      }
    }
    if (sinceTimestamp && e.timestamp <= sinceTimestamp) {
      return false;
    }
    if (patternId && e.pattern_id && e.pattern_id !== patternId) {
      return false;
    }
    if (agentId) {
      if (e.scope === "agent") {
        if (e.from_agent !== agentId && e.to_agent !== agentId) {
          return false;
        }
      } else if (e.to_agent && e.to_agent !== agentId) {
        return false;
      }
    }
    return true;
  });
}
function formatEntry(entry) {
  const parts = [`[${entry.type.toUpperCase()}]`];
  if (entry.from_agent) {
    parts.push(`from:${entry.from_agent}`);
  }
  if (entry.to_agent) {
    parts.push(`to:${entry.to_agent}`);
  }
  if (entry.formal) {
    parts.push(`| ${entry.formal}`);
  } else if (entry.prose) {
    parts.push(`| ${entry.prose}`);
  }
  if (entry.data && Object.keys(entry.data).length > 0) {
    parts.push(`| data=${JSON.stringify(entry.data)}`);
  }
  if (entry.artifacts && entry.artifacts.length > 0) {
    parts.push(`| files=${entry.artifacts.join(",")}`);
  }
  return parts.join(" ");
}
function formatSystemReminder(entries) {
  if (entries.length === 0) {
    return "";
  }
  const priorityOrder = {
    critical: 0,
    high: 1,
    normal: 2,
    low: 3
  };
  const sorted = [...entries].sort((a, b) => {
    const pa = priorityOrder[a.priority] ?? 2;
    const pb = priorityOrder[b.priority] ?? 2;
    if (pa !== pb) return pa - pb;
    return a.timestamp.localeCompare(b.timestamp);
  });
  const limited = sorted.slice(0, MAX_ENTRIES_IN_REMINDER);
  const lines = ["<blackboard>"];
  for (const entry of limited) {
    lines.push(formatEntry(entry));
  }
  if (sorted.length > MAX_ENTRIES_IN_REMINDER) {
    lines.push(`... and ${sorted.length - MAX_ENTRIES_IN_REMINDER} more entries`);
  }
  lines.push("</blackboard>");
  return lines.join("\n");
}
async function readStdin() {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => {
      resolve(data);
    });
  });
}
async function main() {
  const input = JSON.parse(await readStdin());
  const output = { result: "continue" };
  const { agentId, patternId, projectId } = getAgentContext();
  if (!projectId) {
    console.log(JSON.stringify(output));
    return;
  }
  const sinceTimestamp = getLastPollTimestamp(input.session_id);
  const entries = readBlackboardCache(projectId);
  const filtered = filterEntries(entries, agentId, patternId, sinceTimestamp);
  if (filtered.length > 0) {
    const reminder = formatSystemReminder(filtered);
    output.message = `Blackboard update:
${reminder}`;
    const maxTimestamp = filtered.reduce(
      (max, e) => e.timestamp > max ? e.timestamp : max,
      ""
    );
    if (maxTimestamp) {
      saveLastPollTimestamp(input.session_id, maxTimestamp);
    }
  }
  console.log(JSON.stringify(output));
}
main().catch((err) => {
  console.error("Hook error:", err);
  console.log(JSON.stringify({ result: "continue" }));
});
