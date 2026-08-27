#!/usr/bin/env node

// src/pre-tool-use-broadcast.ts
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

// src/pre-tool-use-broadcast.ts
import { spawnSync } from "child_process";
import { join } from "path";
var SAFE_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;
async function main() {
  const input = readStdinSync();
  JSON.parse(input);
  const swarmId = process.env.SWARM_ID;
  if (!swarmId) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (!SAFE_ID_PATTERN.test(swarmId)) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const agentId = process.env.AGENT_ID || "unknown";
  if (agentId !== "unknown" && !SAFE_ID_PATTERN.test(agentId)) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const dbPath = join(
    projectDir,
    ".claude",
    "cache",
    "agentica-coordination",
    "coordination.db"
  );
  if (!existsSync(dbPath)) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  try {
    const query = `
import sqlite3
import json
import sys

db_path = sys.argv[1]
swarm_id = sys.argv[2]
agent_id = sys.argv[3]

conn = sqlite3.connect(db_path)
# Set busy_timeout to prevent indefinite blocking (Finding 3: STARVATION_FINDINGS.md)
conn.execute("PRAGMA busy_timeout = 5000")
conn.execute("PRAGMA journal_mode = WAL")
conn.row_factory = sqlite3.Row
cursor = conn.execute('''
    SELECT sender_agent, broadcast_type, payload, created_at
    FROM broadcasts
    WHERE swarm_id = ? AND sender_agent != ?
    ORDER BY created_at DESC
    LIMIT 10
''', (swarm_id, agent_id))

broadcasts = []
for row in cursor.fetchall():
    broadcasts.append({
        'sender': row['sender_agent'],
        'type': row['broadcast_type'],
        'payload': json.loads(row['payload']),
        'time': row['created_at']
    })

print(json.dumps(broadcasts))
`;
    const result = spawnSync("python3", ["-c", query, dbPath, swarmId, agentId], {
      encoding: "utf-8",
      maxBuffer: 1024 * 1024
    });
    if (result.status !== 0) {
      console.log(JSON.stringify({ result: "continue" }));
      return;
    }
    const broadcasts = JSON.parse(result.stdout.trim() || "[]");
    if (broadcasts.length > 0) {
      let contextMessage = "\n--- SWARM BROADCASTS ---\n";
      for (const b of broadcasts) {
        contextMessage += `[${b.type.toUpperCase()}] from ${b.sender}:
`;
        contextMessage += `  ${JSON.stringify(b.payload)}
`;
      }
      contextMessage += "------------------------\n";
      console.log(JSON.stringify({
        result: "continue",
        message: contextMessage
      }));
    } else {
      console.log(JSON.stringify({ result: "continue" }));
    }
  } catch (err) {
    console.error("Broadcast query error:", err);
    console.log(JSON.stringify({ result: "continue" }));
  }
}
main().catch((err) => {
  console.error("Uncaught error:", err);
  console.log(JSON.stringify({ result: "continue" }));
});
