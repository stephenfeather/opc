import { describe, expect, it, beforeEach } from "vitest";
import { spawnSync } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BIN = path.resolve(__dirname, "..", "..", "dist", "working-on-sync.mjs");

const SESSION = "abcdef12-3456-7890-abcd-ef1234567890";

function run(payload: object, home: string) {
  return spawnSync("node", [BIN], {
    encoding: "utf-8",
    input: JSON.stringify(payload),
    // Isolate cache writes and starve the detached DB write (no DB URL).
    env: {
      ...process.env,
      HOME: home,
      CONTINUOUS_CLAUDE_DB_URL: "",
      DATABASE_URL: "",
      OPC_POSTGRES_URL: "",
    },
  });
}

function cacheFile(home: string) {
  return path.join(home, ".claude", "cache", "working-on", `${SESSION}.json`);
}

describe("working-on-sync CLI", () => {
  let home: string;
  beforeEach(() => {
    home = fs.mkdtempSync(path.join(os.tmpdir(), "wos-"));
  });

  it("always emits {result: continue} and never throws", () => {
    const res = run({ session_id: SESSION, tool_name: "Read", tool_input: {} }, home);
    expect(res.status).toBe(0);
    expect(JSON.parse(res.stdout.trim()).result).toBe("continue");
  });

  it("TaskCreate persists the label to the session cache", () => {
    const res = run(
      {
        session_id: SESSION,
        tool_name: "TaskCreate",
        tool_input: { activeForm: "Fixing 65" },
        tool_response: "Task #2 created successfully",
      },
      home,
    );
    expect(res.status).toBe(0);
    const cache = JSON.parse(fs.readFileSync(cacheFile(home), "utf-8"));
    expect(cache.tasks["2"]).toBe("Fixing 65");
  });

  it("TaskCreate then TaskUpdate->in_progress sets currentId across invocations", () => {
    run(
      {
        session_id: SESSION,
        tool_name: "TaskCreate",
        tool_input: { activeForm: "Backporting" },
        tool_response: "Task #1 created",
      },
      home,
    );
    run(
      { session_id: SESSION, tool_name: "TaskUpdate", tool_input: { taskId: "1", status: "in_progress" } },
      home,
    );
    const cache = JSON.parse(fs.readFileSync(cacheFile(home), "utf-8"));
    expect(cache.currentId).toBe("1");
    expect(cache.tasks["1"]).toBe("Backporting");
  });

  it("ignores an invalid session id", () => {
    const res = run({ session_id: "../etc", tool_name: "TaskCreate", tool_input: {} }, home);
    expect(res.status).toBe(0);
    expect(fs.existsSync(path.join(home, ".claude", "cache", "working-on"))).toBe(false);
  });
});
