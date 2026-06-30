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

  it("still emits continue (exit 0) on a DB-writing event when no DB URL is set", () => {
    // With no DB URL set, the #265 backend gate resolves to sqlite and
    // updateWorkingOnDetached no-ops gracefully (no spawn, no throw); the hook
    // must still emit continue and never fail the tool call.
    run(
      {
        session_id: SESSION,
        tool_name: "TaskCreate",
        tool_input: { activeForm: "Backporting" },
        tool_response: "Task #1 created",
      },
      home,
    );
    const res = run(
      { session_id: SESSION, tool_name: "TaskUpdate", tool_input: { taskId: "1", status: "in_progress" } },
      home,
    );
    expect(res.status).toBe(0);
    expect(JSON.parse(res.stdout.trim()).result).toBe("continue");
  });

  it("falls back to USERPROFILE when HOME is unset (Windows)", () => {
    const env: Record<string, string> = {
      ...(process.env as Record<string, string>),
      USERPROFILE: home,
      CONTINUOUS_CLAUDE_DB_URL: "",
      DATABASE_URL: "",
      OPC_POSTGRES_URL: "",
    };
    delete env.HOME;
    const res = spawnSync("node", [BIN], {
      encoding: "utf-8",
      input: JSON.stringify({
        session_id: SESSION,
        tool_name: "TaskCreate",
        tool_input: { activeForm: "Fixing 65" },
        tool_response: "Task #2 created",
      }),
      env,
    });
    expect(res.status).toBe(0);
    const cache = JSON.parse(fs.readFileSync(cacheFile(home), "utf-8"));
    expect(cache.tasks["2"]).toBe("Fixing 65");
  });

  it("tolerates a corrupt cache (non-string task values) without crashing", () => {
    const dir = path.join(home, ".claude", "cache", "working-on");
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(
      cacheFile(home),
      JSON.stringify({ tasks: { "1": { nested: "obj" }, "2": "Real" }, currentId: null }),
    );
    const res = run(
      { session_id: SESSION, tool_name: "TaskUpdate", tool_input: { taskId: "1", status: "in_progress" } },
      home,
    );
    expect(res.status).toBe(0);
    expect(JSON.parse(res.stdout.trim()).result).toBe("continue");
    // the non-string entry was dropped on read; the good one survives the rewrite
    const cache = JSON.parse(fs.readFileSync(cacheFile(home), "utf-8"));
    expect(cache.tasks["1"]).toBeUndefined();
    expect(cache.tasks["2"]).toBe("Real");
  });

  it("ignores an invalid session id", () => {
    const res = run({ session_id: "../etc", tool_name: "TaskCreate", tool_input: {} }, home);
    expect(res.status).toBe(0);
    expect(fs.existsSync(path.join(home, ".claude", "cache", "working-on"))).toBe(false);
  });
});
